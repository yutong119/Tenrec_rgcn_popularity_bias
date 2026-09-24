from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
import torch.nn.functional as F

from tenrec_rgcn_dataset import TenRecFullGraphDataset
from tenrec_rgcn_fullmap import TenRecFullGraphRGCN

from torch import nn
SOURCE_DOMAIN = "QK-video"
TARGET_DOMAIN = "QK-article"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)

def align_source_users_to_target(
        source_dataset: TenRecFullGraphDataset,
        target_dataset: TenRecFullGraphDataset,
        source_user_repr: torch.Tensor,
) -> torch.Tensor:
    """
    Align source user representation to the target user index order
    using raw TenRec user IDs.
    """

    source_indices = []
    missing_users = []

    for raw_uid in target_dataset.idx_to_raw_user:
        raw_uid = int(raw_uid)

        source_idx = source_dataset.raw_user_to_idx.get(raw_uid)

        if source_idx is None:
            missing_users.append(raw_uid)
        else:
            source_indices.append(source_idx)

    if missing_users:
        raise ValueError(
            f"{len(missing_users)} target users are missing "
            "from the source domain."
        )

    source_indices = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=source_user_repr.device,
    )

    aligned = source_user_repr[source_indices]

    if aligned.shape[0] != target_dataset.num_users:
        raise RuntimeError(
            "Aligned source user count does not match target user count."
        )

    return aligned

def load_frozen_source_user_repr(
        args:argparse.Namespace,
        device: torch.device,
        source_dataset: TenRecFullGraphDataset,
        target_dataset: TenRecFullGraphDataset,
) -> torch.Tensor | None:

    if args.transfer_mode == "target_only":
        return None

    source_checkpoint_path = Path(args.source_checkpoint)

    if not source_checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing source checkpoint:\n"
            f"{source_checkpoint_path}"
        )

    checkpoint = torch.load(
        source_checkpoint_path,
        map_location=device,
    )

    source_args = checkpoint["args"]

    if source_args.get("domain") != SOURCE_DOMAIN:
        raise ValueError(
            "Source checkpoint must be trained on QK-video."
        )

    source_embedding_dim = int(
        source_args["embedding_dim"]
    )

    if source_embedding_dim != args.embedding_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"source={source_embedding_dim}, "
            f"target={args.embedding_dim}"
        )

    source_model = TenRecFullGraphRGCN(
        num_users=source_dataset.num_users,
        num_items=source_dataset.num_items,
        embedding_dim=source_embedding_dim,
        num_relations=8,
        num_bases=source_args.get("num_bases"),
        dropout=source_args.get("dropout", 0.0),
    ).to(device)

    source_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    source_model.eval()

    for parameter in source_model.parameters():
        parameter.requires_grad = False

    source_edge_index = (
        source_dataset.edge_index.to(device)
    )

    source_edge_type = (
        source_dataset.edge_type.to(device)
    )

    with torch.no_grad():
        source_user_repr, _ = source_model.encode(
            source_edge_index,
            source_edge_type,
        )

    source_user_repr = (
        source_user_repr.detach()
    )

    aligned = align_source_users_to_target(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_user_repr=source_user_repr,
    )

    if args.transfer_mode == "shuffled":
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            args.seed + 9999
        )

        permutation = torch.randperm(
            target_dataset.num_users,
            generator=generator,
        ).to(device)

        aligned = aligned[permutation]

    return aligned.detach()


@torch.no_grad()
def compute_representation_path_change(
    model: TenRecFullGraphRGCN,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    projector: nn.Module,
    source_user_repr_aligned: torch.Tensor,
) -> float:
    """
    Verify that projected Source context changes Target user
    representations for the same current Target model.
    """

    model_was_training = model.training
    projector_was_training = projector.training

    model.eval()
    projector.eval()

    user_without_source, _ = model.encode(
        edge_index,
        edge_type,
    )

    external_user_context = projector(
        source_user_repr_aligned
    )

    user_with_source, _ = model.encode(
        edge_index,
        edge_type,
        external_user_context=external_user_context,
    )

    mean_change = (
        user_with_source
        - user_without_source
    ).norm(dim=1).mean()

    if model_was_training:
        model.train()

    if projector_was_training:
        projector.train()

    return float(mean_change.item())


def bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """The ONLY training loss in Baseline + Diagnosis v1."""
    return -F.logsigmoid(
        positive_scores - negative_scores
    ).mean()


@torch.no_grad()
def evaluate(
    model: TenRecFullGraphRGCN,
    dataset: TenRecFullGraphDataset,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    device: torch.device,
    split: Literal["val", "test"],
    projector: nn.Module | None = None,
    source_user_repr_aligned: torch.Tensor | None = None,
    k: int = 10,
    user_batch_size: int = 128,
) -> Dict[str, float]:
    """
    Evaluate target-domain recommendation performance and
    post-hoc popularity bias.

    If source_user_repr_aligned is provided, the frozen source-domain
    user representation is projected into the target embedding space
    and injected into the target RGCN input.

    Popularity is diagnosis-only and never affects training or ranking.
    """

    model.eval()

    if projector is not None:
        projector.eval()

    # ------------------------------------------------------------
    # Build external source-domain user context.
    #
    # target_only:
    #     source_user_repr_aligned = None
    #     external_user_context = None
    #
    # correct / shuffled:
    #     source_user_repr_aligned exists
    #     projector maps Source embedding -> Target embedding space
    # ------------------------------------------------------------

    external_user_context = None

    if source_user_repr_aligned is not None:
        if projector is None:
            raise RuntimeError(
                "source_user_repr_aligned is provided, "
                "but projector is None."
            )

        external_user_context = projector(
            source_user_repr_aligned
        )

    # ------------------------------------------------------------
    # Target RGCN encoding.
    #
    # target_only:
    #     equivalent to the original RGCN baseline.
    #
    # cross-domain:
    #     target initial user embedding
    #          +
    #     projected source user context
    #          ↓
    #     Target RGCN
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # One full Target-domain RGCN convolution per optimizer update.
    #
    # Source information enters BEFORE Target RGCN propagation.
    # ------------------------------------------------------------

    user_repr, item_repr = model.encode(
        edge_index,
        edge_type,
        external_user_context=external_user_context,
    )

    if split == "val":
        target_by_user = dataset.val_item_by_user
        other_holdout_by_user = dataset.test_item_by_user

    elif split == "test":
        target_by_user = dataset.test_item_by_user
        other_holdout_by_user = dataset.val_item_by_user

    else:
        raise ValueError(
            "split must be 'val' or 'test'."
        )

    # ------------------------------------------------------------
    # Popularity diagnosis only.
    # ------------------------------------------------------------

    pop_raw = dataset.item_popularity_raw.to(device)
    pop_norm = dataset.item_popularity_norm.to(device)

    profile_pop_raw = (
        dataset.user_profile_popularity(False).to(device)
    )

    profile_pop_norm = (
        dataset.user_profile_popularity(True).to(device)
    )

    tail_mask = dataset.tail_item_mask.to(device)
    head_mask = ~tail_mask

    recall_sum = 0.0
    ndcg_sum = 0.0

    rec_pop_raw_sum = 0.0
    rec_pop_norm_sum = 0.0

    profile_pop_raw_sum = 0.0
    profile_pop_norm_sum = 0.0

    head_slots = 0
    tail_slots = 0
    total_slots = 0

    all_recommended_items: List[torch.Tensor] = []

    user_indices = np.arange(
        dataset.num_users,
        dtype=np.int64,
    )

    for start in range(
        0,
        dataset.num_users,
        user_batch_size,
    ):
        batch_np = user_indices[
            start:start + user_batch_size
        ]

        batch_users = (
            torch.from_numpy(batch_np)
            .long()
            .to(device)
        )

        scores = model.score_user_batch_against_all_items(
            batch_users,
            user_repr,
            item_repr,
        )

        # --------------------------------------------------------
        # Remove training-positive items.
        #
        # Also remove the OTHER held-out item so validation and
        # test positives do not compete against each other.
        # --------------------------------------------------------

        for local_row, user_idx in enumerate(batch_np):

            seen = dataset.train_seen_items_by_user[
                int(user_idx)
            ]

            if seen:
                seen_idx = torch.tensor(
                    list(seen),
                    dtype=torch.long,
                    device=device,
                )

                scores[
                    local_row,
                    seen_idx,
                ] = -torch.inf

            other_item = int(
                other_holdout_by_user[
                    int(user_idx)
                ]
            )

            scores[
                local_row,
                other_item,
            ] = -torch.inf

        topk = torch.topk(
            scores,
            k=min(k, dataset.num_items),
            dim=1,
        ).indices

        all_recommended_items.append(
            topk.detach().cpu()
        )

        # --------------------------------------------------------
        # Recall + NDCG
        # --------------------------------------------------------

        for local_row, user_idx in enumerate(batch_np):

            target = int(
                target_by_user[
                    int(user_idx)
                ]
            )

            matches = (
                topk[local_row] == target
            ).nonzero(
                as_tuple=False
            )

            if matches.numel() > 0:

                rank_zero_based = int(
                    matches[0].item()
                )

                recall_sum += 1.0

                ndcg_sum += (
                    1.0
                    / math.log2(
                        rank_zero_based + 2
                    )
                )

        # --------------------------------------------------------
        # Popularity diagnosis
        # --------------------------------------------------------

        batch_rec_pop_raw = (
            pop_raw[topk].mean(dim=1)
        )

        batch_rec_pop_norm = (
            pop_norm[topk].mean(dim=1)
        )

        rec_pop_raw_sum += float(
            batch_rec_pop_raw.sum().item()
        )

        rec_pop_norm_sum += float(
            batch_rec_pop_norm.sum().item()
        )

        profile_pop_raw_sum += float(
            profile_pop_raw[
                batch_users
            ].sum().item()
        )

        profile_pop_norm_sum += float(
            profile_pop_norm[
                batch_users
            ].sum().item()
        )

        head_slots += int(
            head_mask[topk].sum().item()
        )

        tail_slots += int(
            tail_mask[topk].sum().item()
        )

        total_slots += int(
            topk.numel()
        )

    rec_matrix = torch.cat(
        all_recommended_items,
        dim=0,
    )

    unique_recommended = torch.unique(
        rec_matrix
    )

    tail_item_count = int(
        dataset.tail_item_mask.sum().item()
    )

    rec_tail_mask = (
        dataset.tail_item_mask[
            rec_matrix
        ]
    )

    unique_tail_recommended = torch.unique(
        rec_matrix[
            rec_tail_mask
        ]
    )

    n_users = dataset.num_users

    avg_rec_pop_raw = (
        rec_pop_raw_sum / n_users
    )

    avg_rec_pop_norm = (
        rec_pop_norm_sum / n_users
    )

    avg_profile_pop_raw = (
        profile_pop_raw_sum / n_users
    )

    avg_profile_pop_norm = (
        profile_pop_norm_sum / n_users
    )

    return {
        "Recall@K":
            recall_sum / n_users,

        "NDCG@K":
            ndcg_sum / n_users,

        "ARP_raw@K":
            avg_rec_pop_raw,

        "ARP_norm@K":
            avg_rec_pop_norm,

        "ProfilePopularity_raw":
            avg_profile_pop_raw,

        "ProfilePopularity_norm":
            avg_profile_pop_norm,

        "PopularityLift_raw@K":
            avg_rec_pop_raw
            - avg_profile_pop_raw,

        "PopularityLift_norm@K":
            avg_rec_pop_norm
            - avg_profile_pop_norm,

        "HeadRatio@K":
            (
                head_slots / total_slots
                if total_slots
                else 0.0
            ),

        "TailRatio@K":
            (
                tail_slots / total_slots
                if total_slots
                else 0.0
            ),

        "TailCoverage@K":
            (
                len(unique_tail_recommended)
                / tail_item_count
                if tail_item_count
                else 0.0
            ),

        "CatalogCoverage@K":
            (
                len(unique_recommended)
                / dataset.num_items
                if dataset.num_items
                else 0.0
            ),
    }
def save_history(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_one_run(args: argparse.Namespace) -> Dict[str, float]:
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = TenRecFullGraphDataset(
        prepared_root=args.prepared_root,
        subset_size=args.subset,
        domain=SOURCE_DOMAIN,
        seed=args.seed,
        min_clicks_both_domains=args.min_clicks,
    )

    target_dataset = TenRecFullGraphDataset(
        prepared_root = args.prepared_root,
        subset_size = args.subset,
        domain = TARGET_DOMAIN,
        seed = args.seed,
        min_clicks_both_domains = args.min_clicks,
    )

    print("\n===== SOURCE: QK-video =====")
    source_dataset.print_summary()

    print("\n===== TARGET: QK-article =====")
    target_dataset.print_summary()

    edge_index = target_dataset.edge_index.to(device)
    edge_type = target_dataset.edge_type.to(device)

    source_user_repr_aligned = (
        load_frozen_source_user_repr(
            args=args,
            device=device,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
        )
    )

    # ============================================================
    # Sanity checks for transferred Source representation
    # ============================================================

    if source_user_repr_aligned is not None:

        expected_shape = (
            target_dataset.num_users,
            args.embedding_dim,
        )

        actual_shape = tuple(
            source_user_repr_aligned.shape
        )

        print(
            "\nTransferred Source "
            f"user representation shape: "
            f"{actual_shape}"
        )

        if actual_shape != expected_shape:
            raise RuntimeError(
                "Transferred Source user "
                "representation has wrong shape.\n"
                f"Expected: {expected_shape}\n"
                f"Actual  : {actual_shape}"
            )

        if source_user_repr_aligned.requires_grad:
            raise RuntimeError(
                "Frozen Source representation "
                "must not require gradients."
            )

        assert source_user_repr_aligned.requires_grad is False

    else:
        print(
            "\nNo Source user representation "
            "will be used (target_only)."
        )

    dataset = target_dataset

    dataset.print_summary()

    split_path = output_dir / (
        f"split_{TARGET_DOMAIN}_{args.subset}_seed{args.seed}.csv"
    )
    target_dataset.split_manifest().to_csv(split_path, index=False)

    set_seed(args.seed)

    model = TenRecFullGraphRGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=args.embedding_dim,
        num_relations=8,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)

    projector =None
    trainable_parameters = list(
        model.parameters()
    )

    if source_user_repr_aligned is not None:

        projector = nn.Linear(
            args.embedding_dim,
            args.embedding_dim,
            bias=False,
        ).to(device)

        trainable_parameters += list(
            projector.parameters()
        )

    # ============================================================
    # Transfer-mode consistency checks
    # ============================================================

    if args.transfer_mode == "target_only":

        if source_user_repr_aligned is not None:
            raise RuntimeError(
                "target_only mode must not "
                "contain Source representations."
            )

        if projector is not None:
            raise RuntimeError(
                "target_only mode must not "
                "create a projector."
            )

    else:

        if source_user_repr_aligned is None:
            raise RuntimeError(
                f"{args.transfer_mode} mode requires "
                "Source user representations."
            )

        if projector is None:
            raise RuntimeError(
                f"{args.transfer_mode} mode requires "
                "a Source-to-Target projector."
            )

    representation_mean_change: float | None = None

    if (
        args.transfer_mode == "correct"
        and projector is not None
        and source_user_repr_aligned is not None
    ):
        representation_mean_change = compute_representation_path_change(
            model=model,
            edge_index=edge_index,
            edge_type=edge_type,
            projector=projector,
            source_user_repr_aligned=source_user_repr_aligned,
        )

        print(
            "\nRepresentation path sanity "
            f"mean_change={representation_mean_change:.8f}"
        )

        if representation_mean_change <= 0.0:
            raise RuntimeError(
                "Projected Source context did not change "
                "Target user representations."
            )


    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rng = np.random.default_rng(args.seed + 1000)

    best_val_ndcg = -float("inf")
    best_epoch = -1
    patience_counter = 0

    # ============================================================
    # Unique experiment name
    # ============================================================

    run_name = (
        f"{args.transfer_mode}_"
        f"{SOURCE_DOMAIN}_to_{TARGET_DOMAIN}_"
        f"{args.subset}_"
        f"seed{args.seed}"
    )

    checkpoint_path = (
            output_dir
            / f"best_{run_name}.pt"
    )

    history_path = (
            output_dir
            / f"history_{run_name}.csv"
    )

    result_path = (
            output_dir
            / f"result_{run_name}.json"
    )


    history_rows: List[Dict[str, float]] = []
    projector_grad_checked = False

    print("\n" + "=" * 72)
    print("ONE-WAY CROSS-DOMAIN RGCN VALIDATION")
    print("=" * 72)
    print(
        f"Source domain       : "
        f"{SOURCE_DOMAIN}"
    )
    print(
        f"Target domain       : "
        f"{TARGET_DOMAIN}"
    )
    print(
        f"Transfer mode       : "
        f"{args.transfer_mode}"
    )
    print(
        f"Source frozen       : YES"
    )
    print(
        f"Source representation: "
        f"{'USED' if source_user_repr_aligned is not None else 'NOT USED'}"
    )
    print(
        f"Projector           : "
        f"{'TRAINABLE' if projector is not None else 'NONE'}"
    )
    print(
        f"Device              : "
        f"{device}"
    )
    print(
        f"Embedding dim       : "
        f"{args.embedding_dim}"
    )
    print(
        f"Epochs              : "
        f"{args.epochs}"
    )
    print(
        f"Updates / epoch     : "
        f"{args.updates_per_epoch}"
    )
    print(
        f"BPR batch size      : "
        f"{args.batch_size}"
    )
    print(
        "Training objective  : "
        "TARGET BPR ONLY"
    )
    print(
        "Popularity in loss  : NO"
    )
    print(
        "Popularity rerank   : NO"
    )
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        model.train()

        if projector is not None:
            projector.train()

        epoch_bpr = 0.0
        start_time = time.time()

        for _ in range(args.updates_per_epoch):
            users, positives, negatives = (
                target_dataset.sample_bpr_batch(
                batch_size=args.batch_size,
                rng=rng,
            ))
            users = users.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad(set_to_none=True)

            # One full-graph convolution per optimizer update.
            external_user_context = None

            if source_user_repr_aligned is not None:

                if projector is None:
                    raise RuntimeError(
                        "Source user representations exist "
                        "but projector is None."
                    )

                external_user_context = projector(
                    source_user_repr_aligned
                )

            user_repr, item_repr = model.encode(
                edge_index,
                edge_type,
                external_user_context=external_user_context,
            )

            positive_scores = model.score_pairs(
                users, positives, user_repr, item_repr
            )
            negative_scores = model.score_pairs(
                users, negatives, user_repr, item_repr
            )

            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()

            if projector is not None and not projector_grad_checked:
                projector_grad = projector.weight.grad

                if projector_grad is None:
                    raise RuntimeError(
                        "Projector gradient is None after "
                        "the first backward pass."
                    )

                projector_grad_norm = float(
                    projector_grad.norm().item()
                )

                print(
                    "Projector gradient sanity "
                    f"norm={projector_grad_norm:.8f}"
                )

                if projector_grad_norm <= 0.0:
                    raise RuntimeError(
                        "Projector gradient norm is zero after "
                        "the first backward pass."
                    )

                projector_grad_checked = True

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=args.grad_clip,
                )

            optimizer.step()
            epoch_bpr += float(loss.item())

        train_seconds = time.time() - start_time
        row: Dict[str, float] = {
            "epoch": epoch,
            "train_bpr_loss": epoch_bpr / args.updates_per_epoch,
            "train_seconds": train_seconds,
            "optimizer_updates": epoch * args.updates_per_epoch,
        }

        should_eval = (epoch % args.eval_every == 0 or epoch == args.epochs)
        if should_eval:
            val_metrics = evaluate(
                model=model,
                dataset=target_dataset,
                edge_index=edge_index,
                edge_type=edge_type,
                device=device,
                split="val",
                projector=projector,
                source_user_repr_aligned=(
                    source_user_repr_aligned
                ),
                k=args.topk,
                user_batch_size=(
                    args.eval_user_batch_size
                ),
            )
            for key, value in val_metrics.items():
                row[f"val_{key}"] = value

            print(
                f"Epoch {epoch:03d} | "
                f"steps={epoch * args.updates_per_epoch:04d} | "
                f"BPR={row['train_bpr_loss']:.6f} | "
                f"Recall@{args.topk}={val_metrics['Recall@K']:.6f} | "
                f"NDCG@{args.topk}={val_metrics['NDCG@K']:.6f} | "
                f"ARP={val_metrics['ARP_raw@K']:.3f} | "
                f"Lift={val_metrics['PopularityLift_raw@K']:.3f} | "
                f"Head={val_metrics['HeadRatio@K']:.3f} | "
                f"TailCov={val_metrics['TailCoverage@K']:.4f} | "
                f"{train_seconds:.1f}s"
            )

            val_ndcg = val_metrics["NDCG@K"]
            if val_ndcg > best_val_ndcg + args.min_delta:
                best_val_ndcg = val_ndcg
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "target_model_state_dict":
                            model.state_dict(),

                        "projector_state_dict":
                            (
                                projector.state_dict()
                                if projector is not None
                                else None
                            ),

                        "args":
                            vars(args),

                        "dataset_stats":
                            asdict(
                                target_dataset.stats
                            ),

                        "best_val_ndcg":
                            best_val_ndcg,

                        "best_epoch":
                            best_epoch,

                        "source_checkpoint":
                            (
                                args.source_checkpoint
                                if source_user_repr_aligned
                                   is not None
                                else None
                            ),

                        "source_domain":
                            SOURCE_DOMAIN,

                        "target_domain":
                            TARGET_DOMAIN,

                        "transfer_mode":
                            args.transfer_mode,
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"steps={epoch * args.updates_per_epoch:04d} | "
                f"BPR={row['train_bpr_loss']:.6f} | "
                f"{train_seconds:.1f}s"
            )

        history_rows.append(row)
        save_history(history_rows, history_path)

        if (
            should_eval
            and args.patience > 0
            and patience_counter >= args.patience
        ):
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}."
            )
            break

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No validation checkpoint was created. "
            "Check eval_every / epochs / validation metrics."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        checkpoint["target_model_state_dict"]
    )

    if projector is not None:
        projector_state = checkpoint["projector_state_dict"]

        if projector_state is None:
            raise RuntimeError(
                "Best checkpoint is missing projector_state_dict."
            )

        projector.load_state_dict(projector_state)

    test_metrics = evaluate(
        model=model,
        dataset=target_dataset,
        edge_index=edge_index,
        edge_type=edge_type,
        device=device,
        split="test",
        projector=projector,
        source_user_repr_aligned=(
            source_user_repr_aligned
        ),
        k=args.topk,
        user_batch_size=(
            args.eval_user_batch_size
        ),
    )

    result = {
        **vars(args),

        **asdict(
            target_dataset.stats
        ),

        # --------------------------------------------------------
        # Cross-domain experiment identity
        # --------------------------------------------------------

        "source_domain":
            SOURCE_DOMAIN,

        "target_domain":
            TARGET_DOMAIN,

        "transfer_direction":
            (
                f"{SOURCE_DOMAIN}"
                f" -> "
                f"{TARGET_DOMAIN}"
            ),

        "transfer_mode":
            args.transfer_mode,

        "source_frozen":
            True,

        "source_information_used":
            (
                    source_user_repr_aligned
                    is not None
            ),

        "source_user_alignment":
            (
                "none"
                if args.transfer_mode
                   == "target_only"
                else args.transfer_mode
            ),

        "projector_used":
            projector is not None,

        # --------------------------------------------------------
        # Best validation checkpoint
        # --------------------------------------------------------

        "best_epoch":
            int(
                checkpoint[
                    "best_epoch"
                ]
            ),

        "best_val_ndcg":
            float(
                checkpoint[
                    "best_val_ndcg"
                ]
            ),
    }

    for key, value in test_metrics.items():
        result[f"test_{key}"] = float(value)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)

    print(
        "BEST ONE-WAY CDR MODEL "
        "- TEST + POPULARITY DIAGNOSIS"
    )

    print("=" * 72)

    print(
        f"Transfer mode                 : "
        f"{args.transfer_mode}"
    )

    print(
        f"Transfer direction            : "
        f"{SOURCE_DOMAIN} -> {TARGET_DOMAIN}"
    )
    print(f"Best epoch                    : {result['best_epoch']}")
    print("\nRecommendation performance:")
    print(f"Recall@{args.topk}                    : {test_metrics['Recall@K']:.6f}")
    print(f"NDCG@{args.topk}                      : {test_metrics['NDCG@K']:.6f}")
    print("\nPopularity diagnosis (post-hoc only):")
    print(f"Recommendation ARP raw@{args.topk}    : {test_metrics['ARP_raw@K']:.6f}")
    print(f"Profile popularity raw         : {test_metrics['ProfilePopularity_raw']:.6f}")
    print(f"Popularity lift raw@{args.topk}       : {test_metrics['PopularityLift_raw@K']:.6f}")
    print(f"Recommendation ARP norm@{args.topk}   : {test_metrics['ARP_norm@K']:.6f}")
    print(f"Profile popularity norm        : {test_metrics['ProfilePopularity_norm']:.6f}")
    print(f"Popularity lift norm@{args.topk}      : {test_metrics['PopularityLift_norm@K']:.6f}")
    print(f"Head ratio@{args.topk}                : {test_metrics['HeadRatio@K']:.6f}")
    print(f"Tail ratio@{args.topk}                : {test_metrics['TailRatio@K']:.6f}")
    print(f"Tail coverage@{args.topk}             : {test_metrics['TailCoverage@K']:.6f}")
    print(f"Catalog coverage@{args.topk}          : {test_metrics['CatalogCoverage@K']:.6f}")
    print(
        "\nIMPORTANT: popularity was never used in training or reranking. "
        "All popularity values above are diagnosis measurements only."
    )
    print(f"\nCheckpoint: {checkpoint_path.resolve()}")
    print(f"History   : {history_path.resolve()}")
    print(f"Result    : {result_path.resolve()}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pure TenRec multi-relational FullGraph RGCN "
            "Baseline + Popularity Diagnosis v1."
        )
    )

    parser.add_argument("--prepared_root", type=str, default="tenrec_rgcn_prepared")
    parser.add_argument("--subset", type=int, choices=[3000, 5000], default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_clicks", type=int, default=5)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_bases", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--updates_per_epoch",
        type=int,
        default=2,
        help=(
            "Each optimizer update recomputes the whole graph once. "
            "Compare experiments using the same total update budget."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_user_batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / mps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tenrec_baseline_diagnosis_v1",
    )
    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default=(
            "outputs/control_video_3k_rgcn/"
            "best_2layer_rgcn_QK-video_3000_seed42.pt"
        ),
    )
    parser.add_argument(
        "--transfer_mode",
        type=str,
        choices=[
            "target_only",
            "correct",
            "shuffled",
        ],
        default="correct",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_one_run(parse_args())
