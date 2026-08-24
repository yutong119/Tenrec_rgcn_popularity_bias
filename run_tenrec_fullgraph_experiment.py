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
    k: int = 10,
    user_batch_size: int = 128,
) -> Dict[str, float]:
    """
    Performance + POST-HOC popularity diagnosis.

    Popularity is used only as a measuring ruler here. It never changes
    model parameters, raw recommendation scores, or ranking.
    """
    model.eval()

    user_repr, item_repr = model.encode(edge_index, edge_type)

    if split == "val":
        target_by_user = dataset.val_item_by_user
        other_holdout_by_user = dataset.test_item_by_user
    elif split == "test":
        target_by_user = dataset.test_item_by_user
        other_holdout_by_user = dataset.val_item_by_user
    else:
        raise ValueError("split must be 'val' or 'test'.")

    # Diagnosis-only popularity statistics.
    pop_raw = dataset.item_popularity_raw.to(device)
    pop_norm = dataset.item_popularity_norm.to(device)
    profile_pop_raw = dataset.user_profile_popularity(False).to(device)
    profile_pop_norm = dataset.user_profile_popularity(True).to(device)
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

    user_indices = np.arange(dataset.num_users, dtype=np.int64)

    for start in range(0, dataset.num_users, user_batch_size):
        batch_np = user_indices[start:start + user_batch_size]
        batch_users = torch.from_numpy(batch_np).long().to(device)

        scores = model.score_user_batch_against_all_items(
            batch_users,
            user_repr,
            item_repr,
        )

        # Remove training positive-relation items from candidates.
        # Also hide the other held-out positive so val/test do not compete.
        for local_row, user_idx in enumerate(batch_np):
            seen = dataset.train_seen_items_by_user[int(user_idx)]
            if seen:
                seen_idx = torch.tensor(
                    list(seen),
                    dtype=torch.long,
                    device=device,
                )
                scores[local_row, seen_idx] = -torch.inf

            other_item = int(other_holdout_by_user[int(user_idx)])
            scores[local_row, other_item] = -torch.inf

        topk = torch.topk(
            scores,
            k=min(k, dataset.num_items),
            dim=1,
        ).indices

        all_recommended_items.append(topk.detach().cpu())

        for local_row, user_idx in enumerate(batch_np):
            target = int(target_by_user[int(user_idx)])
            matches = (topk[local_row] == target).nonzero(as_tuple=False)
            if matches.numel() > 0:
                rank_zero_based = int(matches[0].item())
                recall_sum += 1.0
                ndcg_sum += 1.0 / math.log2(rank_zero_based + 2)

        batch_rec_pop_raw = pop_raw[topk].mean(dim=1)
        batch_rec_pop_norm = pop_norm[topk].mean(dim=1)
        rec_pop_raw_sum += float(batch_rec_pop_raw.sum().item())
        rec_pop_norm_sum += float(batch_rec_pop_norm.sum().item())

        profile_pop_raw_sum += float(profile_pop_raw[batch_users].sum().item())
        profile_pop_norm_sum += float(profile_pop_norm[batch_users].sum().item())

        head_slots += int(head_mask[topk].sum().item())
        tail_slots += int(tail_mask[topk].sum().item())
        total_slots += int(topk.numel())

    rec_matrix = torch.cat(all_recommended_items, dim=0)
    unique_recommended = torch.unique(rec_matrix)

    tail_item_count = int(dataset.tail_item_mask.sum().item())
    rec_tail_mask = dataset.tail_item_mask[rec_matrix]
    unique_tail_recommended = torch.unique(rec_matrix[rec_tail_mask])

    n_users = dataset.num_users
    avg_rec_pop_raw = rec_pop_raw_sum / n_users
    avg_rec_pop_norm = rec_pop_norm_sum / n_users
    avg_profile_pop_raw = profile_pop_raw_sum / n_users
    avg_profile_pop_norm = profile_pop_norm_sum / n_users

    return {
        "Recall@K": recall_sum / n_users,
        "NDCG@K": ndcg_sum / n_users,
        "ARP_raw@K": avg_rec_pop_raw,
        "ARP_norm@K": avg_rec_pop_norm,
        "ProfilePopularity_raw": avg_profile_pop_raw,
        "ProfilePopularity_norm": avg_profile_pop_norm,
        "PopularityLift_raw@K": avg_rec_pop_raw - avg_profile_pop_raw,
        "PopularityLift_norm@K": avg_rec_pop_norm - avg_profile_pop_norm,
        "HeadRatio@K": head_slots / total_slots if total_slots else 0.0,
        "TailRatio@K": tail_slots / total_slots if total_slots else 0.0,
        "TailCoverage@K": (
            len(unique_tail_recommended) / tail_item_count
            if tail_item_count else 0.0
        ),
        "CatalogCoverage@K": (
            len(unique_recommended) / dataset.num_items
            if dataset.num_items else 0.0
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
    set_seed(args.seed)
    device = choose_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TenRecFullGraphDataset(
        prepared_root=args.prepared_root,
        subset_size=args.subset,
        domain=args.domain,
        seed=args.seed,
        min_clicks_both_domains=args.min_clicks,
    )
    dataset.print_summary()

    split_path = output_dir / (
        f"split_{args.domain}_{args.subset}_seed{args.seed}.csv"
    )
    dataset.split_manifest().to_csv(split_path, index=False)

    edge_index = dataset.edge_index.to(device)
    edge_type = dataset.edge_type.to(device)

    model = TenRecFullGraphRGCN(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=args.embedding_dim,
        num_relations=8,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    rng = np.random.default_rng(args.seed + 1000)

    best_val_ndcg = -float("inf")
    best_epoch = -1
    patience_counter = 0

    checkpoint_path = output_dir / (
        f"best_baseline_{args.domain}_{args.subset}_seed{args.seed}.pt"
    )
    history_path = output_dir / (
        f"history_baseline_{args.domain}_{args.subset}_seed{args.seed}.csv"
    )
    result_path = output_dir / (
        f"result_baseline_{args.domain}_{args.subset}_seed{args.seed}.json"
    )

    history_rows: List[Dict[str, float]] = []

    print("\n" + "=" * 72)
    print("PURE BASELINE TRAINING")
    print("=" * 72)
    print(f"Device             : {device}")
    print(f"Embedding dim      : {args.embedding_dim}")
    print(f"Epochs             : {args.epochs}")
    print(f"Updates / epoch    : {args.updates_per_epoch}")
    print(f"BPR batch size     : {args.batch_size}")
    print("Training objective : BPR ONLY")
    print("Popularity in loss : NO")
    print("Popularity rerank  : NO")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_bpr = 0.0
        start_time = time.time()

        for _ in range(args.updates_per_epoch):
            users, positives, negatives = dataset.sample_bpr_batch(
                batch_size=args.batch_size,
                rng=rng,
            )
            users = users.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad(set_to_none=True)

            # One full-graph convolution per optimizer update.
            user_repr, item_repr = model.encode(edge_index, edge_type)

            positive_scores = model.score_pairs(
                users, positives, user_repr, item_repr
            )
            negative_scores = model.score_pairs(
                users, negatives, user_repr, item_repr
            )

            loss = bpr_loss(positive_scores, negative_scores)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
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
                dataset=dataset,
                edge_index=edge_index,
                edge_type=edge_type,
                device=device,
                split="val",
                k=args.topk,
                user_batch_size=args.eval_user_batch_size,
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
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "dataset_stats": asdict(dataset.stats),
                        "best_val_ndcg": best_val_ndcg,
                        "best_epoch": best_epoch,
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
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataset=dataset,
        edge_index=edge_index,
        edge_type=edge_type,
        device=device,
        split="test",
        k=args.topk,
        user_batch_size=args.eval_user_batch_size,
    )

    result = {
        **vars(args),
        **asdict(dataset.stats),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_ndcg": float(checkpoint["best_val_ndcg"]),
    }
    for key, value in test_metrics.items():
        result[f"test_{key}"] = float(value)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("BEST BASELINE MODEL - TEST + POPULARITY DIAGNOSIS")
    print("=" * 72)
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
    parser.add_argument(
        "--domain",
        type=str,
        choices=["QK-video", "QK-article"],
        default="QK-video",
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    train_one_run(parse_args())
