from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch


RELATIONS: Tuple[str, ...] = ("click", "like", "share", "follow")

FORWARD_RELATION_ID: Dict[str, int] = {
    "click": 0,
    "like": 1,
    "share": 2,
    "follow": 3,
}

REVERSE_RELATION_ID: Dict[str, int] = {
    "click": 4,
    "like": 5,
    "share": 6,
    "follow": 7,
}

NUM_RELATIONS = 8


@dataclass
class TenRecSplitStats:
    subset_size: int
    domain: str
    seed: int
    min_clicks_both_domains: int
    num_users: int
    num_items: int
    num_train_click_pairs: int
    num_val_users: int
    num_test_users: int
    num_forward_edges: int
    num_bidirectional_edges: int
    cold_val_items: int
    cold_test_items: int
    warm_split_attempt: int


class TenRecFullGraphDataset:
    """
    Dataset layer for the first formal TenRec FullGraph RGCN experiment.

    Design:
    - Input canonical_user_item.csv from tenrec_rgcn_prepared.
    - Keep users with enough clicked items in BOTH QK-video and QK-article.
    - Candidate items are items with at least one positive relation in the
      eligible cohort.
    - Hold out one clicked pair for validation and one for test per user.
    - Remove ALL relations of held-out pairs from the training graph.
    - BPR positive = training click=1 pair.
    - BPR negative = item with NO positive relation for that user.
    - Popularity = unique TRAINING click-user degree.
    - Relations:
        0 click
        1 like
        2 share
        3 follow
        4 rev_click
        5 rev_like
        6 rev_share
        7 rev_follow
    """

    def __init__(
        self,
        prepared_root: str | Path = "tenrec_rgcn_prepared",
        subset_size: int = 3000,
        domain: str = "QK-video",
        seed: int = 42,
        min_clicks_both_domains: int = 5,
    ) -> None:
        if domain not in {"QK-video", "QK-article"}:
            raise ValueError("domain must be 'QK-video' or 'QK-article'.")

        if min_clicks_both_domains < 3:
            raise ValueError(
                "min_clicks_both_domains must be >= 3 because "
                "we need train + validation + test clicked items."
            )

        self.prepared_root = Path(prepared_root)
        self.subset_size = int(subset_size)
        self.domain = domain
        self.seed = int(seed)
        self.min_clicks_both_domains = int(min_clicks_both_domains)

        self.video_path = (
            self.prepared_root
            / str(self.subset_size)
            / "QK-video"
            / "canonical_user_item.csv"
        )
        self.article_path = (
            self.prepared_root
            / str(self.subset_size)
            / "QK-article"
            / "canonical_user_item.csv"
        )

        self._check_paths()

        self.eligible_raw_user_ids = self._compute_shared_eligible_users()

        self.raw_user_to_idx = {
            int(uid): idx
            for idx, uid in enumerate(self.eligible_raw_user_ids)
        }
        self.idx_to_raw_user = np.asarray(
            self.eligible_raw_user_ids,
            dtype=np.int64,
        )

        canonical = self._load_domain_canonical()
        canonical = canonical[
            canonical["user_id"].isin(self.raw_user_to_idx)
        ].copy()

        # Candidate item universe:
        # items with at least one positive relation somewhere in the
        # eligible cohort. This excludes globally exposure-only items
        # in QK-video from the first symmetric source/target setup.
        any_positive = canonical[list(RELATIONS)].sum(axis=1) > 0
        candidate_raw_items = np.sort(
            canonical.loc[any_positive, "item_id"]
            .astype("int64")
            .unique()
        )

        self.raw_item_to_idx = {
            int(iid): idx
            for idx, iid in enumerate(candidate_raw_items)
        }
        self.idx_to_raw_item = candidate_raw_items.astype(np.int64)

        canonical = canonical[
            canonical["item_id"].isin(self.raw_item_to_idx)
        ].copy()

        canonical["user_idx"] = (
            canonical["user_id"].map(self.raw_user_to_idx).astype("int64")
        )
        canonical["item_idx"] = (
            canonical["item_id"].map(self.raw_item_to_idx).astype("int64")
        )

        self.canonical = canonical.reset_index(drop=True)

        self.num_users = len(self.raw_user_to_idx)
        self.num_items = len(self.raw_item_to_idx)

        (
            self.train_canonical,
            self.val_item_by_user,
            self.test_item_by_user,
        ) = self._split_clicked_pairs()

        # Full positive sets are used to block negatives. This includes
        # validation/test positives so they can never be sampled as negatives.
        self.blocked_negative_items_by_user = (
            self._build_full_positive_item_sets(
                self.canonical,
                num_users=self.num_users,
            )
        )

        # Training seen sets are used for evaluation masking.
        self.train_seen_items_by_user = (
            self._build_full_positive_item_sets(
                self.train_canonical,
                num_users=self.num_users,
            )
        )

        # Training clicked items per user are used ONLY to measure each
        # user's historical popularity preference during Diagnosis.
        self.train_click_items_by_user = (
            self._build_train_click_item_sets()
        )

        self.train_user_idx, self.train_pos_item_idx = (
            self._build_train_click_pairs()
        )

        self.edge_index, self.edge_type = self._build_training_graph()

        (
            self.item_popularity_raw,
            self.item_popularity_norm,
        ) = self._compute_training_popularity()

        self.tail_item_mask = self._build_tail_item_mask(
            head_ratio=0.20
        )

        self.stats = self._build_stats()

        self._validate_internal_consistency()

    # ------------------------------------------------------------------
    # Loading and eligibility
    # ------------------------------------------------------------------

    def _check_paths(self) -> None:
        for path in (self.video_path, self.article_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing prepared TenRec file:\n{path}\n\n"
                    "Expected output from prepare_tenrec_rgcn_dataset.py."
                )

    @staticmethod
    def _load_canonical_file(path: Path) -> pd.DataFrame:
        usecols = [
            "user_id",
            "item_id",
            "click",
            "like",
            "share",
            "follow",
        ]
        df = pd.read_csv(path, usecols=usecols)

        df["user_id"] = pd.to_numeric(
            df["user_id"], errors="raise"
        ).astype("int64")
        df["item_id"] = pd.to_numeric(
            df["item_id"], errors="raise"
        ).astype("int64")

        for rel in RELATIONS:
            df[rel] = pd.to_numeric(
                df[rel], errors="raise"
            ).astype("int8")

        if df.duplicated(["user_id", "item_id"]).any():
            raise ValueError(
                f"{path} is not canonical: duplicated user-item pairs remain."
            )

        return df

    def _compute_shared_eligible_users(self) -> np.ndarray:
        """
        Build ONE shared user cohort that is suitable for warm-start
        leave-two-out splitting in BOTH domains.

        A user must satisfy two conditions in BOTH QK-video and QK-article:

        1) at least min_clicks_both_domains clicked items;
        2) at least TWO clicked items whose current cohort click degree >= 2.

        The second condition is recomputed iteratively because removing users
        can reduce item degrees. We stop only when the shared cohort is stable.

        This does NOT use popularity for model training. It is purely a
        data-splitting constraint so validation/test targets remain warm.
        """
        video = self._load_canonical_file(self.video_path)
        article = self._load_canonical_file(self.article_path)

        def clicked_count(df: pd.DataFrame) -> pd.Series:
            return (
                df.loc[df["click"] == 1]
                .groupby("user_id")["item_id"]
                .nunique()
            )

        video_counts = clicked_count(video)
        article_counts = clicked_count(article)

        eligible: Set[int] = (
            set(
                video_counts[
                    video_counts >= self.min_clicks_both_domains
                ].index.astype("int64")
            )
            &
            set(
                article_counts[
                    article_counts >= self.min_clicks_both_domains
                ].index.astype("int64")
            )
        )

        if not eligible:
            raise RuntimeError(
                "No shared users satisfy the minimum click threshold."
            )

        def warm_capable_users(
            df: pd.DataFrame,
            current_users: Set[int],
        ) -> Set[int]:
            # Work only with clicked pairs from the CURRENT candidate cohort.
            clicked = (
                df.loc[
                    (df["click"] == 1)
                    & df["user_id"].isin(current_users),
                    ["user_id", "item_id"],
                ]
                .drop_duplicates()
            )

            # Click degree within the current cohort.
            item_degree = (
                clicked.groupby("item_id")["user_id"]
                .nunique()
            )

            # A candidate held-out item must have at least two clickers before
            # splitting so one click can be held out while >=1 remains.
            warm_items = set(
                item_degree[item_degree >= 2]
                .index.astype("int64")
            )

            if not warm_items:
                return set()

            warm_count_per_user = (
                clicked.loc[
                    clicked["item_id"].isin(warm_items)
                ]
                .groupby("user_id")["item_id"]
                .nunique()
            )

            # Need TWO distinct warm candidates: one validation + one test.
            return set(
                warm_count_per_user[
                    warm_count_per_user >= 2
                ].index.astype("int64")
            )

        iteration = 0

        while True:
            iteration += 1

            video_warm = warm_capable_users(
                video,
                eligible,
            )
            article_warm = warm_capable_users(
                article,
                eligible,
            )

            new_eligible = (
                eligible
                & video_warm
                & article_warm
            )

            if not new_eligible:
                raise RuntimeError(
                    "Warm-start eligibility removed all shared users."
                )

            if new_eligible == eligible:
                break

            eligible = new_eligible

        self.warm_eligibility_iterations = iteration

        return np.asarray(
            sorted(eligible),
            dtype=np.int64,
        )

    def _load_domain_canonical(self) -> pd.DataFrame:
        path = (
            self.video_path
            if self.domain == "QK-video"
            else self.article_path
        )
        return self._load_canonical_file(path)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _split_clicked_pairs(
        self,
    ) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
        """
        Warm-start constrained leave-two-out split.

        Goal
        ----
        Each eligible user still gets:
            1 validation clicked item
            1 test clicked item

        But a pair can be held out only if its item will still have at least
        ONE training click after all holdouts are assigned.

        Why this matters
        ----------------
        We are diagnosing popularity bias, not cold-start recommendation.
        Therefore validation/test targets should have a defined training
        click degree and should not disappear from the training click graph.

        Implementation
        --------------
        - Compute click degree before splitting.
        - Treat each item as having holdout capacity = degree - 1.
        - Assign two distinct clicked items per user.
        - Never consume the final remaining click of an item.
        - Retry deterministic randomized greedy assignment if necessary.
        - Remove ALL relations of held-out (user,item) pairs from the graph
          to avoid leakage through like/share/follow.
        """
        clicked = (
            self.canonical.loc[
                self.canonical["click"] == 1,
                ["user_idx", "item_idx"],
            ]
            .drop_duplicates()
        )

        clicked_by_user = {
            int(user_idx): np.sort(
                grp["item_idx"].to_numpy(dtype=np.int64)
            )
            for user_idx, grp in clicked.groupby(
                "user_idx",
                sort=True,
            )
        }

        initial_degree = np.bincount(
            clicked["item_idx"].to_numpy(dtype=np.int64),
            minlength=self.num_items,
        ).astype(np.int64)

        # Pre-check guaranteed by shared warm eligibility, but verify again
        # after domain-specific item remapping.
        warm_candidate_count = np.zeros(
            self.num_users,
            dtype=np.int64,
        )

        for user_idx in range(self.num_users):
            items = clicked_by_user.get(user_idx)

            if items is None:
                raise RuntimeError(
                    f"Eligible user {user_idx} has no clicked items."
                )

            warm_candidate_count[user_idx] = int(
                np.sum(
                    initial_degree[items] >= 2
                )
            )

            if warm_candidate_count[user_idx] < 2:
                raise RuntimeError(
                    f"Eligible user {user_idx} has fewer than two "
                    "warm-start candidate clicked items."
                )

        max_attempts = 100
        successful = False

        for attempt in range(max_attempts):
            rng = np.random.default_rng(
                self.seed + attempt
            )

            remaining_degree = initial_degree.copy()

            val_item_by_user: Dict[int, int] = {}
            test_item_by_user: Dict[int, int] = {}
            heldout_pairs: Set[Tuple[int, int]] = set()

            # Assign the most constrained users first. Random jitter gives a
            # reproducible tie-break and lets later attempts explore a
            # different feasible assignment.
            jitter = rng.random(
                self.num_users
            )

            user_order = sorted(
                range(self.num_users),
                key=lambda u: (
                    int(warm_candidate_count[u]),
                    float(jitter[u]),
                ),
            )

            failed = False

            for user_idx in user_order:
                items = clicked_by_user[user_idx]

                # Eligible NOW means holding this click out still leaves at
                # least one click on the item in the future training set.
                available = items[
                    remaining_degree[items] > 1
                ]

                if len(available) < 2:
                    failed = True
                    break

                chosen = rng.permutation(
                    available
                )[:2]

                val_item = int(chosen[0])
                test_item = int(chosen[1])

                val_item_by_user[user_idx] = val_item
                test_item_by_user[user_idx] = test_item

                heldout_pairs.add(
                    (user_idx, val_item)
                )
                heldout_pairs.add(
                    (user_idx, test_item)
                )

                # Consume one click of holdout capacity from each item.
                remaining_degree[val_item] -= 1
                remaining_degree[test_item] -= 1

            if not failed:
                successful = True
                self.warm_split_attempt = attempt + 1
                break

        if not successful:
            raise RuntimeError(
                "Could not construct a warm-start leave-two-out split "
                f"after {max_attempts} deterministic attempts. "
                "Consider a stricter warm-item eligibility threshold or "
                "inspect highly constrained users."
            )

        pair_index = pd.MultiIndex.from_arrays(
            [
                self.canonical["user_idx"].to_numpy(),
                self.canonical["item_idx"].to_numpy(),
            ]
        )

        heldout_index = pd.MultiIndex.from_tuples(
            sorted(heldout_pairs)
        )

        is_heldout = pair_index.isin(
            heldout_index
        )

        train_canonical = self.canonical.loc[
            ~is_heldout
        ].copy()

        # Final hard guarantee: every held-out target has >=1 TRAIN click.
        train_clicked = (
            train_canonical.loc[
                train_canonical["click"] == 1,
                ["user_idx", "item_idx"],
            ]
            .drop_duplicates()
        )

        train_degree = np.bincount(
            train_clicked["item_idx"].to_numpy(dtype=np.int64),
            minlength=self.num_items,
        )

        for item_idx in val_item_by_user.values():
            if train_degree[int(item_idx)] < 1:
                raise AssertionError(
                    "Warm-start constraint failed for a validation item."
                )

        for item_idx in test_item_by_user.values():
            if train_degree[int(item_idx)] < 1:
                raise AssertionError(
                    "Warm-start constraint failed for a test item."
                )

        return (
            train_canonical.reset_index(drop=True),
            val_item_by_user,
            test_item_by_user,
        )

    # ------------------------------------------------------------------
    # Positive / negative sets
    # ------------------------------------------------------------------

    @staticmethod
    def _build_full_positive_item_sets(
        df: pd.DataFrame,
        num_users: int,
    ) -> List[Set[int]]:
        result: List[Set[int]] = [
            set() for _ in range(num_users)
        ]

        any_positive = df[list(RELATIONS)].sum(axis=1) > 0

        for row in df.loc[
            any_positive,
            ["user_idx", "item_idx"],
        ].itertuples(index=False):
            result[int(row.user_idx)].add(int(row.item_idx))

        return result

    def _build_train_click_pairs(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        train_clicks = self.train_canonical.loc[
            self.train_canonical["click"] == 1,
            ["user_idx", "item_idx"],
        ].drop_duplicates()

        return (
            train_clicks["user_idx"].to_numpy(dtype=np.int64),
            train_clicks["item_idx"].to_numpy(dtype=np.int64),
        )

    def _build_train_click_item_sets(self) -> List[Set[int]]:
        """Training clicked items per user; diagnosis support only."""
        result: List[Set[int]] = [
            set() for _ in range(self.num_users)
        ]

        clicks = self.train_canonical.loc[
            self.train_canonical["click"] == 1,
            ["user_idx", "item_idx"],
        ].drop_duplicates()

        for row in clicks.itertuples(index=False):
            result[int(row.user_idx)].add(int(row.item_idx))

        return result

    def sample_bpr_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        positive = clicked training item

        negative is sampled uniformly from candidate items, subject to:
            negative NOT IN any positive-relation item for this user

        So like/share/follow-only items and validation/test positives
        cannot accidentally become BPR negatives.
        """
        if len(self.train_user_idx) == 0:
            raise RuntimeError("No training click pairs.")

        pair_positions = rng.integers(
            low=0,
            high=len(self.train_user_idx),
            size=batch_size,
        )

        users = self.train_user_idx[pair_positions]
        positives = self.train_pos_item_idx[pair_positions]
        negatives = np.empty(batch_size, dtype=np.int64)

        for k, user_idx in enumerate(users):
            blocked = self.blocked_negative_items_by_user[int(user_idx)]

            if len(blocked) >= self.num_items:
                raise RuntimeError(
                    f"User {user_idx} has no available negative item."
                )

            while True:
                candidate = int(rng.integers(0, self.num_items))
                if candidate not in blocked:
                    negatives[k] = candidate
                    break

        return (
            torch.from_numpy(users).long(),
            torch.from_numpy(positives).long(),
            torch.from_numpy(negatives).long(),
        )

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_training_graph(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_nodes: List[np.ndarray] = []
        dst_nodes: List[np.ndarray] = []
        rel_types: List[np.ndarray] = []

        for rel in RELATIONS:
            mask = self.train_canonical[rel] == 1
            sub = self.train_canonical.loc[
                mask,
                ["user_idx", "item_idx"],
            ]

            if len(sub) == 0:
                continue

            u = sub["user_idx"].to_numpy(dtype=np.int64)
            i = sub["item_idx"].to_numpy(dtype=np.int64)
            item_nodes = self.num_users + i

            # user -> item
            src_nodes.append(u)
            dst_nodes.append(item_nodes)
            rel_types.append(
                np.full(
                    len(sub),
                    FORWARD_RELATION_ID[rel],
                    dtype=np.int64,
                )
            )

            # item -> user
            src_nodes.append(item_nodes)
            dst_nodes.append(u)
            rel_types.append(
                np.full(
                    len(sub),
                    REVERSE_RELATION_ID[rel],
                    dtype=np.int64,
                )
            )

        if not src_nodes:
            raise RuntimeError(
                "Training graph has no positive relation edges."
            )

        src = np.concatenate(src_nodes)
        dst = np.concatenate(dst_nodes)
        rel = np.concatenate(rel_types)

        edge_index = torch.tensor(
            np.vstack([src, dst]),
            dtype=torch.long,
        )

        edge_type = torch.tensor(
            rel,
            dtype=torch.long,
        )

        return edge_index, edge_type

    # ------------------------------------------------------------------
    # Popularity
    # ------------------------------------------------------------------

    def _compute_training_popularity(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DIAGNOSIS ONLY.

        Pop(i) = number of unique TRAINING users with click=1 on item i.
        This statistic is never fed into the baseline RGCN, never used in
        BPR loss, and never used to rerank recommendations.
        """
        clicks = self.train_canonical.loc[
            self.train_canonical["click"] == 1,
            ["user_idx", "item_idx"],
        ].drop_duplicates()

        raw = np.zeros(
            self.num_items,
            dtype=np.float32,
        )

        counts = (
            clicks.groupby("item_idx")["user_idx"]
            .nunique()
        )

        raw[
            counts.index.to_numpy(dtype=np.int64)
        ] = counts.to_numpy(dtype=np.float32)

        min_v = float(raw.min()) if len(raw) else 0.0
        max_v = float(raw.max()) if len(raw) else 0.0

        if max_v > min_v:
            norm = (
                (raw - min_v)
                / (max_v - min_v)
            )
        else:
            norm = np.zeros_like(raw)

        return (
            torch.from_numpy(raw),
            torch.from_numpy(
                norm.astype(np.float32)
            ),
        )

    def _build_tail_item_mask(
        self,
        head_ratio: float = 0.20,
    ) -> torch.Tensor:
        """
        Head = top 20% by TRAINING click degree.
        Tail = remaining 80%.
        """
        pop = self.item_popularity_raw.numpy()

        num_head = max(
            1,
            int(np.ceil(self.num_items * head_ratio)),
        )

        order = np.argsort(
            -pop,
            kind="stable",
        )

        head_idx = order[:num_head]

        tail_mask = np.ones(
            self.num_items,
            dtype=bool,
        )

        tail_mask[head_idx] = False

        return torch.from_numpy(tail_mask)

    def user_profile_popularity(
        self,
        normalized: bool = False,
    ) -> torch.Tensor:
        """
        Per-user mean popularity of TRAINING clicked items.

        Diagnosis only. This lets us compare recommendation popularity
        against what each user actually clicked in their training history.
        """
        popularity = (
            self.item_popularity_norm
            if normalized
            else self.item_popularity_raw
        )

        result = torch.zeros(
            self.num_users,
            dtype=torch.float32,
        )

        for user_idx, items in enumerate(self.train_click_items_by_user):
            if not items:
                continue

            item_idx = torch.tensor(
                sorted(items),
                dtype=torch.long,
            )
            result[user_idx] = popularity[item_idx].mean()

        return result

    # ------------------------------------------------------------------
    # Reports / checks
    # ------------------------------------------------------------------

    def _build_stats(self) -> TenRecSplitStats:
        train_click_degree = (
            self.item_popularity_raw.numpy()
        )

        cold_val_items = sum(
            1
            for item in self.val_item_by_user.values()
            if train_click_degree[item] == 0
        )

        cold_test_items = sum(
            1
            for item in self.test_item_by_user.values()
            if train_click_degree[item] == 0
        )

        return TenRecSplitStats(
            subset_size=self.subset_size,
            domain=self.domain,
            seed=self.seed,
            min_clicks_both_domains=self.min_clicks_both_domains,
            num_users=self.num_users,
            num_items=self.num_items,
            num_train_click_pairs=len(self.train_user_idx),
            num_val_users=len(self.val_item_by_user),
            num_test_users=len(self.test_item_by_user),
            num_forward_edges=int(self.edge_type.numel() // 2),
            num_bidirectional_edges=int(self.edge_type.numel()),
            cold_val_items=int(cold_val_items),
            cold_test_items=int(cold_test_items),
            warm_split_attempt=int(self.warm_split_attempt),
        )

    def _validate_internal_consistency(self) -> None:
        if self.edge_index.shape[0] != 2:
            raise AssertionError(
                "edge_index must have shape [2, E]."
            )

        if (
            self.edge_index.shape[1]
            != self.edge_type.numel()
        ):
            raise AssertionError(
                "edge_index / edge_type length mismatch."
            )

        if (
            int(self.edge_type.min()) < 0
            or int(self.edge_type.max()) >= NUM_RELATIONS
        ):
            raise AssertionError(
                "edge_type outside [0, 7]."
            )

        if (
            len(self.blocked_negative_items_by_user)
            != self.num_users
        ):
            raise AssertionError(
                "Blocked negative sets do not cover all users."
            )

        if (
            len(self.train_seen_items_by_user)
            != self.num_users
        ):
            raise AssertionError(
                "Training seen sets do not cover all users."
            )

        train_pairs = set(
            zip(
                self.train_canonical["user_idx"].astype(int),
                self.train_canonical["item_idx"].astype(int),
            )
        )

        for u, i in self.val_item_by_user.items():
            if (u, i) in train_pairs:
                raise AssertionError(
                    "Validation pair leaked into training graph."
                )

        for u, i in self.test_item_by_user.items():
            if (u, i) in train_pairs:
                raise AssertionError(
                    "Test pair leaked into training graph."
                )

        # Warm-start split guarantee: every validation/test target must
        # retain at least one TRAINING click after holdout.
        train_click_degree = self.item_popularity_raw.numpy()

        for item_idx in self.val_item_by_user.values():
            if train_click_degree[int(item_idx)] < 1:
                raise AssertionError(
                    "Validation target became cold after splitting."
                )

        for item_idx in self.test_item_by_user.values():
            if train_click_degree[int(item_idx)] < 1:
                raise AssertionError(
                    "Test target became cold after splitting."
                )

    def print_summary(self) -> None:
        s = self.stats

        print("=" * 72)
        print(
            f"TENREC FULLGRAPH DATASET - {s.domain} - "
            f"subset={s.subset_size:,}"
        )
        print("=" * 72)
        print(f"Seed                         : {s.seed}")
        print(
            f"Min clicks in BOTH domains   : "
            f"{s.min_clicks_both_domains}"
        )
        print(f"Eligible shared users        : {s.num_users:,}")
        print(f"Candidate items              : {s.num_items:,}")
        print(f"Train click pairs            : {s.num_train_click_pairs:,}")
        print(f"Validation users             : {s.num_val_users:,}")
        print(f"Test users                   : {s.num_test_users:,}")
        print(f"Forward relation edges       : {s.num_forward_edges:,}")
        print(f"Bidirectional graph edges    : {s.num_bidirectional_edges:,}")
        print(
            f"Cold validation items        : "
            f"{s.cold_val_items:,} "
            f"({s.cold_val_items / max(1, s.num_val_users):.2%})"
        )
        print(
            f"Cold test items              : "
            f"{s.cold_test_items:,} "
            f"({s.cold_test_items / max(1, s.num_test_users):.2%})"
        )
        print(
            f"Warm split assignment try    : "
            f"{s.warm_split_attempt}"
        )
        print(
            "Warm-start rule              : every held-out target keeps "
            ">=1 TRAINING click"
        )
        print(
            "Popularity usage              : DIAGNOSIS ONLY (not training)"
        )

        relation_counts = torch.bincount(
            self.edge_type,
            minlength=NUM_RELATIONS,
        ).tolist()

        names = [
            "click",
            "like",
            "share",
            "follow",
            "rev_click",
            "rev_like",
            "rev_share",
            "rev_follow",
        ]

        print("\nRelation edge counts:")
        for rid, name in enumerate(names):
            print(
                f"  {rid}: {name:<10} "
                f"{relation_counts[rid]:>10,}"
            )

    def split_manifest(self) -> pd.DataFrame:
        rows = []

        for user_idx in range(self.num_users):
            val_item = self.val_item_by_user[user_idx]
            test_item = self.test_item_by_user[user_idx]

            rows.append(
                {
                    "user_idx": user_idx,
                    "raw_user_id": int(
                        self.idx_to_raw_user[user_idx]
                    ),
                    "val_item_idx": int(val_item),
                    "val_raw_item_id": int(
                        self.idx_to_raw_item[val_item]
                    ),
                    "test_item_idx": int(test_item),
                    "test_raw_item_id": int(
                        self.idx_to_raw_item[test_item]
                    ),
                }
            )

        return pd.DataFrame(rows)
