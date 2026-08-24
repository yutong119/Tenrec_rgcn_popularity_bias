
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# This script assumes you have already run the shared-user subset script.
INPUT_ROOT = Path("../../tenrec_shared_subset_output") / "subsets"

# Start with 3k for debugging and 5k as the main candidate.
SUBSET_SIZES = [3_000, 5_000]

DOMAINS = ["QK-video", "QK-article"]

RELATIONS = ["click", "like", "share", "follow"]

FORWARD_RELATION_ID = {
    "click": 0,
    "like": 1,
    "share": 2,
    "follow": 3,
}

REVERSE_RELATION_ID = {
    "click": 4,
    "like": 5,
    "share": 6,
    "follow": 7,
}

OUTPUT_ROOT = Path("../../tenrec_rgcn_prepared")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Thresholds used only for split-feasibility audit.
# No users are filtered by these thresholds in this script.
ACTIVITY_THRESHOLDS = [2, 3, 5, 10]


# ============================================================
# HELPERS
# ============================================================

def to_binary(series: pd.Series) -> pd.Series:
    """
    Normalize TenRec binary-like values to integer 0/1.

    Supports:
      0 / 1
      0.0 / 1.0
      False / True
      "False" / "True"
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype("int8")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return (numeric != 0).astype("int8")

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive_values = {
        "1",
        "1.0",
        "true",
        "t",
        "yes",
        "y",
    }

    return normalized.isin(positive_values).astype("int8")


def load_and_normalize(path: Path) -> pd.DataFrame:
    """
    Read one subset file and keep only fields needed by the first RGCN version.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            f"Run the shared-user subset preparation first."
        )

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
        df["user_id"],
        errors="raise",
    ).astype("int64")

    df["item_id"] = pd.to_numeric(
        df["item_id"],
        errors="raise",
    ).astype("int64")

    for rel in RELATIONS:
        df[rel] = to_binary(df[rel])

    return df


def aggregate_duplicate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize repeated (user_id, item_id) rows.

    Rule:
        relation(u, i) = max over all repeated rows

    Example:
        u1,i5,click=1,like=0
        u1,i5,click=1,like=1

    becomes:
        u1,i5,click=1,like=1

    This guarantees that each (user,item,relation) contributes at most
    one forward graph edge.
    """
    canonical = (
        df.groupby(
            ["user_id", "item_id"],
            as_index=False,
            sort=False,
        )[RELATIONS]
        .max()
    )

    for rel in RELATIONS:
        canonical[rel] = canonical[rel].astype("int8")

    return canonical


def build_forward_relation_edges(canonical: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide canonical user-item rows into one forward edge per
    positive relation.

    Output:
        user_id
        item_id
        relation
        relation_id
    """
    edge_parts = []

    for rel in RELATIONS:
        mask = canonical[rel] == 1

        if not mask.any():
            continue

        part = canonical.loc[
            mask,
            ["user_id", "item_id"],
        ].copy()

        part["relation"] = rel
        part["relation_id"] = FORWARD_RELATION_ID[rel]

        edge_parts.append(part)

    if not edge_parts:
        return pd.DataFrame(
            columns=[
                "user_id",
                "item_id",
                "relation",
                "relation_id",
            ]
        )

    edges = pd.concat(
        edge_parts,
        ignore_index=True,
    )

    edges["relation_id"] = edges["relation_id"].astype("int8")

    return edges


def build_bidirectional_rgcn_edges(
    forward_edges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an explicit bidirectional relation-edge table.

    Forward:
        user -> item
        relation IDs 0..3

    Reverse:
        item -> user
        relation IDs 4..7

    Node IDs remain raw TenRec IDs.
    src_node_type / dst_node_type disambiguate user IDs from item IDs.
    Later, the model loader can remap IDs to contiguous tensor indices.
    """
    if forward_edges.empty:
        return pd.DataFrame(
            columns=[
                "src_node_type",
                "src_id",
                "dst_node_type",
                "dst_id",
                "relation",
                "relation_id",
            ]
        )

    forward = pd.DataFrame({
        "src_node_type": "user",
        "src_id": forward_edges["user_id"].astype("int64"),
        "dst_node_type": "item",
        "dst_id": forward_edges["item_id"].astype("int64"),
        "relation": forward_edges["relation"],
        "relation_id": forward_edges["relation_id"].astype("int8"),
    })

    reverse_relation_names = {
        "click": "rev_click",
        "like": "rev_like",
        "share": "rev_share",
        "follow": "rev_follow",
    }

    reverse = pd.DataFrame({
        "src_node_type": "item",
        "src_id": forward_edges["item_id"].astype("int64"),
        "dst_node_type": "user",
        "dst_id": forward_edges["user_id"].astype("int64"),
        "relation": forward_edges["relation"].map(reverse_relation_names),
        "relation_id": forward_edges["relation"].map(
            REVERSE_RELATION_ID
        ).astype("int8"),
    })

    return pd.concat(
        [forward, reverse],
        ignore_index=True,
    )


def compute_popularity(canonical: pd.DataFrame) -> pd.DataFrame:
    """
    First-version popularity definition:

        Pop(i) = number of unique users with click(u,i)=1

    Because canonical already has one row per user-item pair,
    counting click-positive rows per item is equivalent to counting
    unique clicking users.
    """
    clicked = canonical.loc[
        canonical["click"] == 1,
        ["user_id", "item_id"],
    ].drop_duplicates()

    popularity = (
        clicked.groupby("item_id")["user_id"]
        .nunique()
        .rename("unique_click_users")
        .reset_index()
        .sort_values(
            "unique_click_users",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return popularity


def compute_user_activity(
    canonical: pd.DataFrame,
    domain: str,
) -> pd.DataFrame:
    """
    Per-user activity used only to audit whether train/val/test splitting
    is feasible later.

    click_items:
        number of distinct clicked items

    any_positive_items:
        number of distinct items with at least one positive relation

    relation_edges:
        total number of positive relation types over the user's items
        (click + like + share + follow after canonicalization)
    """
    users = pd.DataFrame({
        "user_id": sorted(canonical["user_id"].unique())
    })

    clicked = canonical[canonical["click"] == 1]

    click_counts = (
        clicked.groupby("user_id")["item_id"]
        .nunique()
        .rename(f"{domain}_click_items")
    )

    any_positive_mask = (
        canonical[RELATIONS].sum(axis=1) > 0
    )

    any_positive_counts = (
        canonical.loc[any_positive_mask]
        .groupby("user_id")["item_id"]
        .nunique()
        .rename(f"{domain}_any_positive_items")
    )

    relation_edge_counts = (
        canonical.assign(
            relation_edge_count=canonical[RELATIONS].sum(axis=1)
        )
        .groupby("user_id")["relation_edge_count"]
        .sum()
        .rename(f"{domain}_relation_edges")
    )

    users = (
        users
        .merge(
            click_counts,
            left_on="user_id",
            right_index=True,
            how="left",
        )
        .merge(
            any_positive_counts,
            left_on="user_id",
            right_index=True,
            how="left",
        )
        .merge(
            relation_edge_counts,
            left_on="user_id",
            right_index=True,
            how="left",
        )
    )

    count_cols = [
        f"{domain}_click_items",
        f"{domain}_any_positive_items",
        f"{domain}_relation_edges",
    ]

    users[count_cols] = (
        users[count_cols]
        .fillna(0)
        .astype("int64")
    )

    return users


def activity_distribution(series: pd.Series) -> dict:
    values = series.to_numpy(dtype=float)

    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def save_relation_mapping(output_dir: Path):
    rows = [
        {
            "relation": "click",
            "relation_id": 0,
            "direction": "user_to_item",
        },
        {
            "relation": "like",
            "relation_id": 1,
            "direction": "user_to_item",
        },
        {
            "relation": "share",
            "relation_id": 2,
            "direction": "user_to_item",
        },
        {
            "relation": "follow",
            "relation_id": 3,
            "direction": "user_to_item",
        },
        {
            "relation": "rev_click",
            "relation_id": 4,
            "direction": "item_to_user",
        },
        {
            "relation": "rev_like",
            "relation_id": 5,
            "direction": "item_to_user",
        },
        {
            "relation": "rev_share",
            "relation_id": 6,
            "direction": "item_to_user",
        },
        {
            "relation": "rev_follow",
            "relation_id": 7,
            "direction": "item_to_user",
        },
    ]

    pd.DataFrame(rows).to_csv(
        output_dir / "relation_mapping.csv",
        index=False,
    )


# ============================================================
# PREPARE ONE DOMAIN
# ============================================================

def prepare_domain(
    subset_size: int,
    domain: str,
):
    input_path = (
        INPUT_ROOT
        / str(subset_size)
        / f"{domain}.csv"
    )

    output_dir = (
        OUTPUT_ROOT
        / str(subset_size)
        / domain
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PREPARING {domain} — {subset_size:,} shared users")
    print("=" * 80)

    raw = load_and_normalize(input_path)

    raw_rows = len(raw)
    raw_users = raw["user_id"].nunique()
    raw_items = raw["item_id"].nunique()
    unique_pairs_before = raw[
        ["user_id", "item_id"]
    ].drop_duplicates().shape[0]

    canonical = aggregate_duplicate_pairs(raw)

    canonical_rows = len(canonical)
    duplicate_rows_removed = (
        raw_rows - canonical_rows
    )

    if canonical.duplicated(
        subset=["user_id", "item_id"]
    ).any():
        raise AssertionError(
            "Canonicalization failed: duplicated user-item pairs remain."
        )

    forward_edges = build_forward_relation_edges(
        canonical
    )

    bidirectional_edges = build_bidirectional_rgcn_edges(
        forward_edges
    )

    popularity = compute_popularity(
        canonical
    )

    user_activity = compute_user_activity(
        canonical,
        domain=domain,
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    canonical.to_csv(
        output_dir / "canonical_user_item.csv",
        index=False,
    )

    positive_pair_mask = (
        canonical[RELATIONS].sum(axis=1) > 0
    )

    canonical.loc[
        positive_pair_mask
    ].to_csv(
        output_dir / "positive_user_item_pairs.csv",
        index=False,
    )

    forward_edges.to_csv(
        output_dir / "forward_relation_edges.csv",
        index=False,
    )

    bidirectional_edges.to_csv(
        output_dir / "rgcn_edges_bidirectional.csv",
        index=False,
    )

    popularity.to_csv(
        output_dir / "item_popularity_click_degree.csv",
        index=False,
    )

    user_activity.to_csv(
        output_dir / "user_activity.csv",
        index=False,
    )

    save_relation_mapping(output_dir)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    relation_counts = {
        rel: int(canonical[rel].sum())
        for rel in RELATIONS
    }

    positive_pair_count = int(
        positive_pair_mask.sum()
    )

    graph_users = int(
        canonical.loc[
            positive_pair_mask,
            "user_id",
        ].nunique()
    )

    graph_items = int(
        canonical.loc[
            positive_pair_mask,
            "item_id",
        ].nunique()
    )

    summary = {
        "subset_size": subset_size,
        "domain": domain,

        "raw_rows": int(raw_rows),
        "raw_users": int(raw_users),
        "raw_items": int(raw_items),

        "unique_user_item_pairs_before": int(
            unique_pairs_before
        ),

        "canonical_pairs_after_aggregation": int(
            canonical_rows
        ),

        "duplicate_rows_removed": int(
            duplicate_rows_removed
        ),

        "positive_user_item_pairs": positive_pair_count,

        "click_edges_forward": relation_counts["click"],
        "like_edges_forward": relation_counts["like"],
        "share_edges_forward": relation_counts["share"],
        "follow_edges_forward": relation_counts["follow"],

        "total_forward_relation_edges": int(
            len(forward_edges)
        ),

        "total_bidirectional_rgcn_edges": int(
            len(bidirectional_edges)
        ),

        "positive_graph_users": graph_users,
        "positive_graph_items": graph_items,
        "positive_graph_nodes": graph_users + graph_items,

        "clicked_items": int(
            len(popularity)
        ),

        "max_item_click_degree": (
            int(popularity["unique_click_users"].max())
            if not popularity.empty
            else 0
        ),
    }

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    expected_forward_edges = sum(
        relation_counts.values()
    )

    assert (
        len(forward_edges)
        == expected_forward_edges
    ), (
        "Forward edge count mismatch."
    )

    assert (
        len(bidirectional_edges)
        == 2 * len(forward_edges)
    ), (
        "Bidirectional edge count must be exactly 2x forward edges."
    )

    assert (
        forward_edges.duplicated(
            subset=[
                "user_id",
                "item_id",
                "relation_id",
            ]
        ).sum()
        == 0
    ), (
        "Duplicate (user,item,relation) edges remain."
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(f"Raw rows                         : {raw_rows:,}")
    print(f"Raw users                        : {raw_users:,}")
    print(f"Raw items                        : {raw_items:,}")
    print(
        f"Unique user-item pairs           : "
        f"{unique_pairs_before:,}"
    )
    print(
        f"Canonical pairs                  : "
        f"{canonical_rows:,}"
    )
    print(
        f"Duplicate rows removed           : "
        f"{duplicate_rows_removed:,}"
    )

    print("\nForward relation edges:")
    for rel in RELATIONS:
        print(
            f"  {rel:<6}: "
            f"{relation_counts[rel]:>12,}"
        )

    print(
        f"\nForward relation edges total     : "
        f"{len(forward_edges):,}"
    )
    print(
        f"Bidirectional RGCN edges         : "
        f"{len(bidirectional_edges):,}"
    )
    print(
        f"Positive graph users             : "
        f"{graph_users:,}"
    )
    print(
        f"Positive graph items             : "
        f"{graph_items:,}"
    )
    print(
        f"Positive graph nodes             : "
        f"{graph_users + graph_items:,}"
    )

    print(f"\nSaved to: {output_dir.resolve()}")

    return summary, user_activity


# ============================================================
# CROSS-DOMAIN USER-ACTIVITY / SPLIT FEASIBILITY AUDIT
# ============================================================

def cross_domain_activity_audit(
    subset_size: int,
    video_activity: pd.DataFrame,
    article_activity: pd.DataFrame,
):
    output_dir = (
        OUTPUT_ROOT
        / str(subset_size)
        / "cross_domain_audit"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged = video_activity.merge(
        article_activity,
        on="user_id",
        how="outer",
        validate="one_to_one",
    ).fillna(0)

    count_cols = [
        c for c in merged.columns
        if c != "user_id"
    ]

    merged[count_cols] = (
        merged[count_cols]
        .astype("int64")
    )

    video_click_col = "QK-video_click_items"
    article_click_col = "QK-article_click_items"

    # --------------------------------------------------------
    # Threshold flags
    # --------------------------------------------------------

    for threshold in ACTIVITY_THRESHOLDS:
        merged[
            f"video_click_ge_{threshold}"
        ] = (
            merged[video_click_col] >= threshold
        )

        merged[
            f"article_click_ge_{threshold}"
        ] = (
            merged[article_click_col] >= threshold
        )

        merged[
            f"both_domains_click_ge_{threshold}"
        ] = (
            merged[f"video_click_ge_{threshold}"]
            & merged[f"article_click_ge_{threshold}"]
        )

    merged.to_csv(
        output_dir / "cross_domain_user_activity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    n_users = len(merged)

    audit_rows = []

    for threshold in ACTIVITY_THRESHOLDS:
        video_n = int(
            merged[f"video_click_ge_{threshold}"].sum()
        )
        article_n = int(
            merged[f"article_click_ge_{threshold}"].sum()
        )
        both_n = int(
            merged[
                f"both_domains_click_ge_{threshold}"
            ].sum()
        )

        audit_rows.append({
            "subset_size": subset_size,
            "threshold_click_items": threshold,

            "video_users_meeting_threshold": video_n,
            "video_ratio": video_n / n_users,

            "article_users_meeting_threshold": article_n,
            "article_ratio": article_n / n_users,

            "both_domains_users_meeting_threshold": both_n,
            "both_domains_ratio": both_n / n_users,
        })

    threshold_summary = pd.DataFrame(
        audit_rows
    )

    threshold_summary.to_csv(
        output_dir / "split_feasibility_thresholds.csv",
        index=False,
    )

    distributions = {
        "subset_size": subset_size,
        "num_users": n_users,
        "QK-video_click_items": activity_distribution(
            merged[video_click_col]
        ),
        "QK-article_click_items": activity_distribution(
            merged[article_click_col]
        ),
    }

    with open(
        output_dir / "click_activity_distribution.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            distributions,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print(
        f"CROSS-DOMAIN CLICK ACTIVITY AUDIT — "
        f"{subset_size:,} users"
    )
    print("-" * 80)

    video_stats = distributions[
        "QK-video_click_items"
    ]
    article_stats = distributions[
        "QK-article_click_items"
    ]

    print(
        "QK-video click items/user:"
        f" median={video_stats['median']:.1f},"
        f" p25={video_stats['p25']:.1f},"
        f" p75={video_stats['p75']:.1f},"
        f" p90={video_stats['p90']:.1f},"
        f" max={video_stats['max']:.0f}"
    )

    print(
        "QK-article click items/user:"
        f" median={article_stats['median']:.1f},"
        f" p25={article_stats['p25']:.1f},"
        f" p75={article_stats['p75']:.1f},"
        f" p90={article_stats['p90']:.1f},"
        f" max={article_stats['max']:.0f}"
    )

    print("\nUsers meeting click-item thresholds:")

    for _, row in threshold_summary.iterrows():
        t = int(row["threshold_click_items"])
        both_n = int(
            row["both_domains_users_meeting_threshold"]
        )
        both_ratio = row["both_domains_ratio"]

        print(
            f"  >= {t:>2} clicked items in BOTH domains: "
            f"{both_n:>6,} / {n_users:,} "
            f"({both_ratio:.2%})"
        )

    return threshold_summary


# ============================================================
# MAIN
# ============================================================

def main():
    global_summary_records = []
    split_summary_records = []

    for subset_size in SUBSET_SIZES:
        activity_by_domain = {}

        for domain in DOMAINS:
            summary, activity = prepare_domain(
                subset_size=subset_size,
                domain=domain,
            )

            global_summary_records.append(
                summary
            )

            activity_by_domain[domain] = activity

        threshold_summary = (
            cross_domain_activity_audit(
                subset_size=subset_size,
                video_activity=activity_by_domain["QK-video"],
                article_activity=activity_by_domain["QK-article"],
            )
        )

        split_summary_records.append(
            threshold_summary
        )

    # --------------------------------------------------------
    # Global summaries
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        global_summary_records
    )

    summary_path = (
        OUTPUT_ROOT
        / "rgcn_preparation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    if split_summary_records:
        split_df = pd.concat(
            split_summary_records,
            ignore_index=True,
        )

        split_path = (
            OUTPUT_ROOT
            / "split_feasibility_summary.csv"
        )

        split_df.to_csv(
            split_path,
            index=False,
        )
    else:
        split_path = None

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TENREC RGCN DATASET PREPARATION COMPLETE")
    print("=" * 80)

    print(f"Prepared root : {OUTPUT_ROOT.resolve()}")
    print(f"Model summary : {summary_path.resolve()}")

    if split_path is not None:
        print(f"Split summary : {split_path.resolve()}")

    print("\nIMPORTANT:")
    print(
        "1. This script does NOT filter users based on activity thresholds."
    )
    print(
        "2. It does NOT create train/validation/test splits yet."
    )
    print(
        "3. It does NOT modify your RGCN model."
    )
    print(
        "4. It only canonicalizes duplicates, constructs relation tables, "
        "computes popularity, and audits split feasibility."
    )

    print(
        "\nNext step: inspect rgcn_preparation_summary.csv and "
        "split_feasibility_summary.csv before changing the model."
    )


if __name__ == "__main__":
    main()
