from pathlib import Path
import math
import shutil
import numpy as np
import pandas as pd


TENREC_DIR = Path(r"D:\datasets\TenRec")
VIDEO_RAW = TENREC_DIR / "QK-video.csv"
ARTICLE_RAW = TENREC_DIR / "QK-article.csv"

OUTPUT_DIR = Path("../../tenrec_shared_subset_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MASTER_N = 10_000
SUBSET_SIZES = [1_000, 3_000, 5_000, 10_000]
RELATIONS = ["click", "like", "share", "follow"]
KEEP_COLUMNS = ["user_id", "item_id", "click", "like", "share", "follow"]

USER_SCAN_CHUNK_SIZE = 1_000_000
FILTER_CHUNK_SIZE = 500_000

SHARED_USERS_FILE = OUTPUT_DIR / "shared_users_all.csv"
MASTER_USERS_FILE = OUTPUT_DIR / "master_shared_users_10k.csv"
VERIFICATION_DIR = OUTPUT_DIR / "verification"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 收集 / 读取 shared users
# ============================================================
def collect_unique_users(path: Path, name: str) -> set:
    users = set()
    total_rows = 0

    for chunk_id, chunk in enumerate(
        pd.read_csv(path, usecols=["user_id"], chunksize=USER_SCAN_CHUNK_SIZE),
        start=1,
    ):
        ids = (
            pd.to_numeric(chunk["user_id"], errors="coerce")
            .dropna()
            .astype("int64")
            .unique()
        )
        users.update(ids)
        total_rows += len(chunk)

        if chunk_id % 10 == 0:
            print(
                f"{name}: {total_rows:,} rows processed, "
                f"{len(users):,} unique users"
            )

    print(f"{name}: DONE — {total_rows:,} rows, {len(users):,} unique users")
    return users


def get_shared_users() -> np.ndarray:
    if SHARED_USERS_FILE.exists():
        print(f"Loading cached shared users: {SHARED_USERS_FILE}")
        shared = pd.read_csv(SHARED_USERS_FILE)["user_id"].astype("int64").to_numpy()
        print(f"Loaded {len(shared):,} shared users")
        return shared

    print("\nScanning QK-video users...")
    video_users = collect_unique_users(VIDEO_RAW, "QK-video")

    print("\nScanning QK-article users...")
    article_users = collect_unique_users(ARTICLE_RAW, "QK-article")

    shared = np.array(sorted(video_users & article_users), dtype=np.int64)

    print("\n" + "=" * 60)
    print("FULL SHARED-USER STATISTICS")
    print("=" * 60)
    print(f"QK-video unique users  : {len(video_users):,}")
    print(f"QK-article unique users: {len(article_users):,}")
    print(f"Shared users           : {len(shared):,}")
    print(f"Video overlap ratio    : {len(shared) / len(video_users):.4%}")
    print(f"Article overlap ratio  : {len(shared) / len(article_users):.4%}")

    pd.DataFrame({"user_id": shared}).to_csv(SHARED_USERS_FILE, index=False)
    print(f"Saved: {SHARED_USERS_FILE}")
    return shared


# ============================================================
# 2. 固定 seed 建 10k master cohort
# ============================================================
def build_master_users(shared_users: np.ndarray) -> np.ndarray:
    if MASTER_USERS_FILE.exists():
        df = pd.read_csv(MASTER_USERS_FILE)
        master = df["user_id"].astype("int64").to_numpy()
        if len(master) != MASTER_N:
            raise ValueError(
                f"Existing master file contains {len(master)} users, expected {MASTER_N}."
            )
        print(f"Loaded existing master cohort: {MASTER_USERS_FILE}")
        return master

    if len(shared_users) < MASTER_N:
        raise ValueError(
            f"Only {len(shared_users):,} shared users available; cannot sample {MASTER_N:,}."
        )

    rng = np.random.default_rng(SEED)
    # 保留随机顺序，这样 first 1k ⊂ first 3k ⊂ first 5k ⊂ first 10k
    master = rng.permutation(shared_users)[:MASTER_N]

    pd.DataFrame(
        {"rank": np.arange(1, MASTER_N + 1), "user_id": master}
    ).to_csv(MASTER_USERS_FILE, index=False)

    print(f"Saved 10k master cohort: {MASTER_USERS_FILE}")
    return master


# ============================================================
# 3. 原始大文件只扫描一次，过滤到 10k master users
# ============================================================
def filter_raw_to_master(
    raw_path: Path,
    output_path: Path,
    master_users: np.ndarray,
    domain_name: str,
):
    if output_path.exists():
        print(f"{domain_name}: master subset already exists, skip: {output_path}")
        return

    master_set = set(int(x) for x in master_users)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    first_write = True

    print(f"\nFiltering {domain_name} to 10k master users...")

    for chunk_id, chunk in enumerate(
        pd.read_csv(raw_path, usecols=KEEP_COLUMNS, chunksize=FILTER_CHUNK_SIZE),
        start=1,
    ):
        total_rows += len(chunk)
        chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce").astype("Int64")
        filtered = chunk[chunk["user_id"].isin(master_set)].copy()

        if not filtered.empty:
            filtered["user_id"] = filtered["user_id"].astype("int64")
            filtered.to_csv(
                output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )
            first_write = False
            kept_rows += len(filtered)

        if chunk_id % 10 == 0:
            print(
                f"{domain_name}: scanned {total_rows:,} raw rows, "
                f"kept {kept_rows:,}"
            )

    if first_write:
        raise RuntimeError(
            f"{domain_name}: no rows found for selected users. Check IDs/path."
        )

    print(f"{domain_name}: DONE — scanned {total_rows:,}, kept {kept_rows:,}")
    print(f"Saved: {output_path}")


# ============================================================
# 4. 从 10k master 派生 1k/3k/5k/10k 嵌套 subset
# ============================================================
def derive_nested_subsets(
    master_file: Path,
    master_users: np.ndarray,
    domain_name: str,
):
    master_df = pd.read_csv(master_file, usecols=KEEP_COLUMNS)
    master_df["user_id"] = master_df["user_id"].astype("int64")

    for n in SUBSET_SIZES:
        subset_dir = OUTPUT_DIR / "subsets" / str(n)
        subset_dir.mkdir(parents=True, exist_ok=True)
        output_path = subset_dir / f"{domain_name}.csv"

        if n == MASTER_N:
            shutil.copy2(master_file, output_path)
            print(f"{domain_name} {n:,}: copied -> {output_path}")
            continue

        selected_users = set(int(x) for x in master_users[:n])
        subset_df = master_df[master_df["user_id"].isin(selected_users)].copy()
        subset_df.to_csv(output_path, index=False)

        print(
            f"{domain_name} {n:,}: "
            f"{subset_df['user_id'].nunique():,} users, "
            f"{len(subset_df):,} rows -> {output_path}"
        )


def save_nested_user_lists(master_users: np.ndarray):
    for n in SUBSET_SIZES:
        subset_dir = OUTPUT_DIR / "subsets" / str(n)
        subset_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"rank": np.arange(1, n + 1), "user_id": master_users[:n]}
        ).to_csv(subset_dir / "shared_user_ids.csv", index=False)


# ============================================================
# 5. relation 标准化
# ============================================================
def to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype("int8")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return (numeric != 0).astype("int8")

    normalized = series.astype(str).str.strip().str.lower()
    positive_values = {"1", "1.0", "true", "t", "yes", "y"}
    return normalized.isin(positive_values).astype("int8")


# ============================================================
# 6. long-tail / popularity helpers
# ============================================================
def gini_coefficient(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0

    x = np.sort(x)
    n = len(x)
    cumulative = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def compute_head_click_share(popularity: pd.Series, head_ratio=0.20) -> float:
    if len(popularity) == 0 or popularity.sum() == 0:
        return float("nan")
    sorted_pop = popularity.sort_values(ascending=False)
    head_n = max(1, math.ceil(len(sorted_pop) * head_ratio))
    return float(sorted_pop.iloc[:head_n].sum() / sorted_pop.sum())


def estimate_graph_storage_mb(num_nodes: int, bidirectional_edges: int, embedding_dim: int):
    # edge_index [2,E] int64 = 16 bytes/edge
    # edge_type  [E]   int64 =  8 bytes/edge
    # node embeddings float32 = 4*N*d bytes
    edge_bytes = bidirectional_edges * 24
    embedding_bytes = num_nodes * embedding_dim * 4
    return {
        "edge_tensors_mb_lower_bound": edge_bytes / (1024 ** 2),
        "embedding_mb_lower_bound": embedding_bytes / (1024 ** 2),
        "combined_mb_lower_bound": (edge_bytes + embedding_bytes) / (1024 ** 2),
    }


# ============================================================
# 7. 验证单个 domain / subset
# ============================================================
def verify_subset_file(csv_path: Path, domain: str, requested_users: int):
    print("\n" + "=" * 70)
    print(f"{domain} — requested shared users: {requested_users:,}")
    print("=" * 70)

    df = pd.read_csv(csv_path, usecols=KEEP_COLUMNS)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce").astype("Int64")

    missing_user = int(df["user_id"].isna().sum())
    missing_item = int(df["item_id"].isna().sum())
    if missing_user == 0:
        df["user_id"] = df["user_id"].astype("int64")
    if missing_item == 0:
        df["item_id"] = df["item_id"].astype("int64")

    relation_df = pd.DataFrame({rel: to_binary(df[rel]) for rel in RELATIONS})

    row_count = len(df)
    users_present = int(df["user_id"].nunique(dropna=True))
    all_items_present = int(df["item_id"].nunique(dropna=True))

    # 注意：keep=False 会把所有重复 occurrence 都计数，不是“重复组数”。
    duplicate_user_item_occurrences = int(
        df.duplicated(subset=["user_id", "item_id"], keep=False).sum()
    )
    exact_duplicate_occurrences = int(
        df.duplicated(subset=KEEP_COLUMNS, keep=False).sum()
    )

    relation_counts = {rel: int(relation_df[rel].sum()) for rel in RELATIONS}
    relation_rates = {
        rel: (relation_counts[rel] / row_count if row_count else float("nan"))
        for rel in RELATIONS
    }

    positive_relation_edges = int(sum(relation_counts.values()))
    bidirectional_edges = int(2 * positive_relation_edges)

    any_positive_relation = relation_df[RELATIONS].sum(axis=1) > 0
    positive_graph_users = int(df.loc[any_positive_relation, "user_id"].nunique())
    positive_graph_items = int(df.loc[any_positive_relation, "item_id"].nunique())
    positive_graph_nodes = positive_graph_users + positive_graph_items

    # Popularity = unique clicking users per item
    click_mask = relation_df["click"] == 1
    clicked_pairs = (
        df.loc[click_mask, ["user_id", "item_id"]]
        .dropna()
        .drop_duplicates()
    )
    popularity = (
        clicked_pairs.groupby("item_id")["user_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    clicked_items = len(popularity)
    unique_click_pairs = len(clicked_pairs)

    if clicked_items:
        x = popularity.to_numpy(dtype=float)
        pop_min = float(np.min(x))
        pop_mean = float(np.mean(x))
        pop_median = float(np.median(x))
        pop_max = float(np.max(x))
        p50 = float(np.percentile(x, 50))
        p80 = float(np.percentile(x, 80))
        p90 = float(np.percentile(x, 90))
        p95 = float(np.percentile(x, 95))
        p99 = float(np.percentile(x, 99))
        degree1_ratio = float(np.mean(x == 1))
        gini = gini_coefficient(x)
        head20_share = compute_head_click_share(popularity, 0.20)
    else:
        pop_min = pop_mean = pop_median = pop_max = float("nan")
        p50 = p80 = p90 = p95 = p99 = float("nan")
        degree1_ratio = gini = head20_share = float("nan")

    pop_path = VERIFICATION_DIR / f"popularity_{domain}_{requested_users}.csv"
    popularity.rename("unique_click_users").reset_index().to_csv(pop_path, index=False)

    print(f"Rows                      : {row_count:,}")
    print(f"Users present             : {users_present:,}")
    print(f"All items in rows         : {all_items_present:,}")
    print(f"Missing user IDs          : {missing_user:,}")
    print(f"Missing item IDs          : {missing_item:,}")
    print(f"Duplicate user-item occ.  : {duplicate_user_item_occurrences:,}")
    print(f"Exact duplicate occ.      : {exact_duplicate_occurrences:,}")

    print("\nPositive relation counts:")
    for rel in RELATIONS:
        print(f"  {rel:<6}: {relation_counts[rel]:>12,} ({relation_rates[rel]:.4%})")

    print(f"\nForward relation edges    : {positive_relation_edges:,}")
    print(f"Bidirectional RGCN edges   : {bidirectional_edges:,}")
    print(f"Positive-graph users       : {positive_graph_users:,}")
    print(f"Positive-graph items       : {positive_graph_items:,}")
    print(f"Positive-graph nodes       : {positive_graph_nodes:,}")

    print("\nPopularity = unique clicking users / item")
    print(f"  Clicked items            : {clicked_items:,}")
    print(f"  Unique click pairs       : {unique_click_pairs:,}")
    print(f"  Min / Mean / Median / Max: {pop_min:.2f} / {pop_mean:.2f} / {pop_median:.2f} / {pop_max:.2f}")
    print(f"  P50/P80/P90/P95/P99      : {p50:.2f}/{p80:.2f}/{p90:.2f}/{p95:.2f}/{p99:.2f}")
    print(f"  Degree=1 ratio           : {degree1_ratio:.4%}")
    print(f"  Gini                     : {gini:.4f}")
    print(f"  Top20% items click share : {head20_share:.4%}")

    est64 = estimate_graph_storage_mb(positive_graph_nodes, bidirectional_edges, 64)
    est128 = estimate_graph_storage_mb(positive_graph_nodes, bidirectional_edges, 128)

    print("\nVery rough tensor-storage lower bound (NOT full training memory):")
    print(
        f"  d=64 : edges={est64['edge_tensors_mb_lower_bound']:.2f} MB, "
        f"emb={est64['embedding_mb_lower_bound']:.2f} MB, "
        f"combined={est64['combined_mb_lower_bound']:.2f} MB"
    )
    print(
        f"  d=128: edges={est128['edge_tensors_mb_lower_bound']:.2f} MB, "
        f"emb={est128['embedding_mb_lower_bound']:.2f} MB, "
        f"combined={est128['combined_mb_lower_bound']:.2f} MB"
    )

    return {
        "subset_users_requested": requested_users,
        "domain": domain,
        "rows": row_count,
        "users_present": users_present,
        "all_items_present": all_items_present,
        "missing_user_ids": missing_user,
        "missing_item_ids": missing_item,
        "duplicate_user_item_occurrences": duplicate_user_item_occurrences,
        "exact_duplicate_occurrences": exact_duplicate_occurrences,
        "click_edges_forward": relation_counts["click"],
        "like_edges_forward": relation_counts["like"],
        "share_edges_forward": relation_counts["share"],
        "follow_edges_forward": relation_counts["follow"],
        "click_positive_rate": relation_rates["click"],
        "like_positive_rate": relation_rates["like"],
        "share_positive_rate": relation_rates["share"],
        "follow_positive_rate": relation_rates["follow"],
        "total_positive_relation_edges_forward": positive_relation_edges,
        "total_edges_bidirectional": bidirectional_edges,
        "positive_graph_users": positive_graph_users,
        "positive_graph_items": positive_graph_items,
        "positive_graph_nodes": positive_graph_nodes,
        "clicked_items": clicked_items,
        "unique_click_pairs": unique_click_pairs,
        "pop_min": pop_min,
        "pop_mean": pop_mean,
        "pop_median": pop_median,
        "pop_max": pop_max,
        "pop_p50": p50,
        "pop_p80": p80,
        "pop_p90": p90,
        "pop_p95": p95,
        "pop_p99": p99,
        "pop_degree1_ratio": degree1_ratio,
        "pop_gini": gini,
        "pop_top20_click_share": head20_share,
        "edge_tensor_mb_lower_bound": est64["edge_tensors_mb_lower_bound"],
        "combined_lower_bound_mb_d64": est64["combined_mb_lower_bound"],
        "combined_lower_bound_mb_d128": est128["combined_mb_lower_bound"],
    }


# ============================================================
# 8. 验证两个 domain 确实使用同一批 users
# ============================================================
def verify_cross_domain_user_alignment(n: int):
    subset_dir = OUTPUT_DIR / "subsets" / str(n)
    expected_users = set(
        pd.read_csv(subset_dir / "shared_user_ids.csv")["user_id"]
        .astype("int64")
        .tolist()
    )
    video_users = set(
        pd.read_csv(subset_dir / "QK-video.csv", usecols=["user_id"])["user_id"]
        .astype("int64")
        .unique()
        .tolist()
    )
    article_users = set(
        pd.read_csv(subset_dir / "QK-article.csv", usecols=["user_id"])["user_id"]
        .astype("int64")
        .unique()
        .tolist()
    )

    print("\n" + "-" * 70)
    print(f"CROSS-DOMAIN USER ALIGNMENT — N={n:,}")
    print("-" * 70)
    print(f"Expected users         : {len(expected_users):,}")
    print(f"Video users            : {len(video_users):,}")
    print(f"Article users          : {len(article_users):,}")
    print(f"Video ∩ Article        : {len(video_users & article_users):,}")
    print(f"Video == expected      : {video_users == expected_users}")
    print(f"Article == expected    : {article_users == expected_users}")
    print(f"Video users == Article : {video_users == article_users}")

    return {
        "subset_users_requested": n,
        "expected_users": len(expected_users),
        "video_users": len(video_users),
        "article_users": len(article_users),
        "video_article_overlap": len(video_users & article_users),
        "video_equals_expected": video_users == expected_users,
        "article_equals_expected": article_users == expected_users,
        "video_equals_article": video_users == article_users,
    }


# ============================================================
# 9. MAIN
# ============================================================
def main():
    if not VIDEO_RAW.exists():
        raise FileNotFoundError(f"Missing: {VIDEO_RAW}")
    if not ARTICLE_RAW.exists():
        raise FileNotFoundError(f"Missing: {ARTICLE_RAW}")

    # A. shared users
    shared_users = get_shared_users()

    # B. fixed 10k master cohort
    master_users = build_master_users(shared_users)

    # C. raw -> master 10k
    video_master = OUTPUT_DIR / "master_10k" / "QK-video.csv"
    article_master = OUTPUT_DIR / "master_10k" / "QK-article.csv"

    filter_raw_to_master(VIDEO_RAW, video_master, master_users, "QK-video")
    filter_raw_to_master(ARTICLE_RAW, article_master, master_users, "QK-article")

    # D. nested subsets
    save_nested_user_lists(master_users)
    derive_nested_subsets(video_master, master_users, "QK-video")
    derive_nested_subsets(article_master, master_users, "QK-article")

    # E. verify
    summary_records = []
    alignment_records = []

    for n in SUBSET_SIZES:
        alignment_records.append(verify_cross_domain_user_alignment(n))
        for domain in ["QK-video", "QK-article"]:
            path = OUTPUT_DIR / "subsets" / str(n) / f"{domain}.csv"
            summary_records.append(verify_subset_file(path, domain, n))

    summary_df = pd.DataFrame(summary_records)
    summary_path = VERIFICATION_DIR / "subset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    alignment_df = pd.DataFrame(alignment_records)
    alignment_path = VERIFICATION_DIR / "user_alignment_summary.csv"
    alignment_df.to_csv(alignment_path, index=False)

    print("\n" + "=" * 70)
    print("ALL SHARED-USER SUBSET VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Summary CSV   : {summary_path.resolve()}")
    print(f"Alignment CSV : {alignment_path.resolve()}")
    print(f"Popularity CSVs are in: {VERIFICATION_DIR.resolve()}")
    print("\nSTOP HERE. Do not modify RGCN yet. Send me subset_summary.csv or terminal output.")


if __name__ == "__main__":
    main()
