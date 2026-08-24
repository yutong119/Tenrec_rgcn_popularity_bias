from pathlib import Path
import pandas as pd
from collections import Counter

TENREC_DIR = Path(r"D:\datasets\TenRec")

video_path = TENREC_DIR / "QK-video.csv"
article_path = TENREC_DIR / "QK-article.csv"

relations = ["click", "like", "share", "follow"]

CHUNK_SIZE = 500_000
MAX_ROWS = 5_000_000


def inspect_relations(path, domain_name):
    counters = {r: Counter() for r in relations}
    null_counts = {r: 0 for r in relations}
    total_rows = 0

    usecols = ["user_id", "item_id"] + relations

    for chunk_id, chunk in enumerate(
            pd.read_csv(
                path,
                usecols=usecols,
                chunksize=CHUNK_SIZE
            )
    ):
        total_rows += len(chunk)

        for relation in relations:
            null_counts[relation] += chunk[relation].isna().sum()

            values = chunk[relation].dropna()
            counters[relation].update(values.astype(str))

        print(
            f"{domain_name}: processed "
            f"{total_rows:,} rows"
        )

        if total_rows >= MAX_ROWS:
            print(
                f"{domain_name}: reached sample limit "
                f"of {MAX_ROWS:,} rows."
            )
            break

    print("\n" + "=" * 60)
    print(domain_name)
    print("=" * 60)
    print(f"Total rows: {total_rows:,}")

    for relation in relations:
        print(f"\nRelation: {relation}")
        print("Values:", counters[relation])
        print("Missing:", null_counts[relation])


inspect_relations(video_path, "QK-video")
inspect_relations(article_path, "QK-article")