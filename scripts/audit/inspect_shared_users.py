from pathlib import Path
import pandas as pd

TENREC_DIR = Path(r"D:\datasets\TenRec")

video_path = TENREC_DIR / "QK-video.csv"
article_path = TENREC_DIR / "QK-article.csv"

CHUNK_SIZE = 1_000_000


def collect_unique_users(path, name):
    users = set()
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        usecols=["user_id"],
        chunksize=CHUNK_SIZE
    ):
        users.update(
            chunk["user_id"]
            .dropna()
            .astype("int64")
            .unique()
        )

        total_rows += len(chunk)

        print(
            f"{name}: "
            f"{total_rows:,} rows processed, "
            f"{len(users):,} unique users"
        )

    return users


print("\nReading QK-video users...")
video_users = collect_unique_users(
    video_path,
    "QK-video"
)

print("\nReading QK-article users...")
article_users = collect_unique_users(
    article_path,
    "QK-article"
)

shared_users = video_users & article_users

print("\n" + "=" * 60)
print("USER STATISTICS")
print("=" * 60)

print(
    f"QK-video unique users: "
    f"{len(video_users):,}"
)

print(
    f"QK-article unique users: "
    f"{len(article_users):,}"
)

print(
    f"Shared users: "
    f"{len(shared_users):,}"
)

print(
    f"Video overlap ratio: "
    f"{len(shared_users) / len(video_users):.4%}"
)

print(
    f"Article overlap ratio: "
    f"{len(shared_users) / len(article_users):.4%}"
)