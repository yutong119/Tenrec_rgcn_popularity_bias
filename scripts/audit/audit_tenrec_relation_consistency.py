from pathlib import Path
import pandas as pd


DATA_PATH = Path(
    "../../tenrec_rgcn_prepared"
) / "5000" / "QK-video" / "canonical_user_item.csv"


df = pd.read_csv(DATA_PATH)

relations = [
    "like",
    "share",
    "follow",
]


print("=" * 70)
print("QK-video RELATION CONSISTENCY AUDIT")
print("=" * 70)

print(f"Canonical user-item pairs: {len(df):,}")


for relation in relations:

    mask = (
        (df["click"] == 0)
        &
        (df[relation] == 1)
    )

    subset = df.loc[mask]

    print(
        f"\nclick=0 but {relation}=1"
    )

    print(
        f"Pairs : {len(subset):,}"
    )

    print(
        f"Users : {subset['user_id'].nunique():,}"
    )

    print(
        f"Items : {subset['item_id'].nunique():,}"
    )

    print(
        f"Ratio : {len(subset) / len(df):.4%}"
    )


strong_mask = (
    (df["click"] == 0)
    &
    (
        (df["like"] == 1)
        |
        (df["share"] == 1)
        |
        (df["follow"] == 1)
    )
)

strong_subset = df.loc[strong_mask]

print("\n" + "-" * 70)

print(
    "ANY like/share/follow = 1 "
    "while click = 0"
)

print("-" * 70)

print(
    f"Pairs : {len(strong_subset):,}"
)

print(
    f"Users : "
    f"{strong_subset['user_id'].nunique():,}"
)

print(
    f"Items : "
    f"{strong_subset['item_id'].nunique():,}"
)

print(
    f"Ratio : "
    f"{len(strong_subset) / len(df):.4%}"
)


print("\nDetailed relation combinations:")

combination_counts = (
    strong_subset
    .groupby(
        ["click", "like", "share", "follow"]
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(combination_counts)