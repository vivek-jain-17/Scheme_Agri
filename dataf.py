import pandas as pd
import re
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT = Path("Schemes.csv")
OUTPUT = Path("agriculture_schemes.csv")


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Original dataset: {len(df)} schemes")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 3. REMOVE DUPLICATE SCHEMES
# ============================================================

df = df.drop_duplicates(
    subset=["name"]
).copy()

print(f"After duplicate removal: {len(df)} schemes")


# ============================================================
# 4. FILTER AGRICULTURE SCHEMES
# ============================================================

mask = (
    df["category"]
    .fillna("")
    .astype(str)
    .str.contains(
        "Agriculture",
        case=False,
        regex=False
    )
)

agriculture_df = df.loc[mask].copy()

print(f"Agriculture schemes: {len(agriculture_df)}")


# ============================================================
# 5. CREATE COMBINED SEMANTIC TEXT
# ============================================================

text_cols = [
    "name",
    "description",
    "benefits",
    "category",
    "beneficiary_type",
    "eligibility_text"
]

# Make sure missing values don't create NaN
for col in text_cols:
    agriculture_df[col] = (
        agriculture_df[col]
        .fillna("")
        .astype(str)
    )


agriculture_df["scheme_text"] = (
    agriculture_df[text_cols]
    .agg(" ".join, axis=1)
)


# ============================================================
# 6. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove markdown symbols
    text = re.sub(
        r"[*#>`_]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


agriculture_df["scheme_text"] = (
    agriculture_df["scheme_text"]
    .apply(clean_text)
)


# ============================================================
# 7. BASIC QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("AGRICULTURE DATASET")
print("=" * 60)

print(
    agriculture_df[
        [
            "name",
            "category",
            "beneficiary_type"
        ]
    ].head(10)
)

print("\nMissing values in important fields:")

check_cols = [
    "name",
    "description",
    "benefits",
    "eligibility_text",
    "application_process",
    "documents_required"
]

print(
    agriculture_df[check_cols]
    .isnull()
    .sum()
)


# ============================================================
# 8. SAVE FINAL AGRICULTURE DATASET
# ============================================================

agriculture_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 0 COMPLETE")
print("=" * 60)

print(f"Original schemes       : {len(df)}")
print(f"Agriculture schemes    : {len(agriculture_df)}")
print(f"Final dataset          : {OUTPUT}")

print("\nDataset is ready for Phase 1.")