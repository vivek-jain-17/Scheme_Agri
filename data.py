import pandas as pd
import numpy as np
import re
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT = Path("Schemes.csv")
OUTPUT_DIR = Path("phase0_output")

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"Original dataset shape : {df.shape}")
print(f"Number of schemes      : {len(df)}")
print(f"Number of columns      : {len(df.columns)}")


# ============================================================
# 3. BASIC DATASET AUDIT
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET AUDIT")
print("=" * 70)

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2}. {col}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum().sort_values(ascending=False)

missing_percentage = (
    df.isnull().mean() * 100
).sort_values(ascending=False)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percentage": missing_percentage.round(2)
})

print(missing_report)


# Save missingness report
missing_report.to_csv(
    OUTPUT_DIR / "dataset_missingness_report.csv"
)


# ============================================================
# 4. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

total_duplicates = df.duplicated().sum()

print(f"Complete duplicate rows: {total_duplicates}")

if "name" in df.columns:
    duplicate_names = df["name"].duplicated().sum()
    print(f"Duplicate scheme names : {duplicate_names}")
else:
    print("WARNING: 'name' column not found.")


# ============================================================
# 5. AGRICULTURE FILTER — PRIMARY SCOPE
# ============================================================

print("\n" + "=" * 70)
print("AGRICULTURE SCHEME FILTER")
print("=" * 70)

if "category" not in df.columns:
    raise ValueError("Column 'category' is required but was not found.")

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

agriculture_df = (
    df.loc[mask]
    .drop_duplicates(subset=["name"])
    .copy()
)

print(f"Original schemes       : {len(df)}")
print(f"Agriculture schemes    : {len(agriculture_df)}")
print(
    f"Percentage of dataset  : "
    f"{len(agriculture_df) / len(df) * 100:.2f}%"
)


# ============================================================
# 6. AGRICULTURE CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("AGRICULTURE CATEGORY DISTRIBUTION")
print("=" * 70)

category_distribution = (
    agriculture_df["category"]
    .fillna("Unknown")
    .value_counts()
)

print(category_distribution)

category_distribution.to_csv(
    OUTPUT_DIR / "agriculture_category_distribution.csv"
)


# ============================================================
# 7. BENEFICIARY TYPE DISTRIBUTION
# ============================================================

if "beneficiary_type" in agriculture_df.columns:

    print("\n" + "=" * 70)
    print("BENEFICIARY TYPE DISTRIBUTION")
    print("=" * 70)

    beneficiary_distribution = (
        agriculture_df["beneficiary_type"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    print(beneficiary_distribution)

    beneficiary_distribution.to_csv(
        OUTPUT_DIR / "agriculture_beneficiary_distribution.csv"
    )


# ============================================================
# 8. STATE DISTRIBUTION
# ============================================================

if "state" in agriculture_df.columns:

    print("\n" + "=" * 70)
    print("STATE DISTRIBUTION")
    print("=" * 70)

    state_distribution = (
        agriculture_df["state"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    print(state_distribution.head(30))

    state_distribution.to_csv(
        OUTPUT_DIR / "agriculture_state_distribution.csv"
    )


# ============================================================
# 9. IMPORTANT TEXT FIELD CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT TEXT FIELD CHECK")
print("=" * 70)

text_cols = [
    "name",
    "description",
    "benefits",
    "category",
    "beneficiary_type",
    "eligibility_text",
    "application_process",
    "documents_required"
]

available_text_cols = [
    col for col in text_cols
    if col in agriculture_df.columns
]

text_missing_report = pd.DataFrame({
    "missing_count": agriculture_df[available_text_cols].isnull().sum(),
    "missing_percentage": (
        agriculture_df[available_text_cols]
        .isnull()
        .mean() * 100
    ).round(2)
})

print(text_missing_report)

text_missing_report.to_csv(
    OUTPUT_DIR / "agriculture_text_field_missingness.csv"
)


# ============================================================
# 10. CREATE COMBINED SCHEME TEXT
# ============================================================

print("\n" + "=" * 70)
print("CREATING SCHEME TEXT")
print("=" * 70)

semantic_text_cols = [
    "name",
    "description",
    "benefits",
    "category",
    "beneficiary_type",
    "eligibility_text"
]

available_semantic_cols = [
    col for col in semantic_text_cols
    if col in agriculture_df.columns
]

for col in available_semantic_cols:
    agriculture_df[col] = (
        agriculture_df[col]
        .fillna("")
        .astype(str)
    )


# Combine important information into one field
agriculture_df["scheme_text"] = (
    agriculture_df[available_semantic_cols]
    .agg(" ".join, axis=1)
)


# ============================================================
# 11. CLEAN SCHEME TEXT
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

    # Replace URLs with space
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize whitespace
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
# 12. TEXT LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TEXT LENGTH ANALYSIS")
print("=" * 70)

agriculture_df["text_length"] = (
    agriculture_df["scheme_text"]
    .str.len()
)

print(
    agriculture_df["text_length"]
    .describe()
)

print("\nShortest schemes:")

print(
    agriculture_df[
        ["name", "text_length"]
    ]
    .sort_values("text_length")
    .head(10)
)


# ============================================================
# 13. IDENTIFY BROADER AGRICULTURE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("BROAD AGRICULTURE CANDIDATE SEARCH")
print("=" * 70)

# These candidates are NOT automatically accepted.
# They are created for review because a scheme may be
# agriculture-related even when its category does not
# explicitly contain "Agriculture".

agriculture_keywords = [

    "agricultur",
    "farmer",
    "farming",
    "crop",
    "irrigation",
    "fertilizer",
    "fertiliser",
    "soil",
    "horticulture",
    "livestock",
    "dairy",
    "fisher",
    "fisheries",
    "poultry",
    "beekeep",
    "apiculture",
    "kisan",
    "farm machinery",
    "farm equipment",
    "seed",
    "seeds",
    "organic farming",
    "natural farming",
    "agri",
    "agro",
    "cultivation",
    "harvest",
    "farmland",
    "agricultural",
    "animal husbandry"
]


candidate_search_cols = [
    "name",
    "description",
    "benefits",
    "category",
    "beneficiary_type",
    "eligibility_text",
    "ministry",
    "department"
]

available_candidate_cols = [
    col for col in candidate_search_cols
    if col in df.columns
]


combined_candidate_text = (
    df[available_candidate_cols]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)


pattern = "|".join(
    re.escape(keyword.lower())
    for keyword in agriculture_keywords
)


candidate_mask = combined_candidate_text.str.contains(
    pattern,
    regex=True,
    na=False
)


agriculture_candidates = (
    df.loc[candidate_mask]
    .drop_duplicates(subset=["name"])
    .copy()
)

print(
    f"Keyword-based agriculture candidates: "
    f"{len(agriculture_candidates)}"
)


# ============================================================
# 14. FIND ADDITIONAL CANDIDATES NOT IN CATEGORY FILTER
# ============================================================

print("\n" + "=" * 70)
print("ADDITIONAL AGRICULTURE CANDIDATES")
print("=" * 70)

agriculture_names = set(
    agriculture_df["name"]
    .dropna()
    .astype(str)
)

extra_candidates = agriculture_candidates[
    ~agriculture_candidates["name"]
    .astype(str)
    .isin(agriculture_names)
].copy()

print(
    f"Additional candidates outside "
    f"category-based subset: {len(extra_candidates)}"
)


# ============================================================
# 15. DISPLAY ADDITIONAL CANDIDATES FOR MANUAL REVIEW
# ============================================================

if len(extra_candidates) > 0:

    display_cols = [
        col for col in [
            "name",
            "category",
            "beneficiary_type",
            "description",
            "benefits"
        ]
        if col in extra_candidates.columns
    ]

    print("\nFirst 30 additional candidates:\n")

    print(
        extra_candidates[display_cols]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 16. SAVE INITIAL AGRICULTURE DATASET
# ============================================================

initial_output = (
    OUTPUT_DIR /
    "agriculture_schemes_clean_v1.csv"
)

agriculture_df.to_csv(
    initial_output,
    index=False
)


# ============================================================
# 17. SAVE BROADER CANDIDATES
# ============================================================

candidate_output = (
    OUTPUT_DIR /
    "agriculture_candidates_for_review.csv"
)

extra_candidates.to_csv(
    candidate_output,
    index=False
)


# ============================================================
# 18. SAVE ALL AGRICULTURE KEYWORD CANDIDATES
# ============================================================

all_candidate_output = (
    OUTPUT_DIR /
    "all_agriculture_keyword_candidates.csv"
)

agriculture_candidates.to_csv(
    all_candidate_output,
    index=False
)


# ============================================================
# 19. SAVE FINAL DATASET PREVIEW
# ============================================================

preview_output = (
    OUTPUT_DIR /
    "agriculture_dataset_preview.csv"
)

agriculture_df.head(50).to_csv(
    preview_output,
    index=False
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 0 COMPLETE")
print("=" * 70)

print(f"""
Original dataset:
    {len(df)} schemes

Category-based agriculture subset:
    {len(agriculture_df)} schemes

Keyword-based agriculture candidates:
    {len(agriculture_candidates)} schemes

Additional candidates requiring review:
    {len(extra_candidates)} schemes

Final working agriculture dataset:
    {len(agriculture_df)} schemes

Output directory:
    {OUTPUT_DIR.resolve()}
""")

print("Generated files:")

print(f"1. {initial_output}")
print(f"2. {candidate_output}")
print(f"3. {all_candidate_output}")
print(f"4. {preview_output}")
print(f"5. {OUTPUT_DIR / 'dataset_missingness_report.csv'}")
print(f"6. {OUTPUT_DIR / 'agriculture_category_distribution.csv'}")
print(f"7. {OUTPUT_DIR / 'agriculture_beneficiary_distribution.csv'}")
print(f"8. {OUTPUT_DIR / 'agriculture_state_distribution.csv'}")
print(f"9. {OUTPUT_DIR / 'agriculture_text_field_missingness.csv'}")

print("\nNext step:")
print("Review the additional agriculture candidates before adding them.")
print("Then proceed to Phase 1: TF-IDF baseline -> Semantic Embeddings.")