# ============================================================
# PHASE 1A
# TF-IDF + COSINE SIMILARITY
# Government Agriculture Scheme Recommendation System
# ============================================================

import pandas as pd
import numpy as np
import re

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT = Path("agriculture_schemes.csv")

TOP_K = 10


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("PHASE 1A - TF-IDF BASELINE RECOMMENDER")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nDataset loaded successfully.")
print(f"Number of schemes: {len(df)}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMN
# ============================================================

if "scheme_text" not in df.columns:
    raise ValueError(
        "Column 'scheme_text' not found in dataset."
    )

if "name" not in df.columns:
    raise ValueError(
        "Column 'name' not found in dataset."
    )


# ============================================================
# 4. BASIC TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove markdown formatting
    text = re.sub(
        r"[*#>`_]",
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


df["scheme_text"] = (
    df["scheme_text"]
    .fillna("")
    .apply(clean_text)
)


# ============================================================
# 5. REMOVE EMPTY TEXT RECORDS
# ============================================================

before = len(df)

df = df[
    df["scheme_text"].str.strip() != ""
].copy()

after = len(df)

print(f"Removed empty text records: {before - after}")
print(f"Usable schemes: {after}")


# ============================================================
# 6. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING TF-IDF VECTORS")
print("=" * 70)


vectorizer = TfidfVectorizer(

    # Ignore extremely rare words
    min_df=1,

    # Ignore words appearing in almost every document
    max_df=0.95,

    # Include single words and two-word phrases
    ngram_range=(1, 2),

    # Limit vocabulary size
    max_features=30000,

    # Normalize vectors
    norm="l2",

    # Ignore common English stopwords
    stop_words="english"
)


scheme_vectors = vectorizer.fit_transform(
    df["scheme_text"]
)


print(f"TF-IDF matrix shape: {scheme_vectors.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")


# ============================================================
# 7. SHOW SAMPLE FEATURES
# ============================================================

feature_names = vectorizer.get_feature_names_out()

print("\nSample TF-IDF features:")

print(
    feature_names[:50]
)


# ============================================================
# 8. RECOMMENDATION FUNCTION
# ============================================================

def recommend_schemes(
    user_query,
    top_k=10
):

    # --------------------------------------------------------
    # Clean user query
    # --------------------------------------------------------

    user_query = clean_text(
        user_query
    )

    if not user_query:
        print("Please enter a valid query.")
        return None


    # --------------------------------------------------------
    # Convert user query into TF-IDF vector
    # --------------------------------------------------------

    query_vector = vectorizer.transform(
        [user_query]
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        query_vector,
        scheme_vectors
    ).flatten()


    # --------------------------------------------------------
    # Get top K scheme indices
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]


    # --------------------------------------------------------
    # Create recommendation dataframe
    # --------------------------------------------------------

    results = df.iloc[
        top_indices
    ].copy()

    results["similarity_score"] = (
        similarity_scores[top_indices]
    )

    results["rank"] = range(
        1,
        len(results) + 1
    )


    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    preferred_columns = [
        "rank",
        "name",
        "similarity_score",
        "category",
        "beneficiary_type",
        "benefits",
        "eligibility_text",
        "application_process"
    ]

    available_columns = [
        col
        for col in preferred_columns
        if col in results.columns
    ]

    results = results[
        available_columns
    ]


    return results


# ============================================================
# 9. TEST RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("TEST RECOMMENDATION")
print("=" * 70)


test_query = (
    "I am a small farmer looking for financial "
    "assistance for crop cultivation."
)


print(f"\nUser query:")
print(test_query)


results = recommend_schemes(
    test_query,
    TOP_K
)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print(f"TOP {TOP_K} RECOMMENDED SCHEMES")
print("=" * 70)


if results is not None:

    for _, row in results.iterrows():

        print(
            f"\nRank {int(row['rank'])}"
        )

        print(
            f"Scheme: {row['name']}"
        )

        print(
            f"Similarity: "
            f"{row['similarity_score']:.4f}"
        )

        if "category" in row:
            print(
                f"Category: {row['category']}"
            )

        if "beneficiary_type" in row:
            print(
                f"Beneficiary: "
                f"{row['beneficiary_type']}"
            )

        if "benefits" in row:
            print(
                f"Benefits: {row['benefits']}"
            )


# ============================================================
# 11. INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE RECOMMENDATION")
print("=" * 70)

print(
    "\nEnter your requirement in natural language."
)

print(
    "Type 'exit' to stop."
)


while True:

    user_query = input(
        "\nYour requirement: "
    ).strip()

    if user_query.lower() == "exit":
        print("\nExiting recommendation system.")
        break

    if not user_query:
        print(
            "Please enter a requirement."
        )
        continue


    results = recommend_schemes(
        user_query,
        TOP_K
    )


    if results is None:
        continue


    print(
        "\n" + "-" * 70
    )

    print(
        f"TOP {TOP_K} RECOMMENDATIONS"
    )

    print(
        "-" * 70
    )


    for _, row in results.iterrows():

        print(
            f"\n{int(row['rank'])}. "
            f"{row['name']}"
        )

        print(
            f"   Similarity: "
            f"{row['similarity_score']:.4f}"
        )

        if "beneficiary_type" in row:

            print(
                f"   Beneficiary: "
                f"{row['beneficiary_type']}"
            )

        if "benefits" in row:

            print(
                f"   Benefits: "
                f"{row['benefits']}"
            )


# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1A COMPLETE")
print("=" * 70)