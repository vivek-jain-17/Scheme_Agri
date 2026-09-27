# ============================================================
# PHASE 1B
# SEMANTIC EMBEDDING RECOMMENDATION
#
# Government Agriculture Scheme Recommendation System
#
# Model:
# intfloat/multilingual-e5-base
# ============================================================

import pandas as pd
import numpy as np
import re

from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT = Path("agriculture_schemes.csv")

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 10

BATCH_SIZE = 32


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("PHASE 1B - SEMANTIC EMBEDDING RECOMMENDER")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nDataset loaded successfully.")
print(f"Number of schemes: {len(df)}")


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "name",
    "scheme_text"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"was not found in dataset."
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
# 5. REMOVE EMPTY RECORDS
# ============================================================

before = len(df)

df = df[
    df["scheme_text"].str.strip() != ""
].copy()

after = len(df)

print(f"Removed empty records: {before - after}")
print(f"Usable schemes: {after}")


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME
)

print("Model loaded successfully.")


# ============================================================
# 7. PREPARE DOCUMENTS
# ============================================================

# E5 models work better when documents are
# explicitly marked as passages.

scheme_documents = [
    "passage: " + text
    for text in df["scheme_text"]
]


# ============================================================
# 8. GENERATE SCHEME EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SCHEME EMBEDDINGS")
print("=" * 70)

scheme_embeddings = model.encode(

    scheme_documents,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True
)


print(
    f"\nEmbedding matrix shape: "
    f"{scheme_embeddings.shape}"
)


# ============================================================
# 9. SAVE EMBEDDINGS
# ============================================================

embedding_output = (
    Path("agriculture_scheme_embeddings.npy")
)

np.save(
    embedding_output,
    scheme_embeddings
)

print(
    f"Embeddings saved to: "
    f"{embedding_output}"
)


# ============================================================
# 10. SEMANTIC RECOMMENDATION FUNCTION
# ============================================================

def recommend_schemes(
    user_query,
    top_k=10
):

    # --------------------------------------------------------
    # Clean query
    # --------------------------------------------------------

    user_query = clean_text(
        user_query
    )

    if not user_query:

        print(
            "Please enter a valid query."
        )

        return None


    # --------------------------------------------------------
    # E5 query format
    # --------------------------------------------------------

    query_text = (
        "query: " + user_query
    )


    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(

        [query_text],

        normalize_embeddings=True,

        convert_to_numpy=True
    )


    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(

        query_embedding,

        scheme_embeddings

    ).flatten()


    # --------------------------------------------------------
    # Get top K
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]


    # --------------------------------------------------------
    # Create results
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
# 11. TEST QUERIES
# ============================================================

test_queries = [

    "I am a small farmer looking for "
    "financial assistance for crop cultivation.",

    "I need government support for fishing "
    "and fisheries activities.",

    "I am looking for an agriculture scheme "
    "for senior citizens.",

    "I am a girl studying agriculture "
    "and looking for financial assistance."
]


# ============================================================
# 12. RUN TEST QUERIES
# ============================================================

for query_number, query in enumerate(
    test_queries,
    start=1
):

    print("\n" + "=" * 70)

    print(
        f"TEST QUERY {query_number}"
    )

    print("=" * 70)

    print(
        f"\nUser query:\n{query}"
    )


    results = recommend_schemes(
        query,
        TOP_K
    )


    if results is None:

        continue


    print(
        f"\nTOP {TOP_K} SEMANTIC RECOMMENDATIONS"
    )

    print("-" * 70)


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
                f"Category: "
                f"{row['category']}"
            )

        if "beneficiary_type" in row:

            print(
                f"Beneficiary: "
                f"{row['beneficiary_type']}"
            )

        if "benefits" in row:

            print(
                f"Benefits: "
                f"{row['benefits']}"
            )


# ============================================================
# 13. INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE SEMANTIC RECOMMENDATION")
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

        print(
            "\nExiting semantic recommender."
        )

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
        f"TOP {TOP_K} SEMANTIC RECOMMENDATIONS"
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
print("PHASE 1B COMPLETE")
print("=" * 70)