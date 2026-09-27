# ============================================================
# EVALUATION VERSION
# BGE-M3 + COSINE SIMILARITY
# Government Agriculture Scheme Recommendation System
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

DATASET = Path("agriculture_schemes.csv")
EVALUATION_QUERIES = Path("evaluation_queries.csv")

MODEL_NAME = "BAAI/bge-m3"

EMBEDDINGS_FILE = Path(
    "agriculture_scheme_embeddings_bge_m3.npy"
)

OUTPUT_FILE = Path(
    "evaluation_results_bge_m3.csv"
)

TOP_K = 10
BATCH_SIZE = 16


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("BGE-M3 EVALUATION")
print("=" * 70)

df = pd.read_csv(DATASET)

print(f"\nDataset loaded successfully.")
print(f"Raw schemes: {len(df)}")


# ============================================================
# 3. VALIDATE DATASET
# ============================================================

required_columns = ["name", "scheme_text"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' was not found in dataset."
        )

df["scheme_id"] = np.arange(len(df))


# ============================================================
# 4. TEXT CLEANING
# ============================================================

# BGE is now intentionally using the SAME cleaning pipeline
# as TF-IDF and E5 so that all three models receive the same
# underlying scheme corpus.

def clean_text(text):
    """Apply the same text normalization across all three models."""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[*#>`_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["scheme_text"] = (
    df["scheme_text"]
    .fillna("")
    .apply(clean_text)
)

before = len(df)

df = df[
    df["scheme_text"].str.strip() != ""
].copy()

df = df.reset_index(drop=True)

print(f"Removed empty schemes: {before - len(df)}")
print(f"Usable schemes: {len(df)}")


# ============================================================
# 5. LOAD EVALUATION QUERIES
# ============================================================

queries = pd.read_csv(EVALUATION_QUERIES)

required_query_columns = ["query_id", "query"]

for column in required_query_columns:
    if column not in queries.columns:
        raise ValueError(
            f"Required query column '{column}' was not found "
            f"in evaluation_queries.csv."
        )

queries = queries[
    ["query_id", "query"]
].copy()

queries["query"] = queries["query"].fillna("").apply(clean_text)

if queries["query"].str.strip().eq("").any():
    raise ValueError("One or more evaluation queries are empty.")

print(f"Evaluation queries: {len(queries)}")


# ============================================================
# 6. LOAD BGE-M3 MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BGE-M3 MODEL")
print("=" * 70)

print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# 7. GENERATE SCHEME EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING BGE-M3 SCHEME EMBEDDINGS")
print("=" * 70)

print("This may take some time depending on hardware.")

scheme_embeddings = model.encode(
    df["scheme_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("\nEmbedding generation completed.")

print(
    f"Embedding matrix shape: "
    f"{scheme_embeddings.shape}"
)


# ============================================================
# 8. SAVE EMBEDDINGS
# ============================================================

np.save(
    EMBEDDINGS_FILE,
    scheme_embeddings
)

print(f"Embeddings saved to: {EMBEDDINGS_FILE}")


# ============================================================
# 9. RUN STANDARDIZED EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING EVALUATION QUERIES")
print("=" * 70)

all_results = []

for query_number, row in enumerate(
    queries.itertuples(index=False),
    start=1
):
    query_id = row.query_id
    user_query = row.query

    print(
        f"\n[{query_number}/{len(queries)}] "
        f"{query_id}: {user_query}"
    )

    query_embedding = model.encode(
        [user_query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarity_scores = cosine_similarity(
        query_embedding,
        scheme_embeddings
    ).flatten()

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:TOP_K]

    for rank, idx in enumerate(top_indices, start=1):
        result_row = df.iloc[idx]

        all_results.append({
            "model": "BGE-M3",
            "query_id": query_id,
            "query": user_query,
            "rank": rank,
            "scheme_id": int(result_row["scheme_id"]),
            "scheme_name": result_row["name"],
            "similarity_score": float(similarity_scores[idx]),
            "category": result_row.get("category", ""),
            "beneficiary_type": result_row.get(
                "beneficiary_type", ""
            )
        })


# ============================================================
# 10. SAVE STANDARDIZED RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df[
    [
        "model",
        "query_id",
        "query",
        "rank",
        "scheme_id",
        "scheme_name",
        "similarity_score",
        "category",
        "beneficiary_type"
    ]
]

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"Queries evaluated: {len(queries)}")
print(f"Results generated: {len(results_df)}")
print(f"Expected results: {len(queries) * TOP_K}")
print(f"Output file: {OUTPUT_FILE}")

print("\nFirst query results:")
print(
    results_df[
        results_df["query_id"] == queries.iloc[0]["query_id"]
    ][
        ["rank", "scheme_name", "similarity_score"]
    ].to_string(index=False)
)

print("\nBGE-M3 evaluation finished successfully.")
