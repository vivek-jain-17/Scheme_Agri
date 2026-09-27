import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path("agriculture_schemes.csv")
EMBEDDINGS_FILE = Path("agriculture_scheme_embeddings_bge_m3.npy")

MODEL_NAME = "BAAI/bge-m3"

TOP_K = 10


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BGE-M3 SEMANTIC RECOMMENDATION ENGINE")
print("=" * 70)

print("\nLoading agriculture scheme dataset...")

df = pd.read_csv(DATASET)

print(f"Total schemes loaded: {len(df)}")


# ============================================================
# PREPARE SCHEME TEXT
# ============================================================

print("\nPreparing scheme text...")

df["scheme_text"] = (
    df["scheme_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove schemes with empty text
df = df[df["scheme_text"] != ""].reset_index(drop=True)

print(f"Usable schemes: {len(df)}")


# ============================================================
# LOAD BGE-M3 MODEL
# ============================================================

print("\nLoading BGE-M3 model...")
print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# GENERATE SCHEME EMBEDDINGS
# ============================================================

print("\nGenerating scheme embeddings...")
print("This may take some time on CPU.")

scheme_embeddings = model.encode(
    df["scheme_text"].tolist(),
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

print("\nEmbedding generation completed.")

print(f"Embedding shape: {scheme_embeddings.shape}")


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    EMBEDDINGS_FILE,
    scheme_embeddings
)

print(f"Embeddings saved to: {EMBEDDINGS_FILE}")


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_schemes(user_query, top_k=TOP_K):

    # Generate query embedding
    query_embedding = model.encode(
        [user_query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_embedding,
        scheme_embeddings
    ).flatten()

    # Get top K indices
    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    # Create result dataframe
    results = df.iloc[top_indices].copy()

    results["similarity_score"] = (
        similarity_scores[top_indices]
    )

    # Reset ranking
    results = results.reset_index(drop=True)

    results.insert(
        0,
        "rank",
        range(1, len(results) + 1)
    )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(user_query, results):

    print("\n" + "=" * 70)
    print("USER QUERY")
    print("=" * 70)

    print(user_query)

    print("\n" + "=" * 70)
    print("TOP RECOMMENDED AGRICULTURE SCHEMES")
    print("=" * 70)

    for _, row in results.iterrows():

        print(
            f"\n{int(row['rank'])}. {row['name']}"
        )

        print(
            f"   Similarity Score: "
            f"{row['similarity_score']:.4f}"
        )

        print(
            f"   Category: "
            f"{row.get('category', '')}"
        )

        print(
            f"   Beneficiary Type: "
            f"{row.get('beneficiary_type', '')}"
        )

        # Display benefits if available
        benefits = str(row.get("benefits", ""))

        if benefits and benefits.lower() != "nan":

            # Prevent extremely long output
            if len(benefits) > 300:
                benefits = benefits[:300] + "..."

            print(
                f"   Benefits: {benefits}"
            )


# ============================================================
# TEST QUERIES
# ============================================================

test_queries = [

    "I am a small farmer looking for financial assistance for crop cultivation.",

    "I am looking for an agriculture scheme related to fishing and fisheries.",

    "I am looking for an agriculture scheme for senior citizens.",

    "I am a girl studying agriculture and looking for a government scheme."
]


# ============================================================
# RUN TEST QUERIES
# ============================================================

print("\n" + "=" * 70)
print("RUNNING TEST QUERIES")
print("=" * 70)

for query in test_queries:

    results = recommend_schemes(
        query,
        top_k=TOP_K
    )

    display_results(
        query,
        results
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

print("\n" + "=" * 70)
print("INTERACTIVE RECOMMENDATION MODE")
print("=" * 70)

print("Enter a query to get agriculture scheme recommendations.")
print("Type 'exit' to stop.")

while True:

    user_query = input("\nEnter your query: ").strip()

    if user_query.lower() == "exit":
        print("\nExiting BGE-M3 recommendation engine.")
        break

    if not user_query:
        print("Please enter a valid query.")
        continue

    results = recommend_schemes(
        user_query,
        top_k=TOP_K
    )

    display_results(
        user_query,
        results
    )