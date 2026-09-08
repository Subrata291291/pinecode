import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)


# ============================================================
# RERANKER PRACTICE
# ============================================================
#
# Stage 1:
#
# User Question
#       ↓
# Embedding Model
#       ↓
# Pinecone
#       ↓
# Top-K Candidates
#
#
# Stage 2:
#
# Question + Candidate Chunk
#       ↓
# Cross Encoder
#       ↓
# Reranking Score
#       ↓
# Better Ranking
#
# ============================================================


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

pinecone_api_key = os.getenv(
    "PINECONE_API_KEY"
)

if not pinecone_api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found.\n"
        "Check your .env file."
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

INDEX_NAME = "demo-index"

NAMESPACE = "chunk-practice"

# Retrieve more candidates initially.
#
# We will retrieve 5 from Pinecone,
# then rerank them and keep the best 3.

RETRIEVAL_TOP_K = 5

FINAL_TOP_K = 3


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)


# ============================================================
# 4. RERANKER MODEL
# ============================================================
#
# CrossEncoder takes TWO pieces of text:
#
#     Question
#     +
#     Candidate Chunk
#
# and calculates how relevant the chunk is
# to the question.
#
# ============================================================

RERANKER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "ms-marco-MiniLM-L6-v2"
)

# ============================================================
# 5. USER QUESTION
# ============================================================

question = (
    "What is Retrieval Augmented Generation "
    "and how does it work?"
)


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_PATH
)

print(
    "Embedding model loaded successfully."
)


# ============================================================
# 7. LOAD RERANKER
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL_PATH
)

print(
    "Reranker model loaded successfully."
)


# ============================================================
# 8. CONNECT TO PINECONE
# ============================================================

print("\nConnecting to Pinecone...")

pc = Pinecone(
    api_key=pinecone_api_key
)

index = pc.Index(
    INDEX_NAME
)

print("Connected to Pinecone.")

print("Index:", INDEX_NAME)

print("Namespace:", NAMESPACE)


# ============================================================
# 9. CREATE QUERY EMBEDDING
# ============================================================

print("\n========================================")
print("USER QUESTION")
print("========================================")

print(question)


print("\n========================================")
print("QUERY EMBEDDING")
print("========================================")


query_embedding = embedding_model.encode(
    question
).tolist()


print("Query embedding created.")

print(
    "Dimensions:",
    len(query_embedding)
)


# ============================================================
# 10. STAGE 1 — PINECONE RETRIEVAL
# ============================================================

print("\n========================================")
print("STAGE 1: PINECONE RETRIEVAL")
print("========================================")


results = index.query(

    namespace=NAMESPACE,

    vector=query_embedding,

    top_k=RETRIEVAL_TOP_K,

    include_metadata=True,

    include_values=False
)


matches = results["matches"]


if not matches:

    print(
        "\nNo chunks were retrieved."
    )

    raise SystemExit


print(
    "Candidates retrieved:",
    len(matches)
)


# ============================================================
# 11. DISPLAY ORIGINAL PINECONE RANKING
# ============================================================

print("\n========================================")
print("ORIGINAL PINECONE RANKING")
print("========================================")


for rank, match in enumerate(
    matches,
    start=1
):

    print("\n----------------------------------------")

    print("Rank:", rank)

    print(
        "Chunk ID:",
        match["id"]
    )

    print(
        "Pinecone Score:",
        round(match["score"], 4)
    )

    print("Text:")

    print(
        match["metadata"]["text"]
    )


# ============================================================
# 12. PREPARE QUESTION + CHUNK PAIRS
# ============================================================
#
# CrossEncoder receives:
#
# [
#     [question, chunk1],
#     [question, chunk2],
#     [question, chunk3]
# ]
#
# ============================================================

pairs = []


for match in matches:

    chunk_text = (
        match["metadata"]["text"]
    )

    pairs.append(
        [
            question,
            chunk_text
        ]
    )


# ============================================================
# 13. STAGE 2 — RERANK
# ============================================================

print("\n========================================")
print("STAGE 2: RERANKING")
print("========================================")


reranker_scores = reranker.predict(
    pairs
)


# ============================================================
# 14. COMBINE CHUNKS + RERANK SCORES
# ============================================================

reranked_results = []


for match, score in zip(
    matches,
    reranker_scores
):

    reranked_results.append(
        {
            "id": match["id"],

            "pinecone_score": match["score"],

            "reranker_score": float(score),

            "text": match["metadata"]["text"],

            "metadata": match["metadata"]
        }
    )


# ============================================================
# 15. SORT BY RERANKER SCORE
# ============================================================

reranked_results.sort(
    key=lambda item: item["reranker_score"],
    reverse=True
)


# ============================================================
# 16. DISPLAY RERANKED RESULTS
# ============================================================

print("\n========================================")
print("RERANKED RESULTS")
print("========================================")


for rank, result in enumerate(
    reranked_results,
    start=1
):

    print("\n----------------------------------------")

    print("New Rank:", rank)

    print(
        "Chunk ID:",
        result["id"]
    )

    print(
        "Pinecone Score:",
        round(
            result["pinecone_score"],
            4
        )
    )

    print(
        "Reranker Score:",
        round(
            result["reranker_score"],
            4
        )
    )

    print("Text:")

    print(
        result["text"]
    )


# ============================================================
# 17. SELECT FINAL TOP-K
# ============================================================

final_results = reranked_results[
    :FINAL_TOP_K
]


# ============================================================
# 18. BUILD FINAL CONTEXT
# ============================================================

context_parts = []


for rank, result in enumerate(
    final_results,
    start=1
):

    context_parts.append(

        f"[Context {rank}]\n"
        f"{result['text']}"

    )


context = "\n\n".join(
    context_parts
)


# ============================================================
# 19. DISPLAY FINAL CONTEXT
# ============================================================

print("\n========================================")
print("FINAL RERANKED CONTEXT")
print("========================================")


print(context)


# ============================================================
# 20. SHOW FINAL PIPELINE
# ============================================================

print("\n========================================")
print("TWO-STAGE RETRIEVAL PIPELINE")
print("========================================")


print(
    """
User Question
      |
      v
Embedding Model
      |
      v
Pinecone Vector Search
      |
      v
Top 5 Candidates
      |
      v
Cross Encoder Reranker
      |
      v
Reranked Candidates
      |
      v
Best 3 Chunks
      |
      v
Final Context
      |
      v
NEXT STEP: SEND TO LLM
"""
)


print(
    "\nReranking practice completed successfully."
)