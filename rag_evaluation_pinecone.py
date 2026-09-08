# rag_evaluation_pinecone.py

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "demo-index"
NAMESPACE = "chunk-practice"

MODEL_NAME = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)

TOP_K = 3


# ============================================================
# 2. EVALUATION DATASET
# ============================================================
#
# These are our "ground truth" answers for retrieval.
#
# For each question, we manually specify which chunks
# should be considered relevant.
#
# Later, in a professional system, this dataset can be
# much larger and can be maintained separately.
# ============================================================

evaluation_dataset = [
    {
        "question": "What is Retrieval Augmented Generation?",
        "relevant_chunks": ["chunk-0"],
    },
    {
        "question": "Why is chunk overlap useful in RAG?",
        "relevant_chunks": ["chunk-4"],
    },
    {
        "question": "What is hybrid search?",
        "relevant_chunks": ["chunk-4", "chunk-5"],
    },
]


# ============================================================
# 3. PRECISION@K
# ============================================================

def precision_at_k(retrieved, relevant, k):
    """
    Precision@K tells us:

    Out of the top K retrieved chunks,
    how many are actually relevant?

    Formula:

        Precision@K =
        Relevant retrieved chunks / K
    """

    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        1
        for chunk in retrieved_at_k
        if chunk in relevant
    )

    return relevant_count / k


# ============================================================
# 4. RECALL@K
# ============================================================

def recall_at_k(retrieved, relevant, k):
    """
    Recall@K tells us:

    Out of ALL relevant chunks,
    how many did our retriever find
    within the top K?

    Formula:

        Recall@K =
        Relevant retrieved chunks /
        Total relevant chunks
    """

    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        1
        for chunk in retrieved_at_k
        if chunk in relevant
    )

    return relevant_count / len(relevant)


# ============================================================
# 5. RECIPROCAL RANK
# ============================================================

def reciprocal_rank(retrieved, relevant):
    """
    Reciprocal Rank tells us how high
    the FIRST relevant result appears.

    If first relevant result is rank 1:

        RR = 1 / 1 = 1.0

    If first relevant result is rank 2:

        RR = 1 / 2 = 0.5

    If first relevant result is rank 3:

        RR = 1 / 3 = 0.333
    """

    for rank, chunk in enumerate(retrieved, start=1):

        if chunk in relevant:
            return 1 / rank

    return 0


# ============================================================
# 6. LOAD ENVIRONMENT
# ============================================================

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY not found in .env file."
    )


# ============================================================
# 7. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# 8. CONNECT TO PINECONE
# ============================================================

print("\nConnecting to Pinecone...")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone.")


# ============================================================
# 9. STORAGE FOR OVERALL METRICS
# ============================================================

all_precision = []
all_recall = []
all_rr = []


# ============================================================
# 10. START EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("ACTUAL PINECONE RAG EVALUATION")
print("=" * 70)


for item in evaluation_dataset:

    question = item["question"]

    relevant = item["relevant_chunks"]


    # ========================================================
    # A. CREATE QUERY EMBEDDING
    # ========================================================

    query_vector = model.encode(
        question
    ).tolist()


    # ========================================================
    # B. QUERY PINECONE
    # ========================================================

    results = index.query(
        namespace=NAMESPACE,
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )


    # ========================================================
    # C. EXTRACT CHUNK IDs
    # ========================================================
    #
    # IMPORTANT:
    #
    # Pinecone metadata currently stores:
    #
    #     chunk_id = 0
    #     chunk_id = 4
    #     chunk_id = 5
    #
    # But our evaluation dataset uses:
    #
    #     chunk-0
    #     chunk-4
    #     chunk-5
    #
    # Therefore we normalize the ID here.
    # ========================================================

    retrieved = []

    for match in results.matches:

        chunk_id = match.metadata.get("chunk_id")

        normalized_chunk_id = f"chunk-{chunk_id}"

        retrieved.append(
            normalized_chunk_id
        )


    # ========================================================
    # D. CALCULATE METRICS
    # ========================================================

    precision = precision_at_k(
        retrieved,
        relevant,
        TOP_K
    )

    recall = recall_at_k(
        retrieved,
        relevant,
        TOP_K
    )

    rr = reciprocal_rank(
        retrieved,
        relevant
    )


    # ========================================================
    # E. STORE RESULTS
    # ========================================================

    all_precision.append(precision)
    all_recall.append(recall)
    all_rr.append(rr)


    # ========================================================
    # F. DISPLAY QUESTION
    # ========================================================

    print("\nQuestion:")
    print(question)


    # ========================================================
    # G. DISPLAY EXPECTED RESULTS
    # ========================================================

    print("\nExpected relevant chunks:")
    print(relevant)


    # ========================================================
    # H. DISPLAY ACTUAL PINECONE RESULTS
    # ========================================================

    print("\nActual Pinecone results:")

    for rank, match in enumerate(
        results.matches,
        start=1
    ):

        chunk_id = match.metadata.get("chunk_id")

        score = match.score

        normalized_chunk_id = f"chunk-{chunk_id}"

        is_relevant = (
            normalized_chunk_id in relevant
        )

        status = "YES" if is_relevant else "NO"

        print(
            f"{rank}. "
            f"{normalized_chunk_id} "
            f"(score={score:.4f}) "
            f"[Relevant: {status}]"
        )


    # ========================================================
    # I. DISPLAY METRICS
    # ========================================================

    print("\nMetrics:")

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.3f}"
    )

    print(
        f"Recall@{TOP_K}:    "
        f"{recall:.3f}"
    )

    print(
        f"Reciprocal Rank:   "
        f"{rr:.3f}"
    )

    print("-" * 70)


# ============================================================
# 11. CALCULATE OVERALL METRICS
# ============================================================

mean_precision = (
    sum(all_precision)
    / len(all_precision)
)

mean_recall = (
    sum(all_recall)
    / len(all_recall)
)

mrr = (
    sum(all_rr)
    / len(all_rr)
)


# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION REPORT")
print("=" * 70)

print(
    f"\nMean Precision@{TOP_K}: "
    f"{mean_precision:.3f}"
)

print(
    f"Mean Recall@{TOP_K}:    "
    f"{mean_recall:.3f}"
)

print(
    f"MRR:                     "
    f"{mrr:.3f}"
)

print("\n")
print("=" * 70)
print("Evaluation completed successfully.")
print("=" * 70)