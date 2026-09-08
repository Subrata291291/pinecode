# rag_evaluation_practice.py

# ==========================================
# RAG Evaluation Practice
# Precision@K
# Recall@K
# MRR
# ==========================================


# -------------------------------------------------
# 1. Evaluation Dataset
# -------------------------------------------------

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


# -------------------------------------------------
# 2. Simulated Retrieval Results
# -------------------------------------------------

# এগুলো আপাতত manually দেওয়া হচ্ছে।
# পরে আমরা Pinecone থেকে automatically নিয়ে আসব.

retrieval_results = {
    "What is Retrieval Augmented Generation?": [
        "chunk-0",
        "chunk-5",
        "chunk-3",
    ],

    "Why is chunk overlap useful in RAG?": [
        "chunk-4",
        "chunk-0",
        "chunk-3",
    ],

    "What is hybrid search?": [
        "chunk-4",
        "chunk-5",
        "chunk-2",
    ],
}


# -------------------------------------------------
# 3. Precision@K
# -------------------------------------------------

def precision_at_k(retrieved, relevant, k):
    """
    Precision@K =
    Relevant retrieved documents / K
    """

    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        1 for chunk in retrieved_at_k
        if chunk in relevant
    )

    return relevant_count / k


# -------------------------------------------------
# 4. Recall@K
# -------------------------------------------------

def recall_at_k(retrieved, relevant, k):
    """
    Recall@K =
    Relevant retrieved documents / Total relevant documents
    """

    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        1 for chunk in retrieved_at_k
        if chunk in relevant
    )

    return relevant_count / len(relevant)


# -------------------------------------------------
# 5. Reciprocal Rank
# -------------------------------------------------

def reciprocal_rank(retrieved, relevant):
    """
    Reciprocal Rank =
    1 / rank of first relevant result
    """

    for rank, chunk in enumerate(retrieved, start=1):

        if chunk in relevant:
            return 1 / rank

    return 0


# -------------------------------------------------
# 6. Evaluate Each Question
# -------------------------------------------------

K = 3

all_precision = []
all_recall = []
all_rr = []


print("\n")
print("=" * 60)
print("RAG EVALUATION")
print("=" * 60)


for item in evaluation_dataset:

    question = item["question"]
    relevant = item["relevant_chunks"]

    retrieved = retrieval_results[question]

    precision = precision_at_k(
        retrieved,
        relevant,
        K
    )

    recall = recall_at_k(
        retrieved,
        relevant,
        K
    )

    rr = reciprocal_rank(
        retrieved,
        relevant
    )

    all_precision.append(precision)
    all_recall.append(recall)
    all_rr.append(rr)

    print("\nQuestion:")
    print(question)

    print("\nRelevant chunks:")
    print(relevant)

    print("\nRetrieved chunks:")
    print(retrieved)

    print("\nMetrics:")

    print(f"Precision@{K}: {precision:.3f}")
    print(f"Recall@{K}:    {recall:.3f}")
    print(f"Reciprocal Rank: {rr:.3f}")

    print("-" * 60)


# -------------------------------------------------
# 7. Mean Metrics
# -------------------------------------------------

mean_precision = sum(all_precision) / len(all_precision)
mean_recall = sum(all_recall) / len(all_recall)
mrr = sum(all_rr) / len(all_rr)


print("\n")
print("=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"\nMean Precision@{K}: {mean_precision:.3f}")
print(f"Mean Recall@{K}:    {mean_recall:.3f}")
print(f"MRR:                 {mrr:.3f}")

print("\n")
print("=" * 60)