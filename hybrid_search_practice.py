import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# HYBRID SEARCH PRACTICE
# ============================================================
#
# Hybrid Search =
#
#       Vector Search
#            +
#       Keyword Search
#            ↓
#        RRF Fusion
#            ↓
#      Combined Ranking
#
#
# Vector Search:
#     Understands semantic meaning.
#
# BM25:
#     Strong at exact words / keywords.
#
# RRF:
#     Combines the rankings from both systems.
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

VECTOR_TOP_K = 5

BM25_TOP_K = 5

FINAL_TOP_K = 5


# ============================================================
# 3. LOCAL EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)


# ============================================================
# 4. DOCUMENTS
# ============================================================
#
# These are the same documents that we stored in Pinecone.
#
# We keep a local copy here because BM25 needs access to
# the actual text.
#
# ============================================================

documents = [
    {
        "id": "rag-document-001-chunk-0",
        "text": (
            "Retrieval Augmented Generation, commonly called RAG, "
            "is a technique that combines information retrieval "
            "with a language model. A RAG system usually starts "
            "with a collection of documents. Large documents are "
            "split into smaller pieces called chunks. Each chunk "
            "is converted"
        )
    },

    {
        "id": "rag-document-001-chunk-1",
        "text": (
            "split into smaller pieces called chunks. Each chunk "
            "is converted into a numerical representation called "
            "an embedding. The embeddings are stored in a vector "
            "database such as Pinecone. When a user asks a question, "
            "the question is also converted into"
        )
    },

    {
        "id": "rag-document-001-chunk-2",
        "text": (
            "user asks a question, the question is also converted "
            "into an embedding. The vector database compares the "
            "query embedding with the stored chunk embeddings and "
            "retrieves the most relevant chunks. The retrieved "
            "chunks are then provided to a Large Language"
        )
    },

    {
        "id": "rag-document-001-chunk-3",
        "text": (
            "The retrieved chunks are then provided to a Large "
            "Language Model as context. The language model uses "
            "this context to generate an answer. Good chunking "
            "is important because chunks that are too large may "
            "contain unrelated information, while chunks that"
        )
    },

    {
        "id": "rag-document-001-chunk-4",
        "text": (
            "are too large may contain unrelated information, "
            "while chunks that are too small may lose useful "
            "context. Chunk overlap can help preserve context "
            "between neighboring chunks. A production RAG system "
            "may also use metadata filtering, reranking, hybrid "
            "search, evaluation, citations,"
        )
    },

    {
        "id": "rag-document-001-chunk-5",
        "text": (
            "may also use metadata filtering, reranking, hybrid "
            "search, evaluation, citations, and access control."
        )
    }
]


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_PATH
)

print(
    "Embedding model loaded successfully."
)


# ============================================================
# 6. CONNECT TO PINECONE
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
# 7. USER QUERY
# ============================================================

query = (
    "How does hybrid search work in a RAG system?"
)


print("\n========================================")
print("USER QUERY")
print("========================================")

print(query)


# ============================================================
# 8. VECTOR SEARCH
# ============================================================

print("\n========================================")
print("STAGE 1: VECTOR SEARCH")
print("========================================")


query_embedding = embedding_model.encode(
    query
).tolist()


vector_results = index.query(

    namespace=NAMESPACE,

    vector=query_embedding,

    top_k=VECTOR_TOP_K,

    include_metadata=True,

    include_values=False
)


vector_matches = vector_results["matches"]


print(
    "Vector candidates:",
    len(vector_matches)
)


for rank, match in enumerate(
    vector_matches,
    start=1
):

    print("\n----------------------------------------")

    print("Rank:", rank)

    print(
        "ID:",
        match["id"]
    )

    print(
        "Vector Score:",
        round(match["score"], 4)
    )


# ============================================================
# 9. BM25 TOKENIZATION
# ============================================================
#
# BM25 works with tokens/words.
#
# Example:
#
# "How does hybrid search work?"
#
# becomes:
#
# ["how", "does", "hybrid", "search", "work"]
#
# ============================================================

print("\n========================================")
print("STAGE 2: BM25 KEYWORD SEARCH")
print("========================================")


tokenized_documents = [
    document["text"].lower().split()
    for document in documents
]


bm25 = BM25Okapi(
    tokenized_documents
)


query_tokens = query.lower().split()


bm25_scores = bm25.get_scores(
    query_tokens
)


# ============================================================
# 10. CREATE BM25 RESULTS
# ============================================================

bm25_results = []


for index_number, score in enumerate(
    bm25_scores
):

    bm25_results.append(
        {
            "id": documents[index_number]["id"],

            "bm25_score": float(score),

            "text": documents[index_number]["text"]
        }
    )


# Sort highest BM25 score first.

bm25_results.sort(
    key=lambda item: item["bm25_score"],
    reverse=True
)


# ============================================================
# 11. DISPLAY BM25 RANKING
# ============================================================

for rank, result in enumerate(
    bm25_results[:BM25_TOP_K],
    start=1
):

    print("\n----------------------------------------")

    print("Rank:", rank)

    print(
        "ID:",
        result["id"]
    )

    print(
        "BM25 Score:",
        round(
            result["bm25_score"],
            4
        )
    )


# ============================================================
# 12. CREATE RANK MAPS
# ============================================================
#
# We don't directly combine:
#
#     Vector Score + BM25 Score
#
# because they use different scoring systems.
#
# Instead, we combine their RANKS.
#
# ============================================================

vector_rank = {}

for rank, match in enumerate(
    vector_matches,
    start=1
):

    vector_rank[match["id"]] = rank


bm25_rank = {}

for rank, result in enumerate(
    bm25_results[:BM25_TOP_K],
    start=1
):

    bm25_rank[result["id"]] = rank


# ============================================================
# 13. RECIPROCAL RANK FUSION
# ============================================================
#
# RRF formula:
#
# RRF score =
#
#       1 / (k + rank)
#
#
# We use k = 60, a common smoothing constant.
#
# ============================================================

print("\n========================================")
print("STAGE 3: RRF FUSION")
print("========================================")


RRF_K = 60


all_ids = set(
    vector_rank.keys()
).union(
    bm25_rank.keys()
)


rrf_results = []


for document_id in all_ids:

    score = 0.0


    # --------------------------------------------------------
    # Vector contribution
    # --------------------------------------------------------

    if document_id in vector_rank:

        score += 1 / (
            RRF_K +
            vector_rank[document_id]
        )


    # --------------------------------------------------------
    # BM25 contribution
    # --------------------------------------------------------

    if document_id in bm25_rank:

        score += 1 / (
            RRF_K +
            bm25_rank[document_id]
        )


    # --------------------------------------------------------
    # Find document text
    # --------------------------------------------------------

    document_text = next(
        document["text"]
        for document in documents
        if document["id"] == document_id
    )


    rrf_results.append(
        {
            "id": document_id,

            "rrf_score": score,

            "vector_rank": vector_rank.get(
                document_id
            ),

            "bm25_rank": bm25_rank.get(
                document_id
            ),

            "text": document_text
        }
    )


# ============================================================
# 14. SORT RRF RESULTS
# ============================================================

rrf_results.sort(
    key=lambda item: item["rrf_score"],
    reverse=True
)


# ============================================================
# 15. DISPLAY HYBRID RANKING
# ============================================================

print("\n========================================")
print("HYBRID / RRF RANKING")
print("========================================")


for rank, result in enumerate(
    rrf_results,
    start=1
):

    print("\n----------------------------------------")

    print("Final Rank:", rank)

    print(
        "ID:",
        result["id"]
    )

    print(
        "RRF Score:",
        round(
            result["rrf_score"],
            6
        )
    )

    print(
        "Vector Rank:",
        result["vector_rank"]
    )

    print(
        "BM25 Rank:",
        result["bm25_rank"]
    )

    print("Text:")

    print(
        result["text"]
    )


# ============================================================
# 16. SELECT FINAL RESULTS
# ============================================================

final_results = rrf_results[
    :FINAL_TOP_K
]


# ============================================================
# 17. BUILD FINAL CONTEXT
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
# 18. DISPLAY FINAL CONTEXT
# ============================================================

print("\n========================================")
print("FINAL HYBRID CONTEXT")
print("========================================")


print(context)


# ============================================================
# 19. FINAL PIPELINE
# ============================================================

print("\n========================================")
print("HYBRID SEARCH PIPELINE")
print("========================================")


print(
    """
User Query
      |
      +----------------------+
      |                      |
      v                      v
Vector Search            BM25 Search
      |                      |
      v                      v
Vector Ranking           Keyword Ranking
      |                      |
      +----------+-----------+
                 |
                 v
             RRF Fusion
                 |
                 v
          Combined Ranking
                 |
                 v
           Final Context
                 |
                 v
            NEXT: LLM
"""
)


print(
    "\nHybrid search practice completed successfully."
)