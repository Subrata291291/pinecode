import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ============================================================
# CHUNK -> EMBEDDING -> PINECONE -> RETRIEVAL
# ============================================================
#
# This is our first small end-to-end RAG retrieval pipeline.
#
# Flow:
#
# Document
#    |
#    v
# Chunking
#    |
#    v
# Chunk Embeddings
#    |
#    v
# Pinecone
#    |
#    v
# User Query
#    |
#    v
# Query Embedding
#    |
#    v
# Top-K Relevant Chunks
#
# ============================================================


# ============================================================
# 1. OFFLINE MODEL SETTINGS
# ============================================================

# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found.\n"
        "Make sure your .env file exists."
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

INDEX_NAME = "demo-index"

MODEL_NAME = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)

# Use a separate namespace for this practice.
# This prevents mixing our new chunk vectors with
# the original doc-0 ... doc-4 vectors.
NAMESPACE = "chunk-practice"

CHUNK_SIZE = 40

CHUNK_OVERLAP = 10


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded successfully.")


# ============================================================
# 5. CONNECT TO PINECONE
# ============================================================

print("\nConnecting to Pinecone...")

pc = Pinecone(api_key=api_key)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone.")
print("Index:", INDEX_NAME)
print("Namespace:", NAMESPACE)


# ============================================================
# 6. SAMPLE DOCUMENT
# ============================================================

document_id = "rag-document-001"

document = """
Retrieval Augmented Generation, commonly called RAG, is a
technique that combines information retrieval with a language
model.

A RAG system usually starts with a collection of documents.
Large documents are split into smaller pieces called chunks.
Each chunk is converted into a numerical representation called
an embedding.

The embeddings are stored in a vector database such as
Pinecone. When a user asks a question, the question is also
converted into an embedding.

The vector database compares the query embedding with the
stored chunk embeddings and retrieves the most relevant chunks.

The retrieved chunks are then provided to a Large Language
Model as context. The language model uses this context to
generate an answer.

Good chunking is important because chunks that are too large
may contain unrelated information, while chunks that are too
small may lose useful context. Chunk overlap can help preserve
context between neighboring chunks.

A production RAG system may also use metadata filtering,
reranking, hybrid search, evaluation, citations, and access
control.
"""


# ============================================================
# 7. CHUNKING FUNCTION
# ============================================================

def word_chunks_with_overlap(text, chunk_size, overlap):

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    words = text.split()

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(words):

        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            break

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# 8. CREATE CHUNKS
# ============================================================

print("\n========================================")
print("CHUNKING DOCUMENT")
print("========================================")

chunks = word_chunks_with_overlap(
    document,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

print("Document ID:", document_id)

print("Chunk size:", CHUNK_SIZE)

print("Overlap:", CHUNK_OVERLAP)

print("Number of chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    print("\n----------------------------------------")

    print("Chunk ID:", i)

    print("Word count:", len(chunk.split()))

    print("Text:")

    print(chunk)


# ============================================================
# 9. CREATE EMBEDDINGS + PREPARE VECTORS
# ============================================================

print("\n========================================")
print("EMBEDDING + VECTOR PREPARATION")
print("========================================")

vectors = []


for i, chunk in enumerate(chunks):

    # --------------------------------------------------------
    # Convert chunk into embedding
    # --------------------------------------------------------

    embedding = model.encode(chunk).tolist()


    # --------------------------------------------------------
    # Create metadata
    # --------------------------------------------------------

    metadata = {

        "text": chunk,

        "document_id": document_id,

        "chunk_id": i,

        "word_count": len(chunk.split())
    }


    # --------------------------------------------------------
    # Create unique vector ID
    # --------------------------------------------------------

    vector_id = f"{document_id}-chunk-{i}"


    # --------------------------------------------------------
    # Add vector to our list
    # --------------------------------------------------------

    vectors.append(
        (
            vector_id,
            embedding,
            metadata
        )
    )


    print(
        f"Prepared {vector_id} | "
        f"dimensions={len(embedding)}"
    )


# ============================================================
# 10. UPSERT ALL CHUNKS INTO PINECONE
# ============================================================

print("\n========================================")
print("UPSERTING CHUNKS INTO PINECONE")
print("========================================")


index.upsert(
    vectors=vectors,
    namespace=NAMESPACE
)


print("\nAll chunks inserted into Pinecone.")


# ============================================================
# 11. CREATE USER QUERY
# ============================================================

query = "What is Retrieval Augmented Generation?"


print("\n========================================")
print("RETRIEVAL")
print("========================================")

print("Query:")

print(query)


# ============================================================
# 12. CONVERT QUERY INTO EMBEDDING
# ============================================================

query_embedding = model.encode(
    query
).tolist()


print("\nQuery embedding created.")

print(
    "Embedding dimensions:",
    len(query_embedding)
)


# ============================================================
# 13. SEARCH PINECONE
# ============================================================

results = index.query(

    namespace=NAMESPACE,

    vector=query_embedding,

    top_k=3,

    include_metadata=True
)


# ============================================================
# 14. DISPLAY RETRIEVED CHUNKS
# ============================================================

print("\nTop 3 relevant chunks:")


for rank, match in enumerate(
    results["matches"],
    start=1
):

    print("\n----------------------------------------")

    print("Rank:", rank)

    print("ID:", match["id"])

    print(
        "Score:",
        round(match["score"], 4)
    )

    print(
        "Document ID:",
        match["metadata"]["document_id"]
    )

    print(
        "Chunk ID:",
        match["metadata"]["chunk_id"]
    )

    print("Text:")

    print(match["metadata"]["text"])


# ============================================================
# 15. SECOND QUERY
# ============================================================

query2 = "Why is chunk overlap useful in RAG?"


print("\n========================================")
print("RETRIEVAL QUERY 2")
print("========================================")

print("Query:")

print(query2)


# ============================================================
# 16. EMBED SECOND QUERY
# ============================================================

query2_embedding = model.encode(
    query2
).tolist()


# ============================================================
# 17. SEARCH FOR SECOND QUERY
# ============================================================

results2 = index.query(

    namespace=NAMESPACE,

    vector=query2_embedding,

    top_k=2,

    include_metadata=True
)


# ============================================================
# 18. DISPLAY SECOND RESULTS
# ============================================================

print("\nTop 2 relevant chunks:")


for rank, match in enumerate(
    results2["matches"],
    start=1
):

    print("\n----------------------------------------")

    print("Rank:", rank)

    print(
        "Score:",
        round(match["score"], 4)
    )

    print(
        "Chunk ID:",
        match["metadata"]["chunk_id"]
    )

    print("Text:")

    print(match["metadata"]["text"])


# ============================================================
# 19. FINAL CONCEPT
# ============================================================

print("\n========================================")
print("PIPELINE COMPLETED")
print("========================================")


print(
    """
Document
   |
   v
Chunking
   |
   v
Each Chunk
   |
   v
Embedding Model
   |
   v
Vector + Metadata
   |
   v
Pinecone
   |
   v
User Query
   |
   v
Query Embedding
   |
   v
Similarity Search
   |
   v
Relevant Chunks
   |
   v
NEXT STEP: SEND THESE CHUNKS TO AN LLM
"""
)