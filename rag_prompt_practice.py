import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# ============================================================
# RAG PROMPT PRACTICE
# ============================================================
#
# Flow:
#
# User Question
#       |
#       v
# Query Embedding
#       |
#       v
# Pinecone Retrieval
#       |
#       v
# Top-K Relevant Chunks
#       |
#       v
# Context Builder
#       |
#       v
# RAG Prompt
#
# NOTE:
# We are NOT calling an LLM yet.
#
# This file only teaches us how to prepare the
# retrieved context and final prompt.
#
# ============================================================


# ============================================================
# 1. MODEL SETTINGS
# ============================================================

# Use the project-local model downloaded by
# chunk_to_pinecone.py.


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

NAMESPACE = "chunk-practice"

MODEL_NAME = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)

TOP_K = 3


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
# 6. USER QUESTION
# ============================================================

question = "What is Retrieval Augmented Generation and how does it work?"


print("\n========================================")
print("USER QUESTION")
print("========================================")

print(question)


# ============================================================
# 7. CREATE QUERY EMBEDDING
# ============================================================

print("\n========================================")
print("QUERY EMBEDDING")
print("========================================")

query_embedding = model.encode(
    question
).tolist()

print("Query embedding created.")

print(
    "Dimensions:",
    len(query_embedding)
)


# ============================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ============================================================

print("\n========================================")
print("PINECONE RETRIEVAL")
print("========================================")

results = index.query(

    namespace=NAMESPACE,

    vector=query_embedding,

    top_k=TOP_K,

    include_metadata=True,

    include_values=False
)


# ============================================================
# 9. CHECK RESULTS
# ============================================================

matches = results["matches"]

print("Retrieved chunks:", len(matches))


if not matches:

    print("\nNo relevant chunks were found.")

    raise SystemExit


# ============================================================
# 10. DISPLAY RETRIEVED CHUNKS
# ============================================================

print("\n========================================")
print("RETRIEVED CHUNKS")
print("========================================")


for rank, match in enumerate(
    matches,
    start=1
):

    print("\n----------------------------------------")

    print("Rank:", rank)

    print("Chunk ID:", match["id"])

    print(
        "Score:",
        round(match["score"], 4)
    )

    print("Text:")

    print(
        match["metadata"]["text"]
    )


# ============================================================
# 11. BUILD CONTEXT
# ============================================================
#
# We now take the retrieved chunks and combine them
# into one context string.
#
# This context will later be sent to the LLM.
#
# ============================================================

context_parts = []


for rank, match in enumerate(
    matches,
    start=1
):

    chunk_text = match["metadata"]["text"]

    context_parts.append(
        f"[Context {rank}]\n"
        f"{chunk_text}"
    )


context = "\n\n".join(
    context_parts
)


# ============================================================
# 12. DISPLAY CONTEXT
# ============================================================

print("\n========================================")
print("COMBINED CONTEXT")
print("========================================")

print(context)


# ============================================================
# 13. BUILD RAG PROMPT
# ============================================================
#
# This is the important part.
#
# We combine:
#
# System instruction
# +
# Retrieved context
# +
# User question
#
# Later this exact prompt structure will be
# sent to an LLM.
#
# ============================================================

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the
information provided in the context below.

If the answer cannot be found in the context,
say that the information is not available
in the provided context.

Do not make up information.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

USER QUESTION:

{question}

ANSWER:
""".strip()


# ============================================================
# 14. DISPLAY FINAL PROMPT
# ============================================================

print("\n========================================")
print("FINAL RAG PROMPT")
print("========================================")

print(prompt)


# ============================================================
# 15. SHOW RAG PIPELINE
# ============================================================

print("\n========================================")
print("RAG PIPELINE")
print("========================================")

print(
    """
User Question
      |
      v
Query Embedding
      |
      v
Pinecone
      |
      v
Top-K Relevant Chunks
      |
      v
Context Builder
      |
      v
RAG Prompt
      |
      v
NEXT STEP: LLM
"""
)


print("\nRAG prompt preparation completed successfully.")