import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# COMPLETE RAG PIPELINE
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
#       |
#       v
# OpenAI LLM
#       |
#       v
# Final Answer
#
# ============================================================


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


pinecone_api_key = os.getenv(
    "PINECONE_API_KEY"
)

openai_api_key = os.getenv(
    "OPENAI_API_KEY"
)


if not pinecone_api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found.\n"
        "Check your .env file."
    )


if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found.\n"
        "Check your .env file."
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

INDEX_NAME = "demo-index"

NAMESPACE = "chunk-practice"

TOP_K = 3


# Local embedding model
MODEL_NAME = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)


# LLM model
LLM_MODEL = "gpt-5.6-luna"


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print("Embedding model loaded successfully.")


# ============================================================
# 4. CONNECT TO PINECONE
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
# 5. CONNECT TO OPENAI
# ============================================================

print("\nConnecting to OpenAI...")

client = OpenAI(
    api_key=openai_api_key
)

print("OpenAI client initialized.")


# ============================================================
# 6. USER QUESTION
# ============================================================

question = (
    "What is Retrieval Augmented Generation "
    "and how does it work?"
)


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

query_embedding = embedding_model.encode(
    question
).tolist()


print("Query embedding created.")

print(
    "Dimensions:",
    len(query_embedding)
)


# ============================================================
# 8. RETRIEVE FROM PINECONE
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
# 9. GET MATCHES
# ============================================================

matches = results["matches"]


print(
    "Retrieved chunks:",
    len(matches)
)


if not matches:

    print(
        "\nNo relevant chunks were found."
    )

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

    print(
        "Chunk ID:",
        match["id"]
    )

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

context_parts = []


for rank, match in enumerate(
    matches,
    start=1
):

    chunk_text = (
        match["metadata"]["text"]
    )

    context_parts.append(
        f"[Context {rank}]\n"
        f"{chunk_text}"
    )


context = "\n\n".join(
    context_parts
)


# ============================================================
# 12. BUILD RAG PROMPT
# ============================================================

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the
context, clearly say that the information
is not available in the provided context.

Do not make up information.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

USER QUESTION:

{question}
""".strip()


# ============================================================
# 13. DISPLAY PROMPT
# ============================================================

print("\n========================================")
print("RAG PROMPT")
print("========================================")

print(prompt)


# ============================================================
# 14. SEND PROMPT TO LLM
# ============================================================

print("\n========================================")
print("CALLING LLM")
print("========================================")


response = client.responses.create(

    model=LLM_MODEL,

    input=prompt
)


# ============================================================
# 15. GET FINAL ANSWER
# ============================================================

answer = response.output_text


# ============================================================
# 16. DISPLAY FINAL ANSWER
# ============================================================

print("\n========================================")
print("FINAL RAG ANSWER")
print("========================================")

print(answer)


# ============================================================
# 17. FINAL PIPELINE
# ============================================================

print("\n========================================")
print("COMPLETE RAG PIPELINE")
print("========================================")

print(
    """
User Question
      |
      v
Embedding Model
      |
      v
Query Vector
      |
      v
Pinecone
      |
      v
Top-K Relevant Chunks
      |
      v
Context
      |
      v
RAG Prompt
      |
      v
OpenAI LLM
      |
      v
FINAL ANSWER
"""
)


print("\nRAG completed successfully.")