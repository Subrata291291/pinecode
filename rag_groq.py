import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# COMPLETE RAG WITH GROQ
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
# Context
#       |
#       v
# RAG Prompt
#       |
#       v
# Groq LLM
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

groq_api_key = os.getenv(
    "GROQ_API_KEY"
)


if not pinecone_api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found.\n"
        "Check your .env file."
    )


if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found.\n"
        "Check your .env file."
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

INDEX_NAME = "demo-index"

NAMESPACE = "chunk-practice"

TOP_K = 3


# ============================================================
# 3. LOCAL EMBEDDING MODEL
# ============================================================

MODEL_NAME = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)


# ============================================================
# 4. GROQ MODEL
# ============================================================

LLM_MODEL = "openai/gpt-oss-20b"


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
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
# 7. CONNECT TO GROQ
# ============================================================

print("\nConnecting to Groq...")

client = Groq(
    api_key=groq_api_key
)

print("Groq client initialized.")

print("LLM model:", LLM_MODEL)


# ============================================================
# 8. USER QUESTION
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
# 9. CREATE QUERY EMBEDDING
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
# 10. RETRIEVE FROM PINECONE
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
# 11. GET MATCHES
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
# 12. DISPLAY RETRIEVED CHUNKS
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
# 13. BUILD CONTEXT
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
# 14. BUILD RAG PROMPT
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
# 15. DISPLAY PROMPT
# ============================================================

print("\n========================================")
print("RAG PROMPT")
print("========================================")

print(prompt)


# ============================================================
# 16. CALL GROQ LLM
# ============================================================

print("\n========================================")
print("CALLING GROQ LLM")
print("========================================")


response = client.chat.completions.create(

    model=LLM_MODEL,

    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful RAG assistant. "
                "Use only the provided context."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ============================================================
# 17. GET FINAL ANSWER
# ============================================================

answer = response.choices[0].message.content


# ============================================================
# 18. DISPLAY FINAL ANSWER
# ============================================================

print("\n========================================")
print("FINAL RAG ANSWER")
print("========================================")

print(answer)


# ============================================================
# 19. SHOW USAGE
# ============================================================

if response.usage:

    print("\n========================================")
    print("TOKEN USAGE")
    print("========================================")

    print(
        "Prompt tokens:",
        response.usage.prompt_tokens
    )

    print(
        "Completion tokens:",
        response.usage.completion_tokens
    )

    print(
        "Total tokens:",
        response.usage.total_tokens
    )


# ============================================================
# 20. FINAL PIPELINE
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
Groq LLM
      |
      v
FINAL ANSWER
"""
)

print("\nRAG completed successfully.")