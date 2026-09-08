import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# RAG + METADATA FILTERING
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
# Pinecone
#       |
#       +---- Metadata Filter
#       |
#       v
# Filtered Candidates
#       |
#       v
# Top-K Relevant Chunks
#       |
#       v
# Context
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

llm_client = Groq(
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
# 10. DEFINE METADATA FILTER
# ============================================================
#
# We want only chunks belonging to:
#
# rag-document-001
#
# Pinecone metadata filtering uses expressions
# such as:
#
# {"document_id": {"$eq": "rag-document-001"}}
#
# ============================================================

metadata_filter = {
    "document_id": {
        "$eq": "rag-document-001"
    }
}


print("\n========================================")
print("METADATA FILTER")
print("========================================")

print(metadata_filter)


# ============================================================
# 11. PINECONE RETRIEVAL WITH FILTER
# ============================================================

print("\n========================================")
print("PINECONE FILTERED RETRIEVAL")
print("========================================")


results = index.query(

    namespace=NAMESPACE,

    vector=query_embedding,

    top_k=TOP_K,

    filter=metadata_filter,

    include_metadata=True,

    include_values=False
)


# ============================================================
# 12. GET MATCHES
# ============================================================

matches = results["matches"]


print(
    "Retrieved chunks:",
    len(matches)
)


if not matches:

    print(
        "\nNo chunks matched the metadata filter."
    )

    raise SystemExit


# ============================================================
# 13. DISPLAY FILTERED RESULTS
# ============================================================

print("\n========================================")
print("FILTERED RESULTS")
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

    print(
        "Document ID:",
        match["metadata"]["document_id"]
    )

    print("Text:")

    print(
        match["metadata"]["text"]
    )


# ============================================================
# 14. BUILD CONTEXT
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
# 15. DISPLAY CONTEXT
# ============================================================

print("\n========================================")
print("FILTERED CONTEXT")
print("========================================")

print(context)


# ============================================================
# 16. BUILD RAG PROMPT
# ============================================================

prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the information provided in the context.

The context was retrieved from the
requested document using metadata filtering.

If the answer cannot be found in the
context, say that the information is not
available in the provided context.

Do not make up information.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

USER QUESTION:

{question}
""".strip()


# ============================================================
# 17. DISPLAY PROMPT
# ============================================================

print("\n========================================")
print("RAG PROMPT")
print("========================================")

print(prompt)


# ============================================================
# 18. CALL GROQ
# ============================================================

print("\n========================================")
print("CALLING GROQ LLM")
print("========================================")


response = llm_client.chat.completions.create(

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
# 19. GET ANSWER
# ============================================================

answer = (
    response
    .choices[0]
    .message
    .content
)


# ============================================================
# 20. DISPLAY ANSWER
# ============================================================

print("\n========================================")
print("FINAL RAG ANSWER")
print("========================================")

print(answer)


# ============================================================
# 21. TOKEN USAGE
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
# 22. FINAL PIPELINE
# ============================================================

print("\n========================================")
print("RAG + METADATA FILTER PIPELINE")
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
      +---- Metadata Filter
      |
      v
Filtered Vector Search
      |
      v
Top-K Chunks
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


print(
    "\nRAG with metadata filtering "
    "completed successfully."
)