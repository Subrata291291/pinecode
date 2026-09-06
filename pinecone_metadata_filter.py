import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ============================================================
# PINECONE METADATA FILTER PRACTICE
# ============================================================
#
# Uses the existing "demo-index".
# Does NOT insert or update vectors.
#
# Existing metadata from main.py:
#
# doc-0 -> category: ai
# doc-1 -> category: other
# doc-2 -> category: other
# doc-3 -> category: ai
# doc-4 -> category: other
#
# Goal:
# 1. Learn $eq
# 2. Learn $in
# 3. Learn $and
# 4. Learn $or
# 5. Understand FILTER + SEMANTIC SEARCH together
# ============================================================

# ============================================================
# 1. OFFLINE MODEL SETTINGS
# ============================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ============================================================
# 2. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY was not found. Check your .env file.")

# ============================================================
# 3. CONFIGURATION
# ============================================================

INDEX_NAME = "demo-index"
MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================
# 4. LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True
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

# ============================================================
# 6. COMMON QUERY
# ============================================================

query = "AI applications in simulation"
query_embedding = model.encode(query).tolist()

print("\n========================================")
print("COMMON QUERY")
print("========================================")
print("Query:", query)

# ============================================================
# 7. HELPER FUNCTION
# ============================================================

def run_search(title, filter_expression):

    print("\n========================================")
    print(title)
    print("========================================")

    print("Filter:")
    print(filter_expression)

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter=filter_expression
    )

    if not results["matches"]:
        print("\nNo matching documents found.")
        return

    for rank, match in enumerate(results["matches"], start=1):

        print("\nRank:", rank)
        print("ID:", match["id"])
        print("Score:", round(match["score"], 4))
        print("Text:", match["metadata"]["text"])
        print("Category:", match["metadata"]["category"])
        print("Keywords:", match["metadata"]["keywords"])

# ============================================================
# 8. FILTER 1 — $eq
# ============================================================

run_search(
    "FILTER 1 — $eq",
    {
        "category": {
            "$eq": "ai"
        }
    }
)

# ============================================================
# 9. FILTER 2 — $in
# ============================================================

run_search(
    "FILTER 2 — $in",
    {
        "category": {
            "$in": ["ai", "other"]
        }
    }
)

# ============================================================
# 10. FILTER 3 — $and
# ============================================================

run_search(
    "FILTER 3 — $and",
    {
        "$and": [
            {
                "category": {
                    "$eq": "ai"
                }
            },
            {
                "keywords": {
                    "$in": ["simulation"]
                }
            }
        ]
    }
)

# ============================================================
# 11. FILTER 4 — $or
# ============================================================

run_search(
    "FILTER 4 — $or",
    {
        "$or": [
            {
                "category": {
                    "$eq": "ai"
                }
            },
            {
                "keywords": {
                    "$in": ["finance"]
                }
            }
        ]
    }
)

# ============================================================
# 12. FILTER 5 — KEYWORD ONLY
# ============================================================

run_search(
    "FILTER 5 — KEYWORD = healthcare",
    {
        "keywords": {
            "$in": ["healthcare"]
        }
    }
)

# ============================================================
# 13. COMPLETED
# ============================================================

print("\n========================================")
print("PRACTICE COMPLETED")
print("========================================")

print("\nConcept:")
print("User Query -> Embedding -> Pinecone")
print("                     -> Metadata Filter")
print("                     -> Similarity Ranking")
print("                     -> Top-K Results")
