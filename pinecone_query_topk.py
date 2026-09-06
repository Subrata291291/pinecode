import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ============================================================
# PINECONE QUERY + TOP-K PRACTICE
# ============================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Load API key
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY was not found. Check your .env file.")

# Configuration
INDEX_NAME = "demo-index"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True
)

print("Embedding model loaded successfully.")

# Connect to Pinecone
print("\nConnecting to Pinecone...")

pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

print("Connected to Pinecone.")
print("Index:", INDEX_NAME)

# Show index information
print("\n========================================")
print("INDEX INFORMATION")
print("========================================")

stats = index.describe_index_stats()
print("Total vectors:", stats["total_vector_count"])

# ------------------------------------------------------------
# QUERY 1
# ------------------------------------------------------------

query = "AI applications in healthcare"

print("\n========================================")
print("QUERY 1")
print("========================================")

print("User query:")
print(query)

# Query -> embedding
query_embedding = model.encode(query).tolist()

print("\nQuery embedding created.")
print("Embedding dimensions:", len(query_embedding))

# ------------------------------------------------------------
# TOP-K = 1
# ------------------------------------------------------------

print("\n========================================")
print("TOP-K = 1")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=1,
    include_metadata=True
)

for rank, match in enumerate(results["matches"], start=1):
    print("\nRank:", rank)
    print("ID:", match["id"])
    print("Score:", round(match["score"], 4))
    print("Text:", match["metadata"]["text"])

# ------------------------------------------------------------
# TOP-K = 3
# ------------------------------------------------------------

print("\n========================================")
print("TOP-K = 3")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

for rank, match in enumerate(results["matches"], start=1):
    print("\nRank:", rank)
    print("ID:", match["id"])
    print("Score:", round(match["score"], 4))
    print("Text:", match["metadata"]["text"])

# ------------------------------------------------------------
# TOP-K = 5
# ------------------------------------------------------------

print("\n========================================")
print("TOP-K = 5")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

for rank, match in enumerate(results["matches"], start=1):
    print("\nRank:", rank)
    print("ID:", match["id"])
    print("Score:", round(match["score"], 4))
    print("Text:", match["metadata"]["text"])

# ------------------------------------------------------------
# QUERY 2
# ------------------------------------------------------------

query2 = "financial machine learning"

print("\n========================================")
print("QUERY 2")
print("========================================")

print("User query:")
print(query2)

query2_embedding = model.encode(query2).tolist()

results = index.query(
    vector=query2_embedding,
    top_k=3,
    include_metadata=True
)

for rank, match in enumerate(results["matches"], start=1):
    print("\nRank:", rank)
    print("Score:", round(match["score"], 4))
    print("Text:", match["metadata"]["text"])

# ------------------------------------------------------------
# CONCEPT
# ------------------------------------------------------------

print("\n========================================")
print("PRACTICE COMPLETED")
print("========================================")

print("""
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Pinecone
    ↓
Similarity Search
    ↓
top_k
    ↓
Ranked Results

top_k = 1  → best match
top_k = 3  → best 3 matches
top_k = 5  → best 5 matches
""")
