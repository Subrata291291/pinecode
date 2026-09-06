import os
import time

# The model was already downloaded once. Keep Hugging Face/Transformers from
# making network checks on every run, which can fail in restricted shells.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError(
        "PINECONE_API_KEY was not found.\n"
        "Make sure you created a .env file in the same folder as main.py."
    )


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = "demo-index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 creates 384-dimensional embeddings
EMBEDDING_DIMENSION = 384


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

print("Embedding model loaded successfully.")


# ============================================================
# CONNECT TO PINECONE
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(api_key=api_key)

print("Connected to Pinecone successfully.")


# ============================================================
# CHECK / CREATE INDEX
# ============================================================

print("\nChecking Pinecone indexes...")

existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:

    print(f"Index '{INDEX_NAME}' does not exist.")
    print("Creating index...")

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"Index '{INDEX_NAME}' created successfully.")

else:

    print(f"Index '{INDEX_NAME}' already exists.")


# ============================================================
# CONNECT TO INDEX
# ============================================================

index = pc.Index(INDEX_NAME)

print(f"Connected to index: {INDEX_NAME}")


# ============================================================
# SAMPLE DATA
# ============================================================

pinecone_texts = [
    "AI is transforming healthcare",
    "Robotics in industrial automation",
    "Machine learning in finance",
    "AI for simulation and modeling",
    "Cloud computing basics"
]


# ============================================================
# INSERT DATA INTO PINECONE
# ============================================================

print("\n========================================")
print("INSERTING DATA")
print("========================================")

for i, text in enumerate(pinecone_texts):

    # Generate embedding
    embedding = model.encode(text).tolist()

    # Create metadata
    metadata = {
        "text": text,
        "category": "ai" if "AI" in text else "other",
        "created_at": int(time.time()),
        "keywords": text.lower().split()
    }

    # Insert vector
    index.upsert(
        vectors=[
            (
                f"doc-{i}",
                embedding,
                metadata
            )
        ]
    )

    print(f"Inserted doc-{i}: {text}")


print("Data insertion completed.")


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

query = "AI in simulations"

print("\n========================================")
print("QUERY")
print("========================================")

print("Query:", query)

query_embedding = model.encode(query).tolist()


# ============================================================
# 1. BASIC SEMANTIC SEARCH
# ============================================================

print("\n========================================")
print("1. BASIC SEMANTIC SEARCH")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

for match in results["matches"]:

    print("\nID:", match["id"])
    print("Score:", match["score"])
    print("Text:", match["metadata"]["text"])
    print("Category:", match["metadata"]["category"])
    print("Keywords:", match["metadata"]["keywords"])

    print("--------------------------------")


# ============================================================
# 2. FILTER BY CATEGORY
# ============================================================

print("\n========================================")
print("2. FILTER BY CATEGORY")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
    filter={
        "category": {
            "$eq": "ai"
        }
    }
)

for match in results["matches"]:

    print("\nID:", match["id"])
    print("Score:", match["score"])
    print("Text:", match["metadata"]["text"])
    print("Category:", match["metadata"]["category"])

    print("--------------------------------")


# ============================================================
# 3. FILTER BY KEYWORD
# ============================================================

print("\n========================================")
print("3. FILTER BY KEYWORD")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
    filter={
        "keywords": {
            "$in": ["finance"]
        }
    }
)

for match in results["matches"]:

    print("\nID:", match["id"])
    print("Score:", match["score"])
    print("Text:", match["metadata"]["text"])
    print("Keywords:", match["metadata"]["keywords"])

    print("--------------------------------")


# ============================================================
# 4. COMBINED FILTER
# ============================================================

print("\n========================================")
print("4. COMBINED FILTER")
print("========================================")

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True,
    filter={
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

for match in results["matches"]:

    print("\nID:", match["id"])
    print("Score:", match["score"])
    print("Text:", match["metadata"]["text"])
    print("Category:", match["metadata"]["category"])
    print("Keywords:", match["metadata"]["keywords"])

    print("--------------------------------")


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("ALL OPERATIONS COMPLETED")
print("========================================")
