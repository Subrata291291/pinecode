import os

# ============================================================
# EMBEDDING + COSINE SIMILARITY PRACTICE
# ============================================================
#
# Goal:
# 1. Load the local SentenceTransformer model
# 2. Convert text into 384-dimensional embeddings
# 3. Compare sentences using cosine similarity
# 4. Understand why semantic search works
#
# This file does NOT use Pinecone yet.
# First understand embeddings and similarity.
# ============================================================

# Keep Hugging Face / Transformers offline because
# the model has already been downloaded locally.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer, util


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    local_files_only=True
)

print("Embedding model loaded successfully.")


# ============================================================
# 2. CREATE OUR TEST SENTENCES
# ============================================================

texts = [
    "AI is transforming healthcare",
    "Artificial intelligence is changing the healthcare industry",
    "AI is used for simulation and modeling",
    "Machine learning is used in finance",
    "I enjoy cooking pasta",
]


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("\n========================================")
print("CREATING EMBEDDINGS")
print("========================================")

embeddings = model.encode(texts)

for i, (text, embedding) in enumerate(zip(texts, embeddings)):

    print(f"\nText {i}:")
    print(text)

    print("Number of dimensions:")
    print(len(embedding))

    # Print only the first 5 numbers so the terminal
    # does not become too large.
    print("First 5 vector values:")
    print(embedding[:5])


# ============================================================
# 4. COMPARE TWO SENTENCES
# ============================================================

print("\n========================================")
print("PAIRWISE SIMILARITY")
print("========================================")

pairs = [
    (0, 1),  # healthcare vs healthcare
    (0, 2),  # healthcare vs simulation
    (0, 4),  # healthcare vs cooking
    (2, 3),  # simulation vs finance
    (2, 4),  # simulation vs cooking
]

for i, j in pairs:

    score = util.cos_sim(
        embeddings[i],
        embeddings[j]
    ).item()

    print("\n----------------------------------------")
    print("Text A:", texts[i])
    print("Text B:", texts[j])
    print("Cosine similarity:", round(score, 4))


# ============================================================
# 5. FIND THE MOST SIMILAR TEXT TO A QUERY
# ============================================================

print("\n========================================")
print("SEMANTIC SIMILARITY RANKING")
print("========================================")

query = "AI applications in healthcare"

query_embedding = model.encode(query)

# Compare the query with every document.
scores = util.cos_sim(
    query_embedding,
    embeddings
)[0]

# Sort from highest score to lowest score.
ranked_results = sorted(
    enumerate(scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nQuery:")
print(query)

print("\nResults:")

for rank, (index, score) in enumerate(ranked_results, start=1):

    print("\nRank:", rank)
    print("Score:", round(score.item(), 4))
    print("Text:", texts[index])


# ============================================================
# 6. FINAL CONCEPT
# ============================================================

print("\n========================================")
print("PRACTICE COMPLETED")
print("========================================")

print("""
What happened?

Text
  ↓
SentenceTransformer
  ↓
384-dimensional embedding
  ↓
Cosine similarity
  ↓
Most semantically similar text

Next we will connect this concept to Pinecone.
""")
