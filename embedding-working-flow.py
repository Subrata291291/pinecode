from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)

text = "How to reduce employee burnout?"

embedding = model.encode(text)

print(f"Embedding dimensions: {len(embedding)}")
print(embedding)