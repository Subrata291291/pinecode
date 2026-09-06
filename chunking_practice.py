# ============================================================
# CHUNKING PRACTICE - RAG FUNDAMENTALS
# ============================================================
#
# Goal:
# 1. Understand why documents are split into chunks
# 2. Practice character-based chunking
# 3. Practice word-based chunking
# 4. Practice chunk overlap
# 5. See how a long document becomes multiple chunks
#
# This file does NOT use Pinecone.
# First understand chunking before storing chunks as vectors.
# ============================================================

# ============================================================
# 1. SAMPLE DOCUMENT
# ============================================================

document = """
Artificial Intelligence is transforming many industries.
Healthcare uses AI for medical image analysis, diagnosis support,
drug discovery, and patient monitoring.

Machine learning is a major part of Artificial Intelligence.
Machine learning systems learn patterns from data and use those
patterns to make predictions or decisions.

Generative AI can create text, images, code, audio, and other
types of content. Large Language Models are a major technology
behind modern text-based generative AI applications.

Retrieval Augmented Generation, commonly called RAG, combines
information retrieval with a Large Language Model. A RAG system
first retrieves relevant information from a knowledge base and
then provides that information to an LLM as context.

Vector databases such as Pinecone can store embeddings of text
chunks. During retrieval, a user's question is converted into
an embedding and compared with stored embeddings to find
semantically relevant chunks.
"""

# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("========================================")
print("DOCUMENT INFORMATION")
print("========================================")

print("Characters:", len(document))
print("Words:", len(document.split()))

# ============================================================
# 3. SIMPLE CHARACTER CHUNKING
# ============================================================

def character_chunks(text, chunk_size):
    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks


print("\n========================================")
print("CHARACTER CHUNKING")
print("========================================")

chunks = character_chunks(document, 200)

print("Chunk size: 200 characters")
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n----------------------------------------")
    print("Chunk:", i)
    print("Length:", len(chunk))
    print(chunk.strip())

# ============================================================
# 4. WORD CHUNKING
# ============================================================

def word_chunks(text, chunk_size):
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)

    return chunks


print("\n========================================")
print("WORD CHUNKING")
print("========================================")

chunks = word_chunks(document, 30)

print("Chunk size: 30 words")
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n----------------------------------------")
    print("Chunk:", i)
    print("Word count:", len(chunk.split()))
    print(chunk)

# ============================================================
# 5. WORD CHUNKING WITH OVERLAP
# ============================================================
#
# Example:
#
# Chunk 1: words 0 - 29
# Chunk 2: words 20 - 49
# Chunk 3: words 40 - 69
#
# Therefore 10 words are shared between adjacent chunks.
# ============================================================

def word_chunks_with_overlap(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(words):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))
        start += step

    return chunks


print("\n========================================")
print("WORD CHUNKING + OVERLAP")
print("========================================")

chunk_size = 30
overlap = 10

chunks = word_chunks_with_overlap(
    document,
    chunk_size,
    overlap
)

print("Chunk size:", chunk_size, "words")
print("Overlap:", overlap, "words")
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n----------------------------------------")
    print("Chunk:", i)
    print("Word count:", len(chunk.split()))
    print(chunk)

# ============================================================
# 6. COMPARE DIFFERENT CHUNK SIZES
# ============================================================

print("\n========================================")
print("CHUNK SIZE COMPARISON")
print("========================================")

for size in [20, 30, 50]:
    chunks = word_chunks_with_overlap(
        document,
        chunk_size=size,
        overlap=5
    )

    print(
        "Chunk size:",
        size,
        "| overlap: 5",
        "| chunks:",
        len(chunks)
    )

# ============================================================
# 7. RAG CHUNKING CONCEPT
# ============================================================

print("\n========================================")
print("RAG CHUNKING CONCEPT")
print("========================================")

print("""
Original Document
       |
       v
    Chunking
       |
   +---+---+---+---+
   |   |   |   |   |
   v   v   v   v   v
  C1  C2  C3  C4  C5
   |   |   |   |   |
   +---+---+---+---+
           |
           v
   Embedding each chunk
           |
           v
     Vector Database
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
           LLM
           |
           v
       Final Answer
""")

# ============================================================
# 8. IMPORTANT NOTES
# ============================================================

print("========================================")
print("IMPORTANT NOTES")
print("========================================")

print("""
1. Very large chunks:
   - More context
   - But retrieval can become less precise

2. Very small chunks:
   - More precise retrieval
   - But context may be lost

3. Overlap:
   - Preserves context across chunk boundaries
   - But creates more chunks and more storage

4. Real RAG systems usually prefer semantic or
   structure-aware splitting instead of blindly cutting
   every N characters.

Next we will take these chunks, create embeddings for them,
and store the chunks in Pinecone.
""")
