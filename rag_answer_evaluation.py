# rag_answer_evaluation.py

import os
import json

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

INDEX_NAME = "demo-index"
NAMESPACE = "chunk-practice"

EMBEDDING_MODEL = os.path.join(
    os.path.dirname(__file__),
    ".models",
    "all-MiniLM-L6-v2"
)

LLM_MODEL = "openai/gpt-oss-20b"

TOP_K = 3


# ============================================================
# 2. CHECK API KEYS
# ============================================================

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY not found in .env"
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )


# ============================================================
# 3. USER QUESTION
# ============================================================

question = "What is Retrieval Augmented Generation?"


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# 5. CONNECT TO PINECONE
# ============================================================

print("\nConnecting to Pinecone...")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone.")


# ============================================================
# 6. CONNECT TO GROQ
# ============================================================

print("\nConnecting to Groq...")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("Connected to Groq.")


# ============================================================
# 7. CREATE QUESTION EMBEDDING
# ============================================================

query_vector = embedding_model.encode(
    question
).tolist()


# ============================================================
# 8. RETRIEVE FROM PINECONE
# ============================================================

print("\nSearching Pinecone...")

results = index.query(
    namespace=NAMESPACE,
    vector=query_vector,
    top_k=TOP_K,
    include_metadata=True,
    include_values=False
)


# ============================================================
# 9. BUILD CONTEXT
# ============================================================

context_parts = []

print("\nRetrieved chunks:")

for rank, match in enumerate(
    results.matches,
    start=1
):

    chunk_id = match.metadata.get("chunk_id")
    text = match.metadata.get("text")
    score = match.score

    print(
        f"{rank}. chunk-{chunk_id} "
        f"(score={score:.4f})"
    )

    context_parts.append(
        f"[Chunk {chunk_id}]\n{text}"
    )


context = "\n\n".join(context_parts)


# ============================================================
# 10. GENERATE RAG ANSWER
# ============================================================

rag_prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that the information is not available in the context.

Do not invent facts.

USER QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""


print("\nGenerating RAG answer...")

rag_response = groq_client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": rag_prompt
        }
    ],
    temperature=0
)

answer = rag_response.choices[0].message.content


# ============================================================
# 11. DISPLAY ANSWER
# ============================================================

print("\n")
print("=" * 70)
print("GENERATED RAG ANSWER")
print("=" * 70)

print(answer)


# ============================================================
# 12. LLM-AS-A-JUDGE PROMPT
# ============================================================

judge_prompt = f"""
You are an expert evaluator for a Retrieval Augmented Generation
(RAG) system.

Evaluate the generated answer using the question and retrieved
context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}


Evaluate these three dimensions:

1. FAITHFULNESS

Does the answer contain only claims that are supported by
the retrieved context?

Score:
0 = completely unsupported
1 = partially supported
2 = fully supported


2. ANSWER_RELEVANCY

Does the answer directly answer the user's question?

Score:
0 = irrelevant
1 = partially relevant
2 = highly relevant


3. CONTEXT_RELEVANCY

Is the retrieved context useful for answering the question?

Score:
0 = mostly irrelevant
1 = partially useful
2 = highly useful


Return ONLY valid JSON.

Use exactly this format:

{{
    "faithfulness": {{
        "score": 0,
        "reason": "..."
    }},
    "answer_relevancy": {{
        "score": 0,
        "reason": "..."
    }},
    "context_relevancy": {{
        "score": 0,
        "reason": "..."
    }}
}}
"""


# ============================================================
# 13. RUN LLM JUDGE
# ============================================================

print("\nEvaluating answer with LLM judge...")

judge_response = groq_client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": judge_prompt
        }
    ],
    temperature=0
)


judge_text = judge_response.choices[0].message.content


# ============================================================
# 14. DISPLAY RAW JUDGE RESPONSE
# ============================================================

print("\n")
print("=" * 70)
print("RAW JUDGE RESPONSE")
print("=" * 70)

print(judge_text)


# ============================================================
# 15. PARSE JSON
# ============================================================

try:

    evaluation = json.loads(judge_text)

except json.JSONDecodeError:

    print("\nCould not parse judge response as JSON.")

    evaluation = None


# ============================================================
# 16. DISPLAY EVALUATION
# ============================================================

if evaluation:

    print("\n")
    print("=" * 70)
    print("RAG ANSWER EVALUATION")
    print("=" * 70)


    faithfulness = evaluation["faithfulness"]
    answer_relevancy = evaluation["answer_relevancy"]
    context_relevancy = evaluation["context_relevancy"]


    print("\nFaithfulness:")
    print(
        f"Score: {faithfulness['score']}/2"
    )
    print(
        f"Reason: {faithfulness['reason']}"
    )


    print("\nAnswer Relevancy:")
    print(
        f"Score: {answer_relevancy['score']}/2"
    )
    print(
        f"Reason: {answer_relevancy['reason']}"
    )


    print("\nContext Relevancy:")
    print(
        f"Score: {context_relevancy['score']}/2"
    )
    print(
        f"Reason: {context_relevancy['reason']}"
    )


# ============================================================
# 17. TOKEN USAGE
# ============================================================

print("\n")
print("=" * 70)
print("TOKEN USAGE")
print("=" * 70)

print(
    "\nRAG prompt tokens:",
    rag_response.usage.prompt_tokens
)

print(
    "RAG completion tokens:",
    rag_response.usage.completion_tokens
)

print(
    "RAG total tokens:",
    rag_response.usage.total_tokens
)

print(
    "\nJudge prompt tokens:",
    judge_response.usage.prompt_tokens
)

print(
    "Judge completion tokens:",
    judge_response.usage.completion_tokens
)

print(
    "Judge total tokens:",
    judge_response.usage.total_tokens
)


print("\n")
print("=" * 70)
print("Evaluation completed.")
print("=" * 70)