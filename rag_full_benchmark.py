# rag_full_benchmark.py
#
# Reusable RAG Benchmark
#
# Documents / chunks live in Pinecone.
# Evaluation questions and ground-truth chunks are maintained
# separately.
#
# Pipeline:
#
# Question
#    ↓
# Embedding
#    ↓
# Pinecone Retrieval
#    ↓
# Top-K
#    ↓
# Retrieval Metrics
#    ↓
# Context
#    ↓
# Groq
#    ↓
# Generated Answer
#    ↓
# LLM Judge
#    ↓
# Faithfulness
# Answer Relevancy
# Context Relevancy


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

TOP_K = 5


# ============================================================
# 2. EVALUATION DATASET
# ============================================================
#
# IMPORTANT:
#
# This is NOT your document data.
#
# This is only evaluation/ground-truth data.
#
# In a real project, keep this in a separate JSON/CSV file.
#
# Example:
#
# {
#     "question": "...",
#     "relevant_chunks": ["chunk-4", "chunk-5"]
# }
#
# You can later replace this section with:
#
# evaluation_dataset.json
#
# or
#
# evaluation_dataset.csv
# ============================================================

EVALUATION_DATASET = [

    {
        "question": "What is Retrieval Augmented Generation?",
        "relevant_chunks": ["chunk-0"],
    },

    {
        "question": "What does RAG stand for?",
        "relevant_chunks": ["chunk-0"],
    },

    {
        "question": "Why are documents divided into chunks?",
        "relevant_chunks": ["chunk-0"],
    },

    {
        "question": "What are embeddings used for in RAG?",
        "relevant_chunks": ["chunk-1"],
    },

    {
        "question": "What is a vector database?",
        "relevant_chunks": ["chunk-1"],
    },

    {
        "question": "How does Pinecone help with RAG?",
        "relevant_chunks": ["chunk-1"],
    },

    {
        "question": "How is a user query converted for vector search?",
        "relevant_chunks": ["chunk-2"],
    },

    {
        "question": "What happens after a query is embedded?",
        "relevant_chunks": ["chunk-2"],
    },

    {
        "question": "How does retrieval work in RAG?",
        "relevant_chunks": ["chunk-2"],
    },

    {
        "question": "Why does an LLM need context in RAG?",
        "relevant_chunks": ["chunk-3"],
    },

    {
        "question": "What happens after relevant chunks are retrieved?",
        "relevant_chunks": ["chunk-3"],
    },

    {
        "question": "How does RAG provide information to an LLM?",
        "relevant_chunks": ["chunk-3"],
    },

    {
        "question": "Why is chunk overlap useful in RAG?",
        "relevant_chunks": ["chunk-4"],
    },

    {
        "question": "What is chunk size?",
        "relevant_chunks": ["chunk-4"],
    },

    {
        "question": "What is reranking in RAG?",
        "relevant_chunks": ["chunk-4"],
    },

    {
        "question": "What is hybrid search?",
        "relevant_chunks": ["chunk-4", "chunk-5"],
    },

    {
        "question": "What is metadata filtering?",
        "relevant_chunks": ["chunk-5"],
    },

    {
        "question": "Why is reranking useful?",
        "relevant_chunks": ["chunk-4", "chunk-5"],
    },

    {
        "question": "What is RAG evaluation?",
        "relevant_chunks": ["chunk-5"],
    },

    {
        "question": "Why are citations and access control important?",
        "relevant_chunks": ["chunk-5"],
    },
]


# ============================================================
# 3. VALIDATE ENVIRONMENT
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
# 7. RETRIEVAL
# ============================================================

def retrieve(question):

    query_vector = embedding_model.encode(
        question
    ).tolist()

    results = index.query(
        namespace=NAMESPACE,
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )

    retrieved_chunks = []

    context_parts = []

    for match in results.matches:

        chunk_id = match.metadata.get(
            "chunk_id"
        )

        text = match.metadata.get(
            "text",
            ""
        )

        normalized_id = f"chunk-{chunk_id}"

        retrieved_chunks.append(
            normalized_id
        )

        context_parts.append(
            f"[{normalized_id}]\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    return (
        retrieved_chunks,
        context,
        results
    )


# ============================================================
# 8. PRECISION@K
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k
):

    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        1
        for chunk in retrieved_at_k
        if chunk in relevant
    )

    return relevant_count / k


# ============================================================
# 9. RECALL@K
# ============================================================

def recall_at_k(
    retrieved,
    relevant,
    k
):

    retrieved_at_k = retrieved[:k]

    relevant_count = sum(
        1
        for chunk in retrieved_at_k
        if chunk in relevant
    )

    return relevant_count / len(relevant)


# ============================================================
# 10. RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
    retrieved,
    relevant
):

    for rank, chunk in enumerate(
        retrieved,
        start=1
    ):

        if chunk in relevant:

            return 1 / rank

    return 0


# ============================================================
# 11. GENERATE RAG ANSWER
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the
provided context.

Do not invent information.

If the context does not contain enough
information, clearly say that the information
is not available in the provided context.

USER QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    return answer, response


# ============================================================
# 12. LLM-AS-A-JUDGE
# ============================================================

def evaluate_answer(
    question,
    context,
    answer
):

    judge_prompt = f"""
You are an expert evaluator for a
Retrieval Augmented Generation system.

Evaluate the generated answer using ONLY
the question and retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}


-----------------------------------------------
FAITHFULNESS
-----------------------------------------------

Are the claims in the answer supported
by the retrieved context?

0 = unsupported
1 = partially supported
2 = fully supported


-----------------------------------------------
ANSWER RELEVANCY
-----------------------------------------------

Does the answer directly answer the question?

0 = irrelevant
1 = partially relevant
2 = highly relevant


-----------------------------------------------
CONTEXT RELEVANCY
-----------------------------------------------

Is the retrieved context useful for answering
the question?

0 = irrelevant
1 = partially useful
2 = highly useful


Return ONLY valid JSON.

Format:

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

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": judge_prompt
            }
        ],
        temperature=0
    )

    raw_output = (
        response
        .choices[0]
        .message
        .content
    )

    try:

        evaluation = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        print("\nWARNING:")
        print("Judge returned invalid JSON.")

        print("\nRaw response:")
        print(raw_output)

        evaluation = None


    return evaluation, response


# ============================================================
# 13. STORAGE FOR METRICS
# ============================================================

precision_scores = []
recall_scores = []
rr_scores = []

faithfulness_scores = []
answer_relevancy_scores = []
context_relevancy_scores = []

total_rag_tokens = 0
total_judge_tokens = 0


# ============================================================
# 14. RUN BENCHMARK
# ============================================================

print("\n")
print("=" * 75)
print("FULL RAG BENCHMARK")
print("=" * 75)

print(
    f"\nQuestions: "
    f"{len(EVALUATION_DATASET)}"
)

print(
    f"Top-K: "
    f"{TOP_K}"
)


for number, item in enumerate(
    EVALUATION_DATASET,
    start=1
):

    question = item["question"]

    relevant = item[
        "relevant_chunks"
    ]


    # ========================================================
    # RETRIEVAL
    # ========================================================

    retrieved, context, results = retrieve(
        question
    )


    # ========================================================
    # RETRIEVAL METRICS
    # ========================================================

    precision = precision_at_k(
        retrieved,
        relevant,
        TOP_K
    )

    recall = recall_at_k(
        retrieved,
        relevant,
        TOP_K
    )

    rr = reciprocal_rank(
        retrieved,
        relevant
    )


    precision_scores.append(
        precision
    )

    recall_scores.append(
        recall
    )

    rr_scores.append(
        rr
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    answer, rag_response = generate_answer(
        question,
        context
    )


    # ========================================================
    # EVALUATE ANSWER
    # ========================================================

    evaluation, judge_response = (
        evaluate_answer(
            question,
            context,
            answer
        )
    )


    # ========================================================
    # TOKEN USAGE
    # ========================================================

    total_rag_tokens += (
        rag_response
        .usage
        .total_tokens
    )

    total_judge_tokens += (
        judge_response
        .usage
        .total_tokens
    )


    # ========================================================
    # GENERATION METRICS
    # ========================================================

    if evaluation is not None:

        faithfulness = (
            evaluation[
                "faithfulness"
            ]["score"] / 2
        )

        answer_relevancy = (
            evaluation[
                "answer_relevancy"
            ]["score"] / 2
        )

        context_relevancy = (
            evaluation[
                "context_relevancy"
            ]["score"] / 2
        )


        faithfulness_scores.append(
            faithfulness
        )

        answer_relevancy_scores.append(
            answer_relevancy
        )

        context_relevancy_scores.append(
            context_relevancy
        )

    else:

        faithfulness = 0
        answer_relevancy = 0
        context_relevancy = 0


    # ========================================================
    # DISPLAY PROGRESS
    # ========================================================

    print(
        f"\n[{number}/{len(EVALUATION_DATASET)}]"
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Expected: {relevant}"
    )

    print(
        f"Retrieved: {retrieved}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{precision:.3f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{recall:.3f}"
    )

    print(
        f"MRR: "
        f"{rr:.3f}"
    )

    print(
        f"Faithfulness: "
        f"{faithfulness:.3f}"
    )

    print(
        f"Answer Relevancy: "
        f"{answer_relevancy:.3f}"
    )

    print(
        f"Context Relevancy: "
        f"{context_relevancy:.3f}"
    )


# ============================================================
# 15. CALCULATE FINAL METRICS
# ============================================================

mean_precision = (
    sum(precision_scores)
    / len(precision_scores)
)

mean_recall = (
    sum(recall_scores)
    / len(recall_scores)
)

mrr = (
    sum(rr_scores)
    / len(rr_scores)
)


if faithfulness_scores:

    mean_faithfulness = (
        sum(faithfulness_scores)
        / len(faithfulness_scores)
    )

    mean_answer_relevancy = (
        sum(answer_relevancy_scores)
        / len(answer_relevancy_scores)
    )

    mean_context_relevancy = (
        sum(context_relevancy_scores)
        / len(context_relevancy_scores)
    )

else:

    mean_faithfulness = 0
    mean_answer_relevancy = 0
    mean_context_relevancy = 0


# ============================================================
# 16. FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("FINAL RAG BENCHMARK REPORT")
print("=" * 75)


print("\n")
print("RETRIEVAL QUALITY")

print(
    f"Mean Precision@{TOP_K}: "
    f"{mean_precision:.3f}"
)

print(
    f"Mean Recall@{TOP_K}:    "
    f"{mean_recall:.3f}"
)

print(
    f"MRR:                     "
    f"{mrr:.3f}"
)


print("\n")
print("GENERATION QUALITY")

print(
    f"Mean Faithfulness:       "
    f"{mean_faithfulness:.3f}"
)

print(
    f"Mean Answer Relevancy:   "
    f"{mean_answer_relevancy:.3f}"
)

print(
    f"Mean Context Relevancy:  "
    f"{mean_context_relevancy:.3f}"
)


print("\n")
print("TOKEN USAGE")

print(
    f"RAG tokens:              "
    f"{total_rag_tokens}"
)

print(
    f"Judge tokens:            "
    f"{total_judge_tokens}"
)

print(
    f"Total tokens:            "
    f"{total_rag_tokens + total_judge_tokens}"
)


print("\n")
print("=" * 75)
print("BENCHMARK COMPLETED")
print("=" * 75)