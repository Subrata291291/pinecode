# ragas_style_evaluation.py

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
# 2. EVALUATION DATASET
# ============================================================

evaluation_dataset = [
    {
        "question": "What is Retrieval Augmented Generation?",
        "relevant_chunks": ["chunk-0"],
    },
    {
        "question": "Why is chunk overlap useful in RAG?",
        "relevant_chunks": ["chunk-4"],
    },
    {
        "question": "What is hybrid search?",
        "relevant_chunks": ["chunk-4", "chunk-5"],
    },
]


# ============================================================
# 3. CHECK API KEYS
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
# 4. LOAD MODELS / CLIENTS
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


print("\nConnecting to Pinecone...")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone.")


print("\nConnecting to Groq...")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("Connected to Groq.")


# ============================================================
# 5. GENERATE RAG ANSWER
# ============================================================

def generate_rag_answer(question, context):

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

Do not invent information.

If the context does not contain enough information,
say that the information is not available in the context.

QUESTION:
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

    return (
        response.choices[0].message.content,
        response
    )


# ============================================================
# 6. JUDGE THE RAG ANSWER
# ============================================================

def evaluate_with_llm(
    question,
    context,
    answer
):

    judge_prompt = f"""
You are an expert evaluator for a RAG system.

Evaluate the generated answer using the question
and retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}


Evaluate these dimensions.


1. FAITHFULNESS

Does the answer contain claims that are supported
by the retrieved context?

Score:

0 = unsupported
1 = partially supported
2 = fully supported


2. ANSWER_RELEVANCY

Does the answer directly answer the user's question?

Score:

0 = irrelevant
1 = partially relevant
2 = highly relevant


3. CONTEXT_RELEVANCY

Is the retrieved context useful for answering
the question?

Score:

0 = irrelevant
1 = partially useful
2 = highly useful


Return ONLY valid JSON.

Use exactly:

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
        response.choices[0].message.content
    )

    try:

        evaluation = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        print("\nJudge returned invalid JSON:")
        print(raw_output)

        evaluation = None


    return evaluation, response


# ============================================================
# 7. BUILD CONTEXT FROM PINECONE
# ============================================================

def retrieve_context(question):

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
            "text"
        )

        score = match.score

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
# 8. NORMALIZE SCORE
# ============================================================

def normalize_score(score):

    return score / 2


# ============================================================
# 9. MAIN EVALUATION
# ============================================================

all_faithfulness = []
all_answer_relevancy = []
all_context_relevancy = []


print("\n")
print("=" * 75)
print("RAGAS-STYLE AUTOMATED EVALUATION")
print("=" * 75)


for number, item in enumerate(
    evaluation_dataset,
    start=1
):

    question = item["question"]

    expected_chunks = item[
        "relevant_chunks"
    ]


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(f"QUESTION {number}")
    print("=" * 75)

    print("\nQuestion:")
    print(question)


    retrieved_chunks, context, results = (
        retrieve_context(question)
    )


    print("\nExpected relevant chunks:")
    print(expected_chunks)


    print("\nRetrieved chunks:")

    for rank, match in enumerate(
        results.matches,
        start=1
    ):

        chunk_id = match.metadata.get(
            "chunk_id"
        )

        print(
            f"{rank}. chunk-{chunk_id} "
            f"(score={match.score:.4f})"
        )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    print("\nGenerating answer...")

    answer, rag_response = (
        generate_rag_answer(
            question,
            context
        )
    )


    print("\nGenerated answer:")
    print(answer)


    # --------------------------------------------------------
    # LLM JUDGE
    # --------------------------------------------------------

    print("\nEvaluating answer...")

    evaluation, judge_response = (
        evaluate_with_llm(
            question,
            context,
            answer
        )
    )


    if evaluation is None:
        continue


    # --------------------------------------------------------
    # RAW SCORES
    # --------------------------------------------------------

    faithfulness_score = evaluation[
        "faithfulness"
    ]["score"]

    answer_relevancy_score = evaluation[
        "answer_relevancy"
    ]["score"]

    context_relevancy_score = evaluation[
        "context_relevancy"
    ]["score"]


    # --------------------------------------------------------
    # NORMALIZED SCORES
    # --------------------------------------------------------

    faithfulness_normalized = (
        normalize_score(
            faithfulness_score
        )
    )

    answer_relevancy_normalized = (
        normalize_score(
            answer_relevancy_score
        )
    )

    context_relevancy_normalized = (
        normalize_score(
            context_relevancy_score
        )
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_faithfulness.append(
        faithfulness_normalized
    )

    all_answer_relevancy.append(
        answer_relevancy_normalized
    )

    all_context_relevancy.append(
        context_relevancy_normalized
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("-" * 75)

    print("FAITHFULNESS")
    print(
        f"Score: "
        f"{faithfulness_score}/2 "
        f"({faithfulness_normalized:.2f})"
    )

    print(
        f"Reason: "
        f"{evaluation['faithfulness']['reason']}"
    )


    print("\nANSWER RELEVANCY")

    print(
        f"Score: "
        f"{answer_relevancy_score}/2 "
        f"({answer_relevancy_normalized:.2f})"
    )

    print(
        f"Reason: "
        f"{evaluation['answer_relevancy']['reason']}"
    )


    print("\nCONTEXT RELEVANCY")

    print(
        f"Score: "
        f"{context_relevancy_score}/2 "
        f"({context_relevancy_normalized:.2f})"
    )

    print(
        f"Reason: "
        f"{evaluation['context_relevancy']['reason']}"
    )


    print("-" * 75)


# ============================================================
# 10. OVERALL SCORES
# ============================================================

if all_faithfulness:

    mean_faithfulness = (
        sum(all_faithfulness)
        / len(all_faithfulness)
    )

    mean_answer_relevancy = (
        sum(all_answer_relevancy)
        / len(all_answer_relevancy)
    )

    mean_context_relevancy = (
        sum(all_context_relevancy)
        / len(all_context_relevancy)
    )


    overall_score = (
        mean_faithfulness
        + mean_answer_relevancy
        + mean_context_relevancy
    ) / 3


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 75)
    print("FINAL RAG EVALUATION REPORT")
    print("=" * 75)

    print(
        f"\nMean Faithfulness: "
        f"{mean_faithfulness:.3f}"
    )

    print(
        f"Mean Answer Relevancy: "
        f"{mean_answer_relevancy:.3f}"
    )

    print(
        f"Mean Context Relevancy: "
        f"{mean_context_relevancy:.3f}"
    )

    print(
        f"\nOverall RAG Score: "
        f"{overall_score:.3f}"
    )

    print("\n")
    print("=" * 75)
    print("Evaluation completed successfully.")
    print("=" * 75)

else:

    print(
        "\nNo valid evaluation results were produced."
    )