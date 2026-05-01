"""
NovaBite RAG Retrieval Accuracy Evaluation

Runs 5 test queries directly against the RAG agent and checks for expected
keywords in the response. Demonstrates that retrieval is grounded and accurate.

Requirements:
    - FAISS index must exist (run `uvicorn api.main:app` once to auto-build it,
      or run `python -m rag.ingest` directly)
    - GROQ_API_KEY must be set in .env

Usage:
    python eval/rag_eval.py
"""
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.rag_agent import answer  # noqa: E402

TEST_CASES = [
    {
        "query": "Do you have vegan pasta?",
        "expected_keyword": "Pasta Primavera",
        "description": "Vegan pasta retrieval",
    },
    {
        "query": "Is the chicken grilled or fried?",
        "expected_keyword": "grilled",
        "description": "Cooking method retrieval",
    },
    {
        "query": "What are your opening hours on Saturday?",
        "expected_keyword": "Saturday",
        "description": "Weekend hours retrieval",
    },
    {
        "query": "Do you host birthday events?",
        "expected_keyword": "birthday",
        "description": "Private events retrieval",
    },
    {
        "query": "Do you serve sushi?",
        "expected_keyword": "don't have that information",
        "description": "Hallucination guard — out-of-scope query",
    },
]


def run_eval():
    print("=" * 60)
    print("NovaBite RAG Retrieval Accuracy Evaluation")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc['description']}")
        print(f"  Query   : {tc['query']}")

        try:
            response = answer(tc["query"])
            ok = tc["expected_keyword"].lower() in response.lower()
        except Exception as exc:
            response = f"ERROR: {exc}"
            ok = False

        status = "PASS (OK)" if ok else "FAIL (X)"
        print(f"  Status  : {status}")
        print(f"  Response: {response[:150]}{'...' if len(response) > 150 else ''}")

        if not ok:
            print(f"  Missing : expected keyword '{tc['expected_keyword']}' not found")
            failed += 1
        else:
            passed += 1

    print("\n" + "=" * 60)
    print(f"Result: {passed}/{len(TEST_CASES)} passed, {failed}/{len(TEST_CASES)} failed")

    if failed == 0:
        print("All tests passed — retrieval is grounded and accurate.")
    else:
        print("Some tests failed — check FAISS index and prompt configuration.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_eval()
    sys.exit(0 if success else 1)
