"""Read-only BM25 demonstration: no API calls, model downloads, or reports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Set before importing project code, including when run without Python's -B flag.
sys.dont_write_bytecode = True
os.environ.pop("OPENAI_API_KEY", None)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample.ask import answer_question
from sample.evaluate import run_set
from sample.retrieval import BM25, load_chunks


def print_evaluation(filename: str, label: str) -> None:
    """Print the existing evaluator's metrics and every answer/retrieval miss."""
    metrics, questions, answers = run_set(ROOT / "eval" / filename, "bm25")
    print(f"\n=== {label}: eval/{filename} (BM25) ===")
    print(f"Questions: {metrics['questions']} "
          f"({metrics['answerable']} answerable, {metrics['refusal_questions']} refusals)")
    for key, title in (
        ("recall_at_3", "Recall@3"),
        ("key_fact_rate", "Key-fact rate"),
        ("refusal_accuracy", "Refusal accuracy"),
        ("wrong_answer_rate", "Wrong-answer rate"),
    ):
        print(f"{title}: {metrics[key]:.1%}")
    misses = 0
    for question in questions:
        answer = answers[question["id"]]
        correct = (
            "don't know" in answer.lower() if question["refusal"]
            else all(keyword.lower() in answer.lower() for keyword in question["keywords"])
        )
        retrieval_miss = not question["refusal"] and not question["retrieval_hit"]
        if correct and not retrieval_miss:
            continue
        misses += 1
        print(f"\nMISS {question['id']}: {question['question']}")
        print(f"Actual: {answer}")
        expected = question["answer"] if not question["refusal"] else "I don't know based on the available documents."
        print(f"Expected: {expected}")
        if not question["refusal"]:
            print(f"Required keywords: {', '.join(question['keywords'])}")
            print(f"Expected source: {question['source']} — {question['heading']}")
            print(f"Retrieval@3: {'HIT' if question['retrieval_hit'] else 'MISS'}")
    print(f"\nMissed questions: {misses}/{metrics['questions']}")


def main() -> None:
    """Run an answer, an abstention, then both evaluation sets in order."""
    print("Offline BM25 only | API key ignored | No model download | No report files")
    index = BM25(load_chunks(ROOT / "docs"))
    for header, question in (
        ("A. Answer with source", "How long is the warranty on tents?"),
        ("B. Refusal", "What color is the CEO's favorite tent?"),
    ):
        print(f"\n=== {header} ===")
        print(f"Question: {question}")
        print(answer_question(question, index, use_llm=False))
    print_evaluation("questions.jsonl", "C1. Development evaluation")
    print_evaluation("heldout.jsonl", "C2. Held-out evaluation")


if __name__ == "__main__":
    main()
