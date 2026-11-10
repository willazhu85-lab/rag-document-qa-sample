"""Offline evaluation of retrieval, extractive answers, and abstention."""
from __future__ import annotations

import json
from pathlib import Path

from .ask import answer_question
from .retrieval import BM25, Hybrid, load_chunks

ROOT = Path(__file__).resolve().parent.parent


def calculate_metrics(questions: list[dict], answer_for) -> dict:
    """Calculate metrics from supplied questions and answers (easy to test offline)."""
    answerable = [q for q in questions if not q["refusal"]]
    refusals = [q for q in questions if q["refusal"]]
    hits = correct_facts = correct_refusals = wrong_answers = 0
    for q in answerable:
        if q.get("retrieval_hit", False):
            hits += 1
        answer = answer_for(q).lower()
        fact_correct = all(k.lower() in answer for k in q["keywords"])
        if fact_correct:
            correct_facts += 1
        if answer and "don't know" not in answer and not fact_correct:
            wrong_answers += 1
    for q in refusals:
        answer = answer_for(q).lower()
        if "don't know" in answer:
            correct_refusals += 1
        elif answer.strip():
            wrong_answers += 1
    return {
        "questions": len(questions), "answerable": len(answerable), "refusal_questions": len(refusals),
        "recall_at_3": hits / len(answerable) if answerable else 0,
        "key_fact_rate": correct_facts / len(answerable) if answerable else 0,
        "refusal_accuracy": correct_refusals / len(refusals) if refusals else 0,
        "wrong_answer_rate": wrong_answers / len(questions) if questions else 0,
    }


def evaluate(questions_path: Path | None = None, docs_dir: Path | None = None, mode: str = "bm25", embedder=None) -> dict:
    questions_path = questions_path or ROOT / "eval" / "questions.jsonl"
    chunks = load_chunks(docs_dir or ROOT / "docs")
    index = BM25(chunks) if mode == "bm25" else Hybrid(chunks, embedder=embedder)
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line]
    for q in questions:
        if q["refusal"]:
            q["retrieval_hit"] = False
            continue
        results = index.search(q["question"], 3)
        q["retrieval_hit"] = any(c.source == q["source"] and c.heading == q["heading"] for c, _ in results)
    return calculate_metrics(questions, lambda q: answer_question(q["question"], index, use_llm=False))


def run_set(path: Path, mode: str, embedder=None):
    chunks = load_chunks(ROOT / "docs")
    index = BM25(chunks) if mode == "bm25" else Hybrid(chunks, embedder=embedder)
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for q in questions:
        hits = index.search(q["question"], 3)
        q["retrieval_hit"] = (not q["refusal"] and any(c.source == q["source"] and c.heading == q["heading"] for c, _ in hits))
    answers = {q["id"]: answer_question(q["question"], index, use_llm=False) for q in questions}
    return calculate_metrics(questions, lambda q: answers[q["id"]]), questions, answers


def main() -> None:
    results = {}
    detail = {}
    embedder = None
    # Instantiate once so both sets share one local model load.
    embedder = __import__("sample.retrieval", fromlist=["MiniLMEmbedder"]).MiniLMEmbedder()
    for mode in ("bm25", "hybrid"):
        for label, filename in (("dev", "questions.jsonl"), ("held-out", "heldout.jsonl")):
            path = ROOT / "eval" / filename
            metrics, questions, answers = run_set(path, mode, embedder=embedder)
            results[(mode, label)] = metrics
            detail[(mode, label)] = (questions, answers)
    report = "# Retrieval Evaluation\n\n| Mode | Set | Questions | Recall@3 | Key-fact rate | Refusal accuracy | Wrong-answer rate |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for mode in ("bm25", "hybrid"):
        for label in ("dev", "held-out"):
            m = results[(mode, label)]
            report += f"| {mode} | {label} ({m['answerable']} answerable, {m['refusal_questions']} refusals) | {m['questions']} | {m['recall_at_3']:.1%} | {m['key_fact_rate']:.1%} | {m['refusal_accuracy']:.1%} | {m['wrong_answer_rate']:.1%} |\n"
    report += "\nWrong-answer rate is answered-but-incorrect questions (including answered refusals) divided by all questions in that set. Dev alone may guide thresholds; held-out questions and scores are reported without tuning. Hybrid thresholds (RRF >= 0.025; query-to-answer cosine >= 0.45) were selected using dev only.\n\n## Held-out incorrect responses\n"
    for mode in ("bm25", "hybrid"):
        questions, answers = detail[(mode, "held-out")]
        errors = []
        for q in questions:
            ans = answers[q["id"]]
            correct = ("don't know" in ans.lower()) if q["refusal"] else all(k.lower() in ans.lower() for k in q["keywords"])
            if not correct:
                errors.append((q, ans))
        report += f"\n### {mode}\n"
        if not errors:
            report += "\nNone.\n"
        for q, ans in errors:
            report += f"\n#### {q['id']}: {q['question']}\n\n- System answer: {ans}\n- Expected answer: {q['answer']}\n- Reason: {'Answered an unanswerable question.' if q['refusal'] else 'Answer omitted expected facts: ' + ', '.join(q['keywords']) + '.'}\n"
    out = ROOT / "reports" / "eval.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
