"""Question answering entry point: offline extractive by default, optional OpenAI."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from .retrieval import BM25, Hybrid, Chunk, load_chunks, tokenize

ROOT = Path(__file__).resolve().parent.parent


def answer_question(question: str, index, use_llm: bool | None = None, evidence_threshold: float = 0.025,
                    theme_threshold: float = 0.45) -> str:
    results = index.search(question, 3)
    # A zero-overlap query has no evidentiary basis; fail closed.
    if not results:
        return "I don't know based on the available documents."
    if results[0][1] < evidence_threshold:
        return "I don't know based on the available documents."
    if use_llm is None:
        use_llm = bool(os.environ.get("OPENAI_API_KEY"))
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        return _llm_answer(question, results)
    theme_check = (lambda q, s: index.similarities(q, [s])[0]) if isinstance(index, Hybrid) else None
    return _extractive_answer(question, results, theme_threshold=theme_threshold, theme_check=theme_check)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _extractive_answer(question: str, results: list[tuple[Chunk, float]], theme_threshold: float = 0.22,
                       theme_check=None) -> str:
    stop_words = {"a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for", "from", "how", "i", "in", "is", "it", "long", "may", "me", "much", "many", "of", "on", "the", "their", "to", "was", "what", "when", "where", "which", "who", "why", "with"}
    qterms = set(tokenize(question)) - stop_words
    candidates: list[tuple[int, float, int, str, Chunk]] = []
    best_evidence = 0
    for ci, (chunk, score) in enumerate(results):
        for si, sentence in enumerate(_sentences(chunk.text)):
            sentence_terms = set(tokenize(sentence))
            overlap = len(qterms & sentence_terms)
            best_evidence = max(best_evidence, overlap)
            # Numeric questions usually ask for the fact-bearing sentence. Prefer
            # it when the query asks for a quantity and several sentences overlap.
            quantity_query = bool(re.search(r"\b(how (?:much|many|long)|how soon|what (?:is|does|rate|cost))\b", question.lower()))
            quantity_sentence = bool(re.search(r"\d|\$", sentence))
            if quantity_query and quantity_sentence:
                overlap += 2
            if overlap:
                candidates.append((overlap, score, ci * 100 + si, sentence, chunk))
    if not candidates:
        return "I don't know based on the available documents."
    # A lone shared word (for example, "tent" in a question about the CEO's
    # favorite tent) is not enough evidence to claim the retrieved fact answers it.
    # Dense evidence checks that the proposed sentence discusses the question's
    # topic; keyword overlap alone can otherwise select an unrelated policy.
    # Embedder is carried on the index at the call site via this helper below.
    if best_evidence < 2:
        return "I don't know based on the available documents."
    chosen = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    _, _, _, sentence, chunk = chosen
    # Hybrid passes a callable on the index; compare query and answer sentence.
    if theme_check is not None and theme_check(question, sentence) < theme_threshold:
        return "I don't know based on the available documents."
    return f"{sentence}\nSource: {chunk.citation}"


def _llm_answer(question: str, results: list[tuple[Chunk, float]]) -> str:
    # Import and network access are isolated to the explicitly configured path.
    from openai import OpenAI
    context = "\n\n".join(f"[{c.citation}]\n{c.text}" for c, _ in results)
    response = OpenAI().responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        instructions=("Answer only from the supplied excerpts. If they do not contain "
                     "the answer, say you do not know. Cite the supporting filename and heading."),
        input=f"Question: {question}\n\nRetrieved excerpts:\n{context}",
    )
    return response.output_text.strip()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m sample.ask "your question"')
        raise SystemExit(2)
    mode = "hybrid"
    args = sys.argv[1:]
    if "--mode" in args:
        pos = args.index("--mode")
        if pos + 1 >= len(args) or args[pos + 1] not in {"bm25", "hybrid"}:
            raise SystemExit("--mode must be bm25 or hybrid")
        mode = args[pos + 1]
        del args[pos:pos + 2]
    if not args:
        print('Usage: python -m sample.ask "your question" [--mode bm25|hybrid]')
        raise SystemExit(2)
    chunks = load_chunks(ROOT / "docs")
    index = BM25(chunks) if mode == "bm25" else Hybrid(chunks)
    print(answer_question(" ".join(args), index))


if __name__ == "__main__":
    main()
