import math

from sample.ask import answer_question
from sample.evaluate import calculate_metrics, evaluate
from sample.retrieval import BM25, Hybrid, Chunk, load_chunks


def test_markdown_chunks_keep_filename_and_heading(tmp_path):
    (tmp_path / "policy.md").write_text("# Returns\n\n## Window\nUnused items return in 30 days.\n", encoding="utf-8")
    chunks = load_chunks(tmp_path)
    assert len(chunks) == 1
    assert chunks[0].source == "policy.md"
    assert chunks[0].heading == "Window"
    assert chunks[0].text == "Unused items return in 30 days."


def test_bm25_matches_hand_calculated_single_term_score():
    chunks = [Chunk("alpha alpha beta", "a.md", "A"), Chunk("alpha gamma", "b.md", "B")]
    bm25 = BM25(chunks)
    # N=2, df(alpha)=2, tf=2, dl=3, avgdl=2.5, k1=1.5, b=.75.
    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    norm = 2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2.5)
    expected = idf * 2 * (1.5 + 1) / norm
    assert math.isclose(bm25.score("alpha", 0), expected, rel_tol=1e-12)
    assert bm25.search("gamma", 1)[0][0].source == "b.md"


def test_unknown_question_abstains():
    index = BM25([Chunk("Tents have a two year warranty.", "warranty.md", "Coverage")])
    assert "don't know" in answer_question("What is the CEO's favorite color?", index, use_llm=False).lower()


def test_evaluation_metrics_regression_on_dev_set():
    metrics = evaluate()
    assert metrics["questions"] == 30
    assert metrics["answerable"] == 25
    assert metrics["refusal_questions"] == 5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["key_fact_rate"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0


def test_metrics_hand_calculated_on_three_synthetic_questions():
    questions = [
        {"refusal": False, "retrieval_hit": True, "keywords": ["$8", "shipping"]},
        {"refusal": False, "retrieval_hit": False, "keywords": ["30 days"]},
        {"refusal": True, "retrieval_hit": False, "keywords": []},
    ]
    answers = iter(["$8 shipping", "I found 30 days", "I don't know"])
    metrics = calculate_metrics(questions, lambda _q: next(answers))
    assert metrics == {"questions": 3, "answerable": 2, "refusal_questions": 1,
                       "recall_at_3": 0.5, "key_fact_rate": 1.0, "refusal_accuracy": 1.0,
                       "wrong_answer_rate": 0.0}


def test_markdown_chunks_support_nested_headings_blank_lines_and_code(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Main\n\nIntro paragraph.\n\n## Details\n\n### Example\n\n```python\nprint('hello')\n```\n\nAfter code.\n",
        encoding="utf-8")
    chunks = load_chunks(tmp_path)
    assert [(c.heading, c.text) for c in chunks] == [
        ("Main", "Intro paragraph."),
        ("Example", "```python print('hello') ```"),
        ("Example", "After code."),
    ]


def test_refuses_when_only_one_content_word_overlaps():
    index = BM25([Chunk("Tents have a two year warranty.", "warranty.md", "Coverage")])
    answer = answer_question("Who owns this tent?", index, use_llm=False)
    assert "don't know" in answer.lower()


def test_llm_mode_without_key_does_not_call_network(monkeypatch):
    import sample.ask as ask
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ask, "_llm_answer", lambda *_: (_ for _ in ()).throw(AssertionError("network path called")))
    index = BM25([Chunk("Tents have a 2-year warranty.", "warranty.md", "Coverage")])
    assert "2-year warranty" in answer_question("What warranty applies to tents?", index, use_llm=True)


class FakeEmbedder:
    def encode(self, texts):
        import numpy as np
        # Deterministic lexical vectors; test never downloads or loads a model.
        vocab = sorted({w.lower() for t in texts for w in t.split()})
        return np.array([[t.lower().split().count(w) for w in vocab] for t in texts], dtype=float)


def test_hybrid_rrf_combines_sparse_and_dense_rankings_offline():
    chunks = [Chunk("tent warranty policy", "a.md", "A"), Chunk("return unused order", "b.md", "B")]
    index = Hybrid(chunks, embedder=FakeEmbedder())
    assert index.search("tent warranty", 1)[0][0].source == "a.md"


def test_hybrid_similarity_returns_cosine_scores_offline():
    index = Hybrid([Chunk("red tent", "a.md", "A")], embedder=FakeEmbedder())
    assert index.similarities("red tent", ["red tent"])[0] > 0.99


def test_hybrid_search_keeps_source_chunk():
    chunk = Chunk("unused goods returns within thirty days", "returns.md", "Window")
    assert Hybrid([chunk], embedder=FakeEmbedder()).search("unused returns", 1)[0][0] == chunk


def test_wrong_answer_rate_counts_answered_incorrect_answers():
    questions = [{"refusal": False, "retrieval_hit": True, "keywords": ["30 days"]},
                 {"refusal": True, "retrieval_hit": False, "keywords": []}]
    answers = iter(["Wrong vacation policy", "I don't know"])
    assert calculate_metrics(questions, lambda _q: next(answers))["wrong_answer_rate"] == 0.5


def test_bm25_mode_retains_sparse_search_behavior():
    assert BM25([Chunk("tent warranty", "warranty.md", "Coverage")]).search("tent", 1)[0][0].source == "warranty.md"
