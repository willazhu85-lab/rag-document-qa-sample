Sample project for demonstration only.

# Acme Outdoor Gear Document Q&A

This sample shows how a small company knowledge assistant can retrieve answers from internal Markdown documents and measure whether it finds the expected evidence. The included Acme Outdoor Gear policies and employee guidance are fictional.

## Architecture

```text
Markdown docs -> heading-aware paragraph chunks -> BM25 + MiniLM dense search -> RRF top 3
                                                                            -> extractive answer + citation
                                                                            -> abstain on weak / off-topic evidence
JSONL questions -> retrieval and answer checks -> reports/eval.md
```

## Run it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m sample.ask "How long is the warranty on tents?" --mode hybrid
.venv/bin/python -m sample.ask "How long is the warranty on tents?" --mode bm25
.venv/bin/python -m sample.evaluate
```

The CLI defaults to `--mode hybrid`, combining local BM25 with the `sentence-transformers/all-MiniLM-L6-v2` model and reciprocal rank fusion. The first hybrid run downloads model files from Hugging Face; later runs use the local cache. It needs no API key, though the first run needs network access for the model. Use `--mode bm25` to run lexical retrieval alone without loading the model. Answers quote a source sentence and cite its filename and heading; weak or off-topic evidence is rejected. To enable optional generated answers, set `OPENAI_API_KEY`; the key is never stored in the repository. OpenAI mode requires network access and may incur API charges. Evaluation uses offline extractive answers in both retrieval modes.

## Test

```sh
.venv/bin/pytest -q
```

Tests cover source-preserving chunks, BM25 against a hand-calculated score, offline fake-embedder hybrid search, abstention, and evaluation metrics. Evaluation compares BM25 and hybrid on dev (30 questions) and held-out (15 questions). The dev set alone is used to choose thresholds; held-out questions are reported as-is, including every incorrect response. Running `python -m sample.evaluate` downloads/loads MiniLM if needed and writes the comparison table to `reports/eval.md`.

## Why hybrid retrieval and abstention

Keyword search can fail when customers use different wording, such as “guarantee” instead of “warranty” or “send back” instead of “return.” Dense MiniLM retrieval adds semantic matches, and RRF combines those results with BM25. Better retrieval still cannot prove that a passage answers the question, so the answer path checks evidence strength and topic similarity and says “I don't know” when they fall below dev-selected thresholds. Wrong-answer rate counts answered-but-wrong questions, including wrongly answered refusal questions, because a confident wrong policy is worse than an abstention.

## How to adapt to your documents

1. Replace or add Markdown files under `docs/`. Use clear headings and short paragraphs; each answer cites its filename and the most recent heading. Keep customer or employee data out of this public sample.
2. Write representative questions in JSON Lines format. Each line should include a unique `id`, `question`, expected `answer`, `keywords` that must appear in the offline answer, expected `source` and `heading`, and a `refusal` boolean. For questions the documents cannot answer, set `refusal` to `true`, use an empty keyword list, and set expected answer/source fields to `null`.
3. Use `eval/questions.jsonl` as the dev set while adjusting retrieval or answer rules. Create a separate `eval/heldout.jsonl` with paraphrases and realistic user wording, including unanswerable questions. Do not use held-out results to tune rules; use them to report performance on new questions. A score below 100% is a useful finding, not a reason to hide the result.
4. Run `.venv/bin/python -m sample.evaluate` and inspect every held-out miss in `reports/eval.md`. Correct expected evidence only when the question annotation was wrong, and record why.

## Limitations

This is a compact demonstration, not a production service. Hybrid mode needs a one-time model download and local model dependencies; BM25 remains available without that model. Chunks are paragraphs rather than token-limited windows, and offline answers copy a sentence rather than synthesize one. Similarity thresholds cannot guarantee perfect topic matching. The sample corpus and evaluation sets are small, synthetic, and not representative of real customer data. Optional generation is not run by evaluation.

## License

MIT. See [LICENSE](LICENSE).
