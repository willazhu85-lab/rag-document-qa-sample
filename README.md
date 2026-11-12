# Acme Outdoor Gear Document Q&A

A sample project, not client work. All Acme policies, names, and evaluation questions are made up.

This small Python document Q&A example retrieves paragraphs from Markdown policies,
returns an extractive sentence with a filename and heading, and attempts to refuse
questions without enough evidence. It includes BM25 retrieval, optional hybrid retrieval,
and explicit development and held-out evaluation sets. Refusal is imperfect; see the misses below.

## Start with offline BM25

Requires Python 3.10 or newer. The BM25 demo uses only the Python standard library;
no dependency installation, API key, model download, or network connection is needed.
From the project folder, run:

```sh
python demo/loom_demo.py
```

Use `python3` instead if that is your Python command. The demo resolves files relative
to its own location, so it also works when invoked from another working directory.
It ignores any configured `OPENAI_API_KEY`, disables bytecode writes, prints the
warranty answer and source, refuses the CEO question, then evaluates both sets and
lists every MISS with actual versus expected answers. It writes no report files.

For an individual question, explicitly select BM25 and remove the API key from the
process environment to ensure extractive, offline answers (macOS/Linux):

```sh
env -u OPENAI_API_KEY python -B -m sample.ask "How long is the warranty on tents?" --mode bm25
```

Expected answer:

```text
Tents and backpacks have a 2-year limited warranty from the purchase date.
Source: warranty.md — Coverage periods
```

## How it works

```text
Markdown -> heading-aware paragraph chunks -> BM25 top 3
                                         -> sentence selection + filename/heading
                                         -> refusal when evidence is insufficient
JSONL questions -> retrieval, keyword, and refusal checks -> printed demo summary
```

`sample/ask.py` also supports hybrid BM25 + MiniLM dense retrieval, fused with reciprocal
rank fusion. Its CLI defaults to hybrid, so specify `--mode bm25` for lexical-only search.
Hybrid uses `sentence-transformers/all-MiniLM-L6-v2`; its first use can download a model
from Hugging Face. Optional generated answers use the OpenAI API when a key is set;
that path requires network access and can incur charges. Neither path runs in the demo.
The demo and evaluator explicitly pass `use_llm=False`.

## Verified BM25 evaluation

Reproduced with `python demo/loom_demo.py`. Results below are for the shipped public
corpus. Removing two fictional email addresses from policy text did not change these
aggregate scores relative to the source corpus. No thresholds or questions were tuned
against the held-out set during this preparation.

| Set | Questions | Recall@3 | Key-fact rate | Refusal accuracy | Wrong-answer rate |
|---|---|---|---|---|---|
| Dev: `eval/questions.jsonl` | 30 (25 answerable, 5 refusals) | 100.0% (25/25) | 100.0% (25/25) | 100.0% (5/5) | 0.0% (0/30) |
| Held-out: `eval/heldout.jsonl` | 15 (12 answerable, 3 refusals) | 58.3% (7/12) | 33.3% (4/12) | 33.3% (1/3) | 26.7% (4/15) |

Recall@3 requires the expected filename **and heading** among the first three results.
Key-fact rate requires every annotated keyword to occur in the answer, ignoring case;
it is a literal substring check, not a semantic correctness judgment. Refusal accuracy
checks for “don't know” on unanswerable questions. Wrong-answer rate counts non-refusal
answers that fail the keyword check, plus answered unanswerable questions, divided by
all questions. Abstaining on an answerable question is a miss but not a wrong answer.

There are **0/30 dev misses** and **10/15 held-out misses**: six abstentions on
answerable questions and four incorrect answers. Here is every held-out miss.
Actual citations appear in the final column; the demo also prints expected sources
and keywords for answerable questions.

| ID | Expected | Actual | Actual source |
|---|---|---|---|
| h01 | 2-year limited warranty | I don't know based on the available documents. | — |
| h02 | Within 30 days after delivery | Up to 5 unused vacation days may carry into the next year and must be used by March 31. | leave.md — Carryover and sick leave |
| h03 | Within 7 days | I don't know based on the available documents. | — |
| h05 | Orders of $75 or more before tax | I don't know based on the available documents. | — |
| h06 | 4–7 business days after dispatch | Alaska, Hawaii, and PO boxes are excluded from the 4–7 business day estimate; those destinations usually take 7–12 business days. | shipping.md — Expedited and remote destinations |
| h08 | $5 discount | I don't know based on the available documents. | — |
| h11 | Within 30 days with receipt | I don't know based on the available documents. | — |
| h12 | $0.67 per mile | I don't know based on the available documents. | — |
| h14 | Refuse: I don't know based on the available documents. | Acme posts active recalls on its website and contacts registered purchasers by email when contact details are available. | safety.md — Recall notices |
| h15 | Refuse: I don't know based on the available documents. | Customers may request a copy or correction of account information by contacting the Acme privacy team; Acme responds within 30 days after verifying the requester's identity. | privacy.md — Data retention and access |

These scores cover BM25 extractive answers only. OpenAI generation was not evaluated.
`reports/eval.md` is an earlier full run of `python -m sample.evaluate` that also
compares MiniLM hybrid retrieval; it was not re-run for this update. The development
score does not establish performance on new questions.

## Tests and optional dependencies

The included suite has **13 passing tests**, covering chunk citations, a hand-calculated
BM25 score, refusal, metric calculations, and hybrid search with a fake embedder.
Fake-embedder tests use NumPy but never download a model.

To install the pinned dependencies for tests and optional modes in your own environment
(this installation requires network access unless packages are already available):

```sh
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
```

The standalone demo above does not need these packages. `python -m sample.evaluate`
is a separate, optional full comparison command: it loads/downloads MiniLM and writes
`reports/eval.md`. Use the demo for a no-download, no-report BM25 evaluation.

## Adaptation and limitations

- Replace Markdown under `docs/` and keep clear headings and short paragraphs.
  This loader supports Markdown, not PDF ingestion or a hosted chat interface.
- Add representative JSONL questions with `id`, `question`, `answer`, `keywords`,
  `source`, `heading`, and `refusal`. Keep development and held-out questions separate;
  tune only on development data and report held-out failures.
- BM25 is lexical and struggles with paraphrases. Offline answers copy one sentence;
  they do not synthesize across passages. Paragraph chunks are not token limited.
- A citation proves which passage was used, not that it answers the question.
  Heuristic refusal can reject answerable questions and answer unsupported questions.
- The small synthetic corpus and literal keyword evaluation do not establish real-world
  accuracy, privacy controls, production readiness, or client delivery experience.

## Files

- `sample/`: retrieval, Q&A, and evaluation code.
- `demo/loom_demo.py`: one-command, read-only offline demonstration.
- `docs/`: 11 fictional Acme Markdown policy documents.
- `eval/`: 30 development and 15 held-out questions.
- `tests/`: offline unit and regression tests.
- `requirements.txt`, `pytest.ini`, `.gitignore`, `LICENSE`.

## License

MIT. See [LICENSE](LICENSE).
