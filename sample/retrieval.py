"""Small, dependency-free BM25 search with source-preserving Markdown chunks."""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.I)


def tokenize(text: str) -> list[str]:
    tokens = [t.lower().replace("’", "'") for t in TOKEN_RE.findall(text)]
    # A tiny normalization helps common policy paraphrases (meal/meals,
    # reimburse/reimbursed) without adding an NLP dependency.
    normalized = []
    for token in tokens:
        if len(token) > 5 and token.endswith("ment"):
            token = token[:-4]
        elif len(token) > 5 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
            if token.endswith("s"):
                token += "e"
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        normalized.append(token)
    return normalized


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    heading: str

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.heading}"


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Split Markdown into paragraph chunks, retaining filename and heading."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        heading = "Document"
        paragraphs: list[str] = []
        def flush() -> None:
            body = " ".join(p.strip() for p in paragraphs if p.strip())
            if body:
                chunks.append(Chunk(body, path.name, heading))
            paragraphs.clear()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                flush()
                heading = line.lstrip("# ").strip()
            elif not line.strip():
                flush()
            else:
                paragraphs.append(line.strip())
        flush()
    return chunks


class BM25:
    """Okapi BM25 using k1=1.5 and b=0.75; scores are deterministic."""
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.terms = [tokenize(c.text) for c in chunks]
        self.lengths = [len(ts) for ts in self.terms]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.df: dict[str, int] = {}
        for terms in self.terms:
            for term in set(terms):
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query: str, index: int) -> float:
        if not self.chunks or not self.avgdl:
            return 0.0
        score = 0.0
        for term in set(tokenize(query)):
            tf = self.terms[index].count(term)
            if not tf:
                continue
            # Robertson-Sparck Jones IDF with positive-score adjustment.
            idf = math.log(1 + (len(self.chunks) - self.df.get(term, 0) + 0.5) /
                           (self.df.get(term, 0) + 0.5))
            norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avgdl)
            score += idf * tf * (self.k1 + 1) / norm
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        ranked = [(chunk, self.score(query, i), i) for i, chunk in enumerate(self.chunks)]
        ranked.sort(key=lambda row: (-row[1], row[2]))
        return [(chunk, score) for chunk, score, _ in ranked[:top_k] if score > 0]


class MiniLMEmbedder:
    """Lazy local all-MiniLM-L6-v2 encoder; downloads from HF on first use."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        cache = Path(__file__).resolve().parent.parent / ".model-cache"
        os.environ.setdefault("HF_HOME", str(cache))
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class Hybrid:
    """BM25 + dense cosine rankings fused with reciprocal rank fusion."""
    def __init__(self, chunks: list[Chunk], embedder=None, rrf_k: int = 60):
        self.chunks, self.bm25 = chunks, BM25(chunks)
        self.embedder = embedder or MiniLMEmbedder()
        self.rrf_k = rrf_k
        self.vectors = self.embedder.encode([c.text for c in chunks]) if chunks else []

    def similarities(self, query: str, texts: list[str]) -> list[float]:
        import numpy as np
        vectors = self.embedder.encode([query, *texts])
        q = vectors[0]
        return [float(np.dot(q, v)) for v in vectors[1:]]

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        dense_scores = self.similarities(query, [c.text for c in self.chunks])
        dense_rank = sorted(range(len(self.chunks)), key=lambda i: (-dense_scores[i], i))
        sparse_rank = [self.chunks.index(c) for c, _ in self.bm25.search(query, len(self.chunks))]
        scores = defaultdict(float)
        for ranking in (sparse_rank, dense_rank):
            for rank, idx in enumerate(ranking, 1):
                scores[idx] += 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [(self.chunks[i], scores[i]) for i in ranked]
