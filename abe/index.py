"""A small BM25 keyword index stored as JSON.

BM25 needs no embedding model, no API key and no vector database, so a fresh
clone works offline and CI results are deterministic. The Retriever interface is
deliberately narrow (`search(query, k)`), so an embedding-based store can replace
it without touching the agent.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .ingest import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its me my of on or "
    "our should so that the their them then there this to was we what when where which who why "
    "will with you your".split()
)
TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1))
            for term in terms:
                f = tf.get(term)
                if f:
                    score += self._idf[term] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return [(self.chunks[i], round(s, 4)) for i, s in scores[:k]]

    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})

    def save(self, path: str | Path) -> None:
        payload = {"format": "abe-bm25-v1", "chunks": [c.to_dict() for c in self.chunks]}
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in payload["chunks"]])
