"""BM25 (Okapi) ranking implemented from scratch, no external search library."""
import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or",
    "how", "what", "when", "who", "which", "do", "does", "i", "my", "me", "we", "our", "it", "that",
    "this", "can", "if", "at", "by", "with", "from", "as", "per",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(t: str) -> str:
    """Very light plural stemming so 'passwords' matches 'password'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.lens = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        # +1 inside the log keeps IDF positive even for very common terms
        self.idf = {term: math.log((n - f + 0.5) / (f + 0.5) + 1) for term, f in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = []
        for tf, length in zip(self.tf, self.lens):
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                s += self.idf[term] * f * (self.k1 + 1) / denom
            scores.append(s)
        return scores

    def top_k(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        scored = sorted(enumerate(self.score(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]
