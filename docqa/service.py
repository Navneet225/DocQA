from pathlib import Path

from docqa.bm25 import BM25, tokenize
from docqa.chunking import Chunk, chunk_text
from docqa.llm import Answerer, default_answerer

NO_ANSWER = "I don't know based on the provided documents."


class DocQA:
    def __init__(self, answerer: Answerer | None = None, min_coverage: float = 0.4):
        self.answerer = answerer or default_answerer()
        self.min_coverage = min_coverage
        self.chunks: list[Chunk] = []
        self._index: BM25 | None = None

    def add_document(self, text: str, source: str) -> int:
        new = chunk_text(text, source, start_id=len(self.chunks))
        self.chunks.extend(new)
        self._index = BM25([c.text for c in self.chunks])
        return len(new)

    def load_dir(self, folder: str) -> int:
        total = 0
        for p in sorted(Path(folder).glob("*.md")):
            total += self.add_document(p.read_text(encoding="utf-8"), p.name)
        return total

    def retrieve(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if not self._index:
            return []
        return [(self.chunks[i], s) for i, s in self._index.top_k(question, k)]

    def _coverage(self, question: str, chunk: Chunk) -> float:
        """Share of the question's terms found in the best chunk. Unlike a raw BM25 score,
        this does not depend on corpus size, so it works as a refusal threshold."""
        q = set(tokenize(question))
        if not q:
            return 0.0
        return len(q & set(self._index.tokens[chunk.id])) / len(q)

    def ask(self, question: str, k: int = 3) -> dict:
        hits = self.retrieve(question, k)
        if not hits or self._coverage(question, hits[0][0]) < self.min_coverage:
            return {"answer": NO_ANSWER, "sources": []}
        answer = self.answerer.answer(question, [c for c, _ in hits])
        return {
            "answer": answer,
            "sources": [{"id": c.id, "source": c.source, "score": round(s, 3), "text": c.text} for c, s in hits],
        }
