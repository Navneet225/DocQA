"""Split documents into overlapping word chunks so retrieval returns focused passages."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    text: str


def chunk_text(text: str, source: str, start_id: int = 0, size: int = 80, overlap: int = 20) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    chunks: list[Chunk] = []
    step = size - overlap
    for start in range(0, max(len(words), 1), step):
        piece = words[start : start + size]
        if not piece:
            break
        chunks.append(Chunk(start_id + len(chunks), source, " ".join(piece)))
        if start + size >= len(words):
            break
    return chunks
