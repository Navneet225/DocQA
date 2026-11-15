from docqa.chunking import Chunk

SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. "
    "Cite sources like [1] or [2] after each claim. "
    "If the sources do not contain the answer, reply exactly: I don't know based on the provided documents."
)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    sources = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, c in enumerate(chunks, start=1))
    user = f"Sources:\n{sources}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
