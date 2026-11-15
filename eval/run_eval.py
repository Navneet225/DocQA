"""Retrieval evaluation on a small hand-written question set (see eval/qa.json).
Reports top-1 / top-3 hit rate: is a chunk from the correct document in the top results?"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from docqa.service import DocQA  # noqa: E402

qa = DocQA()
qa.load_dir(str(ROOT / "data" / "handbook"))
cases = json.loads((ROOT / "eval" / "qa.json").read_text())

top1 = top3 = 0
misses = []
for c in cases:
    hits = qa.retrieve(c["q"], k=3)
    sources = [h[0].source for h in hits]
    if sources[:1] == [c["source"]]:
        top1 += 1
    if c["source"] in sources:
        top3 += 1
    else:
        misses.append((c["q"], c["source"], sources))

n = len(cases)
out_of_scope = [
    "What is the capital of Mongolia?",
    "Who won the cricket world cup?",
    "How do I bake sourdough bread?",
    "What is the company stock price today?",
    "Explain quantum computing basics.",
    "Which programming language is best for beginners?",
]
refused = sum(qa.ask(q)["answer"] == "I don't know based on the provided documents." for q in out_of_scope)
answerable_refused = sum(qa.ask(c["q"])["answer"] == "I don't know based on the provided documents." for c in cases)
print(f"chunks indexed : {len(qa.chunks)}")
print(f"questions      : {n}")
print(f"top-1 hit rate : {top1}/{n} = {top1 / n:.0%}")
print(f"top-3 hit rate : {top3}/{n} = {top3 / n:.0%}")
print(f"out-of-scope refused : {refused}/{len(out_of_scope)}")
print(f"answerable wrongly refused : {answerable_refused}/{n}")
for q, exp, got in misses:
    print(f"MISS: {q!r} expected {exp}, got {got}")
