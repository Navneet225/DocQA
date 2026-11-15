import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from docqa.service import DocQA


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


def create_app(qa: DocQA | None = None) -> FastAPI:
    qa = qa or DocQA()
    docs_dir = os.environ.get("DOCS_DIR")
    if docs_dir:
        qa.load_dir(docs_dir)
    app = FastAPI(title="DocQA - RAG over your documents")

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": len(qa.chunks)}

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        return {"source": req.source, "chunks_added": qa.add_document(req.text, req.source)}

    @app.post("/ask")
    def ask(req: AskRequest):
        return qa.ask(req.question, req.k)

    return app


app = create_app()
