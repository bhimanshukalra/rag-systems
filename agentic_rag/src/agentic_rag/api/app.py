from fastapi import FastAPI

from agentic_rag.api.routes import router
from agentic_rag.observability.logging import configure_logging
from agentic_rag.observability.tracing import configure_tracing


def create_app() -> FastAPI:
    configure_logging()
    configure_tracing()

    app = FastAPI(title="agentic-rag")
    app.include_router(router)
    return app


app = create_app()
