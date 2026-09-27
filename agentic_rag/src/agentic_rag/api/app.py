from fastapi import FastAPI

from agentic_rag.api.routes import router
from agentic_rag.observability.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="agentic-rag")
    app.include_router(router)
    return app


app = create_app()
