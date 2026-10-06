FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependencies first so this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim
# Both volume mount points are created and chowned here: a named volume copies
# the image directory's ownership on first use, and is otherwise root-owned
# (unwritable by the non-root user).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/app/data/registry.db \
    HF_HOME=/home/app/.cache/huggingface
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

# Exactly one worker: the engine cache and the rate limiter's counters are
# process-local, so more workers would silently break both.
CMD ["uvicorn", "agentic_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
