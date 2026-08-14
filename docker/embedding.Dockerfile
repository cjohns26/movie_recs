# GPU embedding microservice (sentence-transformers now, open_clip in Session 5).
#
# Base tag is deliberate: the RTX 5070 is Blackwell (compute capability 12.0 /
# sm_120), which needs CUDA 12.8+ — the locally cached 12.6 base is a toolkit
# smoke-test image only (plan.md Risk #1). 13.0.1 matches the cu13 wheels the
# lockfile resolves for torch, and the `-base` flavour is enough because those
# wheels ship their own CUDA runtime libraries.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    # Model weights land on a mounted volume, not in the image layer.
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source-only change doesn't re-download ~5 GB of
# CUDA wheels. `--no-default-groups --group serving --group ml` is the whole
# point of the dependency-group split: the API image gets `serving` only.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-default-groups --group serving --group ml

COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-default-groups --group serving --group ml

ENV PATH="/opt/venv/bin:${PATH}"
EXPOSE 8080

CMD ["uvicorn", "movie_recs.embeddings.service:app", "--host", "0.0.0.0", "--port", "8080"]
