# Deploy image — Hugging Face Spaces (Docker SDK) / container host lain.
# HF Spaces menjalankan container sebagai user uid 1000 dan mengekspos app_port
# (kita pakai 7860, konvensi HF). Lihat docs/DEPLOY_HF_SPACES.md.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# User non-root uid 1000 (wajib untuk HF Spaces agar direktori writable).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

# Layer dependensi dulu (cache build tetap valid selama lock tidak berubah).
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Seluruh project (vector store ikut — dikendalikan .dockerignore).
COPY --chown=user . .
RUN uv sync --frozen --no-dev

# Pre-download model reranker (±1 GB) ke dalam image supaya startup di server
# tidak menunggu unduhan model.
RUN uv run python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base', max_length=256)"

EXPOSE 7860
CMD ["uv", "run", "python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "7860"]
