# syntax=docker/dockerfile:1
#
# The deployable unit: the HTTP service from Phase 3C, with every artifact it
# serves built inside the image.
#
# WHY THE BASE IS PINNED BY DIGEST AND BY PATCH VERSION
# ----------------------------------------------------
# `python:3.14-slim` is a floating tag -- it currently resolves to 3.14.7, and
# it will resolve to something else next month. This project's recurring bug
# class is a value that is wrong for its context, internally consistent, and
# therefore silent; a base image that changes underneath a published result is
# exactly that shape. So: 3.14.3, which is the interpreter the published
# results were produced under (see venv/), pinned by index digest.
#
# WHY THE ARTIFACTS ARE BUILT HERE RATHER THAN COPIED
# ---------------------------------------------------
# .dockerignore excludes the committed index, metadata and joblib artifacts, so
# this image CANNOT be serving a copy of the developer's files. It must build
# its own, in the documented order, from the dataset generator's seed-42 output.
# The verification that matters -- adv_08 reproducing the published routing
# decision over HTTP -- is then a real test of a from-scratch build rather than
# a test that a file copied correctly.
#
# train_distilbert.py is deliberately NOT run: ~90 minutes on CPU (measured in
# Phase 8A.1) and it sits on no production path.
#
# Build:
#     docker build -t ticket-triage:8b .
# Run (no key -- classification, retrieval and the escalation gate all work):
#     docker run --rm -p 8000:8000 ticket-triage:8b
# Run with resolution drafting enabled:
#     docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... ticket-triage:8b

FROM python:3.14.3-slim@sha256:5e59aae31ff0e87511226be8e2b94d78c58f05216efda3b07dbbed938ec8583b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

# libgomp1: OpenMP runtime, needed by faiss-cpu and by scikit-learn's threaded
# paths. Nothing else is added -- a smaller surface is a smaller thing to keep
# reproducible.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source change does not re-download torch.
#
# torch comes from the CPU index at the SAME pinned version. The default PyPI
# wheel drags in the whole CUDA stack (several GB) for a service that has never
# run on a GPU and whose published latency figures are CPU, batch size 1.
# 2.13.0+cpu satisfies the `torch==2.13.0` pin in requirements.txt, which is
# then installed unmodified.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install -r requirements.txt

# Source. .dockerignore keeps out venv/, models/, **/*.npy, .env, paper/ and
# the committed FAISS/metadata/joblib artifacts.
COPY . .

# THE GUARD THAT SHOULD HAVE EXISTED FIRST.
#
# The first build of this image shipped with the developer's local embedding
# cache baked in: .dockerignore matches with Go's filepath.Match, `*` does not
# cross `/`, so `*.npy` never matched data/ticket_embeddings_*.npy. The cache
# came in, train_embeddings.py printed `[cache HIT]`, and the image's Tier-2
# classifier and FAISS index were derived from a file encoded on another
# machine months earlier. Nothing failed. The only evidence was one line in a
# build log.
#
# So the ignore rule is no longer the only thing enforcing this. If any
# pre-built artifact reaches the context, the build stops here -- before any
# artifact is built on top of it.
RUN set -eu; \
    stale="$(ls -1 data/*.npy data/*.npz data/ticket_index*.faiss \
                   data/ticket_metadata*.json data/*.joblib 2>/dev/null || true)"; \
    if [ -n "$stale" ] || [ -d models ]; then \
        echo "FATAL: pre-built artifacts reached the build context:"; \
        echo "$stale"; [ -d models ] && echo "models/"; \
        echo ""; \
        echo "This image must build its own artifacts. Something that should"; \
        echo "be in .dockerignore is not -- remember that '*' does not cross"; \
        echo "'/' there, so a nested pattern needs '**/'."; \
        exit 1; \
    fi; \
    echo "OK: no pre-built artifact in the context; everything below is built here."

# ---------------------------------------------------------------------------
# Artifacts, in the documented order. One RUN each: a failure names itself
# instead of arriving as "step 12 failed".
# ---------------------------------------------------------------------------
# 1. The 4,000-ticket synthetic corpus, seed 42.
RUN python data/generate_dataset.py

# 2. Tier-1, fitted on the FULL 4,000 rows (not an 80/20 split) -- that is what
#    the goldens were captured under. Carries a manifest with the dataset
#    sha256, which load_tier1() verifies at every boot.
RUN python src/classification/train_tier1.py

# 3. Tier-2 (BGE). Downloads BAAI/bge-base-en-v1.5 into HF_HOME and writes the
#    shared embedding cache data/ticket_embeddings_bge-base-en-v1-5.npy.
RUN python src/classification/train_embeddings.py

# 4. The FAISS index. Reuses the cache from step 3 by row count, so the 4,000
#    rows are encoded exactly ONCE across the whole build.
RUN python src/rag/build_vector_index.py

# ---------------------------------------------------------------------------
# Fail the BUILD, not a request, on a stale or mismatched artifact.
#
# startup() runs load_artifacts(), which enforces all three hard guards:
# index.ntotal == len(metadata), encoder dim == index.d == configured dim, and
# Tier-1's manifest still matching the dataset it was fitted on. An image that
# would boot on stale artifacts and serve confident wrong answers never gets
# built.
# ---------------------------------------------------------------------------
RUN python -c "from src.service.api import startup; startup()" \
    && python -c "from src.agent.config import config_fingerprint; \
print('config_fingerprint:', config_fingerprint())"

# Non-root. The service writes nothing at runtime, so read-only ownership of
# /app is enough; HF_HOME is handed over because sentence-transformers touches
# its cache directory on load.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/hf
USER appuser

# The weights are baked in. Offline means a running container cannot reach out
# and quietly acquire a DIFFERENT model than the one this image was verified
# with -- the same reasoning as the artifact guards, applied to the network.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# curl is not in slim and is not worth adding; stdlib urllib is already here.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).status == 200 else 1)"]

# From the project root: src/service/ has no __init__.py, so src.service.api
# resolves as a namespace package the same way src.experiments.* does.
CMD ["uvicorn", "src.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
