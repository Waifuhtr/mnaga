# syntax=docker/dockerfile:1.7
###############################################################################
# Manga Image Translator + Hy-MT2-7B  ->  Hugging Face Space image
#
# This file is the ONLY build file the Space repo needs. Application source is
# pulled from GitHub at build time, so the Space repo stays tiny.
#
# LAYER ORDER IS LOAD-BEARING. Layers are ordered stable -> volatile so that an
# application code change never invalidates the 4.6 GB model layer:
#
#   1 OS packages            ~ rarely changes
#   2 llama.cpp CUDA binaries  (pinned release)
#   3 PyTorch cu128            (pinned version)
#   4 manga-image-translator + Python deps (pinned commit)
#   5 M.I.T. detection/OCR/inpainting weights   (build-time download)
#   6 manga-ocr HF weights                      (build-time download)
#   7 Hy-MT2-7B-Q4_K_M.gguf  4.6 GB             (build-time download)
#   8 application source from GitHub            <- only this rebuilds normally
#
# Build on CPU hardware (8 vCPU / 32 GB). Everything heavy is resolved here, so
# switching the Space to T4 Small afterwards starts the container from these
# same layers with no re-download.
###############################################################################

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# 1. OS packages.
#
#    The base has to be 24.04: the official llama.cpp CUDA release binaries are
#    linked against GLIBC 2.38 and 22.04 only provides 2.35. (Their CPU-only
#    tarball needs just 2.29, which is why this bites the CUDA build alone.)
#
#    24.04 ships Python 3.12, but manga-image-translator wants >=3.10,<3.12, so
#    Python 3.11 comes from deadsnakes and all Python work happens in a venv -
#    which also avoids 24.04's PEP 668 "externally managed environment" refusal.
#
#    libglib2.0-0 no longer exists on 24.04; it is libglib2.0-0t64 after the
#    64-bit time_t transition.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        git curl ca-certificates \
        libgomp1 \
        libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 \
        fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Putting the venv first on PATH makes `python` and `pip` resolve to 3.11 for
# every layer below, including the model prefetch steps and the entrypoint.
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/python --version
ENV PATH=/opt/venv/bin:$PATH

# ---------------------------------------------------------------------------
# 2. llama.cpp, official prebuilt CUDA 12.8 binaries.
#    Matches the base image CUDA version exactly, so no extra cudart tarball is
#    needed. Nothing is compiled here and no GPU is touched at build time.
# ---------------------------------------------------------------------------
ARG LLAMA_CPP_RELEASE=b11022
RUN curl -fsSL --retry 5 --retry-delay 3 \
        -o /tmp/llama.tar.gz \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_RELEASE}/llama-${LLAMA_CPP_RELEASE}-bin-ubuntu-cuda-12.8-x64.tar.gz" \
    && mkdir -p /opt/llama \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
    && rm -f /tmp/llama.tar.gz \
    && chmod +x /opt/llama/llama-server \
    && /opt/llama/llama-server --version

# ---------------------------------------------------------------------------
# 3. PyTorch (CUDA 12.8 wheels). Installed before the M.I.T. requirements so
#    that requirements.txt's unpinned `torch`/`torchvision` resolve as already
#    satisfied instead of pulling a second, mismatched build.
# ---------------------------------------------------------------------------
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
RUN python -m pip install --no-cache-dir --upgrade "pip<25" setuptools wheel \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# ---------------------------------------------------------------------------
# 4. manga-image-translator at a pinned commit + its Python dependencies.
#    Repo:   https://github.com/zyddnys/manga-image-translator
#    Commit: 95227a2b ("Merge pull request #1157 ... m2m100-hf", 2026-07-20)
#    g++/cython are needed only to build pydensecrf, then removed again.
# ---------------------------------------------------------------------------
ARG MIT_REPO=https://github.com/zyddnys/manga-image-translator
ARG MIT_COMMIT=95227a2bb0fd306cd4f0c104d57284026f991b3a
ENV MIT_ROOT=/opt/manga-image-translator
RUN git clone --filter=blob:none "${MIT_REPO}" "${MIT_ROOT}" \
    && git -C "${MIT_ROOT}" checkout --detach "${MIT_COMMIT}" \
    && echo "manga-image-translator pinned at ${MIT_COMMIT}"

RUN apt-get update && apt-get install -y --no-install-recommends g++ build-essential \
    && python -m pip install --no-cache-dir cython \
    && python -m pip install --no-cache-dir -r "${MIT_ROOT}/requirements.txt" \
    && apt-get purge -y --auto-remove g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 5. M.I.T. detection / OCR / inpainting weights, baked into the image.
#    docker_prepare.py --models takes "<stage>.<enum value>" keys.
#    Verified against upstream enums: Detector.default="default",
#    Ocr.ocr48px="48px", Inpainter.lama_large="lama_large".
# ---------------------------------------------------------------------------
ARG MIT_MODELS="detector.default,ocr.48px,inpaint.lama_large"
RUN cd "${MIT_ROOT}" \
    && python -u docker_prepare.py --models "${MIT_MODELS}" \
    && du -sh "${MIT_ROOT}/models" || true

# ---------------------------------------------------------------------------
# 6. manga-ocr weights. Upstream pulls these lazily inside _load(), i.e. at
#    RUNTIME, which we do not want. Warm the cache here so selecting the "mocr"
#    OCR in the UI never triggers a download on a running Space.
# ---------------------------------------------------------------------------
ENV HF_HOME=/opt/hf-cache
RUN python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('kha-white/manga-ocr-base'); \
print('manga-ocr cached at', p)"

# ---------------------------------------------------------------------------
# 7. Hy-MT2-7B Q4_K_M (4.62 GB) — the single most expensive layer.
#    Pinned to a repo revision and verified by SHA-256 so the build is
#    deterministic. Downloaded straight to its final path with curl: using
#    hf_hub_download would ALSO keep a copy in the HF cache and double the
#    image size for no benefit.
# ---------------------------------------------------------------------------
ARG HY_MT2_REPO=tencent/Hy-MT2-7B-GGUF
ARG HY_MT2_FILE=Hy-MT2-7B-Q4_K_M.gguf
ARG HY_MT2_REVISION=ab8472660ac61fac25f1af43fac2599d52a8a775
ARG HY_MT2_SHA256=9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b
ENV HY_MT2_MODEL_PATH=/opt/models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf
RUN mkdir -p /opt/models/hy-mt2 \
    && curl -fL --retry 5 --retry-delay 5 --retry-connrefused \
        -o "${HY_MT2_MODEL_PATH}" \
        "https://huggingface.co/${HY_MT2_REPO}/resolve/${HY_MT2_REVISION}/${HY_MT2_FILE}" \
    && echo "${HY_MT2_SHA256}  ${HY_MT2_MODEL_PATH}" | sha256sum -c - \
    && ls -la /opt/models/hy-mt2/

# ---------------------------------------------------------------------------
# 8. Application source from GitHub. Last layer on purpose: a code change only
#    rebuilds from here, leaving layers 1-7 (including the 4.6 GB model) cached.
#    The ADD of the commit ref busts the cache when the branch moves.
# ---------------------------------------------------------------------------
ARG APP_REPO=https://github.com/Waifuhtr/mnaga
ARG APP_REF=claude/hg-space-repo-access-033wjr
ADD "https://api.github.com/repos/Waifuhtr/mnaga/commits/${APP_REF}" /tmp/app-commit.json
RUN git clone --depth 1 --branch "${APP_REF}" "${APP_REPO}" /app \
    && rm -rf /app/.git \
    && chmod +x /app/scripts/*.sh

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
# Point manga-image-translator's custom_openai translator at the local
# Hy-MT2 llama-server. These are the exact env names read by
# manga_translator/translators/keys.py.
ENV CUSTOM_OPENAI_API_BASE=http://127.0.0.1:8081/v1 \
    CUSTOM_OPENAI_API_KEY=hy-mt2-local \
    CUSTOM_OPENAI_MODEL=hy-mt2 \
    CUSTOM_OPENAI_MODEL_CONF= \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8081 \
    APP_PORT=7860 \
    PYTHONPATH=/opt/manga-image-translator:/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Writable dirs for the Space's runtime user.
RUN mkdir -p /data/work /data/results "${MIT_ROOT}/result" \
    && chmod -R 777 /data "${MIT_ROOT}/result"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${APP_PORT}/health" || exit 1

WORKDIR /app
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
