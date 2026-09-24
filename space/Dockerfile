# syntax=docker/dockerfile:1.7
###############################################################################
# Manga Image Translator + Hy-MT2-7B  ->  Hugging Face Space image
#
# This file is the ONLY build file the Space repo needs. Application source is
# pulled from GitHub at build time, so the Space repo stays tiny.
#
# LAYER ORDER IS LOAD-BEARING. Layers run stable -> volatile so that an
# application code change never invalidates the 4.6 GB model layer:
#
#   1 OS packages + Python 3.11        ~ rarely changes
#   2 CUDA libraries (pip)               pinned
#   3 PyTorch                            pinned
#   4 llama.cpp CUDA binaries            pinned, smoke-tested here
#   5 manga-image-translator + deps      pinned commit
#   6 M.I.T. detection/OCR/inpaint weights
#   7 manga-ocr weights
#   8 Hy-MT2-7B-Q4_K_M.gguf  4.6 GB      most expensive layer
#   9 application source from GitHub   <- only this rebuilds normally
#
# SIZE MATTERS: the Space build is OOM-killed while pushing an oversized image,
# so this deliberately keeps ONE copy of the CUDA runtime. An nvidia/cuda base
# image would duplicate ~5 GB of libraries that PyTorch's own nvidia-* wheels
# already provide. Instead the base is plain Ubuntu and llama.cpp is pointed at
# PyTorch's copy. The CUDA libraries are also installed in their own layer,
# separately from PyTorch, to halve the largest single layer.
#
# Build on CPU hardware (8 vCPU / 32 GB), then switch the Space to T4 Small:
# nothing is re-downloaded, because every artifact is already in these layers.
###############################################################################

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# 1. OS packages + Python 3.11.
#
#    Ubuntu 24.04 is required because llama.cpp's official CUDA release
#    binaries are linked against GLIBC 2.38; 22.04 only provides 2.35. (Their
#    CPU-only tarball needs just 2.29, which is why this bites CUDA alone.)
#
#    24.04 ships Python 3.12, but manga-image-translator wants >=3.10,<3.12, so
#    Python 3.11 comes from deadsnakes and all Python work happens in a venv,
#    which also avoids 24.04's PEP 668 "externally managed environment" refusal.
#
#    libglib2.0-0 was renamed libglib2.0-0t64 in 24.04 (64-bit time_t).
#    No CJK system font is installed: manga-image-translator ships its own
#    fonts, and the renderer uses one of those.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
        git curl ca-certificates \
        libgomp1 \
        libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Putting the venv first on PATH makes `python` and `pip` resolve to 3.11 for
# every layer below, including the model prefetch steps and the entrypoint.
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/python --version
ENV PATH=/opt/venv/bin:$PATH
ENV VENV_SITE=/opt/venv/lib/python3.11/site-packages

# ---------------------------------------------------------------------------
# 2. CUDA libraries, as their own layer.
#
#    These are exactly the versions PyTorch 2.9.1+cu128 pins. Installing them
#    up front means the PyTorch layer below only adds torch itself.
#
#    They are then split across FIVE layers rather than one. As a single layer
#    they came to ~2.9 GB, and the image push is OOM-killed on the largest
#    single layer - which is why CUDA was separated from PyTorch in the first
#    place. A cold build (a Factory rebuild reuses nothing) has to materialise
#    and push every layer, so that peak is what decides whether the push
#    survives. Split this way the biggest of them is ~800 MB.
#
#    Order is load-bearing: a package must not be installed before something it
#    depends on, or pip resolves that dependency itself and takes the LATEST
#    version instead of the pin below - and those bytes then sit in the layer
#    forever even though a later layer replaces the package.
#      cudnn    -> cublas
#      cusolver -> cublas, cusparse, nvjitlink
#      cusparse, cufft -> nvjitlink
# ---------------------------------------------------------------------------
RUN python -m pip install --upgrade "pip<25" setuptools wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        nvidia-cuda-runtime-cu12==12.8.90 \
        nvidia-cuda-cupti-cu12==12.8.90 \
        nvidia-cuda-nvrtc-cu12==12.8.93 \
        nvidia-nvtx-cu12==12.8.90 \
        nvidia-nvjitlink-cu12==12.8.93 \
        nvidia-cufile-cu12==1.13.1.3 \
        nvidia-curand-cu12==10.3.9.90

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        nvidia-cublas-cu12==12.8.4.1

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        nvidia-cudnn-cu12==9.10.2.21

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        nvidia-cufft-cu12==11.3.3.83 \
        nvidia-cusparse-cu12==12.5.8.93 \
        nvidia-cusparselt-cu12==0.7.1

RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        nvidia-cusolver-cu12==11.7.3.90 \
        nvidia-nccl-cu12==2.27.5 \
        nvidia-nvshmem-cu12==3.3.20 \
    && test -f "${VENV_SITE}/nvidia/cuda_runtime/lib/libcudart.so.12" \
    && test -f "${VENV_SITE}/nvidia/cublas/lib/libcublas.so.12"

# ---------------------------------------------------------------------------
# 3. PyTorch. Its CUDA dependencies are already satisfied by layer 2.
# ---------------------------------------------------------------------------
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    && python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

# ---------------------------------------------------------------------------
# 4. llama.cpp, official prebuilt CUDA 12.8 binaries.
#
#    Placed after PyTorch so the smoke test below can run with the real CUDA
#    libraries visible. `llama-server --version` is a HARD gate on purpose: an
#    earlier version of this file let it fail silently, which turned a missing
#    libgomp.so.1 into a runtime crash instead of a build error.
#    Nothing is compiled and no GPU is touched at build time.
# ---------------------------------------------------------------------------
ARG LLAMA_CPP_RELEASE=b11022
ENV LLAMA_LD_PATH=/opt/llama:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib
RUN curl -fsSL --retry 8 --retry-delay 3 --retry-all-errors --retry-max-time 600 \
        --speed-limit 1048576 --speed-time 60 \
        -o /tmp/llama.tar.gz \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_RELEASE}/llama-${LLAMA_CPP_RELEASE}-bin-ubuntu-cuda-12.8-x64.tar.gz" \
    && mkdir -p /opt/llama \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
    && rm -f /tmp/llama.tar.gz \
    && chmod +x /opt/llama/llama-server \
    && LD_LIBRARY_PATH="${LLAMA_LD_PATH}" /opt/llama/llama-server --version

# ---------------------------------------------------------------------------
# 5. manga-image-translator at a pinned commit + its Python dependencies.
#    Repo:   https://github.com/zyddnys/manga-image-translator
#    Commit: 95227a2b ("Merge pull request #1157 ... m2m100-hf", 2026-07-20)
#    g++ is needed only to build pydensecrf, then removed again. The checkout's
#    .git directory is dropped: it is dead weight in the final image.
# ---------------------------------------------------------------------------
ARG MIT_REPO=https://github.com/zyddnys/manga-image-translator
ARG MIT_COMMIT=95227a2bb0fd306cd4f0c104d57284026f991b3a
ENV MIT_ROOT=/opt/manga-image-translator
RUN git clone --filter=blob:none "${MIT_REPO}" "${MIT_ROOT}" \
    && git -C "${MIT_ROOT}" checkout --detach "${MIT_COMMIT}" \
    && echo "manga-image-translator pinned at ${MIT_COMMIT}" \
    && rm -rf "${MIT_ROOT}/.git"

# Needed by the prefetch step below as well as at runtime.
ENV PYTHONPATH=/opt/manga-image-translator:/app

RUN apt-get update && apt-get install -y --no-install-recommends g++ build-essential \
    && python -m pip install cython \
    && python -m pip install -r "${MIT_ROOT}/requirements.txt" \
    && apt-get purge -y --auto-remove g++ build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# ---------------------------------------------------------------------------
# 6. M.I.T. detection / OCR / inpainting weights, baked into the image.
#
#    NOT using upstream's docker_prepare.py --models. That filters with
#    f"detector.{k}" where k is a str-Enum member, and Python 3.11 changed
#    Enum.__format__ for mixin enums to include the class name. So the
#    expression evaluates to "detector.Detector.default", the filter matches
#    nothing, and the step exits having downloaded NOTHING while still looking
#    like it worked. (Only Translator defines __str__ upstream, which is why
#    the runtime translator path is unaffected.)
#
#    Selecting by enum member instead is immune to that. The script then
#    asserts that weights actually landed on disk, so the build fails here
#    rather than shipping an image that downloads models on first use.
# ---------------------------------------------------------------------------
RUN python -u - <<'PYEOF'
import asyncio, os, sys
from manga_translator.config import Detector, Inpainter, Ocr
from manga_translator.detection import DETECTORS
from manga_translator.inpainting import INPAINTERS
from manga_translator.ocr import OCRS
from manga_translator.utils import ModelWrapper

WANTED = [
    ("detector", DETECTORS, [Detector.default]),
    # 48px is the default; mocr is offered in the UI, so bake its checkpoint
    # too or picking it would trigger a download on a running Space.
    ("ocr", OCRS, [Ocr.ocr48px, Ocr.mocr]),
    ("inpainter", INPAINTERS, [Inpainter.lama_large]),
]

async def main():
    for stage, registry, keys in WANTED:
        for key in keys:
            cls = registry[key]
            if not issubclass(cls, ModelWrapper):
                print(f"    {stage}.{key.value}: no weights to fetch", flush=True)
                continue
            print(f"--> {stage}.{key.value} ({cls.__name__})", flush=True)
            await cls().download()

asyncio.run(main())

# Assert the weights actually landed. Upstream's filter silently downloading
# nothing is exactly the failure this guards against.
root = os.environ["MIT_ROOT"]
model_dir = os.path.join(root, "models")
total = 0
if os.path.isdir(model_dir):
    for dirpath, _, names in os.walk(model_dir):
        for n in names:
            total += os.path.getsize(os.path.join(dirpath, n))
print(f"baked weights: {total / 1e6:.0f} MB in {model_dir}", flush=True)
if total < 50_000_000:
    sys.exit(f"FATAL: expected baked model weights, found {total} bytes")
PYEOF

# ---------------------------------------------------------------------------
# 7. manga-ocr weights. Upstream pulls these lazily inside _load(), i.e. at
#    RUNTIME, which we do not want. Warm the cache here so selecting the "mocr"
#    OCR in the UI never triggers a download on a running Space.
# ---------------------------------------------------------------------------
ENV HF_HOME=/opt/hf-cache
RUN python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('kha-white/manga-ocr-base'); \
print('manga-ocr cached at', p)"

# ---------------------------------------------------------------------------
# 8. Hy-MT2-7B Q4_K_M (4.62 GB) - the single most expensive layer.
#    Pinned to a repo revision and verified by SHA-256 so the build is
#    deterministic. Downloaded straight to its final path with curl: using
#    hf_hub_download would ALSO keep a copy in the HF cache and add another
#    4.6 GB to an image that is already close to the push limit.
# ---------------------------------------------------------------------------
ARG HY_MT2_REPO=tencent/Hy-MT2-7B-GGUF
ARG HY_MT2_FILE=Hy-MT2-7B-Q4_K_M.gguf
ARG HY_MT2_REVISION=ab8472660ac61fac25f1af43fac2599d52a8a775
ARG HY_MT2_SHA256=9f96256500f3fc1ab4d64336b58f52a949a95ad7516b0c229476eef782f9f77b
ENV HY_MT2_MODEL_PATH=/opt/models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf
# curl's --retry only covers what it calls transient errors: timeouts, FTP 4xx
# and HTTP 408/429/5xx. A connection that dies MID-BODY is CURLE_PARTIAL_FILE
# (exit 18) and is NOT on that list, so plain --retry gives up instantly - this
# download died at 2768 MB of 4410 MB and the build failed without one retry.
#
# --retry-all-errors brings exit 18 into scope, and -C - resumes from the bytes
# already on disk instead of starting the 4.4 GB over on every attempt. If a
# resume ever lands at the wrong offset the sha256 check below fails the build,
# so correctness does not rest on the range request being honoured.
#
# --speed-limit/--speed-time turn a stalled-but-open connection into an error
# after 60s, which --retry-all-errors then retries; otherwise it could hang
# until the builder times out.
RUN mkdir -p /opt/models/hy-mt2 \
    && curl -fL --retry 8 --retry-delay 5 --retry-all-errors --retry-max-time 1800 \
        --continue-at - --speed-limit 1048576 --speed-time 60 \
        -o "${HY_MT2_MODEL_PATH}" \
        "https://huggingface.co/${HY_MT2_REPO}/resolve/${HY_MT2_REVISION}/${HY_MT2_FILE}" \
    && echo "${HY_MT2_SHA256}  ${HY_MT2_MODEL_PATH}" | sha256sum -c - \
    && ls -la /opt/models/hy-mt2/

# ---------------------------------------------------------------------------
# 9. Application source from GitHub.
#
#    Baked here as a FALLBACK only. The copy that actually runs is fetched by
#    /opt/bootstrap.sh when the container starts, because this layer cannot be
#    relied on to be current: the ADD of the commit ref below is meant to bust
#    the cache when the branch moves, and on the Spaces builder it does not -
#    a rebuild kept serving a clone three commits old.
#
#    Fetching at startup also makes code changes cost a RESTART instead of a
#    rebuild: seconds and no credits, against ~7 minutes and a full image push.
#    The models stay baked, which is the part that is big and never changes.
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
# Point manga-image-translator's custom_openai translator at the local Hy-MT2
# llama-server. These are the exact env names read by
# manga_translator/translators/keys.py.
ENV CUSTOM_OPENAI_API_BASE=http://127.0.0.1:8081/v1 \
    CUSTOM_OPENAI_API_KEY=hy-mt2-local \
    CUSTOM_OPENAI_MODEL=hy-mt2 \
    CUSTOM_OPENAI_MODEL_CONF= \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8081 \
    APP_PORT=7860 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Spaces runs the container as UID 1000, and /data only exists at runtime when
# a storage bucket is attached, so scratch space lives in /tmp instead.
ENV WORK_DIR=/tmp/mnaga-work
RUN mkdir -p /tmp/mnaga-work "${MIT_ROOT}/result" \
    && chmod 777 /tmp/mnaga-work "${MIT_ROOT}/result"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${APP_PORT}/health" || exit 1

# ---------------------------------------------------------------------------
# Startup bootstrap. Lives outside /app on purpose: it replaces /app, so it
# cannot be running from inside it.
# ---------------------------------------------------------------------------
RUN cat > /opt/bootstrap.sh <<'BOOTSTRAP'
#!/usr/bin/env bash
# Refresh the application from GitHub, then hand over to its entrypoint.
#
# The image carries a copy of /app baked at build time, but that copy cannot be
# trusted to be current: the Dockerfile's ADD of the commit ref is supposed to
# bust the build cache when the branch moves, and on the Spaces builder it does
# not - a rebuild kept serving a clone three commits old. Fetching here instead
# means the running code is whatever the branch holds right now, and a code
# change costs a restart rather than a rebuild.
#
# If the fetch fails for any reason the baked copy runs, so losing GitHub means
# running slightly older code, never a Space that will not start.
set -uo pipefail
cd /

log() { printf '[bootstrap] %s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

APP_REPO=${APP_REPO:-https://github.com/Waifuhtr/mnaga}
APP_REF=${APP_REF:-claude/hg-space-repo-access-033wjr}
APP_REFRESH=${APP_REFRESH:-1}

if [[ "${APP_REFRESH}" == "0" ]]; then
    log "APP_REFRESH=0 - running the copy baked into the image"
else
    log "fetching ${APP_REF} from ${APP_REPO}"
    rm -rf /tmp/app-new /app.old
    if timeout "${APP_REFRESH_TIMEOUT:-120}" \
            git clone --quiet --depth 1 --branch "${APP_REF}" "${APP_REPO}" /tmp/app-new; then
        sha=$(git -C /tmp/app-new rev-parse HEAD)
        when=$(git -C /tmp/app-new log -1 --format=%cI)
        subject=$(git -C /tmp/app-new log -1 --format=%s)
        rm -rf /tmp/app-new/.git
        if mv /app /app.old && mv /tmp/app-new /app; then
            rm -rf /app.old
            chmod +x /app/scripts/*.sh 2>/dev/null || true
            # /health and the version line in the UI read this file.
            python - "$sha" "$when" "$subject" > /tmp/app-commit.json <<'PYJSON'
import json, sys
print(json.dumps({"sha": sys.argv[1],
                  "commit": {"author": {"date": sys.argv[2]},
                             "message": sys.argv[3]}}))
PYJSON
            log "running ${sha:0:7} - ${subject}"
        else
            log "could not swap /app into place - restoring the baked copy"
            [[ -d /app.old && ! -d /app ]] && mv /app.old /app
        fi
    else
        log "fetch failed - falling back to the copy baked into the image"
        rm -rf /tmp/app-new
    fi
fi

exec /app/scripts/entrypoint.sh
BOOTSTRAP

RUN chmod +x /opt/bootstrap.sh && bash -n /opt/bootstrap.sh

WORKDIR /app
ENTRYPOINT ["/opt/bootstrap.sh"]
