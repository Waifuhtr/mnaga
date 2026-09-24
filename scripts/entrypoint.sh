#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Container entrypoint.
#
# Starts two processes:
#   1. llama-server  -> Hy-MT2-7B behind an OpenAI-compatible API (port 8081)
#   2. uvicorn       -> this project's web UI + translate API (port 7860)
#
# Nothing is downloaded here. Every model file was baked into the image at
# build time, so this works with the network switched off.
# ---------------------------------------------------------------------------
set -euo pipefail

LLAMA_BIN=${LLAMA_BIN:-/opt/llama/llama-server}
MODEL_PATH=${HY_MT2_MODEL_PATH:-/opt/models/hy-mt2/Hy-MT2-7B-Q4_K_M.gguf}
LLAMA_HOST=${LLAMA_HOST:-127.0.0.1}
LLAMA_PORT=${LLAMA_PORT:-8081}
APP_PORT=${APP_PORT:-7860}
MODEL_ALIAS=${CUSTOM_OPENAI_MODEL:-hy-mt2}

# llama.cpp's CUDA backend needs libcudart.so.12 and libcublas.so.12. This
# image deliberately keeps exactly ONE copy of the CUDA runtime - the one
# PyTorch ships in its nvidia-* wheels - instead of also carrying an
# nvidia/cuda base image, which would add ~5 GB of duplicate libraries and
# push the image over the size the Space builder can handle. So point the
# dynamic loader at PyTorch's copy. Globbed rather than hardcoded so a torch
# upgrade that moves the directories does not silently break GPU offload.
NV_LIBS="$(python - <<'PY' 2>/dev/null || true
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(":".join(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))))
PY
)"
export LD_LIBRARY_PATH="/opt/llama${NV_LIBS:+:${NV_LIBS}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Every line carries wall-clock UTC and seconds since this script started, so a
# cold start can be measured from the Space log instead of guessed at. The
# Python side already timestamps its own lines, and these two line up.
START_SECONDS=${SECONDS}
log() {
    printf '[entrypoint] %s (+%ss) %s\n' "$(date -u +%H:%M:%S)" "$(( SECONDS - START_SECONDS ))" "$*"
}

log "================= startup ================="
log "model      : ${MODEL_PATH}"
log "llama bin  : ${LLAMA_BIN}"

if [[ ! -f "${MODEL_PATH}" ]]; then
    log "FATAL: model file missing from the image: ${MODEL_PATH}"
    log "The image was built incorrectly - layer 7 of the Dockerfile did not run."
    exit 1
fi
log "model size : $(du -h "${MODEL_PATH}" | cut -f1)"

# ---------------------------------------------------------------------------
# GPU detection. The image is built on CPU hardware, so GPU presence is decided
# here at runtime, never at build time.
# ---------------------------------------------------------------------------
# Detection is layered on purpose. The base image is plain Ubuntu, not
# nvidia/cuda, so nvidia-smi is only present if the container runtime injects
# it. Relying on nvidia-smi alone would silently fall back to CPU on a GPU
# Space - the user would pay for a T4 and get CPU speed with no warning.
# PyTorch talks to libcuda directly and is the authoritative check here.
GPU_LAYERS=0
GPU_NAME="none"
GPU_VRAM=""
GPU_FOUND=0

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_FOUND=1
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    GPU_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
elif python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_FOUND=1
    GPU_NAME="$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo 'CUDA device')"
    GPU_VRAM="$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GiB')" 2>/dev/null || echo '')"
    log "note       : nvidia-smi unavailable, GPU detected via PyTorch"
elif [[ -e /dev/nvidiactl ]]; then
    GPU_FOUND=1
    GPU_NAME="NVIDIA device (/dev/nvidiactl)"
    log "note       : GPU device node present but neither nvidia-smi nor torch could query it"
fi

if (( GPU_FOUND )); then
    GPU_LAYERS=${LLAMA_N_GPU_LAYERS:-99}
    log "GPU        : ${GPU_NAME} ${GPU_VRAM}"
    log "offload    : ENABLED, n_gpu_layers=${GPU_LAYERS}"
    if [[ -n "${NV_LIBS}" ]]; then
        log "cuda libs  : $(echo "${NV_LIBS}" | tr ':' '\n' | wc -l) dir(s) from PyTorch"
    else
        log "cuda libs  : NOT FOUND - GPU offload will fail, check the venv"
    fi
else
    GPU_LAYERS=0
    log "GPU        : not visible -> CPU fallback"
    log "offload    : DISABLED (translation will be slow but functional)"
fi

# Context budget. Manga lines are short, and on a 16 GB T4 this model competes
# for VRAM with lama inpainting, which is the real memory hog. One slot at 4096
# still matches the upstream translator's _MAX_TOKENS while leaving the KV cache
# small, so keep it tight rather than generous.
CTX_SIZE=${LLAMA_CTX_SIZE:-4096}
PARALLEL=${LLAMA_PARALLEL:-1}
THREADS=${LLAMA_THREADS:-4}

log "context    : ${CTX_SIZE} across ${PARALLEL} slots"
log "==========================================="

"${LLAMA_BIN}" \
    --model "${MODEL_PATH}" \
    --alias "${MODEL_ALIAS}" \
    --host "${LLAMA_HOST}" \
    --port "${LLAMA_PORT}" \
    --ctx-size "${CTX_SIZE}" \
    --parallel "${PARALLEL}" \
    --threads "${THREADS}" \
    --n-gpu-layers "${GPU_LAYERS}" \
    --batch-size 512 \
    --ubatch-size 128 \
    --cont-batching \
    --jinja \
    --metrics \
    > /tmp/llama-server.log 2>&1 &
LLAMA_PID=$!

# ---------------------------------------------------------------------------
# Start the web app NOW, in parallel with llama-server's model load.
#
# These two do not depend on each other while loading: llama-server is reading
# a 4.4 GB GGUF off disk and copying it into VRAM, while the web app is mostly
# importing torch and manga-image-translator. Running them back to back cost
# the sum of both; running them together costs roughly the longer of the two.
#
# The app serves immediately, but refuses to start a translation until
# llama-server answers /health - app/server.py checks that on every job, and
# the UI's status chip shows the same thing. So nothing can silently run
# without a translation backend.
# ---------------------------------------------------------------------------
log "starting web app on 0.0.0.0:${APP_PORT} (in parallel with model load)"
python -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port "${APP_PORT}" \
    --app-dir /app \
    --timeout-keep-alive 120 &
APP_PID=$!

cleanup() {
    log "shutting down (llama pid ${LLAMA_PID}, app pid ${APP_PID})"
    kill "${LLAMA_PID}" "${APP_PID}" 2>/dev/null || true
    wait "${LLAMA_PID}" 2>/dev/null || true
    wait "${APP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Wait for llama-server to finish loading. The app is already up at this point;
# this loop exists to report the load time and to fail the container loudly if
# the model never arrives.
# ---------------------------------------------------------------------------
log "waiting for llama-server on ${LLAMA_HOST}:${LLAMA_PORT} ..."
DEADLINE=$(( SECONDS + ${LLAMA_STARTUP_TIMEOUT:-600} ))
LLAMA_WAIT_START=${SECONDS}
until curl -fsS "http://${LLAMA_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
        log "FATAL: llama-server exited during startup. Last 40 log lines:"
        tail -40 /tmp/llama-server.log || true
        exit 1
    fi
    if ! kill -0 "${APP_PID}" 2>/dev/null; then
        log "FATAL: web app exited during startup."
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        log "FATAL: llama-server did not become healthy in time. Last 40 log lines:"
        tail -40 /tmp/llama-server.log || true
        exit 1
    fi
    sleep 2
done
log "llama-server is healthy (model load took $(( SECONDS - LLAMA_WAIT_START ))s)"

# Report whether CUDA was actually engaged, so a T4 runtime can be verified
# from the Space logs alone.
if grep -qiE 'CUDA[0-9]*|offloading .* layers to GPU' /tmp/llama-server.log; then
    grep -iE "load_tensors:|CUDA[0-9]* model buffer|offloaded .* layers" /tmp/llama-server.log \
        | head -8 | sed 's/^/[llama] /' || true
fi

log "READY - both processes up"

# Neither process is expected to exit. Whichever does, take the container down
# with it so the Space restarts rather than sitting there half-working.
STATUS=0
wait -n "${LLAMA_PID}" "${APP_PID}" || STATUS=$?
if kill -0 "${LLAMA_PID}" 2>/dev/null; then
    log "FATAL: web app exited (status ${STATUS})"
else
    log "FATAL: llama-server exited (status ${STATUS}). Last 40 log lines:"
    tail -40 /tmp/llama-server.log || true
fi
exit "${STATUS}"
