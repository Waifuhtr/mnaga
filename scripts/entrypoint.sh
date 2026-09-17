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

export LD_LIBRARY_PATH="/opt/llama:${LD_LIBRARY_PATH:-}"

log() { echo "[entrypoint] $*"; }

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
GPU_LAYERS=0
GPU_NAME="none"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    GPU_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
    GPU_LAYERS=${LLAMA_N_GPU_LAYERS:-99}
    log "GPU        : ${GPU_NAME} (${GPU_VRAM})"
    log "offload    : ENABLED, n_gpu_layers=${GPU_LAYERS}"
else
    GPU_LAYERS=0
    log "GPU        : not visible -> CPU fallback"
    log "offload    : DISABLED (translation will be slow but functional)"
fi

# Context budget. Manga lines are short; a large context would only waste VRAM.
# 8192 total across 2 slots = 4096 per slot, which matches the upstream
# translator's _MAX_TOKENS.
CTX_SIZE=${LLAMA_CTX_SIZE:-8192}
PARALLEL=${LLAMA_PARALLEL:-2}
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

cleanup() {
    log "shutting down (llama pid ${LLAMA_PID})"
    kill "${LLAMA_PID}" 2>/dev/null || true
    wait "${LLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Wait for llama-server to finish loading before accepting traffic.
# ---------------------------------------------------------------------------
log "waiting for llama-server on ${LLAMA_HOST}:${LLAMA_PORT} ..."
DEADLINE=$(( SECONDS + ${LLAMA_STARTUP_TIMEOUT:-600} ))
until curl -fsS "http://${LLAMA_HOST}:${LLAMA_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
        log "FATAL: llama-server exited during startup. Last 40 log lines:"
        tail -40 /tmp/llama-server.log || true
        exit 1
    fi
    if (( SECONDS > DEADLINE )); then
        log "FATAL: llama-server did not become healthy in time. Last 40 log lines:"
        tail -40 /tmp/llama-server.log || true
        exit 1
    fi
    sleep 2
done
log "llama-server is healthy"

# Report whether CUDA was actually engaged, so a T4 runtime can be verified
# from the Space logs alone.
if grep -qiE 'CUDA[0-9]*|offloading .* layers to GPU' /tmp/llama-server.log; then
    grep -iE "load_tensors:|CUDA[0-9]* model buffer|offloaded .* layers" /tmp/llama-server.log \
        | head -8 | sed 's/^/[llama] /' || true
fi

log "starting web app on 0.0.0.0:${APP_PORT}"
exec python -m uvicorn app.server:app \
    --host 0.0.0.0 \
    --port "${APP_PORT}" \
    --app-dir /app \
    --timeout-keep-alive 120
