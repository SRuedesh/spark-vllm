#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Load configuration
###############################################################################

ENV_FILE="${ENV_FILE:-.qwen38-flash-next.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

###############################################################################
# Configuration defaults
###############################################################################

# Qwen3.8-Flash-Next is too large in BF16/FP8 for one 128 GB DGX Spark.
# This launcher uses the NVFP4 checkpoint plus an NVMe-mapped PLE table.
MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"

# 64K is a practical starting point on one Spark and satisfies Hermes Agent's
# minimum context requirement. The model is natively 262,144 tokens.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}"
# Qwen3.8-Flash-Next QSA in the current vLLM preview requires the main KV
# cache to be BF16. Do not leave this at auto, because auto can resolve to a
# non-BF16 cache path for this ModelOpt checkpoint and QSA will abort at init.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"

# MTP speculative decoding. The upstream baseline defaults to 0. This launcher
# keeps 2 enabled for the 64K Hermes deployment because it improves decode on GB10.
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-2}"

# PLE n-gram embedding mmap settings.
PLE_MMAP_WORKERS="${PLE_MMAP_WORKERS:-32}"
PLE_MMAP_PREWARM="${PLE_MMAP_PREWARM:-0}"

PUBLIC_HOST="${PUBLIC_HOST:-0.0.0.0}"
PUBLIC_PORT="${PUBLIC_PORT:-1234}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"

CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-vllm}"

# Do not inherit a generic VLLM_IMAGE override from older Qwen launchers.
# Qwen3.8-Flash-Next needs model support plus the Spark PLE patch.
# Override this deployment only with QWEN38_VLLM_IMAGE if necessary.
if [ -n "${VLLM_IMAGE:-}" ] && [ "${VLLM_IMAGE}" != "qwen38-flash-dgx" ]; then
  echo "WARNING: Ignoring legacy VLLM_IMAGE='$VLLM_IMAGE'."
  echo "Qwen3.8-Flash-Next requires its dedicated patched image."
fi
QWEN38_VLLM_IMAGE="${QWEN38_VLLM_IMAGE:-qwen38-flash-dgx}"
VLLM_IMAGE="$QWEN38_VLLM_IMAGE"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
WORKDIR="${WORKDIR:-$HOME/qwen38-flash-spark}"

# vLLM waits only 600 s for the engine core by default. A cold GB10 start can
# exceed that while loading the model and JIT-compiling FlashInfer kernels.
VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-2400}"

# Persist FlashInfer's JIT artifacts across container recreation. Without this,
# every RESTART_VLLM=1 can pay the cold sm_121 compilation cost again.
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-$WORKDIR/flashinfer-cache}"

# Single-Spark vLLM patch recipe. The patch serves the large PLE lookup table
# from NVMe instead of keeping it resident in the unified memory pool.
RECIPE_REPO_URL="${RECIPE_REPO_URL:-https://github.com/blazux/qwen3.8-Flash-DGX.git}"
RECIPE_REF="${RECIPE_REF:-main}"
RECIPE_DIR="${RECIPE_DIR:-$WORKDIR/qwen3.8-Flash-DGX}"

# Set to 1 to rebuild the patched image from the recipe.
REBUILD_VLLM_IMAGE="${REBUILD_VLLM_IMAGE:-0}"

# The mmap patch is validated against the RadixArk NVFP4 checkpoint. Set this
# only if you intentionally know that another checkpoint is compatible.
ALLOW_CUSTOM_MODEL_ID="${ALLOW_CUSTOM_MODEL_ID:-0}"

# Set to 1 to run the resumable Hugging Face download check.
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"

# Set to 1 when changing the model, image, or vLLM arguments.
# Set to 0 to retain the loaded model and restart only the proxy.
RESTART_VLLM="${RESTART_VLLM:-1}"

# Qwen reasoning configuration.
ENABLE_REASONING_PARSER="${ENABLE_REASONING_PARSER:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Qwen automatic tool calling.
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# Qwen3.8-Flash-Next thinks by default. Keep the previous launcher's behavior
# and disable thinking globally unless a request overrides it.
if [ -z "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi

PROXY_PID_FILE="$WORKDIR/proxy.pid"
PROXY_LOG_FILE="$WORKDIR/proxy.log"

###############################################################################
# Validation
###############################################################################

if [ "$MODEL_ID" != "RadixArk/Qwen3.8-Flash-Next-NVFP4" ] &&
   [ "$ALLOW_CUSTOM_MODEL_ID" != "1" ]; then
  echo "ERROR: MODEL_ID is '$MODEL_ID'."
  echo "The single-Spark mmap recipe expects:"
  echo "  RadixArk/Qwen3.8-Flash-Next-NVFP4"
  echo
  echo "Your environment file may be overriding MODEL_ID."
  echo "Inspect it with:"
  echo "  grep -n '^MODEL_ID=' '$ENV_FILE'"
  echo
  echo "Remove that override, or set ALLOW_CUSTOM_MODEL_ID=1 only if the"
  echo "alternative checkpoint is known to be compatible with this mmap patch."
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ "$DOWNLOAD_MODEL" = "1" ]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Set it in $ENV_FILE, or set DOWNLOAD_MODEL=0 if the checkpoint is cached."
  exit 1
fi

if ! [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] || (( MAX_MODEL_LEN < 65536 )); then
  echo "ERROR: MAX_MODEL_LEN must be an integer of at least 65536 for Hermes Agent."
  exit 1
fi

if ! [[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]] || (( MAX_NUM_SEQS < 1 )); then
  echo "ERROR: MAX_NUM_SEQS must be a positive integer."
  exit 1
fi

if ! [[ "$MTP_SPECULATIVE_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MTP_SPECULATIVE_TOKENS must be a non-negative integer."
  exit 1
fi

# The current Qwen3.8-Flash-Next QSA implementation requires BF16 for the
# main KV cache. Normalize the common short spelling and reject incompatible
# cache formats before spending time loading the model.
case "$KV_CACHE_DTYPE" in
  bf16)
    KV_CACHE_DTYPE="bfloat16"
    ;;
  bfloat16)
    ;;
  auto)
    echo "WARNING: KV_CACHE_DTYPE=auto is unsafe for Qwen3.8-Flash-Next QSA."
    echo "Forcing KV_CACHE_DTYPE=bfloat16."
    KV_CACHE_DTYPE="bfloat16"
    ;;
  *)
    echo "ERROR: Qwen3.8-Flash-Next QSA currently requires a BF16 main KV cache."
    echo "Set KV_CACHE_DTYPE=bfloat16 (or bf16), not '$KV_CACHE_DTYPE'."
    exit 1
    ;;
esac

if ! [[ "$VLLM_ENGINE_READY_TIMEOUT_S" =~ ^[0-9]+$ ]] || (( VLLM_ENGINE_READY_TIMEOUT_S < 600 )); then
  echo "ERROR: VLLM_ENGINE_READY_TIMEOUT_S must be an integer of at least 600."
  exit 1
fi

if ! [[ "$STARTUP_WAIT_SECONDS" =~ ^[0-9]+$ ]] || (( STARTUP_WAIT_SECONDS <= VLLM_ENGINE_READY_TIMEOUT_S )); then
  echo "ERROR: STARTUP_WAIT_SECONDS must be an integer greater than VLLM_ENGINE_READY_TIMEOUT_S."
  exit 1
fi

for command in docker curl python3 git nvidia-smi; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: $command"
    exit 1
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not running or the current user cannot access it."
  exit 1
fi

echo "Checking Docker GPU access..."

if ! docker run --rm \
  --gpus all \
  nvidia/cuda:12.6.0-base-ubuntu24.04 \
  nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: Docker cannot access the NVIDIA GPU."
  echo "Check the NVIDIA driver and NVIDIA Container Toolkit."
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
if [[ "$GPU_NAME" != *"GB10"* ]]; then
  echo "WARNING: Detected GPU '$GPU_NAME'."
  echo "This configuration is tuned for one DGX Spark / GB10 with 128 GB unified memory."
fi

mkdir -p "$HF_HOME" "$WORKDIR" "$FLASHINFER_CACHE_HOST"

if [ ! -w "$WORKDIR" ]; then
  echo "ERROR: WORKDIR is not writable: $WORKDIR"
  echo "Fix its ownership/permissions, or set WORKDIR to a user-writable directory in $ENV_FILE."
  exit 1
fi

if [ -e "$WORKDIR/proxy.py" ] && [ ! -w "$WORKDIR/proxy.py" ]; then
  echo "ERROR: Existing proxy is not writable: $WORKDIR/proxy.py"
  echo "This is commonly left behind by a previous sudo/root run."
  echo "Either fix ownership or use a fresh WORKDIR in $ENV_FILE."
  exit 1
fi

###############################################################################
# Write proxy
###############################################################################

cat > "$WORKDIR/proxy.py" <<'PY'
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

VLLM_HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=None)
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="vLLM OpenAI-compatible proxy",
    lifespan=lifespan,
)


def filter_request_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }

    headers.setdefault("authorization", "Bearer EMPTY")
    return headers


def filter_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def is_stream_request(body: bytes) -> bool:
    if not body:
        return False

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    return payload.get("stream") is True


async def forward(request: Request, upstream_path: str) -> Response:
    body = await request.body()
    url = f"{VLLM_BASE_URL}{upstream_path}"
    headers = filter_request_headers(request)

    if is_stream_request(body):
        upstream_request = app.state.http_client.build_request(
            method=request.method,
            url=url,
            content=body,
            headers=headers,
            params=request.query_params,
        )

        upstream = await app.state.http_client.send(
            upstream_request,
            stream=True,
        )

        async def stream_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=filter_response_headers(upstream),
            media_type=upstream.headers.get(
                "content-type",
                "text/event-stream",
            ),
        )

    upstream = await app.state.http_client.request(
        method=request.method,
        url=url,
        content=body,
        headers=headers,
        params=request.query_params,
    )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=filter_response_headers(upstream),
        media_type=upstream.headers.get(
            "content-type",
            "application/json",
        ),
    )


@app.get("/")
async def root():
    return {
        "service": "vLLM OpenAI-compatible proxy",
        "status": "ok",
        "upstream": VLLM_BASE_URL,
        "endpoints": [
            "/health",
            "/v1/models",
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/api/v1/chat",
            "/api/v1/chat/completions",
            "/api/v1/completions",
        ],
    }


@app.get("/health")
async def health():
    try:
        upstream = await app.state.http_client.get(
            f"{VLLM_BASE_URL}/health",
            timeout=5.0,
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "upstream_unavailable",
                "upstream": VLLM_BASE_URL,
                "error": str(exc),
            },
        )

    if not upstream.is_success:
        return JSONResponse(
            status_code=503,
            content={
                "status": "upstream_error",
                "upstream": VLLM_BASE_URL,
                "upstream_status": upstream.status_code,
            },
        )

    return {
        "status": "ok",
        "upstream": VLLM_BASE_URL,
        "upstream_status": upstream.status_code,
    }


# Standard OpenAI-compatible routes.

@app.get("/v1/models")
async def models(request: Request):
    return await forward(request, "/v1/models")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await forward(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await forward(request, "/v1/completions")


@app.post("/v1/responses")
async def responses(request: Request):
    return await forward(request, "/v1/responses")


# Compatibility aliases.

@app.post("/api/v1/chat")
async def api_chat(request: Request):
    return await forward(request, "/v1/chat/completions")


@app.post("/api/v1/chat/completions")
async def api_chat_completions(request: Request):
    return await forward(request, "/v1/chat/completions")


@app.post("/api/v1/completions")
async def api_completions(request: Request):
    return await forward(request, "/v1/completions")
PY

cat > "$WORKDIR/requirements.txt" <<'REQ'
fastapi
httpx
uvicorn[standard]
REQ

###############################################################################
# Prepare the patched Qwen3.8-Flash-Next vLLM image and model snapshot
###############################################################################

prepare_recipe() {
  if [ ! -d "$RECIPE_DIR/.git" ]; then
    echo "Cloning the single-Spark Qwen3.8-Flash-Next vLLM recipe..."
    git clone "$RECIPE_REPO_URL" "$RECIPE_DIR"
  fi

  # Qwen3.8-Flash-Next support is moving quickly. Always refresh the recipe so
  # an old local checkout cannot silently rebuild an obsolete base image.
  echo "Refreshing recipe checkout: $RECIPE_REF"
  git -C "$RECIPE_DIR" fetch --tags --prune origin

  if git -C "$RECIPE_DIR" rev-parse --verify --quiet "origin/$RECIPE_REF" >/dev/null; then
    git -C "$RECIPE_DIR" checkout -B "$RECIPE_REF" "origin/$RECIPE_REF" >/dev/null
  else
    git -C "$RECIPE_DIR" checkout --detach "$RECIPE_REF" >/dev/null
  fi

  RECIPE_COMMIT="$(git -C "$RECIPE_DIR" rev-parse HEAD)"
}

verify_image_support() {
  echo "Checking Qwen3.8-Flash-Next support inside $VLLM_IMAGE..."
  # Check vLLM directly. The recipe now uses qwen4_exp, while older preview
  # images use qwen3_8_flash_next. Transformers CONFIG_MAPPING is not required.
  if ! docker run --rm --entrypoint python3 "$VLLM_IMAGE" -c '
import importlib.util
import vllm
from vllm.model_executor.models.registry import ModelRegistry
arch = "Qwen4ExpForConditionalGeneration"
archs = ModelRegistry.get_supported_archs()
assert arch in archs, f"{arch} is missing from the vLLM model registry"
packages = ("vllm.models.qwen4_exp", "vllm.models.qwen3_8_flash_next")
model_package = None
for package in packages:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError as exc:
        if exc.name != package and not package.startswith(f"{exc.name}."):
            raise
        continue
    if spec is not None:
        model_package = package
        break
assert model_package is not None, f"Qwen model package is missing: {packages}"
print("vLLM", vllm.__version__)
print(arch, "registry: OK")
print(model_package, "model package: OK")
' ; then
    echo "ERROR: Qwen3.8-Flash-Next support check failed for $VLLM_IMAGE."
    echo "See the Python error above. The image needs the registered model and"
    echo "its model package. Rebuild from the selected single-Spark recipe."
    echo "Try: REBUILD_VLLM_IMAGE=1 $0"
    exit 1
  fi
}

prepare_image() {
  prepare_recipe

  local image_commit=""
  image_commit="$(docker image inspect \
    --format '{{ index .Config.Labels "org.qwen38.recipe_commit" }}' \
    "$VLLM_IMAGE" 2>/dev/null || true)"

  if [ "$REBUILD_VLLM_IMAGE" = "1" ] ||
     ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 ||
     [ "$image_commit" != "$RECIPE_COMMIT" ]; then
    if [ -n "$image_commit" ] && [ "$image_commit" != "$RECIPE_COMMIT" ]; then
      echo "Recipe changed: ${image_commit:0:12} -> ${RECIPE_COMMIT:0:12}"
    elif docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
      echo "Existing image has no current recipe label. Rebuilding once."
    fi

    echo "Building patched vLLM image: $VLLM_IMAGE"
    docker build --pull \
      --label "org.qwen38.recipe_commit=$RECIPE_COMMIT" \
      -t "$VLLM_IMAGE" "$RECIPE_DIR"
  else
    echo "Using current patched vLLM image: $VLLM_IMAGE"
  fi

  verify_image_support
}

download_model() {
  local repo_dir="$HF_HOME/hub/models--${MODEL_ID//\//--}"

  if [ "$DOWNLOAD_MODEL" = "1" ]; then
    echo "Downloading/checking model snapshot: $MODEL_ID"
    echo "The download is resumable and requires roughly 130 GB of free storage."

    docker rm -f qwen38-model-download >/dev/null 2>&1 || true

    docker run --rm \
      --name qwen38-model-download \
      -e HF_HOME=/hf \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
      -e HF_HUB_DISABLE_XET=1 \
      -v "$HF_HOME:/hf" \
      --entrypoint bash \
      "$VLLM_IMAGE" \
      -lc "hf download '$MODEL_ID' --max-workers 8"
  fi

  if [ ! -d "$repo_dir/snapshots" ]; then
    echo "ERROR: Model snapshot not found under:"
    echo "  $repo_dir"
    echo "Set DOWNLOAD_MODEL=1 or place the checkpoint in HF_HOME."
    exit 1
  fi
}

resolve_snapshot() {
  local repo_dir="$HF_HOME/hub/models--${MODEL_ID//\//--}"
  local revision=""
  local snap_host=""

  if [ -f "$repo_dir/refs/main" ]; then
    revision="$(tr -d '\r\n' < "$repo_dir/refs/main")"
    if [ -n "$revision" ] && [ -d "$repo_dir/snapshots/$revision" ]; then
      snap_host="$repo_dir/snapshots/$revision"
    fi
  fi

  if [ -z "$snap_host" ]; then
    snap_host="$(find "$repo_dir/snapshots" \
      -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | head -1 | cut -d' ' -f2- || true)"
  fi

  if [ -z "$snap_host" ] || [ ! -d "$snap_host" ]; then
    echo "ERROR: Could not resolve a local snapshot for $MODEL_ID."
    exit 1
  fi

  MODEL_SNAPSHOT_HOST="$snap_host"
  MODEL_SNAPSHOT_IN="/hf/hub/models--${MODEL_ID//\//--}/snapshots/$(basename "$snap_host")"
}

###############################################################################
# Start or reuse vLLM
###############################################################################

if [ "$RESTART_VLLM" = "1" ]; then
  prepare_image
  download_model
  resolve_snapshot

  echo "Removing any existing container named $CONTAINER_NAME..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  echo "Starting Qwen3.8-Flash-Next with vLLM."
  echo "Checkpoint: $MODEL_ID"
  echo "Served model name: $SERVED_MODEL_NAME"
  echo "Context: $MAX_MODEL_LEN"
  echo "MTP speculative tokens: $MTP_SPECULATIVE_TOKENS"
  echo "KV cache dtype: $KV_CACHE_DTYPE"
  echo "Engine-ready timeout: ${VLLM_ENGINE_READY_TIMEOUT_S}s"
  echo "FlashInfer JIT cache: $FLASHINFER_CACHE_HOST"
  echo "Internal endpoint: http://$VLLM_HOST:$VLLM_PORT"

  # The mmap PLE lookup performs a host-side gather and must stay outside full
  # CUDA graph capture. PIECEWISE capture keeps the rest of the graph optimized.
  SPLITTING_OPS='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'

  VLLM_ARGS=(
    "$MODEL_SNAPSHOT_IN"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --tensor-parallel-size 1
    --load-format safetensors
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --no-enable-prefix-caching
    --enable-chunked-prefill
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    "-cc.cudagraph_mode=PIECEWISE"
    "-cc.splitting_ops=$SPLITTING_OPS"
    --no-enable-flashinfer-autotune
  )

  if [ "$ENABLE_REASONING_PARSER" = "1" ]; then
    VLLM_ARGS+=(
      --reasoning-parser "$REASONING_PARSER"
    )
  fi

  if [ "$ENABLE_AUTO_TOOL_CHOICE" = "1" ]; then
    VLLM_ARGS+=(
      --enable-auto-tool-choice
      --tool-call-parser "$TOOL_CALL_PARSER"
    )
  fi

  if [ -n "$DEFAULT_CHAT_TEMPLATE_KWARGS" ]; then
    VLLM_ARGS+=(
      --default-chat-template-kwargs
      "$DEFAULT_CHAT_TEMPLATE_KWARGS"
    )
  fi

  if (( MTP_SPECULATIVE_TOKENS > 0 )); then
    VLLM_ARGS+=(
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPECULATIVE_TOKENS}}"
    )
  fi

  docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --ipc=host \
    --shm-size 16g \
    --network host \
    --restart no \
    -e HF_HOME=/hf \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_PLE_MMAP=1 \
    -e VLLM_PLE_MMAP_WORKERS="$PLE_MMAP_WORKERS" \
    -e VLLM_PLE_MMAP_PREWARM="$PLE_MMAP_PREWARM" \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=0 \
    -e VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S" \
    -e FLASHINFER_CACHE_DIR=/root/.cache/flashinfer \
    -v "$HF_HOME:/hf" \
    -v "$FLASHINFER_CACHE_HOST:/root/.cache/flashinfer" \
    "$VLLM_IMAGE" \
    "${VLLM_ARGS[@]}"
else
  echo "RESTART_VLLM=0."
  echo "Keeping the existing vLLM process and loaded weights."

  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "ERROR: Container $CONTAINER_NAME is not running."
    echo "Run once with RESTART_VLLM=1."
    exit 1
  fi
fi

###############################################################################
# Wait for vLLM
###############################################################################

echo "Waiting for vLLM to become ready..."
WAIT_ATTEMPTS=$(( (STARTUP_WAIT_SECONDS + 1) / 2 ))

for attempt in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    echo "vLLM is ready."
    # Enable persistence only after successful initialization. During startup,
    # automatic restart would hide the first failure in a crash loop.
    docker update --restart unless-stopped "$CONTAINER_NAME" >/dev/null
    break
  fi

  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "ERROR: vLLM container exited during startup."
    echo
    docker logs --tail 250 "$CONTAINER_NAME" || true
    exit 1
  fi

  if [ "$attempt" -eq "$WAIT_ATTEMPTS" ]; then
    echo "ERROR: vLLM did not become ready within ${STARTUP_WAIT_SECONDS}s."
    echo
    docker logs --tail 250 "$CONTAINER_NAME" || true
    exit 1
  fi

  if (( attempt % 15 == 0 )); then
    echo "Still waiting. Recent vLLM logs:"
    docker logs --tail 12 "$CONTAINER_NAME" || true
  fi

  sleep 2
done

###############################################################################
# Prepare proxy environment
###############################################################################

echo "Preparing proxy environment..."

if [ ! -x "$WORKDIR/.venv/bin/python" ]; then
  python3 -m venv "$WORKDIR/.venv"
fi

"$WORKDIR/.venv/bin/python" -m pip install \
  --upgrade pip >/dev/null

"$WORKDIR/.venv/bin/python" -m pip install \
  -r "$WORKDIR/requirements.txt" >/dev/null

###############################################################################
# Stop previous proxy
###############################################################################

if [ -f "$PROXY_PID_FILE" ]; then
  OLD_PROXY_PID="$(cat "$PROXY_PID_FILE" 2>/dev/null || true)"

  if [ -n "$OLD_PROXY_PID" ] &&
     kill -0 "$OLD_PROXY_PID" >/dev/null 2>&1; then
    echo "Stopping previous proxy process $OLD_PROXY_PID..."
    kill "$OLD_PROXY_PID" >/dev/null 2>&1 || true

    for _ in $(seq 1 20); do
      if ! kill -0 "$OLD_PROXY_PID" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done

    if kill -0 "$OLD_PROXY_PID" >/dev/null 2>&1; then
      echo "Previous proxy did not stop cleanly. Sending SIGKILL."
      kill -9 "$OLD_PROXY_PID" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "$PROXY_PID_FILE"
fi

if ss -ltn 2>/dev/null | grep -qE "[:.]${PUBLIC_PORT}[[:space:]]"; then
  echo "ERROR: Port $PUBLIC_PORT is already in use."
  echo "Inspect it with:"
  echo "  ss -ltnp | grep ':$PUBLIC_PORT'"
  exit 1
fi

###############################################################################
# Start persistent proxy
###############################################################################

echo "Starting proxy in the background..."

cd "$WORKDIR"

nohup env \
  VLLM_HOST="$VLLM_HOST" \
  VLLM_PORT="$VLLM_PORT" \
  "$WORKDIR/.venv/bin/uvicorn" proxy:app \
  --host "$PUBLIC_HOST" \
  --port "$PUBLIC_PORT" \
  > "$PROXY_LOG_FILE" 2>&1 < /dev/null &

PROXY_PID=$!
echo "$PROXY_PID" > "$PROXY_PID_FILE"

for attempt in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PUBLIC_PORT/health" >/dev/null 2>&1; then
    break
  fi

  if ! kill -0 "$PROXY_PID" >/dev/null 2>&1; then
    echo "ERROR: Proxy exited during startup."
    echo
    tail -n 100 "$PROXY_LOG_FILE" || true
    rm -f "$PROXY_PID_FILE"
    exit 1
  fi

  if [ "$attempt" -eq 30 ]; then
    echo "ERROR: Proxy started but did not become healthy."
    echo
    tail -n 100 "$PROXY_LOG_FILE" || true
    exit 1
  fi

  sleep 1
done

SPARK_IP="$(hostname -I | awk '{print $1}')"

###############################################################################
# Summary
###############################################################################

echo
echo "Service started successfully."
echo
echo "Model:"
echo "  $SERVED_MODEL_NAME"
echo
echo "Checkpoint:"
echo "  $MODEL_ID"
echo
echo "OpenAI-compatible base URL:"
echo "  http://$SPARK_IP:$PUBLIC_PORT/v1"
echo
echo "Chat Completions:"
echo "  http://$SPARK_IP:$PUBLIC_PORT/v1/chat/completions"
echo
echo "Legacy Completions:"
echo "  http://$SPARK_IP:$PUBLIC_PORT/v1/completions"
echo
echo "Compatibility alias:"
echo "  http://$SPARK_IP:$PUBLIC_PORT/api/v1/chat/completions"
echo
echo "Health:"
echo "  http://$SPARK_IP:$PUBLIC_PORT/health"
echo
echo "Proxy PID:"
echo "  $PROXY_PID"
echo
echo "Proxy log:"
echo "  $PROXY_LOG_FILE"
echo
echo "Useful commands:"
echo "  tail -f '$PROXY_LOG_FILE'"
echo "  docker logs -f '$CONTAINER_NAME'"
echo "  kill \"\$(cat '$PROXY_PID_FILE')\""
echo "  docker stop '$CONTAINER_NAME'"

