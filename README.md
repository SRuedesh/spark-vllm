# Qwen3.8-Flash-Next on DGX Spark

Start vLLM and an OpenAI-compatible HTTP proxy on one DGX Spark.
The launcher builds the selected Spark recipe and uses its PLE memory-map patch.

## Setup

1. Install Docker with GPU access, Git, curl, Python 3 with venv support, and the NVIDIA driver. The launcher also uses standard Linux tools, including `ss`.
2. Copy the example configuration:

   ```bash
   cp .qwen38-flash-next.env.example .qwen38-flash-next.env
   ```

3. Set `HF_TOKEN` in `.qwen38-flash-next.env`. Use `DOWNLOAD_MODEL=0` only when the model snapshot is already in `HF_HOME`.
4. Start the launcher from this directory:

   ```bash
   bash ./run-qwen-38-flash-next.sh
   ```

5. Check the service:

   ```bash
   curl --fail http://127.0.0.1:1234/health
   curl --fail http://127.0.0.1:1234/v1/models
   ```

The client base URL is `http://<spark-host>:1234/v1`.
The proxy binds to all interfaces by default and has no access control.
Set `PUBLIC_HOST=127.0.0.1` for local access only.

## Configuration

1. Use `ENV_FILE=/path/to/config.env` to select another configuration file.
2. Use `QWEN38_VLLM_IMAGE` to set the image name. The launcher ignores legacy `VLLM_IMAGE` overrides.
3. Set `REBUILD_VLLM_IMAGE=1` in the configuration file to force a build. Reset it after use.
4. Set `RESTART_VLLM=0` to retain the loaded model and restart the proxy.
5. Keep `STARTUP_WAIT_SECONDS` greater than `VLLM_ENGINE_READY_TIMEOUT_S`.
6. Use a commit ID for `RECIPE_REF` when you need a fixed recipe revision. The default `main` can change.

The configuration file is sourced after inherited environment variables. Its assignments take precedence over command-line variable assignments.
