# runpod-minimax-h3 — MiniMax H3 INT8 serverless worker (CUDA 13.0)
#
# - Comfy-Org INT8 ConvRot DiTs + Qwen3-VL-32B Heretic NVFP4 TE + lightx2v
#   turbo LoRAs, loaded from the network volume (~100GB, needs a 48GB+ GPU)
# - Missing volume files are auto-downloaded at worker boot from
#   models/all.txt (populate-volume.sh, VOLUME_AUTOPOPULATE=false to skip)
# - runpod serverless handler drives ComfyUI per-job (handler.py)
#

FROM huchukato/comfyui-base:cu130

# ──────────────────────────────────────────────────────────────────────────────
# Custom nodes — same stack as the pod image
# ──────────────────────────────────────────────────────────────────────────────
ENV GIT_TERMINAL_PROMPT=0
RUN dl() { \
    local user="$1" repo="$2" branch="${3:-main}" dir="$4"; \
    echo "Cloning $user/$repo ($branch) -> $dir"; \
    git clone --depth 1 --branch "$branch" "https://github.com/$user/$repo.git" "/opt/comfyui-baked/custom_nodes/$dir"; \
  } && \
    cd /opt/comfyui-baked/custom_nodes && \
    dl Kosinkadink ComfyUI-VideoHelperSuite main ComfyUI-VideoHelperSuite && \
    dl yolain ComfyUI-Easy-Use main ComfyUI-Easy-Use && \
    dl huchukato ComfyUI-QwenVL-Mod main ComfyUI-QwenVL-Mod && \
    dl huchukato ComfyUI-RIFE-TensorRT-Auto master ComfyUI-RIFE-TensorRT-Auto && \
    dl huchukato ComfyUI-Upscaler-TensorRT-Auto master ComfyUI-Upscaler-TensorRT-Auto && \
    dl huchukato ComfyUI-TagForge v2.3.0 ComfyUI-TagForge && \
    dl huchukato ComfyUI-PerfectVideoResolution master ComfyUI-PerfectVideoResolution && \
    dl huchukato ComfyUI-MiniMaxH3-Director-Enhanced main ComfyUI-MiniMaxH3-Director-Enhanced && \
    dl pixaroma ComfyUI-Pixaroma main ComfyUI-Pixaroma && \
    dl kijai ComfyUI-KJNodes main ComfyUI-KJNodes && \
    dl ltdrdata ComfyUI-Impact-Pack Main ComfyUI-Impact-Pack && \
    dl ltdrdata ComfyUI-Impact-Subpack main ComfyUI-Impact-Subpack


# ──────────────────────────────────────────────────────────────────────────────
# Requirements + runpod serverless SDK
# ──────────────────────────────────────────────────────────────────────────────
# libgl1/libglib2.0-0: opencv-python (VideoHelperSuite) fails to import without
# them → the whole pack never registers → missing_node_type on VHS_* nodes.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN bash -c 'cd /opt/comfyui-baked/custom_nodes && for node_dir in */; do \
        if [ -f "$node_dir/requirements.txt" ]; then \
            echo "Installing requirements for $node_dir..." && \
            pip install --no-cache-dir -r "$node_dir/requirements.txt" || echo "Failed to install requirements for $node_dir"; \
        fi; \
    done' && \
    pip install --no-cache-dir --no-deps "transformers>=5.2.0" && \
    pip install --no-cache-dir 'runpod>=1.12' requests boto3 "huggingface_hub[cli]" hf_transfer && \
    pip cache purge

# ──────────────────────────────────────────────────────────────────────────────
# Model dirs (empty — real models come from the network volume via
# extra_model_paths.yaml written by the entrypoint)
# ──────────────────────────────────────────────────────────────────────────────
RUN mkdir -p /opt/comfyui-baked/models/{vae,vae_approx,diffusion_models,unet,text_encoders,clip_projections,loras,checkpoints,LLM,clip,clip_vision,ultralytics/bbox,sams}

ENV HF_TOKEN=""

# ──────────────────────────────────────────────────────────────────────────────
# Serverless entrypoint + handler + workflow templates (API format)
# ──────────────────────────────────────────────────────────────────────────────
COPY handler.py manifest_tools.py /opt/serverless/
COPY serverless-entrypoint.sh /opt/serverless/entrypoint.sh
COPY populate-volume.sh /opt/serverless/populate-volume.sh
COPY models/ /opt/serverless/models/
COPY workflows-api/h3/ /opt/workflows/
RUN sed -i 's/\r$//' /opt/serverless/entrypoint.sh /opt/serverless/populate-volume.sh && \
    chmod +x /opt/serverless/entrypoint.sh /opt/serverless/populate-volume.sh

# Build-time validation: manifests well-formed, every URL resolves with the
# expected size, and every model a workflow loads is listed in a manifest.
# (Real byte/sha256 checks run on the volume at populate time — ~100GB of
# models can't be downloaded during the image build.)
# Offline builds: --build-arg SKIP_NET_VALIDATION=1
ARG SKIP_NET_VALIDATION=0
RUN python3 /opt/serverless/manifest_tools.py workflows /opt/workflows /opt/serverless/models/all.txt && \
    if [ "$SKIP_NET_VALIDATION" = "1" ]; then NET=--no-network; else NET=; fi && \
    python3 /opt/serverless/manifest_tools.py static $NET /opt/serverless/models/*.txt

ENV COMFYUI_DIR="/workspace/runpod-slim/ComfyUI" \
    WORKFLOWS_DIR="/opt/workflows" \
    JOB_TIMEOUT_S="1800" \
    VENV_SUFFIX="cu130"

# Base image carries ENTRYPOINT ["/start.sh"] (pod mode) — a bare CMD would be
# passed to it as args and the runpod handler would never start. Override the
# entrypoint so the container runs the serverless handler directly.
ENTRYPOINT ["/bin/bash", "/opt/serverless/entrypoint.sh"]
