![MiniMax H3 Singularity — one model to rule them all](https://raw.githubusercontent.com/huchukato/runpod-minimax-h3/main/media/banner-singularity.jpg)

# MiniMax H3 Turbo — ComfyUI Serverless Worker

[![Runpod](https://api.runpod.io/badge/huchukato/runpod-minimax-h3)](https://console.runpod.io/hub)

A production-ready [Runpod Serverless](https://docs.runpod.io/serverless/overview) worker running **MiniMax H3** (Comfy-Org INT8 ConvRot) on ComfyUI — text/image/first-last-frame/reference-to-video **with audio**, multi-scene **storyboards via the Director**, turbo LoRAs, TensorRT RIFE + upscaling, wildcards and an uncensored Qwen3-VL enhancer. Built for [ForgeHub](https://github.com/huchukato/ForgeHub), but the endpoint speaks plain Runpod API.

## Deploy

1. **Storage first**: create a [network volume](https://console.runpod.io/user/storage) (≥160GB) in a datacenter with 96GB Pro GPUs — e.g. `US-NE-1`.
2. Click **Deploy** → endpoint → **Edit** → Max workers ≥ 1 + attach your volume. Workers auto-download ~100GB of models on first boot (once). No volume = every job fails on missing models.
3. Call it via the Runpod API or ForgeHub below — workflows, wildcards, job queue and outputs included; paste your API key and it detects the endpoint by itself.

[![Download ForgeHub](https://raw.githubusercontent.com/huchukato/runpod-minimax-h3/main/media/forgehub-release.jpeg)](https://github.com/huchukato/ForgeHub/releases/latest)

## Workflows included

Templates live in [workflows-api/h3/](workflows-api/h3) (ComfyUI API format) and are baked into `/opt/workflows`.

| Workflow | Description |
| --- | --- |
| `video_minimax_h3_r2v` | Reference images → video + audio (INT8 ref2va, optional 4-step turbo LoRA) |

## Requirements

- **GPU**: 96GB class (`BLACKWELL_96` pool). The 32B text encoder + video DiT + TensorRT engines need the headroom.
- **Network volume**: ≥160GB at `/runpod-volume`. Models auto-download at boot from `models/all.txt` (~100GB); pick a subset with `VOLUME_MANIFESTS="ref2va"`. Disable with `VOLUME_AUTOPOPULATE=false`.
- **CUDA**: 13.0+ hosts.

## Model manifests

`models/*.txt` — tab-separated: `<models subdir> <filename> <URL> [sha256] [size_bytes]` (`-` = unset).

| Manifest | Contents |
| --- | --- |
| `all` | Everything (default for auto-provisioning) |
| `ref2va` | What `video_minimax_h3_r2v` loads |
| `t2va` | fl2va DiT, Heretic text encoder, fp16 video VAE, turbo + realism LoRAs |

Validation:

- **Image build** (`manifest_tools.py static` + `workflows`): rejects malformed lines, unknown subdirs, non-HF/non-https URLs, bad sha256/size values, conflicting duplicates; HEADs every URL and rejects failed downloads or a remote size that differs from the manifest; fails when a workflow loads a model no manifest lists. Offline builds: `--build-arg SKIP_NET_VALIDATION=1`.
- **Volume** (`populate-volume.sh`): a file that is missing, empty or the wrong size is (re)downloaded; every download is verified (size + sha256, 3 attempts, bad files deleted); the script exits non-zero if anything fails. `VERIFY_HASH=true` also hashes files already present.
- **Job**: before queuing, the handler checks that every model the workflow loads exists on the volume.

Check manually: `python manifest_tools.py static models/*.txt` · `python manifest_tools.py workflows workflows-api/h3 models/all.txt` · `python manifest_tools.py verify /runpod-volume/models models/ref2va.txt --hash`. Tests: `python -m unittest discover -s tests`.

## Job input

Inputs are mapped by **node title**:

```json
{
  "input": {
    "workflow": "video_minimax_h3_r2v",
    "inputs": {
      "Input Text (Prompt)": "A slow camera arc around the subject...",
      "INPUT - Frame Count": 121,
      "Resolution Selector (Size)": {"aspect_ratio": "9:16 (Portrait Widescreen)", "megapixels": 0.4},
      "Seed": 12345,
      "Reference Image": ["<base64 or https URL>", "<second image>"]
    }
  }
}
```

- A scalar sets the node's `value`; an object is merged into the node's inputs. Any other node title works the same way.
- `Seed`: omit or `-1` for random (the used seed is returned). Falls back to `RandomNoise` nodes if none is titled `Seed`.
- `Reference Image`: one image or a list, filled into `LoadImage` nodes in id order; unused slots are removed from the graph. More images than slots is an error.
- `INPUT - Frame Count`: the current template has no such node, so it is reported in `warnings` and skipped. Titles that match no node are skipped the same way.
- Invalid values return `{"error": "..."}`. Success returns `{"outputs": [...], "seed": N, "warnings": [...]}`.
- `{"action": "health"}` — status, workflow and manifest lists, no generation.

### Provision only some models

```json
{"input": {"models_manifest": "ref2va", "models_only": true}}
```

`models_manifest` accepts a manifest name, a list of names, inline manifest text, inline rows `[["loras", "x.safetensors", "https://huggingface.co/...", "<sha256>", 123]]`, or `{"include": ..., "lines": ...}`. Inline entries are restricted to https Hugging Face URLs. Without `models_only` the models are provisioned and then the job runs.

Client: [exec/h3-serverless.ps1](exec/h3-serverless.ps1) is a generic runner — all parameters live in a job file such as [exec/ref2va_01.json](exec/ref2va_01.json) (`endpointId`, `outputDir`, `timeoutMinutes`, `input`). Strings in `input` that start with `@` are replaced by that file's base64. Run `$env:RUNPOD_API_KEY="..."; .\exec\h3-serverless.ps1 .\exec\ref2va_01.json` (`-EndpointId` or `RUNPOD_ENDPOINT_ID` can supply the endpoint instead).

## Custom nodes baked in

`ComfyUI-VideoHelperSuite` · `ComfyUI-Easy-Use` · `ComfyUI-Impact-Pack` · `ComfyUI-Impact-Subpack` · `ComfyUI-QwenVL-Mod` · `ComfyUI-MiniMaxH3-Director-Enhanced` · `ComfyUI-RIFE-TensorRT-Auto` · `ComfyUI-Upscaler-TensorRT-Auto` · `ComfyUI-TagForge` · `ComfyUI-PerfectVideoResolution` · `ComfyUI-Pixaroma` · `ComfyUI-KJNodes`

## Companion stack

- 🖼 Image sibling: [huchukato/runpod-qwen21](https://github.com/huchukato/runpod-qwen21) (Qwen Image 2.1 T2I/Edit + Pony)
