"""RunPod serverless handler for the MiniMax-H3 ComfyUI stack.

Boots ComfyUI once per worker; each job picks a workflow template (API format,
/opt/workflows) and fills it by *node title*, so callers never need node ids.

Job input:
  {
    "workflow": "video_minimax_h3_r2v",              # file in /opt/workflows (.json optional)
    "inputs": {                                       # node title -> value
      "Input Text (Prompt)": "...",
      "INPUT - Frame Count": 121,
      "Resolution Selector (Size)": {"aspect_ratio": "9:16 (Portrait Widescreen)", "megapixels": 0.4},
      "Seed": 12345,                                  # -1 / omitted = random
      "Reference Image": ["https://... or base64"]    # one or several, filled in node-id order
    },
    "models_manifest": "ref2va",                      # optional, see below
    "models_only": false                              # true = only provision models, no generation
  }

Rules for `inputs`:
  - A scalar sets the node's `value` input; an object is merged into the node's inputs.
  - `Seed` falls back to RandomNoise nodes and `Reference Image` to LoadImage nodes
    when no node carries that exact title.
  - A title that matches no node is reported in `warnings` and skipped.

`models_manifest` provisions models onto the volume before the job runs:
  "ref2va"                          named manifest (models/<name>.txt)
  ["ref2va", "t2va"]                several named manifests
  "diffusion_models\\tfoo.safetensors\\thttps://huggingface.co/..."   inline manifest text
  [["loras", "x.safetensors", "https://huggingface.co/...", "<sha256>", 123]]   inline rows
  {"include": "ref2va", "lines": [...]}   both

{"action": "health"} reports status without touching ComfyUI.
"""

import asyncio
import base64
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import manifest_tools

SERVERLESS_DIR = Path(__file__).resolve().parent
COMFY_DIR = os.environ.get("COMFYUI_DIR", "/workspace/runpod-slim/ComfyUI")
COMFY_URL = "http://127.0.0.1:8188"
WORKFLOWS_DIR = Path(os.environ.get("WORKFLOWS_DIR", "/opt/workflows"))
MANIFESTS_DIR = Path(os.environ.get("MANIFESTS_DIR", SERVERLESS_DIR / "models"))
VOL_ROOT = Path(os.environ.get("VOL_ROOT", "/runpod-volume"))
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "1800"))

# Output delivery: base64 inline under these limits, S3 offload above them.
# RunPod's job-done callback drops results over ~20MB, so keep headroom.
INLINE_MAX = int(os.environ.get("OUTPUT_INLINE_MAX_MB", "10")) * 1024 * 1024
INLINE_BUDGET = int(os.environ.get("OUTPUT_INLINE_BUDGET_MB", "15")) * 1024 * 1024
S3_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "outputs")

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
SEED_MAX = 2**53 - 1
MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".gif", ".wav", ".mp3", ".flac")


class JobError(RuntimeError):
    """Invalid job input — reported to the caller as-is."""


def _http(path, payload=None, timeout=30):
    url = COMFY_URL + path
    try:
        if payload is None:
            return urllib.request.urlopen(url, timeout=timeout).read()
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout).read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"ComfyUI {path} → HTTP {e.code}: {body}") from e


_BOOT_DONE = threading.Event()
_BOOT_ERR = None


def _boot_comfy():
    # Extra flags come from COMFYUI_EXTRA_ARGS (space-separated). Keep the
    # default empty: --fast/--use-sage-attention/--async-offload corrupt the
    # int8-convrot Qwen/MMh3 stacks (flat garbage output).
    cmd = [
        "python3", "main.py",
        "--listen", "127.0.0.1",
        "--port", "8188",
        "--disable-auto-launch",
        "--extra-model-paths-config",
        str(Path(COMFY_DIR) / "extra_model_paths.yaml"),
    ] + os.environ.get("COMFYUI_EXTRA_ARGS", "").split()
    comfy_log = open("/workspace/comfy.log", "ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=COMFY_DIR, stdout=comfy_log, stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("ComfyUI exited during boot")
        try:
            _http("/system_stats", timeout=5)
            return proc
        except Exception:
            time.sleep(2)
    raise RuntimeError("ComfyUI did not become ready in 600s")


def _boot_in_bg():
    # ComfyUI boots on a daemon thread so the runpod handler is live (and the
    # `health` action answers) even while ComfyUI is still starting — or on
    # machines where it can't start at all (e.g. Hub test pods without the
    # network volume). Real jobs wait on _BOOT_DONE via _ensure_comfy().
    global _BOOT_ERR
    try:
        _boot_comfy()
    except Exception as exc:
        _BOOT_ERR = exc
        print(f"[handler] ComfyUI boot failed: {exc}", flush=True)
    finally:
        _BOOT_DONE.set()


async def _ensure_comfy():
    await asyncio.to_thread(_BOOT_DONE.wait)
    if _BOOT_ERR is not None:
        raise RuntimeError(f"ComfyUI failed to boot: {_BOOT_ERR} (see /workspace/comfy.log)")


# ── Media ────────────────────────────────────────────────────────────────────

def _media_bytes(value):
    if not isinstance(value, str) or not value.strip():
        raise JobError("image values must be a non-empty URL or base64 string")
    value = value.strip()
    if value.startswith(("http://", "https://")):
        return urllib.request.urlopen(value, timeout=120).read()
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        data = base64.b64decode(value, validate=False)
    except Exception as exc:
        raise JobError(f"image is not valid base64: {exc}") from exc
    if not data:
        raise JobError("image decoded to zero bytes")
    return data


def _upload(filename, data):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        COMFY_URL + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return resp.get("name", filename)


# ── Workflow loading + title-keyed input mapping ────────────────────────────

def _load_workflow(name):
    """Workflow templates must be saved in ComfyUI *API format* (Export API),
    where every node is keyed by id and inputs carry widget names."""
    if not isinstance(name, str) or not SAFE_NAME.match(name):
        raise JobError("'workflow' must be a template file name")
    path = WORKFLOWS_DIR / name
    if not path.is_file():
        path = WORKFLOWS_DIR / f"{name}.json"
    if not path.is_file():
        raise JobError(f"workflow template not found: {name} (available: {_workflow_names()})")
    wf = json.loads(path.read_text(encoding="utf-8"))
    if "nodes" in wf:
        raise JobError(f"{name} is a UI-format graph; re-export it with 'Export API'")
    return wf


def _workflow_names():
    return sorted(p.name for p in WORKFLOWS_DIR.glob("*.json"))


def _node_title(node):
    return str((node.get("_meta") or {}).get("title") or "")


def _nodes_by_title(prompt, title):
    exact = [i for i, n in prompt.items() if _node_title(n) == title]
    if exact:
        return sorted(exact, key=_id_key)
    low = title.lower()
    return sorted((i for i, n in prompt.items() if _node_title(n).lower() == low), key=_id_key)


def _nodes_by_class(prompt, class_type):
    return sorted((i for i, n in prompt.items() if n.get("class_type") == class_type), key=_id_key)


def _id_key(node_id):
    return (0, int(node_id)) if str(node_id).isdigit() else (1, str(node_id))


def _as_int(title, value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise JobError(f"'{title}' must be an integer")
    try:
        number = int(float(value)) if float(value) == int(float(value)) else None
    except (ValueError, OverflowError):
        number = None
    if number is None:
        raise JobError(f"'{title}' must be an integer, got {value!r}")
    if minimum is not None and number < minimum:
        raise JobError(f"'{title}' must be >= {minimum}")
    return number


def _set_value(prompt, node_ids, title, value):
    for node_id in node_ids:
        node_inputs = prompt[node_id].setdefault("inputs", {})
        if isinstance(value, dict):
            node_inputs.update(value)
            continue
        if "value" in node_inputs:
            key = "value"
        else:
            literals = [k for k, v in node_inputs.items() if not isinstance(v, list)]
            if len(literals) != 1:
                raise JobError(f"'{title}' (node {node_id}) has no single value input; pass an object instead")
            key = literals[0]
        node_inputs[key] = value


def _prune_nodes(prompt, node_ids):
    """Drop nodes and every input link that pointed at them."""
    gone = {str(i) for i in node_ids}
    for i in gone:
        prompt.pop(i, None)
    for node in prompt.values():
        inputs = node.get("inputs", {})
        for key in [k for k, v in inputs.items()
                    if isinstance(v, list) and v and str(v[0]) in gone]:
            del inputs[key]


def _apply_reference_images(prompt, title, value, upload, job_id):
    loaders = _nodes_by_title(prompt, title) or _nodes_by_class(prompt, "LoadImage")
    images = value if isinstance(value, list) else [value]
    images = [v for v in images if v not in (None, "")]
    if len(images) > len(loaders):
        raise JobError(f"'{title}': {len(images)} images given but the workflow has {len(loaders)} image slot(s)")
    for i, (node_id, item) in enumerate(zip(loaders, images)):
        prompt[node_id].setdefault("inputs", {})["image"] = upload(f"{job_id}_ref{i}.png", _media_bytes(item))
    _prune_nodes(prompt, loaders[len(images):])
    return len(images)


def apply_inputs(prompt, inputs, upload, job_id="job"):
    """Fill a workflow in place from {node title: value}.

    Returns (warnings, seed). Raises JobError on invalid values."""
    if not isinstance(inputs, dict):
        raise JobError("'inputs' must be an object mapping node titles to values")
    warnings, seed = [], None
    handled_seed = False

    for title, value in inputs.items():
        key = title.strip().lower()

        if key == "seed":
            handled_seed = True
            nodes = _nodes_by_title(prompt, title) or _nodes_by_class(prompt, "RandomNoise")
            if not nodes:
                warnings.append(f"no node found for '{title}' — skipped")
                continue
            seed = _as_int(title, value, -1) if value not in (None, "") else -1
            if seed == -1:
                seed = random.randint(0, SEED_MAX)
            elif seed < 0:
                raise JobError(f"'{title}' must be >= 0 (or -1 for random)")
            for node_id in nodes:
                prompt[node_id].setdefault("inputs", {})["noise_seed"] = seed
            continue

        if key == "reference image":
            if value is None:
                continue
            if not (_nodes_by_title(prompt, title) or _nodes_by_class(prompt, "LoadImage")):
                warnings.append(f"no node found for '{title}' — skipped")
                continue
            _apply_reference_images(prompt, title, value, upload, job_id)
            continue

        nodes = _nodes_by_title(prompt, title)
        if not nodes:
            warnings.append(f"no node found for '{title}' — skipped")
            continue

        if key == "input - frame count":
            value = _as_int(title, value, 1)
        elif key == "input text (prompt)":
            if not isinstance(value, str) or not value.strip():
                raise JobError(f"'{title}' must be a non-empty string")
        elif key == "resolution selector (size)":
            if not isinstance(value, dict):
                raise JobError(f"'{title}' must be an object like {{\"aspect_ratio\": \"16:9 (Widescreen)\", \"megapixels\": 0.4}}")
            if "aspect_ratio" in value and not isinstance(value["aspect_ratio"], str):
                raise JobError(f"'{title}'.aspect_ratio must be a string")
            for field in ("megapixels", "multiple"):
                if field in value and (isinstance(value[field], bool) or not isinstance(value[field], (int, float)) or value[field] <= 0):
                    raise JobError(f"'{title}'.{field} must be a positive number")
        _set_value(prompt, nodes, title, value)

    if not handled_seed:
        # An unchanged seed would hit ComfyUI's result cache and skip execution.
        for node_id in _nodes_by_class(prompt, "RandomNoise"):
            seed = random.randint(0, SEED_MAX)
            prompt[node_id].setdefault("inputs", {})["noise_seed"] = seed
    return warnings, seed


def _check_models_present(prompt):
    """Fail fast with the missing file names instead of a ComfyUI validation dump."""
    if not (VOL_ROOT / "models").is_dir():
        return
    missing = [f"{'/'.join(subdirs)}/{name}" for name, subdirs in manifest_tools.prompt_models(prompt)
               if not any((VOL_ROOT / "models" / s / name).is_file() for s in subdirs)]
    if missing:
        raise JobError("models missing on the volume (send models_manifest to provision them): "
                       + ", ".join(sorted(set(missing))))


# ── Model provisioning ───────────────────────────────────────────────────────

def _resolve_manifest_spec(spec, tmpdir):
    """Turn a job's `models_manifest` into a list of manifest file paths."""
    paths, rows, includes = [], [], []

    def include(name):
        if not isinstance(name, str) or not SAFE_NAME.match(name):
            raise JobError(f"invalid manifest name: {name!r}")
        path = MANIFESTS_DIR / f"{name}.txt"
        if not path.is_file():
            avail = sorted(p.stem for p in MANIFESTS_DIR.glob("*.txt"))
            raise JobError(f"unknown manifest '{name}' (available: {avail})")
        includes.append(str(path))

    def row_text(row):
        if not isinstance(row, (list, tuple)) or not 3 <= len(row) <= 5:
            raise JobError("manifest rows must be [subdir, filename, url, sha256?, size?]")
        return "\t".join(str(c) for c in row)

    def add(item):
        if isinstance(item, str):
            if "\t" in item or "\n" in item:
                rows.append(item)
            else:
                include(item.strip())
        elif isinstance(item, list) and item and all(isinstance(r, (list, tuple)) for r in item):
            rows.extend(row_text(r) for r in item)
        elif isinstance(item, list):
            for sub in item:
                add(sub)
        elif isinstance(item, dict):
            unknown = set(item) - {"include", "lines"}
            if unknown:
                raise JobError(f"unknown models_manifest keys: {sorted(unknown)}")
            for name in ([item["include"]] if isinstance(item.get("include"), str) else item.get("include") or []):
                include(name)
            if item.get("lines"):
                add(item["lines"])
        else:
            raise JobError("models_manifest must be a manifest name, list, inline text or object")

    add(spec)
    paths.extend(includes)
    if rows:
        inline = Path(tmpdir) / "inline.txt"
        inline.write_text("\n".join(rows) + "\n", encoding="utf-8")
        paths.append(str(inline))
    if not paths:
        raise JobError("models_manifest is empty")
    return paths


def _provision_models(spec):
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = _resolve_manifest_spec(spec, tmpdir)
        _, errors = manifest_tools.load_manifests(paths)
        if errors:
            raise JobError("invalid models_manifest: " + "; ".join(errors))
        proc = subprocess.run(
            [str(SERVERLESS_DIR / "populate-volume.sh"), *paths],
            capture_output=True, text=True, env={**os.environ, "VOL_ROOT": str(VOL_ROOT)})
        tail = (proc.stdout + proc.stderr)[-1500:]
        print(f"[handler] populate rc={proc.returncode}\n{tail}", flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"model provisioning failed: {tail}")
        entries, _ = manifest_tools.load_manifests(paths)
        return sorted(f"{e.subdir}/{e.name}" for e in entries)


# ── ComfyUI execution + outputs ─────────────────────────────────────────────

def _interrupt_prompt(prompt_id):
    """Best-effort ComfyUI abort for a cancelled job: dequeue if pending,
    interrupt if mid-flight."""
    for path, payload in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {})):
        try:
            _http(path, payload, timeout=5)
        except Exception as exc:
            print(f"[handler] {path} failed during cancel: {exc}", flush=True)


async def _queue_and_wait(prompt):
    # Async + awaited sleeps: the runpod SDK stops a job by cancelling its
    # task, so the CancelledError can only be delivered at an await — a sync
    # handler would freeze the event loop and never see the stop signal.
    client_id = uuid.uuid4().hex
    resp = json.loads(await asyncio.to_thread(
        _http, "/prompt", {"prompt": prompt, "client_id": client_id}, 60))
    prompt_id = resp["prompt_id"]
    deadline = time.time() + JOB_TIMEOUT_S
    try:
        while time.time() < deadline:
            hist = json.loads(await asyncio.to_thread(_http, f"/history/{prompt_id}", None, 30))
            entry = hist.get(prompt_id)
            if entry and entry.get("status", {}).get("status_str") == "error":
                msgs = [str(m) for m in entry.get("status", {}).get("messages", [])]
                raise RuntimeError(f"ComfyUI job failed: {' | '.join(msgs)[:800]}")
            if entry and entry.get("status", {}).get("completed"):
                return entry
            await asyncio.sleep(3)
    except asyncio.CancelledError:
        # Even if this await gets cancelled again, the orphan thread still
        # completes and ComfyUI gets interrupted.
        await asyncio.to_thread(_interrupt_prompt, prompt_id)
        raise
    raise TimeoutError(f"prompt {prompt_id} exceeded {JOB_TIMEOUT_S}s")


def _s3_cfg(job):
    cfg = job.get("s3Config") or (job.get("input") or {}).get("s3Config")
    if isinstance(cfg, dict) and cfg.get("endpointUrl") and cfg.get("accessId"):
        return cfg
    return None


def _s3_client(cfg):
    import boto3
    from botocore.config import Config
    host = urllib.parse.urlparse(cfg["endpointUrl"]).netloc
    region = cfg.get("region") or host.split(".", 1)[0].removeprefix("s3api-") or "us-east-1"
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpointUrl"],
        aws_access_key_id=cfg["accessId"],
        aws_secret_access_key=cfg["accessSecret"],
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


def _emit_file(filename, data, job, inline_used):
    """Small outputs inline as b64; large ones upload to the job's s3Config
    storage (e.g. the RunPod network volume S3 API) and return a presigned URL."""
    oversized = len(data) > INLINE_MAX or inline_used[0] + len(data) > INLINE_BUDGET
    s3c = _s3_cfg(job)
    if oversized and s3c:
        try:
            key = f"{S3_PREFIX}/{job.get('id', 'job')}/{filename}"
            client = _s3_client(s3c)
            client.put_object(Bucket=s3c["bucketName"], Key=key, Body=data)
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3c["bucketName"], "Key": key},
                ExpiresIn=int(os.environ.get("S3_URL_TTL_S", "3600")),
            )
            print(f"[handler] {filename} -> s3://{s3c['bucketName']}/{key} ({len(data)/1e6:.1f}MB)", flush=True)
            return {"filename": filename, "url": url}
        except Exception as e:
            print(f"[handler] s3 upload failed for {filename}: {e} — falling back to inline", flush=True)
    elif oversized:
        print(f"[handler] {filename}: {len(data)/1e6:.1f}MB over inline limit, no s3Config — result may be dropped by RunPod", flush=True)
    inline_used[0] += len(data)
    return {"filename": filename, "b64": base64.b64encode(data).decode()}


def _prune_volume_outputs(max_age_s=86400):
    """Drop stale volume output dirs."""
    root = VOL_ROOT / S3_PREFIX
    cutoff = time.time() - max_age_s
    try:
        for d in root.iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


def _collect_new_files(since_ts, job, inline_used):
    """Fallback: anything written under ComfyUI's output dir since the job ran."""
    out_dir = Path(COMFY_DIR) / "output"
    try:
        files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.stat().st_mtime >= since_ts)
    except Exception:
        return []
    return [_emit_file(p.name, p.read_bytes(), job, inline_used)
            for p in files if p.suffix.lower() in MEDIA_EXTS]


def _collect_outputs(history_entry, job, inline_used):
    out = []
    for node_id, node_out in (history_entry.get("outputs") or {}).items():
        for key in ("images", "gifs", "videos"):
            for item in node_out.get(key, []) or []:
                params = urllib.parse.urlencode({
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                data = _http(f"/view?{params}", timeout=300)
                out.append(_emit_file(item["filename"], data, job, inline_used))
    return out


async def handler(job):
    print(f"[handler] job {job.get('id')} received", flush=True)
    job_input = job.get("input") or {}
    if job_input.get("action") == "health":
        # Hub smoke test: answers 200 without touching ComfyUI or models, so it
        # also works on an endpoint deployed without the network volume.
        return {
            "status": "ok",
            "comfyui": "failed" if _BOOT_ERR else ("ready" if _BOOT_DONE.is_set() else "booting"),
            "comfyui_error": str(_BOOT_ERR) if _BOOT_ERR else None,
            "workflows": _workflow_names(),
            "manifests": sorted(p.stem for p in MANIFESTS_DIR.glob("*.txt")),
            "volume_mounted": VOL_ROOT.is_dir(),
        }

    try:
        provisioned = None
        if job_input.get("models_manifest"):
            provisioned = await asyncio.to_thread(_provision_models, job_input["models_manifest"])
            if job_input.get("models_only"):
                return {"provisioned": provisioned}
        elif job_input.get("models_only"):
            raise JobError("models_only requires models_manifest")

        prompt = _load_workflow(job_input.get("workflow") or "")
        await _ensure_comfy()
        _prune_volume_outputs()
        # Uploads block (network) — run off-thread so a cancel can still land.
        warnings, seed = await asyncio.to_thread(
            apply_inputs, prompt, job_input.get("inputs") or {}, _upload, str(job.get("id") or "job"))
        _check_models_present(prompt)
    except JobError as exc:
        return {"error": str(exc)}

    for w in warnings:
        print(f"[handler] warning: {w}", flush=True)
    print(f"[handler] queuing prompt ({len(prompt)} nodes)", flush=True)
    started = time.time()
    history = await _queue_and_wait(prompt)
    print(f"[handler] comfy execution took {time.time() - started:.1f}s", flush=True)
    inline_used = [0]
    outputs = await asyncio.to_thread(_collect_outputs, history, job, inline_used)
    if not outputs:
        outputs = _collect_new_files(started, job, inline_used)
        print(f"[handler] fallback sweep — {len(outputs)} file(s) in output/", flush=True)
    print(f"[handler] done — {len(outputs)} output(s)", flush=True)
    result = {"outputs": outputs, "seed": seed, "warnings": warnings}
    if provisioned is not None:
        result["provisioned"] = provisioned
    return result


if __name__ == "__main__":
    import runpod
    threading.Thread(target=_boot_in_bg, daemon=True).start()
    runpod.serverless.start({"handler": handler})
