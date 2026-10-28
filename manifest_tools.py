#!/usr/bin/env python3
"""Model manifest parsing + validation.

Manifest line (tab separated): <subdir> <filename> <url> [sha256] [size_bytes]
"-" or an empty column means "unset".

CLI:
  manifest_tools.py static    MANIFEST... [--no-network]   build-time checks
  manifest_tools.py workflows DIR MANIFEST...              workflow models are listed
  manifest_tools.py verify    ROOT MANIFEST... [--entry subdir/name] [--hash]
                                                           files on the volume are intact
Exit code is non-zero when any check fails.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import namedtuple
from pathlib import Path
from urllib.parse import urlparse

Entry = namedtuple("Entry", "subdir name url sha256 size source line")

ALLOWED_SUBDIRS = {
    "checkpoints", "diffusion_models", "unet", "text_encoders", "clip_projections", "loras",
    "vae", "vae_approx", "LLM", "clip", "clip_vision", "upscale_models",
    "ultralytics/bbox", "ultralytics/segm", "sams",
}
ALLOWED_HOSTS = {"huggingface.co", "hf.co"}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Workflow loader input -> models subdir(s) that may hold the file.
LOADER_FIELDS = {
    "UNETLoader": ("unet_name", ("diffusion_models", "unet")),
    "CLIPLoader": ("clip_name", ("text_encoders", "clip")),
    "VAELoader": ("vae_name", ("vae",)),
    "LoraLoader": ("lora_name", ("loras",)),
    "LoraLoaderModelOnly": ("lora_name", ("loras",)),
}


def parse_manifest(text, source="<inline>"):
    """Return (entries, errors); never raises on malformed lines."""
    entries, errors = [], []
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        where = f"{source}:{no}"
        cols = line.split("\t")
        if not 3 <= len(cols) <= 5:
            errors.append(f"{where}: expected 3-5 tab-separated columns, got {len(cols)}")
            continue
        cols = [c.strip() for c in cols] + [""] * (5 - len(cols))
        subdir, name, url, sha, size = cols
        sha = "" if sha == "-" else sha.lower()
        size = "" if size == "-" else size
        if subdir not in ALLOWED_SUBDIRS:
            errors.append(f"{where}: unknown models subdir '{subdir}'")
        if not NAME_RE.match(name) or name.startswith("."):
            errors.append(f"{where}: invalid filename '{name}'")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            errors.append(f"{where}: URL must be https on {sorted(ALLOWED_HOSTS)}: {url}")
        if sha and not SHA_RE.match(sha):
            errors.append(f"{where}: sha256 must be 64 hex chars")
        if size and not (size.isdigit() and int(size) > 0):
            errors.append(f"{where}: size_bytes must be a positive integer")
        if errors and errors[-1].startswith(where):
            continue
        entries.append(Entry(subdir, name, url, sha, int(size) if size else 0, source, no))
    return entries, errors


def load_manifests(paths):
    entries, errors = [], []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            errors.append(f"{p}: manifest not found")
            continue
        e, err = parse_manifest(p.read_text(encoding="utf-8"), str(p))
        entries += e
        errors += err
    seen = {}
    for e in entries:
        key = (e.subdir, e.name)
        prev = seen.get(key)
        if prev and (prev.url, prev.sha256, prev.size) != (e.url, e.sha256, e.size):
            errors.append(f"{e.source}:{e.line}: {e.subdir}/{e.name} conflicts with {prev.source}:{prev.line}")
        seen.setdefault(key, e)
    return list(seen.values()), errors


def sha256_file(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def check_file(path, entry, with_hash=False):
    """Return an error string, or None when the file matches the entry."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size == 0:
        return "empty file"
    if entry.size and size != entry.size:
        return f"size {size} != expected {entry.size}"
    if with_hash and entry.sha256:
        got = sha256_file(path)
        if got != entry.sha256:
            return f"sha256 {got} != expected {entry.sha256}"
    return None


def remote_size(url, timeout=30):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "manifest-tools"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        linked = r.headers.get("x-linked-size") or r.headers.get("Content-Length")
        return int(linked) if linked else None


def cmd_static(args):
    entries, errors = load_manifests(args.manifests)
    if not entries and not errors:
        errors.append("no manifest entries found")
    if not args.no_network:
        for e in entries:
            try:
                size = remote_size(e.url)
            except Exception as exc:
                errors.append(f"{e.source}:{e.line}: {e.name}: URL check failed: {exc}")
                continue
            if size is None:
                errors.append(f"{e.source}:{e.line}: {e.name}: server reported no size")
            elif e.size and size != e.size:
                errors.append(f"{e.source}:{e.line}: {e.name}: remote size {size} != manifest {e.size}")
            elif size < 1024:
                errors.append(f"{e.source}:{e.line}: {e.name}: remote size {size} is implausibly small")
            else:
                print(f"ok  {e.subdir}/{e.name}  {size} bytes")
    return errors


def workflow_models(path):
    wf = json.loads(Path(path).read_text(encoding="utf-8"))
    return None if "nodes" in wf else prompt_models(wf)


def prompt_models(wf):
    """(filename, candidate subdirs) for every model loader in an API-format graph."""
    found = []
    for node in wf.values():
        spec = LOADER_FIELDS.get(node.get("class_type"))
        val = (node.get("inputs") or {}).get(spec[0]) if spec else None
        if isinstance(val, str):
            found.append((val, spec[1]))
    return found


def cmd_workflows(args):
    entries, errors = load_manifests(args.manifests)
    have = {(e.subdir, e.name) for e in entries}
    for wf in sorted(Path(args.dir).glob("*.json")):
        try:
            models = workflow_models(wf)
        except Exception as exc:
            errors.append(f"{wf.name}: unreadable: {exc}")
            continue
        if models is None:
            errors.append(f"{wf.name}: UI-format graph, expected API format")
            continue
        for name, subdirs in models:
            if not any((s, name) in have for s in subdirs):
                errors.append(f"{wf.name}: model '{name}' ({'/'.join(subdirs)}) is not in the manifest")
    return errors


def cmd_verify(args):
    entries, errors = load_manifests(args.manifests)
    for e in entries:
        if args.entry and f"{e.subdir}/{e.name}" != args.entry:
            continue
        err = check_file(Path(args.root) / e.subdir / e.name, e, with_hash=args.hash)
        if err:
            errors.append(f"{e.subdir}/{e.name}: {err}")
        else:
            print(f"ok  {e.subdir}/{e.name}")
    if args.entry and not any(f"{e.subdir}/{e.name}" == args.entry for e in entries):
        errors.append(f"{args.entry}: not in manifest")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("manifests", nargs="+")
    s.add_argument("--no-network", action="store_true")
    s.set_defaults(fn=cmd_static)
    w = sub.add_parser("workflows")
    w.add_argument("dir")
    w.add_argument("manifests", nargs="+")
    w.set_defaults(fn=cmd_workflows)
    v = sub.add_parser("verify")
    v.add_argument("root")
    v.add_argument("manifests", nargs="+")
    v.add_argument("--entry")
    v.add_argument("--hash", action="store_true")
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args(argv)
    errors = args.fn(args)
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
