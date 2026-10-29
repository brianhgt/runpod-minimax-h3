#!/bin/bash
# Downloads every file listed in one or more model manifests onto the network
# volume and verifies it (size + sha256 when the manifest provides them).
# Idempotent: a file that passes verification is skipped; a present file that
# fails (truncated, wrong size) is deleted and re-downloaded. Safe for
# concurrent workers: a flock on $VOL_ROOT/.populate.lock serializes the pass.
#
# Usage: populate-volume.sh [manifest.txt ...]
#
# Env:
#   VOL_ROOT      mount point of the network volume          (default /runpod-volume)
#   MANIFEST      space-separated manifests when no args     (default: models/all.txt next to this script)
#   HF_TOKEN      optional Hugging Face token for gated repos
#   VERIFY_HASH   "true" to also sha256 files already on the volume (slow; always done after a download)
#   DL_ATTEMPTS   download attempts per file                 (default 3)
#
# Exit code: 0 when every entry is present and verified, 1 otherwise.
set -uo pipefail

VOL_ROOT="${VOL_ROOT:-/runpod-volume}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$SCRIPT_DIR/manifest_tools.py"
if [ $# -gt 0 ]; then
    MANIFESTS=("$@")
else
    read -r -a MANIFESTS <<< "${MANIFEST:-$SCRIPT_DIR/models/all.txt}"
fi
M="$VOL_ROOT/models"
ATTEMPTS="${DL_ATTEMPTS:-3}"

if [ ! -d "$VOL_ROOT" ]; then
    echo "populate: $VOL_ROOT not mounted â€” nothing to do"
    exit 0
fi

# Reject malformed manifests before touching the volume
if ! python3 "$TOOLS" static --no-network "${MANIFESTS[@]}"; then
    echo "populate: manifest validation failed" >&2
    exit 1
fi

exec 9>"$VOL_ROOT/.populate.lock"
flock 9

echo "populate: free space on $VOL_ROOT: $(df -h "$VOL_ROOT" | awk 'NR==2{print $4" of "$2}')"
mkdir -p "$M"

verify() { # verify <subdir/name> [--hash]
    python3 "$TOOLS" verify "$M" "${MANIFESTS[@]}" --entry "$1" ${2:+"$2"} >/dev/null
}

download() { # download <sub> <name> <url> <dest>
    local sub="$1" name="$2" url="$3" dest="$4"
    if command -v hf >/dev/null 2>&1 && [[ "$url" =~ ^https://huggingface\.co/ ]]; then
        # Disable Xet: its chunk cache can exceed the container's disk cap.
        local repo_id repo_path tmp_dir
        repo_id=$(echo "$url" | awk -F/ '{print $4"/"$5}')
        repo_path=$(echo "$url" | sed -E 's#https?://[^/]+/[^/]+/[^/]+/resolve/[^/]+/(.+)#\1#')
        tmp_dir="$M/$sub/.tmp_hf_$name"
        rm -rf "$tmp_dir"; mkdir -p "$tmp_dir"
        export HF_HUB_DISABLE_XET=1
        if hf download "$repo_id" "$repo_path" --local-dir "$tmp_dir" && [ -f "$tmp_dir/$repo_path" ]; then
            if mv -f "$tmp_dir/$repo_path" "$dest"; then
                rm -rf "$tmp_dir"
                return 0
            fi
            echo "populate: failed to move downloaded file into place: $dest" >&2
        fi
        rm -rf "$tmp_dir"
    fi
    local hdr=()
    [ -n "${HF_TOKEN:-}" ] && hdr=(--header="Authorization: Bearer $HF_TOKEN")
    if wget -q --continue --tries=5 --timeout=60 "${hdr[@]}" -O "$dest.tmp" "$url"; then
        mv "$dest.tmp" "$dest"
        return 0
    fi
    return 1
}

total=0; n=0; failed=()
for mf in "${MANIFESTS[@]}"; do
    total=$((total + $(grep -vc '^\s*\(#\|$\)' "$mf")))
done

for mf in "${MANIFESTS[@]}"; do
    while IFS=$'\t' read -r sub name url sha size; do
        case "$sub" in ''|\#*) continue;; esac
        n=$((n + 1))
        key="$sub/$name"
        dest="$M/$sub/$name"
        hashflag=""; [ "${VERIFY_HASH:-false}" = "true" ] && hashflag="--hash"

        if [ -e "$dest" ]; then
            if verify "$key" $hashflag; then
                echo "populate: [$n/$total] ok   $key (verified)"
                continue
            fi
            echo "populate: [$n/$total] BAD  $key â€” removing and re-downloading" >&2
            rm -f "$dest"
        fi

        mkdir -p "$M/$sub"
        ok=0
        for attempt in $(seq 1 "$ATTEMPTS"); do
            echo "populate: [$n/$total] downloading $key (attempt $attempt/$ATTEMPTS)"
            if download "$sub" "$name" "$url" "$dest" && verify "$key" --hash; then
                ok=1; break
            fi
            echo "populate: $key failed download/verification â€” free space: $(df -h "$M" | awk 'NR==2{print $4}')" >&2
            rm -f "$dest" "$dest.tmp"
        done
        [ "$ok" = 1 ] || failed+=("$key")
    done < "$mf"
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "populate: FAILED ${#failed[@]} file(s): ${failed[*]}" >&2
    exit 1
fi
echo "populate: done â€” all $n file(s) verified in $M"
