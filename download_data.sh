#!/usr/bin/env bash
set -euo pipefail

# Download external experiment data into configurable_mdp/data.
# Set DATA_URL before running this script.
: "${DATA_URL:=}"
: "${DATA_SHA256:=}"

if [[ -z "$DATA_URL" ]]; then
    echo "ERROR: DATA_URL is empty."
    echo "Example: DATA_URL=https://example.com/BC-HG-data-v1.tar.gz ./download_data.sh"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$REPO_ROOT/configurable_mdp/data"
TMP_DIR="$(mktemp -d)"
ARCHIVE_PATH="$TMP_DIR/data_archive"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "Downloading data archive..."
curl -L --fail --retry 3 "$DATA_URL" -o "$ARCHIVE_PATH"

if [[ -n "$DATA_SHA256" ]]; then
    echo "Verifying SHA-256..."
    actual_sha="$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')"
    if [[ "$actual_sha" != "$DATA_SHA256" ]]; then
        echo "ERROR: SHA-256 mismatch"
        echo "Expected: $DATA_SHA256"
        echo "Actual:   $actual_sha"
        exit 1
    fi
fi

mkdir -p "$TARGET_DIR"

# Extract based on archive extension.
if [[ "$DATA_URL" == *.tar.gz ]] || [[ "$DATA_URL" == *.tgz ]]; then
    tar -xzf "$ARCHIVE_PATH" -C "$TMP_DIR"
elif [[ "$DATA_URL" == *.zip ]]; then
    unzip -q "$ARCHIVE_PATH" -d "$TMP_DIR"
else
    echo "ERROR: Unsupported archive type in DATA_URL (use .tar.gz/.tgz/.zip)."
    exit 1
fi

if [[ -d "$TMP_DIR/configurable_mdp/data" ]]; then
    SOURCE_DIR="$TMP_DIR/configurable_mdp/data"
else
    # Fallback: archive contains experiment directories directly.
    SOURCE_DIR="$TMP_DIR"
fi

echo "Syncing data into configurable_mdp/data ..."
rsync -a --delete \
    --exclude 'README' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

echo "Done. Data is available at: $TARGET_DIR"
