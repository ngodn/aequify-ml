#!/usr/bin/env bash
# Build APEX GPU kernels (package + bootstrap shared library)
set -e

APEX_DIR="src/aequify/core/apex"
KERNELS_DIR="${APEX_DIR}/kernels"
CACHE_DIR="${KERNELS_DIR}/__mojocache__"
PKG_PATH="${CACHE_DIR}/kernels.mojopkg"
SO_PATH="${CACHE_DIR}/bootstrap.so"
BOOTSTRAP_MOJO="${APEX_DIR}/bootstrap.mojo"

# Check if rebuild is needed (any .mojo file newer than .so)
if [ -f "$SO_PATH" ]; then
    NEWEST_MOJO=$(find "$APEX_DIR" -name "*.mojo" -newer "$SO_PATH" 2>/dev/null | head -1)
    if [ -z "$NEWEST_MOJO" ]; then
        echo "Kernels up to date, skipping build"
        exit 0
    fi
fi

echo "Building APEX kernels..."
mkdir -p "$CACHE_DIR"

# Step 1: Build kernels package
echo "  Building kernels package..."
uv run mojo package "$KERNELS_DIR" -o "$PKG_PATH"

# Step 2: Build bootstrap.mojo with kernels package
echo "  Building bootstrap shared library..."
uv run mojo build "$BOOTSTRAP_MOJO" -I "$CACHE_DIR" --emit shared-lib -o "$SO_PATH"

echo "Kernels built successfully!"
