#!/bin/bash
# Start QuestDB server for production
# Data is stored in data/questdb

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        QUESTDB_ARCH="x86_64"
        ;;
    aarch64|arm64)
        QUESTDB_ARCH="aarch64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Detect OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$OS" in
    linux)
        QUESTDB_OS="linux"
        ;;
    darwin)
        QUESTDB_OS="mac"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

QUESTDB_DIR="$PROJECT_ROOT/bin/$QUESTDB_ARCH/$QUESTDB_OS/questdb"
QUESTDB_BIN="$QUESTDB_DIR/bin/questdb.sh"
QUESTDB_DATA="$PROJECT_ROOT/data/questdb"

# Check if questdb binary exists
if [ ! -f "$QUESTDB_BIN" ]; then
    echo "QuestDB binary not found at: $QUESTDB_BIN"
    echo "Please download QuestDB first."
    exit 1
fi

echo "Using QuestDB: $QUESTDB_DIR"
echo "Data directory: $QUESTDB_DATA"

# Create data directory
mkdir -p "$QUESTDB_DATA"

# Set environment
export QUESTDB_HOME="$QUESTDB_DIR"

echo ""
echo "Starting QuestDB server..."
echo "  HTTP API: http://localhost:9000"
echo "  PostgreSQL: localhost:8812"
echo "  ILP (TCP): localhost:9009"
echo ""
echo "Web Console: http://localhost:9000"
echo "Press Ctrl+C to stop"
echo ""

# Start QuestDB
cd "$QUESTDB_DIR"
exec ./bin/questdb.sh start -d "$QUESTDB_DATA" -f
