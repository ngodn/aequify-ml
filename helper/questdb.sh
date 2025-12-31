#!/usr/bin/env bash
#
# QuestDB Helper Script for Aequify
# Usage: ./helper/questdb.sh [setup|start|stop|restart|status|logs]
#

set -e

# =============================================================================
# Configuration
# =============================================================================

QUESTDB_VERSION="9.2.3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"
DATA_DIR="${PROJECT_DIR}/data/questdb"
CONFIG_DIR="${PROJECT_DIR}/config"
LOG_DIR="${PROJECT_DIR}/logs"

# QuestDB paths
QUESTDB_HOME="${BIN_DIR}/questdb"
QUESTDB_PID_FILE="${DATA_DIR}/questdb.pid"
QUESTDB_LOG_FILE="${LOG_DIR}/questdb.log"

# Download URLs
LINUX_URL="https://github.com/questdb/questdb/releases/download/${QUESTDB_VERSION}/questdb-${QUESTDB_VERSION}-rt-linux-x86-64.tar.gz"
WINDOWS_URL="https://github.com/questdb/questdb/releases/download/${QUESTDB_VERSION}/questdb-${QUESTDB_VERSION}-rt-windows-x86-64.tar.gz"

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Linux*)
            if [[ "$arch" == "x86_64" ]]; then
                echo "linux-x86-64"
            else
                log_error "Unsupported Linux architecture: $arch"
                exit 1
            fi
            ;;
        Darwin*)
            echo "macos"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "windows"
            ;;
        *)
            log_error "Unsupported OS: $os"
            exit 1
            ;;
    esac
}

is_running() {
    # Check if QuestDB Java process is running
    pgrep -f "QuestDB-Runtime" > /dev/null 2>&1
}

get_pid() {
    # Get QuestDB process PID
    pgrep -f "QuestDB-Runtime" 2>/dev/null | head -1
}

# =============================================================================
# Setup Functions
# =============================================================================

setup_directories() {
    log_info "Creating directories..."
    mkdir -p "$BIN_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "${DATA_DIR}/db"
    mkdir -p "${DATA_DIR}/conf"
}

setup_config() {
    local config_file="${CONFIG_DIR}/questdb-server.conf"

    if [[ -f "$config_file" ]]; then
        log_info "Config file already exists: $config_file"
        return
    fi

    # Detect system resources
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    local total_mem_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 8388608)
    local total_mem_gb=$((total_mem_kb / 1024 / 1024))

    # Calculate worker counts (~25% of cores - QuestDB shares machine with main app)
    local network_workers=$((cpu_cores / 4))
    local query_workers=$((cpu_cores / 4))
    local write_workers=$((cpu_cores / 8))
    [[ $network_workers -lt 1 ]] && network_workers=1
    [[ $query_workers -lt 1 ]] && query_workers=1
    [[ $write_workers -lt 1 ]] && write_workers=1

    # Memory settings (~25% of RAM - conservative for shared machine)
    local page_size="8M"
    local max_uncommitted=250000
    if [[ $total_mem_gb -ge 32 ]]; then
        page_size="16M"
        max_uncommitted=500000
    elif [[ $total_mem_gb -ge 16 ]]; then
        page_size="8M"
        max_uncommitted=350000
    elif [[ $total_mem_gb -lt 8 ]]; then
        page_size="4M"
        max_uncommitted=100000
    fi

    log_info "Detected: ${cpu_cores} CPU cores, ${total_mem_gb}GB RAM"
    log_info "Creating QuestDB config: $config_file"

    mkdir -p "$CONFIG_DIR"
    cat > "$config_file" << EOF
# QuestDB Server Configuration
# https://questdb.com/docs/configuration/
# Generated for: ${cpu_cores} CPU cores, ${total_mem_gb}GB RAM

# =============================================================================
# HTTP Server (Web Console & REST API)
# =============================================================================
http.enabled=true
http.bind.to=0.0.0.0:9000
http.net.connection.limit=64
http.net.connection.timeout=300000
http.net.connection.sndbuf=2M
http.net.connection.rcvbuf=2M
http.query.cache.enabled=true
http.security.readonly=false

# =============================================================================
# PostgreSQL Wire Protocol
# =============================================================================
pg.enabled=true
pg.net.bind.to=0.0.0.0:8812
pg.net.connection.limit=64
pg.net.connection.timeout=300000
pg.user=admin
pg.password=quest
pg.select.cache.enabled=true

# =============================================================================
# InfluxDB Line Protocol (ILP) - High-speed ingestion
# =============================================================================
line.tcp.enabled=true
line.tcp.net.bind.to=0.0.0.0:9009
line.tcp.net.connection.limit=256
line.tcp.msg.buffer.size=32768
line.tcp.timestamp=n
line.auto.create.new.tables=true
line.auto.create.new.columns=true
line.default.partition.by=DAY

# =============================================================================
# Worker Pools (tuned for ${cpu_cores} cores)
# =============================================================================
shared.network.worker.count=${network_workers}
shared.query.worker.count=${query_workers}
shared.write.worker.count=${write_workers}

# =============================================================================
# Memory & Performance (tuned for ${total_mem_gb}GB RAM)
# =============================================================================
cairo.writer.data.append.page.size=${page_size}
cairo.sql.map.page.size=4M
cairo.sql.sort.key.page.size=4M
cairo.max.uncommitted.rows=${max_uncommitted}
cairo.o3.max.lag=600000
cairo.o3.min.lag=1000
cairo.sql.parallel.filter.enabled=true
cairo.sql.parallel.groupby.enabled=true

# =============================================================================
# Query Settings
# =============================================================================
query.timeout.sec=60

# =============================================================================
# Telemetry
# =============================================================================
telemetry.enabled=false

# =============================================================================
# Validation
# =============================================================================
config.validation.strict=true
EOF

    log_info "Config created. Edit $config_file to customize."
}

setup_linux() {
    local archive_name="questdb-${QUESTDB_VERSION}-rt-linux-x86-64.tar.gz"
    local archive_path="${BIN_DIR}/${archive_name}"
    local extract_dir="${BIN_DIR}/questdb-${QUESTDB_VERSION}-rt-linux-x86-64"

    # Check if already installed
    if [[ -d "$QUESTDB_HOME" ]]; then
        log_info "QuestDB already installed at $QUESTDB_HOME"
        return
    fi

    # Check if extracted dir exists (interrupted rename)
    if [[ -d "$extract_dir" ]]; then
        log_info "Found extracted directory, renaming..."
        mv "$extract_dir" "$QUESTDB_HOME"
        log_info "QuestDB installed at $QUESTDB_HOME"
        return
    fi

    # Check if archive already downloaded
    if [[ -f "$archive_path" ]]; then
        log_info "Archive already downloaded, extracting..."
    else
        log_info "Downloading QuestDB ${QUESTDB_VERSION} for Linux..."
        curl -L -o "$archive_path" "$LINUX_URL"
    fi

    log_info "Extracting..."
    tar -xzf "$archive_path" -C "$BIN_DIR"

    # Rename to standard path
    mv "$extract_dir" "$QUESTDB_HOME"

    # Cleanup archive
    rm -f "$archive_path"

    log_info "QuestDB installed at $QUESTDB_HOME"
}

setup_macos() {
    if command -v questdb &> /dev/null; then
        log_info "QuestDB already installed via Homebrew"
        return
    fi

    if ! command -v brew &> /dev/null; then
        log_error "Homebrew not found. Install from https://brew.sh"
        exit 1
    fi

    log_info "Installing QuestDB via Homebrew..."
    brew install questdb

    log_info "QuestDB installed via Homebrew"
}

setup_windows() {
    local archive_name="questdb-${QUESTDB_VERSION}-rt-windows-x86-64.tar.gz"
    local archive_path="${BIN_DIR}/${archive_name}"
    local extract_dir="${BIN_DIR}/questdb-${QUESTDB_VERSION}-rt-windows-x86-64"

    # Check if already installed
    if [[ -d "$QUESTDB_HOME" ]]; then
        log_info "QuestDB already installed at $QUESTDB_HOME"
        return
    fi

    # Check if extracted dir exists (interrupted rename)
    if [[ -d "$extract_dir" ]]; then
        log_info "Found extracted directory, renaming..."
        mv "$extract_dir" "$QUESTDB_HOME"
        log_info "QuestDB installed at $QUESTDB_HOME"
        return
    fi

    # Check if archive already downloaded
    if [[ -f "$archive_path" ]]; then
        log_info "Archive already downloaded, extracting..."
    else
        log_info "Downloading QuestDB ${QUESTDB_VERSION} for Windows..."
        curl -L -o "$archive_path" "$WINDOWS_URL"
    fi

    log_info "Extracting..."
    tar -xzf "$archive_path" -C "$BIN_DIR"

    # Rename to standard path
    mv "$extract_dir" "$QUESTDB_HOME"

    # Cleanup archive
    rm -f "$archive_path"

    log_info "QuestDB installed at $QUESTDB_HOME"
}

do_setup() {
    local platform=$(detect_platform)
    log_info "Detected platform: $platform"

    setup_directories
    setup_config

    case "$platform" in
        linux-x86-64)
            setup_linux
            ;;
        macos)
            setup_macos
            ;;
        windows)
            setup_windows
            ;;
    esac

    log_info "Setup complete!"
    echo ""
    echo "Directories:"
    echo "  Binary:  $QUESTDB_HOME"
    echo "  Data:    $DATA_DIR"
    echo "  Config:  ${CONFIG_DIR}/questdb-server.conf"
    echo "  Logs:    $LOG_DIR"
    echo ""
    echo "Next: Run './helper/questdb.sh start' to start QuestDB"
}

# =============================================================================
# Control Functions
# =============================================================================

do_start() {
    local platform=$(detect_platform)

    if is_running; then
        log_warn "QuestDB is already running (PID: $(get_pid))"
        return
    fi

    # Ensure directories exist
    mkdir -p "$DATA_DIR" "$LOG_DIR"

    log_info "Starting QuestDB..."

    case "$platform" in
        linux-x86-64|windows)
            if [[ ! -d "$QUESTDB_HOME" ]]; then
                log_error "QuestDB not installed. Run './helper/questdb.sh setup' first."
                exit 1
            fi

            # Copy config to QuestDB's expected location
            local config_src="${CONFIG_DIR}/questdb-server.conf"
            local config_dst="${DATA_DIR}/conf/server.conf"
            if [[ -f "$config_src" ]]; then
                mkdir -p "${DATA_DIR}/conf"
                cp "$config_src" "$config_dst"
            fi

            # Start QuestDB (it daemonizes itself and manages its own PID file)
            "$QUESTDB_HOME/bin/questdb.sh" start \
                -d "$DATA_DIR" \
                > "$QUESTDB_LOG_FILE" 2>&1

            sleep 3

            if is_running; then
                log_info "QuestDB started (PID: $(get_pid))"
                echo ""
                echo "Web Console: http://localhost:9000"
                echo "PostgreSQL:  localhost:8812 (user: admin, pass: quest)"
                echo "ILP (TCP):   localhost:9009"
            else
                log_error "Failed to start QuestDB. Check logs: $QUESTDB_LOG_FILE"
                exit 1
            fi
            ;;
        macos)
            # Use Homebrew service
            brew services start questdb
            log_info "QuestDB started via Homebrew"
            echo ""
            echo "Web Console: http://localhost:9000"
            echo "PostgreSQL:  localhost:8812"
            echo "ILP (TCP):   localhost:9009"
            ;;
    esac
}

do_stop() {
    local platform=$(detect_platform)

    log_info "Stopping QuestDB..."

    case "$platform" in
        linux-x86-64|windows)
            if [[ -d "$QUESTDB_HOME" ]]; then
                "$QUESTDB_HOME/bin/questdb.sh" stop -d "$DATA_DIR" 2>&1 || true
            fi

            # Force kill if still running
            if is_running; then
                local pid=$(get_pid)
                kill -9 "$pid" 2>/dev/null || true
            fi

            log_info "QuestDB stopped"
            ;;
        macos)
            brew services stop questdb
            log_info "QuestDB stopped via Homebrew"
            ;;
    esac
}

do_restart() {
    do_stop
    sleep 2
    do_start
}

do_status() {
    local platform=$(detect_platform)

    echo "QuestDB Status"
    echo "=============="
    echo "Platform: $platform"
    echo "Data Dir: $DATA_DIR"
    echo ""

    case "$platform" in
        linux-x86-64|windows)
            if is_running; then
                echo "Status: RUNNING (PID: $(get_pid))"
            else
                echo "Status: STOPPED"
            fi
            ;;
        macos)
            brew services info questdb 2>/dev/null || echo "Status: Unknown (check 'brew services list')"
            ;;
    esac

    echo ""
    echo "Endpoints:"
    echo "  Web Console: http://localhost:9000"
    echo "  PostgreSQL:  localhost:8812"
    echo "  ILP (TCP):   localhost:9009"
}

do_logs() {
    local platform=$(detect_platform)

    case "$platform" in
        linux-x86-64|windows)
            if [[ -f "$QUESTDB_LOG_FILE" ]]; then
                tail -f "$QUESTDB_LOG_FILE"
            else
                log_error "Log file not found: $QUESTDB_LOG_FILE"
                exit 1
            fi
            ;;
        macos)
            log_info "For macOS, check: /usr/local/var/log/questdb.log"
            log_info "Or run: brew services info questdb"
            ;;
    esac
}

# =============================================================================
# Main
# =============================================================================

show_usage() {
    echo "QuestDB Helper Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  setup    - Download and install QuestDB binaries"
    echo "  start    - Start QuestDB server"
    echo "  stop     - Stop QuestDB server"
    echo "  restart  - Restart QuestDB server"
    echo "  status   - Show QuestDB status"
    echo "  logs     - Tail QuestDB logs"
    echo ""
    echo "Directories:"
    echo "  Binary:  $BIN_DIR/questdb"
    echo "  Data:    $DATA_DIR"
    echo "  Config:  $CONFIG_DIR/questdb-server.conf"
    echo "  Logs:    $LOG_DIR"
}

case "${1:-}" in
    setup)
        do_setup
        ;;
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    restart)
        do_restart
        ;;
    status)
        do_status
        ;;
    logs)
        do_logs
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
