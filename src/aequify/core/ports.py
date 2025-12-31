"""
Aequify Port Configuration.

All network ports used by Aequify components are defined here.

PORT ALLOCATION RULES:
    - All ports MUST be in the range 9300-9999
    - Reserved ranges:
        - 9300-9399: Core infrastructure (PubSub, internal RPC)
        - 9400-9499: Database connections (QuestDB, TigerBeetle)
        - 9500-9599: Exchange connectors (WebSocket, REST proxies)
        - 9600-9699: ML services (model serving, inference)
        - 9700-9799: Monitoring and metrics
        - 9800-9899: Development/debugging
        - 9900-9999: Reserved for future use

CURRENT PORT ASSIGNMENTS:
    9300 - PubSub server (Engine -> TUI, subscribers)

Usage:
    from aequify.core.ports import PUBSUB_PORT

    server = PubSubServer(port=PUBSUB_PORT)
"""

# =============================================================================
# Core Infrastructure (9300-9399)
# =============================================================================

PUBSUB_PORT = 9300
"""
PubSub server port for real-time messaging.

Used by:
    - Engine: Publishes state updates (engine.state, system.stats)
    - TUI: Subscribes to receive updates
    - Other subscribers: Any component needing real-time updates
"""

# =============================================================================
# Database Connections (9400-9499)
# =============================================================================

# QUESTDB_PORT = 9400  # Reserved for QuestDB
# TIGERBEETLE_PORT = 9401  # Reserved for TigerBeetle

# =============================================================================
# Exchange Connectors (9500-9599)
# =============================================================================

# BINANCE_WS_PROXY_PORT = 9500  # Reserved for Binance WebSocket proxy

# =============================================================================
# ML Services (9600-9699)
# =============================================================================

# MODEL_SERVER_PORT = 9600  # Reserved for model serving

# =============================================================================
# Monitoring (9700-9799)
# =============================================================================

# METRICS_PORT = 9700  # Reserved for Prometheus metrics

# =============================================================================
# Development (9800-9899)
# =============================================================================

# DEBUG_RPC_PORT = 9800  # Reserved for debug RPC

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PUBSUB_PORT",
]
