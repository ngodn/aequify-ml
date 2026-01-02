"""
QuestDB client for cold storage.

Provides async interface for QuestDB using:
- ILP (InfluxDB Line Protocol) for high-speed ingestion (port 9009)
- PostgreSQL wire protocol for queries (port 8812)

Uses aequify.runtime for non-blocking operations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from aequify.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

# Lazy imports for optional dependencies
_questdb_sender = None
_asyncpg = None


def _get_questdb_sender():
    """Lazy import questdb.ingress."""
    global _questdb_sender
    if _questdb_sender is None:
        from questdb.ingress import Sender

        _questdb_sender = Sender
    return _questdb_sender


async def _get_asyncpg():
    """Lazy import asyncpg."""
    global _asyncpg
    if _asyncpg is None:
        import asyncpg

        _asyncpg = asyncpg
    return _asyncpg


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class QuestDBConfig:
    """
    QuestDB connection configuration.

    Attributes:
        host: QuestDB host address.
        ilp_port: ILP (InfluxDB Line Protocol) port for ingestion.
        pg_port: PostgreSQL wire protocol port for queries.
        pg_user: PostgreSQL username.
        pg_password: PostgreSQL password.
        pg_database: PostgreSQL database name.
        pool_min_size: Minimum connection pool size.
        pool_max_size: Maximum connection pool size.
    """

    host: str = "localhost"
    ilp_port: int = 9009
    pg_port: int = 8812
    pg_user: str = "admin"
    pg_password: str = "quest"
    pg_database: str = "qdb"
    pool_min_size: int = 2
    pool_max_size: int = 10

    @property
    def ilp_address(self) -> str:
        """ILP connection string."""
        return f"{self.host}:{self.ilp_port}"

    @property
    def pg_dsn(self) -> str:
        """PostgreSQL DSN."""
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.host}:{self.pg_port}/{self.pg_database}"
        )


def load_questdb_config(config_path: str | Path | None = None) -> QuestDBConfig:
    """
    Load QuestDB configuration from config.yaml.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml.

    Returns:
        QuestDBConfig instance with values from config or defaults.
    """
    if config_path is None:
        # Find project root (where config/ is)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "config" / "config.yaml").exists():
                config_path = parent / "config" / "config.yaml"
                break

    if config_path is None or not Path(config_path).exists():
        return QuestDBConfig()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cold_config = config.get("database", {}).get("cold", {})
    if not cold_config:
        return QuestDBConfig()

    return QuestDBConfig(
        host=cold_config.get("host", "localhost"),
        ilp_port=cold_config.get("ilp_port", 9009),
        pg_port=cold_config.get("pg_port", 8812),
        pg_user=cold_config.get("pg_user", "admin"),
        pg_password=cold_config.get("pg_password", "quest"),
        pg_database=cold_config.get("pg_database", "qdb"),
        pool_min_size=cold_config.get("pool_min_size", 2),
        pool_max_size=cold_config.get("pool_max_size", 10),
    )


# Default configuration (loaded from config.yaml)
DEFAULT_CONFIG = load_questdb_config()


# =============================================================================
# Async QuestDB Client
# =============================================================================


class QuestDBClient:
    """
    Async QuestDB client for cold storage.

    Provides:
    - High-speed ingestion via ILP (InfluxDB Line Protocol)
    - SQL queries via PostgreSQL wire protocol
    - Connection pooling for efficient resource usage

    Usage:
        async with QuestDBClient() as client:
            # Insert trades
            await client.insert_trades(trades)

            # Query data
            rows = await client.query("SELECT * FROM trades LIMIT 10")
    """

    def __init__(self, config: QuestDBConfig | None = None) -> None:
        """
        Initialize client.

        Args:
            config: QuestDB configuration. Uses defaults if None.
        """
        self.config = config or DEFAULT_CONFIG
        self._pool: Any = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Initialize connection pool.

        Must be called before any operations, or use context manager.
        """
        async with self._lock:
            if self._initialized:
                return

            asyncpg = await _get_asyncpg()

            try:
                self._pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.pg_port,
                    user=self.config.pg_user,
                    password=self.config.pg_password,
                    database=self.config.pg_database,
                    min_size=self.config.pool_min_size,
                    max_size=self.config.pool_max_size,
                )
                self._initialized = True
                logger.info(
                    f"QuestDB client initialized (PG: {self.config.host}:{self.config.pg_port})"
                )
            except Exception as e:
                logger.error(f"Failed to initialize QuestDB client: {e}")
                raise

    async def close(self) -> None:
        """Close connection pool."""
        async with self._lock:
            if self._pool:
                await self._pool.close()
                self._pool = None
                self._initialized = False
                logger.info("QuestDB client closed")

    async def __aenter__(self) -> QuestDBClient:
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure client is initialized."""
        if not self._initialized or not self._pool:
            raise RuntimeError("Client not initialized. Call initialize() first.")

    # =========================================================================
    # Query Operations (PostgreSQL)
    # =========================================================================

    async def query(
        self,
        sql: str,
        *args: Any,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a SQL query and return results as list of dicts.

        Args:
            sql: SQL query string.
            *args: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            List of row dicts.
        """
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args, timeout=timeout)
            return [dict(row) for row in rows]

    async def query_one(
        self,
        sql: str,
        *args: Any,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Execute a SQL query and return first row as dict.

        Args:
            sql: SQL query string.
            *args: Query parameters.
            timeout: Query timeout in seconds.

        Returns:
            Row dict or None if no results.
        """
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args, timeout=timeout)
            return dict(row) if row else None

    async def execute(
        self,
        sql: str,
        *args: Any,
        timeout: float | None = None,
    ) -> str:
        """
        Execute a SQL statement (INSERT, UPDATE, DELETE, DDL).

        Args:
            sql: SQL statement.
            *args: Statement parameters.
            timeout: Statement timeout in seconds.

        Returns:
            Status string (e.g., "INSERT 0 1").
        """
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            return await conn.execute(sql, *args, timeout=timeout)

    async def executemany(
        self,
        sql: str,
        args: Sequence[tuple],
        timeout: float | None = None,
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement with $1, $2, etc. placeholders.
            args: Sequence of parameter tuples.
            timeout: Statement timeout in seconds.
        """
        self._ensure_initialized()

        async with self._pool.acquire() as conn:
            await conn.executemany(sql, args, timeout=timeout)

    # =========================================================================
    # High-Speed Ingestion (ILP)
    # =========================================================================

    def create_sender(self):
        """
        Create an ILP sender for high-speed ingestion.

        Returns:
            QuestDB Sender instance.

        Usage:
            with client.create_sender() as sender:
                sender.row(
                    "trades",
                    symbols={"symbol": "BTC/USDT:USDT"},
                    columns={
                        "trade_id": trade_id,
                        "price": price,
                        "quantity": quantity,
                        "is_buyer_maker": is_buyer_maker,
                    },
                    at=timestamp_ns,
                )
                sender.flush()
        """
        Sender = _get_questdb_sender()
        return Sender.from_conf(f"tcp::addr={self.config.ilp_address};")

    # =========================================================================
    # Schema Management
    # =========================================================================

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Table name.

        Returns:
            True if table exists.
        """
        result = await self.query_one(
            "SELECT table_name FROM tables() WHERE table_name = $1",
            table_name,
        )
        return result is not None

    async def create_table_if_not_exists(
        self,
        table_name: str,
        schema: str,
        partition_by: str = "DAY",
    ) -> None:
        """
        Create a table if it doesn't exist.

        Args:
            table_name: Table name.
            schema: Column definitions (without CREATE TABLE wrapper).
            partition_by: Partition strategy (NONE, HOUR, DAY, WEEK, MONTH, YEAR).
        """
        if await self.table_exists(table_name):
            logger.debug(f"Table {table_name} already exists")
            return

        sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {schema}
            ) TIMESTAMP(timestamp) PARTITION BY {partition_by};
        """
        await self.execute(sql)
        logger.info(f"Created table {table_name} (partition by {partition_by})")


# =============================================================================
# Sync Client Wrapper
# =============================================================================


class SyncQuestDBClient:
    """
    Synchronous wrapper for QuestDBClient.

    Uses aequify.runtime.IsolatedLoop to run async operations from sync code.
    Useful for calling from Mojo or other sync contexts.

    Usage:
        from aequify.core.db.cold import SyncQuestDBClient

        client = SyncQuestDBClient()
        client.start()

        try:
            rows = client.query("SELECT * FROM trades LIMIT 10")
        finally:
            client.stop()
    """

    def __init__(
        self,
        config: QuestDBConfig | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize sync client wrapper.

        Args:
            config: QuestDB configuration.
            timeout: Default timeout for operations in seconds.
        """
        from aequify.runtime import IsolatedLoop

        self._config = config or DEFAULT_CONFIG
        self._timeout = timeout
        self._loop = IsolatedLoop("aeq-questdb")
        self._client: QuestDBClient | None = None

    def start(self) -> None:
        """Start the client and its isolated event loop."""
        self._loop.start()
        self._client = QuestDBClient(self._config)
        self._loop.run(self._client.initialize(), timeout=self._timeout)
        logger.info("SyncQuestDBClient started")

    def stop(self) -> None:
        """Stop the client and its isolated event loop."""
        if self._client:
            try:
                self._loop.run(self._client.close(), timeout=self._timeout)
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
            self._client = None
        self._loop.stop()
        logger.info("SyncQuestDBClient stopped")

    def __enter__(self) -> SyncQuestDBClient:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()

    def _ensure_running(self) -> QuestDBClient:
        """Ensure client is running."""
        if not self._client or not self._loop.is_running:
            raise RuntimeError("Client not started. Call start() first.")
        return self._client

    def query(self, sql: str, *args: Any, timeout: float | None = None) -> list[dict]:
        """Execute a SQL query and return results."""
        client = self._ensure_running()
        t = timeout or self._timeout
        return self._loop.run(client.query(sql, *args, timeout=t), timeout=t)

    def query_one(
        self, sql: str, *args: Any, timeout: float | None = None
    ) -> dict | None:
        """Execute a SQL query and return first row."""
        client = self._ensure_running()
        t = timeout or self._timeout
        return self._loop.run(client.query_one(sql, *args, timeout=t), timeout=t)

    def execute(self, sql: str, *args: Any, timeout: float | None = None) -> str:
        """Execute a SQL statement."""
        client = self._ensure_running()
        t = timeout or self._timeout
        return self._loop.run(client.execute(sql, *args, timeout=t), timeout=t)

    def create_sender(self):
        """Create an ILP sender for high-speed ingestion."""
        client = self._ensure_running()
        return client.create_sender()

    @property
    def config(self) -> QuestDBConfig:
        """Get configuration."""
        return self._config
