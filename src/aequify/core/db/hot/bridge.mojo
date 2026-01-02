"""
HotStore Python Bridge - Exposes HotStore to Python code.

Uses a HotStoreRegistry struct exposed to Python to manage per-symbol stores.

Usage from Python:
    import hot

    # Create registry
    registry = hot.HotStoreRegistry()

    # Add trade
    registry.add_trade_raw("BTCUSDT", 12345, 50000.0, 0.1, 1704067200000, False)

    # Query window
    window = registry.get_window("BTCUSDT", 0, 60)
    print(window)  # {"start_idx": 0, "end_idx": 100, "count": 100}
"""

from os import abort
from memory import UnsafePointer, alloc

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder

from .models import Trade
from .store import HotStore


# =============================================================================
# Constants
# =============================================================================

comptime MAX_SYMBOLS: Int = 50


# =============================================================================
# HotStoreRegistry - Exposed to Python
# =============================================================================


struct HotStoreRegistry(Defaultable, Movable, Representable):
    """
    Registry of HotStores, one per symbol.

    Manages multiple HotStore instances for different trading pairs.
    Exposed to Python as a type.

    Uses UnsafePointer for heap-allocated HotStore instances since
    HotStore contains List which is not Copyable.
    """

    # Heap-allocated HotStore pointers (since HotStore is not Copyable)
    var _store_ptrs: List[UnsafePointer[HotStore, MutOrigin.external]]
    var _symbols: List[String]

    fn __init__(out self):
        """Create empty registry."""
        self._store_ptrs = List[UnsafePointer[HotStore, MutOrigin.external]]()
        self._symbols = List[String]()

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self._store_ptrs = other._store_ptrs^
        self._symbols = other._symbols^

    fn __del__(deinit self):
        """Destructor - free all heap-allocated HotStore instances."""
        for i in range(len(self._store_ptrs)):
            if self._store_ptrs[i]:
                self._store_ptrs[i].destroy_pointee()
                self._store_ptrs[i].free()

    fn __repr__(self) -> String:
        return String("HotStoreRegistry(symbols=", len(self._symbols), ")")

    # -------------------------------------------------------------------------
    # Helper to get mutable self pointer
    # -------------------------------------------------------------------------

    @staticmethod
    fn _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self, MutAnyOrigin]:
        """Get mutable pointer to self from Python object."""
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Failed to downcast self: ", e))

    fn _find_index(self, symbol: String) -> Int:
        """Find store index for symbol, returns -1 if not found."""
        for i in range(len(self._symbols)):
            if self._symbols[i] == symbol:
                return i
        return -1

    fn _get_or_create_index(mut self, symbol: String) raises -> Int:
        """Get or create store index for symbol."""
        var idx = self._find_index(symbol)
        if idx >= 0:
            return idx

        if len(self._symbols) >= MAX_SYMBOLS:
            raise Error("Maximum number of symbols reached: " + String(MAX_SYMBOLS))

        # Allocate new HotStore on heap using freestanding alloc()
        var store_ptr = alloc[HotStore](1)
        store_ptr.init_pointee_move(HotStore())

        idx = len(self._store_ptrs)
        self._store_ptrs.append(store_ptr)
        self._symbols.append(symbol)
        return idx

    # -------------------------------------------------------------------------
    # Python-exposed instance methods (using def_py_method pattern)
    # -------------------------------------------------------------------------

    @staticmethod
    fn add_trade_raw(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Add a trade from raw values.
        Args: symbol, trade_id, price, quantity, timestamp_ms, is_buyer_maker.
        """
        var self_ptr = Self._get_self_ptr(py_self)

        # Parse positional arguments
        var symbol = py_args[0]
        var trade_id = py_args[1]
        var price = py_args[2]
        var quantity = py_args[3]
        var timestamp_ms = py_args[4]
        var is_buyer_maker = py_args[5]

        var sym = String(symbol)
        var idx = self_ptr[]._get_or_create_index(sym)

        var trade = Trade(
            trade_id=Int64(Int(trade_id)),
            price=Float64(price),
            quantity=Float64(quantity),
            timestamp_ms=Int64(Int(timestamp_ms)),
            is_buyer_maker=Bool(is_buyer_maker),
        )

        self_ptr[]._store_ptrs[idx][].add_trade(trade)

        return PythonObject(True)

    @staticmethod
    fn get_window(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get trade window indices. Args: symbol, start_minute, end_minute."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var start_minute = py_args[1]
        var end_minute = py_args[2]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()

        if idx < 0:
            result.__setitem__("start_idx", value=0)
            result.__setitem__("end_idx", value=0)
            result.__setitem__("count", value=0)
            return result

        var window = self_ptr[]._store_ptrs[idx][].get_window(
            Int(start_minute), Int(end_minute)
        )

        result.__setitem__("start_idx", value=window.start_idx)
        result.__setitem__("end_idx", value=window.end_idx)
        result.__setitem__("count", value=len(window))

        return result

    @staticmethod
    fn get_1m(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get 1-minute window. Args: symbol, hour, minute."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var hour = py_args[1]
        var minute = py_args[2]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()

        if idx < 0:
            result.__setitem__("start_idx", value=0)
            result.__setitem__("end_idx", value=0)
            result.__setitem__("count", value=0)
            return result

        var window = self_ptr[]._store_ptrs[idx][].get_1m(Int(hour), Int(minute))

        result.__setitem__("start_idx", value=window.start_idx)
        result.__setitem__("end_idx", value=window.end_idx)
        result.__setitem__("count", value=len(window))

        return result

    @staticmethod
    fn get_5m(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get 5-minute window. Args: symbol, hour, block (0-11)."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var hour = py_args[1]
        var block = py_args[2]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()

        if idx < 0:
            result.__setitem__("start_idx", value=0)
            result.__setitem__("end_idx", value=0)
            result.__setitem__("count", value=0)
            return result

        var window = self_ptr[]._store_ptrs[idx][].get_5m(Int(hour), Int(block))

        result.__setitem__("start_idx", value=window.start_idx)
        result.__setitem__("end_idx", value=window.end_idx)
        result.__setitem__("count", value=len(window))

        return result

    @staticmethod
    fn get_15m(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get 15-minute window. Args: symbol, hour, quarter (0-3)."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var hour = py_args[1]
        var quarter = py_args[2]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()

        if idx < 0:
            result.__setitem__("start_idx", value=0)
            result.__setitem__("end_idx", value=0)
            result.__setitem__("count", value=0)
            return result

        var window = self_ptr[]._store_ptrs[idx][].get_15m(Int(hour), Int(quarter))

        result.__setitem__("start_idx", value=window.start_idx)
        result.__setitem__("end_idx", value=window.end_idx)
        result.__setitem__("count", value=len(window))

        return result

    @staticmethod
    fn get_1h(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get 1-hour window. Args: symbol, hour."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var hour = py_args[1]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()

        if idx < 0:
            result.__setitem__("start_idx", value=0)
            result.__setitem__("end_idx", value=0)
            result.__setitem__("count", value=0)
            return result

        var window = self_ptr[]._store_ptrs[idx][].get_1h(Int(hour))

        result.__setitem__("start_idx", value=window.start_idx)
        result.__setitem__("end_idx", value=window.end_idx)
        result.__setitem__("count", value=len(window))

        return result

    @staticmethod
    fn get_stats(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get store statistics for a symbol. Args: symbol."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()

        if idx < 0:
            result.__setitem__("symbol", value=sym)
            result.__setitem__("exists", value=False)
            return result

        # Access via pointer to avoid copying (HotStore is not Copyable)
        var store_ptr = self_ptr[]._store_ptrs[idx]

        result.__setitem__("symbol", value=sym)
        result.__setitem__("exists", value=True)
        result.__setitem__("trade_count", value=len(store_ptr[]))
        result.__setitem__("active_minutes", value=store_ptr[].active_minutes())
        result.__setitem__("total_added", value=store_ptr[].total_added)
        result.__setitem__("total_cleared", value=store_ptr[].total_cleared)
        result.__setitem__("memory_bytes", value=store_ptr[].memory_bytes())

        return result

    @staticmethod
    fn list_symbols(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """List all registered symbols."""
        var self_ptr = Self._get_self_ptr(py_self)

        var result = Python.list()

        for i in range(len(self_ptr[]._symbols)):
            result.append(self_ptr[]._symbols[i])

        return result

    @staticmethod
    fn clear_before(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Clear trades before a minute of day. Args: symbol, minute."""
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var minute = py_args[1]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        if idx < 0:
            return PythonObject(False)

        self_ptr[]._store_ptrs[idx][].clear_before(Int(minute))

        return PythonObject(True)

    @staticmethod
    fn symbol_count(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """Get number of registered symbols."""
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(len(self_ptr[]._symbols))

    @staticmethod
    fn get_trades_in_window(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """
        Get all trades in a time window as a list of dicts.

        Args: symbol, start_minute, end_minute.

        Returns: List of dicts with trade_id, price, quantity, timestamp_ms, is_buyer_maker.
        """
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var start_minute = py_args[1]
        var end_minute = py_args[2]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.list()

        if idx < 0:
            return result

        var store_ptr = self_ptr[]._store_ptrs[idx]
        var window = store_ptr[].get_window(Int(start_minute), Int(end_minute))

        if window.is_empty():
            return result

        # Extract trades in window
        for i in range(window.start_idx, window.end_idx):
            var trade = store_ptr[].trades[i]
            var trade_dict = Python.dict()
            trade_dict.__setitem__("trade_id", value=Int(trade.trade_id))
            trade_dict.__setitem__("price", value=Float64(trade.price))
            trade_dict.__setitem__("quantity", value=Float64(trade.quantity))
            trade_dict.__setitem__("timestamp_ms", value=Int(trade.timestamp_ms))
            trade_dict.__setitem__("is_buyer_maker", value=Bool(trade.is_buyer_maker))
            result.append(trade_dict)

        return result

    @staticmethod
    fn flush_window(
        py_self: PythonObject,
        py_args: PythonObject,
        py_kwargs: PythonObject,
    ) raises -> PythonObject:
        """
        Get trades in window and clear them from hot store.

        Args: symbol, start_minute, end_minute.

        Returns: Dict with 'trades' (list of trade dicts) and 'count'.
        """
        var self_ptr = Self._get_self_ptr(py_self)

        var symbol = py_args[0]
        var start_minute = py_args[1]
        var end_minute = py_args[2]

        var sym = String(symbol)
        var idx = self_ptr[]._find_index(sym)

        var result = Python.dict()
        var trades_list = Python.list()

        if idx < 0:
            result.__setitem__("trades", value=trades_list)
            result.__setitem__("count", value=0)
            return result

        var store_ptr = self_ptr[]._store_ptrs[idx]
        var window = store_ptr[].get_window(Int(start_minute), Int(end_minute))

        if window.is_empty():
            result.__setitem__("trades", value=trades_list)
            result.__setitem__("count", value=0)
            return result

        # Extract trades in window
        for i in range(window.start_idx, window.end_idx):
            var trade = store_ptr[].trades[i]
            var trade_dict = Python.dict()
            trade_dict.__setitem__("trade_id", value=Int(trade.trade_id))
            trade_dict.__setitem__("price", value=Float64(trade.price))
            trade_dict.__setitem__("quantity", value=Float64(trade.quantity))
            trade_dict.__setitem__("timestamp_ms", value=Int(trade.timestamp_ms))
            trade_dict.__setitem__("is_buyer_maker", value=Bool(trade.is_buyer_maker))
            trades_list.append(trade_dict)

        var count = len(window)

        # Clear the window from hot store
        store_ptr[].clear_before(Int(end_minute))

        result.__setitem__("trades", value=trades_list)
        result.__setitem__("count", value=count)

        return result


# =============================================================================
# Python Module Init
# =============================================================================


@export
fn PyInit_hot() -> PythonObject:
    """Initialize the hot Python module."""
    try:
        var m = PythonModuleBuilder("hot")

        # Add HotStoreRegistry type using def_py_method for all methods
        _ = (
            m.add_type[HotStoreRegistry]("HotStoreRegistry")
            .def_init_defaultable[HotStoreRegistry]()
            .def_py_method[HotStoreRegistry.add_trade_raw]("add_trade_raw")
            .def_py_method[HotStoreRegistry.get_window]("get_window")
            .def_py_method[HotStoreRegistry.get_1m]("get_1m")
            .def_py_method[HotStoreRegistry.get_5m]("get_5m")
            .def_py_method[HotStoreRegistry.get_15m]("get_15m")
            .def_py_method[HotStoreRegistry.get_1h]("get_1h")
            .def_py_method[HotStoreRegistry.get_stats]("get_stats")
            .def_py_method[HotStoreRegistry.list_symbols]("list_symbols")
            .def_py_method[HotStoreRegistry.clear_before]("clear_before")
            .def_py_method[HotStoreRegistry.symbol_count]("symbol_count")
            .def_py_method[HotStoreRegistry.get_trades_in_window]("get_trades_in_window")
            .def_py_method[HotStoreRegistry.flush_window]("flush_window")
        )

        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
