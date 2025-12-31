"""Trade model - single trade from exchange stream."""


@fieldwise_init
struct Trade(Copyable, Movable, ImplicitlyCopyable, Writable):
    """
    Single trade from exchange stream.

    Memory: ~33 bytes per trade.

    Fields:
        trade_id: Exchange trade ID (8 bytes).
        price: Execution price (8 bytes).
        quantity: Trade quantity (8 bytes).
        timestamp_ms: Unix timestamp in milliseconds (8 bytes).
        is_buyer_maker: True if buyer was maker (1 byte).
    """
    var trade_id: Int64
    var price: Float64
    var quantity: Float64
    var timestamp_ms: Int64
    var is_buyer_maker: Bool

    fn __init__(out self):
        """Empty trade (for array initialization)."""
        self.trade_id = 0
        self.price = 0.0
        self.quantity = 0.0
        self.timestamp_ms = 0
        self.is_buyer_maker = False

    fn notional(self) -> Float64:
        """Price * Quantity."""
        return self.price * self.quantity

    fn side(self) -> String:
        """Taker's side: 'buy' if taker bought, 'sell' if taker sold."""
        if self.is_buyer_maker:
            return "sell"
        return "buy"

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Trade(id=", self.trade_id,
            ", price=", self.price,
            ", qty=", self.quantity,
            ", ts=", self.timestamp_ms,
            ", side=", self.side(), ")"
        )
