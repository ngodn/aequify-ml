"""Window indices model - result of time window queries."""


@fieldwise_init
struct WindowIndices(Copyable, Movable, ImplicitlyCopyable, Sized):
    """
    Result of a window query - start and end indices into trades array.

    Fields:
        start_idx: Starting index (inclusive).
        end_idx: Ending index (exclusive).
    """
    var start_idx: Int
    var end_idx: Int

    fn __init__(out self):
        self.start_idx = 0
        self.end_idx = 0

    fn __len__(self) -> Int:
        """Number of trades in this window."""
        if self.end_idx > self.start_idx:
            return self.end_idx - self.start_idx
        return 0

    fn is_empty(self) -> Bool:
        """Check if window contains no trades."""
        return self.start_idx >= self.end_idx
