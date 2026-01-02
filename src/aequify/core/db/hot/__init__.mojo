"""Hot storage module - In-memory trade storage with time-window queries."""

from .models import Trade, WindowIndices
from .store import HotStore
from .bridge import PyInit_hot
