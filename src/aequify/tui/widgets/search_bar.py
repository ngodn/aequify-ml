"""
Search bar widget for symbol filtering.

Simplified from the old UrlBar - provides symbol search functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Label

from aequify.tui.help_data import HelpData
from aequify.tui.widgets.view_selection import ViewSelector


class SearchInput(Input):
    """Symbol search input."""

    help = HelpData(
        "Search Bar",
        """\
Enter a symbol to search for filtered trading pairs.
- Type to filter the symbol list
- Press Enter to search
- Use ctrl+f to quickly focus this bar
""",
    )

    BINDING_GROUP_TITLE = "Search Input"

    BINDINGS = [
        Binding("down", "app.focus_next", "Focus next", show=False),
        Binding("enter", "submit_search", "Search", show=False),
    ]

    @dataclass
    class SearchSubmitted(Message):
        """Posted when search is submitted."""

        query: str
        input: "SearchInput"

        @property
        def control(self) -> "SearchInput":
            return self.input

    def action_submit_search(self) -> None:
        """Submit the search query."""
        self.post_message(self.SearchSubmitted(query=self.value, input=self))


class SearchButton(Button, can_focus=False):
    """Button for submitting search."""

    pass


class SearchBar(Vertical):
    """
    Search bar for symbol filtering.

    Contains:
    - ViewSelector for switching between views (Symbols/Account)
    - Search input for filtering symbols
    - Search button
    """

    DEFAULT_CSS = """
    SearchBar {
        height: auto;
        padding: 0 1;
    }

    SearchBar #main-row {
        height: 3;
        width: 100%;
    }

    SearchBar #view-selector {
        width: auto;
        margin-right: 1;
    }

    SearchBar #search-input {
        width: 1fr;
    }

    SearchBar #search-button {
        width: auto;
        min-width: 10;
    }

    SearchBar #result-count {
        width: auto;
        margin-left: 1;
        color: $text-muted;
    }
    """

    # Reactive state
    result_count: reactive[int] = reactive(0)

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-row"):
            yield ViewSelector(id="view-selector")
            yield SearchInput(
                placeholder="Search symbols...",
                id="search-input",
            )
            yield Label("", id="result-count")
            yield SearchButton("Search", id="search-button")

    def watch_result_count(self, count: int) -> None:
        """Update result count label."""
        label = self.query_one("#result-count", Label)
        if count > 0:
            label.update(f"{count} results")
        else:
            label.update("")

    @on(SearchInput.SearchSubmitted)
    def on_search_submitted(self, event: SearchInput.SearchSubmitted) -> None:
        """Handle search submission."""
        # Bubble up to app level
        pass

    @on(Button.Pressed, "#search-button")
    def on_search_button_pressed(self, event: Button.Pressed) -> None:
        """Handle search button press."""
        search_input = self.query_one("#search-input", SearchInput)
        search_input.action_submit_search()

    @property
    def search_input(self) -> SearchInput:
        """Get the search input widget."""
        return self.query_one("#search-input", SearchInput)

    @property
    def query(self) -> str:
        """Get the current search query."""
        return self.search_input.value

    def set_query(self, query: str) -> None:
        """Set the search query."""
        self.search_input.value = query

    def focus_input(self) -> None:
        """Focus the search input."""
        self.search_input.focus()
