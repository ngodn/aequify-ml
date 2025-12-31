from dataclasses import dataclass
from typing import Literal

from rich.console import RenderableType
from textual import on
from textual.message import Message
from textual.widgets import Select

from aequify.tui.help_data import HelpData
from aequify.tui.widgets.select import AequifySelect

ViewType = Literal["symbols", "account"]


class ViewSelector(AequifySelect[str]):
    """Selector for switching between Symbols and Account views."""

    help = HelpData(
        title="View Selector",
        description="""\
Select the view to display.
- Symbols: View and manage trading symbols
- Account: View account information
""",
    )

    BINDING_GROUP_TITLE = "View Selector"

    def __init__(
        self,
        *,
        prompt: str = "View",
        value: str = "symbols",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        tooltip: RenderableType | None = None,
    ):
        super().__init__(
            [
                ("Symbols", "symbols"),
                ("Account", "account"),
            ],
            prompt=prompt,
            allow_blank=False,
            value=value,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            tooltip=tooltip,
        )

    @dataclass
    class ViewChanged(Message):
        value: ViewType
        select: "ViewSelector"

        @property
        def control(self) -> "ViewSelector":
            return self.select

    @on(Select.Changed)
    def view_selected(self, event: Select.Changed) -> None:
        event.stop()
        if event.value is not Select.BLANK:
            self.post_message(ViewSelector.ViewChanged(value=event.value, select=self))
