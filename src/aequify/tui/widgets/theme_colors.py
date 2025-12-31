"""
Theme-aware color helpers for Rich Text styling.

Provides a mixin class that fetches colors from Textual theme variables
and provides them for use in Rich Text objects.

NOTE: Textual theme_variables can contain CSS-style values like "auto 60%"
which are NOT valid Rich style strings. This mixin only uses the core
color variables (success, error, warning, accent, primary, secondary)
which are guaranteed to be valid hex colors.
"""

from __future__ import annotations


class ThemeColorsMixin:
    """
    Mixin class providing theme-aware color methods for widgets.

    Use this mixin in widgets that need to create Rich Text with
    colors that match the current Textual theme.

    Example usage:
        class MyPane(VerticalScroll, ThemeColorsMixin):
            def _build_display(self) -> Text:
                text = Text()
                text.append("Success!", style=self.success_style)
                text.append("Error!", style=self.error_style)
                return text
    """

    def _get_theme_vars(self) -> dict[str, str]:
        """Get theme variables from the app."""
        if hasattr(self, "app"):
            return getattr(self.app, "theme_variables", {})
        return {}

    # Primary semantic colors - these are guaranteed to be valid hex colors
    @property
    def success_color(self) -> str:
        """Green/success color from theme."""
        return self._get_theme_vars().get("success", "#00FF00")

    @property
    def error_color(self) -> str:
        """Red/error color from theme."""
        return self._get_theme_vars().get("error", "#FF0000")

    @property
    def warning_color(self) -> str:
        """Yellow/warning color from theme."""
        return self._get_theme_vars().get("warning", "#FFFF00")

    @property
    def accent_color(self) -> str:
        """Accent color from theme."""
        return self._get_theme_vars().get("accent", "#00FFFF")

    @property
    def primary_color(self) -> str:
        """Primary color from theme."""
        return self._get_theme_vars().get("primary", "#FFFFFF")

    @property
    def secondary_color(self) -> str:
        """Secondary color from theme."""
        return self._get_theme_vars().get("secondary", "#AAAAAA")

    # Rich style strings (for use in Text.append())
    @property
    def success_style(self) -> str:
        """Style string for success/positive values."""
        return self.success_color

    @property
    def error_style(self) -> str:
        """Style string for error/negative values."""
        return self.error_color

    @property
    def warning_style(self) -> str:
        """Style string for warning values."""
        return self.warning_color

    @property
    def accent_style(self) -> str:
        """Style string for accent values."""
        return self.accent_color

    @property
    def muted_style(self) -> str:
        """Style string for muted/dim text. Uses Rich 'dim' style."""
        return "dim"

    @property
    def bold_success_style(self) -> str:
        """Bold success style."""
        return f"bold {self.success_color}"

    @property
    def bold_error_style(self) -> str:
        """Bold error style."""
        return f"bold {self.error_color}"

    @property
    def bold_warning_style(self) -> str:
        """Bold warning style."""
        return f"bold {self.warning_color}"

    @property
    def bold_accent_style(self) -> str:
        """Bold accent style."""
        return f"bold {self.accent_color}"

    # Convenience methods for common patterns
    def pnl_style(self, value: float) -> str:
        """Get style for PnL value (green if positive, red if negative)."""
        if value >= 0:
            return self.success_color
        return self.error_color

    def bold_pnl_style(self, value: float) -> str:
        """Get bold style for PnL value."""
        if value >= 0:
            return self.bold_success_style
        return self.bold_error_style

    def side_style(self, is_long: bool) -> str:
        """Get style for position side (green for long, red for short)."""
        if is_long:
            return self.success_color
        return self.error_color

    def bold_side_style(self, is_long: bool) -> str:
        """Get bold style for position side."""
        if is_long:
            return self.bold_success_style
        return self.bold_error_style

    def change_style(self, value: float) -> str:
        """Get style for change/delta value (green if positive, red if negative)."""
        return self.pnl_style(value)
