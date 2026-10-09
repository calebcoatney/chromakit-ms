from PySide6.QtWidgets import QTableWidgetItem


class NumericItem(QTableWidgetItem):
    """Table item that shows formatted text but sorts by its numeric value."""

    def __init__(self, value, text: str):
        super().__init__(text)
        self._value = value

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return self._value < other._value
        return super().__lt__(other)
