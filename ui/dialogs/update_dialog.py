"""What to do about an available update, matched to how ChromaKit was installed."""
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from logic.update_check import git_pull_command, pip_upgrade_command


class UpdateDialog(QDialog):
    """Shows the update command (git, editable) or just the release page (frozen, unknown).

    ChromaKit never runs pip or git itself; the user copies the command into their own terminal.
    """

    def __init__(self, version: str, url: str, kind: str, source_dir: str | None = None,
                 parent=None, executable: str | None = None, platform: str | None = None):
        super().__init__(parent)
        self.kind, self.url, self.version = kind, url, version
        self.setWindowTitle(f"ChromaKit {version} is available")
        self.command = None
        if kind == "git":
            self.command = pip_upgrade_command(executable or sys.executable)
            text = (f"ChromaKit {version} is available. Close ChromaKit, run this in a terminal, "
                    "then start it again:")
        elif kind == "editable" and source_dir:
            self.command = git_pull_command(source_dir)
            text = (f"ChromaKit {version} is available. This copy runs from the source folder "
                    f"{source_dir}. Close ChromaKit, run this in a terminal to update it, "
                    "then start it again:")
        else:
            self.kind = "frozen" if kind == "frozen" else "unknown"
            text = f"ChromaKit {version} is available. Download it from the release page."

        layout = QVBoxLayout(self)
        self._body = QLabel(text)
        self._body.setWordWrap(True)
        layout.addWidget(self._body)

        buttons = QHBoxLayout()
        if self.command:
            row = QHBoxLayout()
            field = QLineEdit(self.command)
            field.setReadOnly(True)
            field.setAccessibleName("Update command")
            field.setMinimumWidth(520)
            row.addWidget(field)
            copy = QPushButton("Copy")
            copy.setToolTip("Copy the command to the clipboard")
            copy.clicked.connect(lambda: self._copy_command(copy))
            row.addWidget(copy)
            layout.addLayout(row)
            if kind == "git" and (platform or sys.platform) == "win32":
                # PowerShell rejects a line starting with a quoted exe path unless it begins with &
                hint = QLabel("In PowerShell, type & and a space before the command.")
                hint.setWordWrap(True)
                layout.addWidget(hint)
            notes = QPushButton("Open release notes")
        else:
            notes = QPushButton("Open release page")
        notes.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.url)))
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(notes)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _copy_command(self, button: QPushButton) -> None:
        QGuiApplication.clipboard().setText(self.command)
        button.setText("Copied")      # status feedback (WCAG 4.1.3); stays until the dialog closes

    def body_text(self) -> str:
        return self._body.text()
