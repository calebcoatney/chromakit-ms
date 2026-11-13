import pytest

pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton

from logic.update_check import pip_upgrade_command, git_pull_command
from ui.dialogs.update_dialog import UpdateDialog

URL = "https://github.com/calebcoatney/chromakit-ms/releases/tag/v9.9.9"


def _button(dlg, text):
    return next((b for b in dlg.findChildren(QPushButton) if b.text() == text), None)


def test_git_dialog_shows_exact_command_read_only_with_copy(qtbot, monkeypatch):
    dlg = UpdateDialog("9.9.9", URL, "git", None, executable="/opt/py/python")
    qtbot.addWidget(dlg)
    field = dlg.findChild(QLineEdit)
    assert field.isReadOnly() and field.text() == pip_upgrade_command("/opt/py/python")
    _button(dlg, "Copy").click()
    assert QApplication.clipboard().text() == pip_upgrade_command("/opt/py/python")
    assert _button(dlg, "Open release notes") is not None


def test_editable_dialog_shows_path_and_git_pull(qtbot):
    dlg = UpdateDialog("9.9.9", URL, "editable", "/src/chromakit-qt")
    qtbot.addWidget(dlg)
    assert dlg.findChild(QLineEdit).text() == git_pull_command("/src/chromakit-qt")
    assert "/src/chromakit-qt" in dlg.body_text()
    _button(dlg, "Copy").click()
    assert QApplication.clipboard().text() == 'git -C "/src/chromakit-qt" pull'


@pytest.mark.parametrize("kind", ["frozen", "unknown"])
def test_frozen_and_unknown_offer_only_the_release_page(qtbot, monkeypatch, kind):
    opened = []
    monkeypatch.setattr("ui.dialogs.update_dialog.QDesktopServices.openUrl", lambda u: opened.append(u.toString()))
    dlg = UpdateDialog("9.9.9", URL, kind, None)
    qtbot.addWidget(dlg)
    assert dlg.findChild(QLineEdit) is None and _button(dlg, "Copy") is None
    assert "9.9.9" in dlg.body_text()
    _button(dlg, "Open release page").click()
    assert opened == [URL]


def test_release_notes_button_opens_url(qtbot, monkeypatch):
    opened = []
    monkeypatch.setattr("ui.dialogs.update_dialog.QDesktopServices.openUrl", lambda u: opened.append(u.toString()))
    dlg = UpdateDialog("9.9.9", URL, "git", None)
    qtbot.addWidget(dlg)
    _button(dlg, "Open release notes").click()
    assert opened == [URL]


from PySide6.QtWidgets import QLabel

HINT = "In PowerShell, type & and a space before the command."


def _has_hint(dlg):
    return any(l.text() == HINT for l in dlg.findChildren(QLabel))


def test_powershell_hint_only_on_windows_git(qtbot):
    win = UpdateDialog("9.9.9", URL, "git", None, platform="win32")
    mac = UpdateDialog("9.9.9", URL, "git", None, platform="darwin")
    ed = UpdateDialog("9.9.9", URL, "editable", "C:\\src", platform="win32")
    for d in (win, mac, ed):
        qtbot.addWidget(d)
    assert _has_hint(win) and not _has_hint(mac) and not _has_hint(ed)
    assert win.findChild(QLineEdit).text() == pip_upgrade_command(__import__("sys").executable)


def test_copy_button_says_copied_after_copying(qtbot):
    # status feedback for the copy (WCAG 4.1.3): the button's text, and so its name, changes
    dlg = UpdateDialog("9.9.9", URL, "git", None, executable="/opt/py/python")
    qtbot.addWidget(dlg)
    copy = _button(dlg, "Copy")
    copy.click()
    assert copy.text() == "Copied"
    assert copy.accessibleName() in ("", "Copied")       # nothing pins a stale name
    assert QApplication.clipboard().text() == pip_upgrade_command("/opt/py/python")


def test_editable_dialog_also_shows_the_reinstall(qtbot):
    # an editable install from before 0.2.0 can't import chromakit_version after a pull alone
    from logic.update_check import editable_reinstall_command
    dlg = UpdateDialog("9.9.9", URL, "editable", "/src/chromakit-qt", executable="/opt/py/python")
    qtbot.addWidget(dlg)
    fields = dlg.findChildren(QLineEdit)
    assert [f.text() for f in fields] == [git_pull_command("/src/chromakit-qt"),
                                          editable_reinstall_command("/opt/py/python", "/src/chromakit-qt")]
    assert fields[1].text() == '"/opt/py/python" -m pip install -e "/src/chromakit-qt"'
    assert "in order" in dlg.body_text()
    copies = [b for b in dlg.findChildren(QPushButton) if b.text() == "Copy"]
    copies[1].click()
    assert QApplication.clipboard().text() == fields[1].text()
    assert copies[1].text() == "Copied" and copies[0].text() == "Copy"


def test_editable_dialog_on_windows_hints_powershell(qtbot):
    dlg = UpdateDialog("9.9.9", URL, "editable", "C:\\src", platform="win32", executable="C:\\py\\python.exe")
    qtbot.addWidget(dlg)
    assert any("PowerShell" in l.text() for l in dlg.findChildren(QLabel))
