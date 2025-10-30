import logging

from PyQt6.QtGui import QClipboard, QIcon, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QPlainTextEdit,
    QLabel,
    QDialogButtonBox,
    QPushButton,
)
from black.trans import Callable

from potentio_gui.ui.PotentiometrieWidgets import OptionalDatapoint


class ExcelImport(QDialog):
    def __init__(
        self,
        on_accept: Callable[[list[OptionalDatapoint]], None],
        parent: QWidget = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.on_accept = on_accept
        super().__init__(parent=parent)

        self.setWindowTitle("Template Erstellen")
        self.setWindowIcon(parent.windowIcon())

        layout = QVBoxLayout(self)
        self.label = QLabel("Excel Dateien hier rein pasten (Strg + V)")
        layout.addWidget(self.label)
        paste_button = QPushButton(
            icon=QIcon.fromTheme("edit-paste"),
            parent=self,
            text="Aus Zwischenablage einfügen",
        )
        paste_button.clicked.connect(self.__paste)
        layout.addWidget(paste_button)
        self.text_frame = QPlainTextEdit(parent=self)
        layout.addWidget(self.text_frame)

        # Close button
        self.buttonbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)
        layout.addWidget(self.buttonbox)
        self.setLayout(layout)

    def __paste(self):
        text = QGuiApplication.clipboard().text(QClipboard.Mode.Clipboard)
        if text != "":
            self.text_frame.setPlainText(text)

    def accept(self):
        self.logger.debug("Accepting with text '%s'" % self.text_frame.toPlainText())
        text = self.text_frame.toPlainText()
        try:
            data = [row.split("\t") for row in text.splitlines(False)]
            return_data = [
                OptionalDatapoint(float(row[0]), float(row[1])) for row in data
            ]
            self.logger.debug(return_data)
            self.on_accept(return_data)
            super().accept()
        except IndexError as err:
            self.logger.exception(f"Could not parse copied Data! IndexError! {err}")
        except ValueError as err:
            self.logger.exception(f"Could not parse copied Data! ValueError! {err}")

    def reject(self):
        super().reject()
