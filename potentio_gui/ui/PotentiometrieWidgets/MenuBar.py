import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QPushButton,
    QSpacerItem,
    QSizePolicy,
)
from black.trans import Callable

from potentio_gui.ui.PotentiometrieWidgets import OptionalDatapoint
from potentio_gui.ui.dialog.ExcelImport import ExcelImport


class MenuBar(QWidget):
    def __init__(
        self,
        add_row: Callable[[], None],
        change_import_data: Callable[[list[OptionalDatapoint]], None],
        generate_plot: Callable[[], None],
        parent: QWidget = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.change_import_data = change_import_data
        super().__init__(parent)

        self.layout = QHBoxLayout()

        import_excel_button = QPushButton(
            icon=QIcon.fromTheme("document-import"),
            parent=self,
            text="Import aus Excel",
        )
        import_excel_button.clicked.connect(self.import_button)
        add_button = QPushButton(
            icon=QIcon.fromTheme("list-add"),
            parent=self,
            text="Neuen Datenpunkt hinzufügen",
        )
        add_button.clicked.connect(add_row)
        generate_plot_button = QPushButton(
            icon=QIcon.fromTheme("viewimage"),
            parent=self,
            text="Plot Generieren",
        )
        generate_plot_button.clicked.connect(generate_plot)

        self.layout.setAlignment(None, Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(import_excel_button)
        self.layout.addWidget(add_button)
        self.layout.addSpacerItem(
            QSpacerItem(
                0, 0, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum
            )
        )
        self.layout.addWidget(generate_plot_button)
        self.setLayout(self.layout)

    def import_button(self):
        self.logger.debug("Starting Excel Import...")

        def on_accept(data: list[OptionalDatapoint]):
            self.logger.debug("Received Excel data!")
            self.change_import_data(data)

        ExcelImport(on_accept, parent=self).showNormal()
