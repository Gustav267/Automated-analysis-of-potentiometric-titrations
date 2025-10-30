from xml.etree.ElementInclude import include

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton, QTableWidget, QDoubleSpinBox, QCheckBox
from PyQt6.QtCore import Qt, QSize

from potentio_gui.ui.PotentiometrieWidgets import OptionalDatapoint


class DataInput(QTableWidget):
    def __init__(self, update_callback, parent=None):
        super().__init__(0, 4, parent)
        self.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)

        self.update_callback = update_callback

        # --- Table setup ---
        self.setHorizontalHeaderLabels(["Volumen", "pH-Wert", "Einbeziehen?", ""])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)

        # Styling header bold
        font = self.horizontalHeader().font()
        font.setBold(True)
        self.horizontalHeader().setFont(font)

    def add_row(self, volume=0.0, ph=0.0, included=True):
        row_position = self.rowCount()
        self.insertRow(row_position)

        # Volumen input
        volumen_input = QDoubleSpinBox()
        volumen_input.setDecimals(2)
        volumen_input.setRange(0, 9999)
        volumen_input.setValue(volume)
        volumen_input.valueChanged.connect(self._trigger_update)
        self.setCellWidget(row_position, 0, volumen_input)

        # pH-Wert input
        ph_input = QDoubleSpinBox()
        ph_input.setDecimals(2)
        ph_input.setRange(0, 14)
        ph_input.setValue(ph)
        ph_input.valueChanged.connect(self._trigger_update)
        self.setCellWidget(row_position, 1, ph_input)

        # Checkbox
        checkbox = QCheckBox()
        checkbox.stateChanged.connect(self._trigger_update)
        checkbox.setChecked(included)
        self.setCellWidget(row_position, 2, checkbox)

        # Remove button
        remove_button = QPushButton(
            icon=QIcon.fromTheme("delete"), parent=self, text=""
        )
        remove_button.setFixedWidth(30)
        remove_button.clicked.connect(lambda _, btn=remove_button: self.remove_row(btn))
        self.setCellWidget(row_position, 3, remove_button)

        self._trigger_update()

    def remove_row(self, button):
        """
        Find the row containing `button` in column 3 and remove it.
        This is robust against shifting indices because it searches the
        table at removal time.
        """
        for row in range(self.rowCount()):
            widget = self.cellWidget(row, 3)
            if widget is button:
                self.removeRow(row)
                self._trigger_update()
                return
        # If we get here, button wasn't found (shouldn't normally happen)
        self._trigger_update()

    def _trigger_update(self):
        """Call the provided update callback with the current table data."""
        data = []
        for row in range(self.rowCount()):
            volumen = self.cellWidget(row, 0).value()
            ph = self.cellWidget(row, 1).value()
            if included := self.cellWidget(row, 2):
                data.append(OptionalDatapoint(volumen, ph, included.isChecked()))
            else:
                data.append(OptionalDatapoint(volumen, ph, True))
        self.update_callback(data)

    def reset_data(self, data: list[OptionalDatapoint] = None):
        if data is None:
            data = []

        # Delete all rows
        for _ in range(self.rowCount()):
            self.removeRow(0)

        # Populate table with new data
        for row in data:
            self.add_row(row.volume, row.ph, row.enabled)

        self._trigger_update()

    def sizeHint(self) -> QSize:
        # Calculate total height dynamically based on rows + header
        height = self.horizontalHeader().height()
        for row in range(self.rowCount()):
            height += self.rowHeight(row)
        # Add some space for frame, margins, etc.
        height += 4
        return QSize(super().sizeHint().width(), height)
