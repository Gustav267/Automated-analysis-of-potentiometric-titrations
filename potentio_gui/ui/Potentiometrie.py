import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout

from potentio_gui.lib.potentiometrie import create_plot
from potentio_gui.ui.Line import QHLine, QVLine
from potentio_gui.ui.PotentiometrieWidgets import OptionalDatapoint
from potentio_gui.ui.PotentiometrieWidgets.DataInput import DataInput
from potentio_gui.ui.PotentiometrieWidgets.MenuBar import MenuBar
from potentio_gui.ui.PotentiometrieWidgets.PlotCanvas import PlotCanvas


class PotentiometrieWindow(QWidget):
    def __init__(
        self,
        parent: QWidget,
    ) -> None:
        # Set up general stuff
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.logger = logging.getLogger(__name__)
        self.parent = parent
        self.layout = QVBoxLayout()

        self.data_input = DataInput(self.update_data, parent=self)
        self.data = []

        # Setup Menu Bar
        self.layout.addWidget(
            MenuBar(
                self.data_input.add_row,
                self.data_input.reset_data,
                self.generate_plot,
                parent=self,
            )
        )
        self.layout.addWidget(QHLine())

        subwidget = QWidget(self)
        sublayout = QHBoxLayout(self)
        sublayout.addWidget(self.data_input)
        sublayout.addWidget(QVLine())
        self.plot_canvas = PlotCanvas(parent=self)
        sublayout.addWidget(
            self.plot_canvas,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        subwidget.setLayout(sublayout)
        self.layout.addWidget(subwidget)

        # Pin all the items to the top
        self.layout.insertStretch(-1, 1)
        self.setLayout(self.layout)

    def update_data(self, data: list[OptionalDatapoint]):
        self.logger.debug("Updating Input data...")
        self.logger.debug(data)
        self.data = data

    def generate_plot(self):
        vol_naoh = [
            x.volume for x in self.data if x.enabled and (x.ph <= 4 or x.ph >= 10)
        ]
        ph_wert = [x.ph for x in self.data if x.enabled and (x.ph <= 4 or x.ph >= 10)]
        self.logger.debug(vol_naoh)
        self.logger.debug(ph_wert)

        create_plot(vol_naoh, ph_wert, self.plot_canvas.get_drawing_canvas())
        self.plot_canvas.draw_canvas()
