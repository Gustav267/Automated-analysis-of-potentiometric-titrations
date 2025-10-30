from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width, height, dpi):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)


class PlotCanvas(QWidget):
    def __init__(self, width=5, height=4, dpi=100, parent=None):
        super().__init__(parent=parent)
        self.__canvas = MplCanvas(width, height, dpi)

        toolbar = NavigationToolbar2QT(self.__canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.__canvas)
        self.setLayout(layout)

    def get_drawing_canvas(self) -> Axes:
        return self.__canvas.figure.subplots()

    def draw_canvas(self):
        self.__canvas.draw()

    def clear_canvas(self):
        self.__canvas.axes.cla()
