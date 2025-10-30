"""Holds the main window of the application."""

import logging

from PyQt6.QtWidgets import QMainWindow

from potentio_gui.ui.Homepage import Homepage
from potentio_gui.ui.MainScrollArea import MainScrollArea
from potentio_gui.ui.Potentiometrie import PotentiometrieWindow

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """The Main Window of the application.

        This is the main window of the application. It holds the homepage or the wizard and the menubar.
        """
        super().__init__()

        self.setWindowTitle("Potentiometrie GUI")
        self.setMinimumSize(600, 400)
        self.resize(800, 600)

        def none() -> None:
            pass

        self.homepage = Homepage(self, none)
        self.potentiometrie = PotentiometrieWindow(self)
        self.scroll_area = MainScrollArea(self.potentiometrie)
        self.setCentralWidget(self.scroll_area)

        # self.setMenuBar(Menubar(self))
