"""Holds the Menubar of the application."""

import logging
from typing import Callable

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction, QIcon, QDesktopServices
from PyQt6.QtWidgets import QMenuBar, QApplication, QMessageBox, QMainWindow

from potentio_gui.ui.dialog.AboutDialog import AboutDialog

logger = logging.getLogger(__name__)


class Menubar(QMenuBar):
    def __init__(
            self,
            clear_data: Callable[[], None],
            import_data: Callable[[], None],
            add_data_row: Callable[[], None],
            generate_plot: Callable[[], None],
            parent: QMainWindow,
    ) -> None:
        """The Menubar of the application.

        :param parent: The parent widget of the menubar, should the main window.
        """
        super().__init__(parent)
        self.__clear_data = clear_data
        self.__import = import_data
        self.__add_row = add_data_row
        self.__generate_plot = generate_plot
        self.parent = parent

        self.__file_menu()
        self.__help_menu()

    def __help_menu(self) -> None:
        """A helper function to create the help menu and add it to the menubar."""
        help_menu = self.addMenu("&Help")

        help_action = QAction(QIcon.fromTheme("help"), "&Help", self)
        help_action.setShortcut("F1")
        help_action.setStatusTip("Show help")
        help_action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://cta.gym-altona.de/"))
        )
        help_menu.addAction(help_action)

        about_action = QAction("&About", self)
        about_action.setStatusTip("Über Potentiometrie Auswertung")
        about_action.triggered.connect(lambda: AboutDialog(self.parent).show())
        help_menu.addAction(about_action)

    def __file_menu(self) -> None:
        """A helper function to create the file menu and add it to the menubar."""
        file_menu = self.addMenu("&File")

        import_action = QAction(QIcon.fromTheme("document-import"), "Import", self)
        import_action.setShortcut("Ctrl+I")
        import_action.setStatusTip("Daten aus Excel importieren")
        import_action.triggered.connect(self.__import)
        file_menu.addAction(import_action)

        reset_action = QAction(QIcon.fromTheme("delete"), "Reset", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.setStatusTip("Daten Zurücksetzen")
        reset_action.triggered.connect(self.__clear_data)
        file_menu.addAction(reset_action)

        add_action = QAction(QIcon.fromTheme("list-add"), "Datenpunkt hinzufügen", self)
        add_action.setShortcut("Ctrl+N")
        add_action.setStatusTip("Datenpunkt hinzufügen")
        add_action.triggered.connect(self.__add_row)
        file_menu.addAction(add_action)

        generate_action = QAction(QIcon.fromTheme("viewimage"), "Plot generieren", self)
        generate_action.setShortcut("Ctrl+G")
        generate_action.setStatusTip("Plot generieren")
        generate_action.triggered.connect(self.__generate_plot)
        file_menu.addAction(generate_action)

        file_menu.addSeparator()

        exit_action = QAction(QIcon.fromTheme("application-exit"), "Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.__exit)
        file_menu.addAction(exit_action)

    def __exit(self) -> None:
        """A helper function to close the application."""
        self.parent.close()
        QApplication.exit(0)

    def __not_implemented(self) -> None:
        """A helper function to show a message box when a feature is not yet implemented."""
        logger.warning("Not implemented")
        QMessageBox.warning(self, "Not implemented", "Not implemented")
