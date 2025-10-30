"""Holds the homepage of the application. This is the first screen that the user sees when they open the application."""

import logging
import math
from typing import Callable

from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QWidget, QGridLayout

from potentio_gui.ui.TemplateDelegate import TemplateDelegate


class Homepage(QWidget):
    def __init__(self, parent: QWidget, navigate: Callable[[], None]) -> None:
        """Initializes a new Homepage

        :param parent: The parent widget
        :param navigate: A function that will be called with a config object to navigate to the wizard and start it
        """
        super().__init__()

        self.parent = parent
        self.logger = logging.getLogger(__name__)
        self.layout = QGridLayout()

        # Library will be initialized in self.load_templates()
        self.library_templates = []
        self.library_templates_widgets = []
        self.templates_per_row = 0

        self.create_layout()

    def reload(self) -> None:
        """Reloads the homepage and the templates from disk"""
        self.create_layout()

    def create_layout(self, templates_per_row: int = 3) -> None:
        """Creates the layout to display the templates in a grid on the hompage

        :param templates_per_row: How many templates should be displayed per row
        """
        idx, idy = 0, 0
        for template in self.library_templates_widgets:
            self.layout.removeWidget(template)
            self.layout.addWidget(template, idy, idx)
            idx += 1
            if idx >= templates_per_row:
                idx = 0
                idy += 1
        self.layout.setRowStretch(templates_per_row, 1)
        self.layout.setColumnStretch(idy + 1, 1)
        self.setLayout(self.layout)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """The event that is called when the window is resized. This will update the layout of the templates to fit the new size

        :param event: Event properties
        """
        templates_per_row = math.floor(event.size().width() / TemplateDelegate.WIDTH)

        if self.templates_per_row == templates_per_row:
            event.accept()
            return

        self.create_layout(templates_per_row)
        self.templates_per_row = templates_per_row
        event.accept()
