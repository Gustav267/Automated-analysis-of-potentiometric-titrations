import sys

import lmfit
import matplotlib
import numpy
import pandas
import scipy
from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QWidget,
    QDialogButtonBox,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QApplication,
    QHBoxLayout,
)

from potentio_gui.ui.Line import QHLine


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget):
        super().__init__(parent)

        self.setWindowTitle("Über Potentiometrie Auswertung")
        self.setWindowIcon(parent.windowIcon())

        font_bold_underline = QFont()
        font_bold_underline.setBold(True)
        font_bold_underline.setUnderline(True)

        # Dialog layout
        layout = QVBoxLayout()

        # Logo and self
        title = QLabel("Potentiometrie Auswertung")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 24))
        app_logo = QLabel()
        app_logo.setPixmap(QIcon.fromTheme("template").pixmap(32, 32))
        name_layout = QHBoxLayout()
        name_layout.addWidget(app_logo)
        name_layout.addWidget(title)
        name_layout.addStretch(0)
        name_widget = QWidget(self)
        name_widget.setLayout(name_layout)
        layout.addWidget(name_widget)

        # Application information
        layout.addWidget(
            QLabel(f"Potentiometrie Auswertung ({QApplication.applicationVersion()})")
        )
        app_homepage = QLabel(
            'Homepage: <a href="https://github.com/Gustav267/Automated-analysis-of-potentiometric-titrations/">https://github.com/Gustav267/Automated-analysis-of-potentiometric-titrations/</a>'
        )
        app_homepage.setOpenExternalLinks(True)
        layout.addWidget(app_homepage)
        app_issues = QLabel(
            'Issues/Bugs: <a href="https://github.com/Gustav267/Automated-analysis-of-potentiometric-titrations/issues/">https://github.com/Gustav267/Automated-analysis-of-potentiometric-titrations/issues/</a>'
        )
        app_issues.setOpenExternalLinks(True)
        layout.addWidget(app_issues)

        layout.addWidget(QHLine())

        # Authors
        authors = QLabel("Entwickler")
        authors.setFont(font_bold_underline)
        layout.addWidget(authors)
        author1 = QLabel(
            'Max Fehlinger &lt;<a href="mailto:mafelp@proton.me">mafelp@proton.me</a>&gt;'
        )
        author1.setOpenExternalLinks(True)
        layout.addWidget(author1)
        author2 = QLabel("Gustav Brauer")
        layout.addWidget(author2)

        layout.addWidget(QHLine())

        # Components
        components = QLabel("Benutzte Bibliotheken")
        components.setFont(font_bold_underline)
        layout.addWidget(components)

        libraries = [
            {
                "name": "lmfit",
                "version": lmfit.__version__,
                "link": "https://lmfit.github.io//lmfit-py/",
                "license": "BSD-3-Clause",
            },
            {
                "name": "matplotlib",
                "version": matplotlib.__version__,
                "link": "https://matplotlib.org/",
                "license": "PSFLv2",
            },
            {
                "name": "numpy",
                "version": numpy.__version__,
                "link": "https://numpy.org/",
                "license": "BSD License",
            },
            {
                "name": "pandas",
                "version": pandas.__version__,
                "link": "https://pandas.pydata.org/",
                "license": "BSD-3-Clause",
            },
            {
                "name": "pyxdg",
                "version": "0.28",
                "link": "http://freedesktop.org/wiki/Software/pyxdg",
                "license": "LGPLv2",
            },
            {
                "name": "scipy",
                "version": scipy.__version__,
                "link": "https://scipy.org/",
                "license": "BSD License",
            },
            {
                "name": "Python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "link": "https://python.org/",
                "license": "PSFv2",
            },
            {
                "name": "PyQt6",
                "version": PYQT_VERSION_STR,
                "link": "https://www.riverbankcomputing.com/software/pyqt/",
                "license": "GPLv3",
            },
            {
                "name": "Qt",
                "version": QT_VERSION_STR,
                "link": "https://doc.qt.io/qt-6/",
                "license": "GPLv3",
            },
        ]
        self.library_widgets: list[QWidget] = []
        for lib in libraries:
            label = QLabel(
                f'<a href="{lib["link"]}"> {lib["name"]} ({lib["version"]})</a>, {lib["license"]}',
                parent=self,
            )
            label.setOpenExternalLinks(True)
            self.library_widgets.append(label)
        for widget in self.library_widgets:
            layout.addWidget(widget)

        self.about_qt_button = QPushButton("About Qt")
        self.about_qt_button.clicked.connect(QApplication.aboutQt)
        layout.addWidget(self.about_qt_button)

        layout.addWidget(QHLine())

        # Copyright
        copyright_label = QLabel("© 2025 Potentiometrie Auswertung Contributors")
        layout.addWidget(copyright_label)

        # Close button
        self.ok_button = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.accepted.connect(self.accept)
        layout.addWidget(self.ok_button)

        # Finalize layout
        self.setLayout(layout)
