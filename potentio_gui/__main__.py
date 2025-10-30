import logging
import sys

from PyQt6.QtWidgets import QApplication

from potentio_gui.ui.MainWindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        level="DEBUG",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Potentiometrie GUI version %s" % "v0.0.1")

    main_window = MainWindow()
    main_window.show()

    try:
        app.exec()
    except Exception as e:
        logger.critical("An unexpected error occurred", exc_info=e)
