import logging

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QWidget, QMessageBox


def error_exit(
    parent: QWidget, msg: str, e: Exception, exit_code: int, logger: logging.Logger
) -> None:
    """Exits the application, but shows an error message beforehand.

    :param parent: The parent element that will be used by PyQt
    :param msg: The message that will be shown to the user
    :param e: The exception that was raised
    :param exit_code: The exit code for the application
    :param logger: The logger object to log the error message
    """
    logger.critical(msg, exc_info=e)
    QMessageBox.critical(parent, "Error", msg + f"\n\n{e}")
    QCoreApplication.exit(exit_code)
