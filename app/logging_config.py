import logging

from app.config import APP_NAME, LOG_LEVEL


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            f"{APP_NAME} | "
            "%(name)s | "
            "%(message)s"
        ),
    )