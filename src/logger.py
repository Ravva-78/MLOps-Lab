#!/usr/bin/env python3
"""
Unified logging setup for MLOps pipeline stages.
"""

import logging
import sys


class ColorFormatter(logging.Formatter):

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)

    formatter = ColorFormatter(
        fmt="[%(levelname)s] %(name)s - %(message)s"
    )

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":

    logger = setup_logger(__name__)

    logger.info("Pipeline started")
    logger.warning("This is a warning")
    logger.error("An error occurred")