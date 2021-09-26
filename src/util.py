import logging


def set_logging_level(logger: logging.Logger, level: str):
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    if level not in levels:
        raise ValueError(
            f'unknown logging level \'{level}\', '
            f'--log must be one of: {" | ".join(levels.keys())}'
        )

    logger.setLevel(levels[level])
