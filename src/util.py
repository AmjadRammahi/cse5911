import logging


def set_logging_level(level: str):
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

    logging.basicConfig(
        level=levels[level],
        format='%(levelname)s - %(message)s'
    )
