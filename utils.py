import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def get_logger(
    global_level: int = logging.WARNING,
    package_level: int = logging.INFO,
) -> logging.Logger:
    """Configure a python logger using Rich

    Args:
        global_level (int, optional): Logging level for all dependencies. Defaults to logging.WARNING.
        package_level (int, optional): Logging level for application specific code. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured Logger object
    """

    log_file = Path(__file__).resolve().parent / "debug.log"
    rich_file_console = Console(
        file=log_file.open("a", encoding="utf-8"),
        force_terminal=False,
        color_system=None,
    )

    # Global settings
    logging.basicConfig(
        level=global_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=rich_file_console)],
        force=True,
    )

    # Our handler for our package
    # If not using a package, write a name instead of __package__ as this will be blank
    log = logging.getLogger(__package__)
    log.setLevel(package_level)

    return log
