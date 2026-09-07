import os
from pathlib import Path

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "finsight.log"


def query_preview(value: str, max_length: int = 120) -> str:
    """Return a bounded, single-line representation for diagnostic logs."""
    preview = " ".join(str(value).split())
    if len(preview) > max_length:
        return preview[: max_length - 3] + "..."
    return preview


def configure_logging() -> None:
    """Configure application sinks once, using environment-controlled verbosity."""
    if getattr(configure_logging, "_configured", False):
        return

    log_level = os.getenv("FINSIGHT_LOG_LEVEL", "INFO").upper()
    log_file = Path(os.getenv("FINSIGHT_LOG_FILE", str(DEFAULT_LOG_FILE))).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {name}:{function}:{line} - {message}",
        colorize=True,
    )
    logger.add(
        str(log_file),
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    )
    configure_logging._configured = True
    logger.info("Logging configured level={} file={}", log_level, log_file)


configure_logging()
