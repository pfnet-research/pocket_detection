import logging
from dataclasses import dataclass, field
from typing import Dict, Optional


def str_to_logger_enum(s):
    if s == "notset":
        return logging.NOTSET
    if s == "debug":
        return logging.DEBUG
    if s == "info":
        return logging.INFO
    if s == "warning":
        return logging.WARNING
    if s == "error":
        return logging.ERROR
    if s == "critical":
        return logging.CRITICAL
    raise RuntimeError(f"unknown log level: {s}")


@dataclass
class LoggerConfig:
    level: str = "info"
    format: str = "[%(levelname)1.1s %(module)s:%(funcName)s] %(message)s"
    modules: Dict[str, str] = field(default_factory=dict)
    rdkit_level: Optional[str] = None


def apply_logger_config(config):
    logging.basicConfig(
        level=str_to_logger_enum(config.level), format=config.format, force=True
    )
    for k, v in config.modules.items():
        logging.getLogger(k).setLevel(str_to_logger_enum(v))
