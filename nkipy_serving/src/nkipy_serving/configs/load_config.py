"""Model load-format config helpers."""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    SHARDED_STATE = "sharded_state"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    MISTRAL = "mistral"
    LAYERED = "layered"
    REMOTE = "remote"


@dataclass
class LoadConfig:
    load_format: str | LoadFormat = LoadFormat.AUTO
    download_dir: str | None = None
    sub_dir: str | None = None
    model_loader_extra_config: str | dict[str, Any] | None = field(default_factory=dict)
    model_class: Any = None
    ignore_patterns: list[str] | str | None = None
    decryption_key_file: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(self.model_loader_extra_config)
        if self.model_loader_extra_config is None:
            self.model_loader_extra_config = {}

        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring patterns when downloading model files: %s",
                self.ignore_patterns,
            )
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if isinstance(self.load_format, LoadFormat):
            return
        if not isinstance(self.load_format, str):
            raise RuntimeError(
                "load_format must be a string or LoadFormat enum, "
                f"got {type(self.load_format)}"
            )
        self.load_format = LoadFormat(self.load_format.lower())
