"""Configuration loading: configs/*.yaml -> nested dataclass-like access."""

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class Config:
    """Read-only attribute access over a nested dict: cfg.debounce.stable_frames."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(f"No config key '{name}'") from exc
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_CONFIG) as f:
        return Config(yaml.safe_load(f))
