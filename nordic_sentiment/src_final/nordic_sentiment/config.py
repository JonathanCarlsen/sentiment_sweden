from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nordic_sentiment.paths import CONFIG_ROOT


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class ProjectConfigs:
    """Final thesis view of the source-file configuration."""

    countries: dict[str, Any]
    sources: dict[str, Any]
    config_root: Path = CONFIG_ROOT

    def resolve_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return (self.config_root / path).resolve()

    def get_source(self, source_group: str, source_name: str | None = None) -> dict[str, Any]:
        group = self.sources.get(source_group, {})
        if source_name is None:
            return group
        return group.get(source_name, {})


def load_project_configs(config_root: Path | None = None) -> ProjectConfigs:
    root = config_root or CONFIG_ROOT
    countries = _read_yaml(root / "countries.yml").get("countries", {})
    sources = _read_yaml(root / "sources.yml").get("sources", {})
    return ProjectConfigs(
        countries=countries,
        sources=sources,
        config_root=root,
    )
