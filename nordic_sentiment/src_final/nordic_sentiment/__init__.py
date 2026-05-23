"""Top-level package exports for the Nordic sentiment project."""

from nordic_sentiment.config import ProjectConfigs, load_project_configs
from nordic_sentiment.paths import PROJECT_ROOT

__all__ = ["PROJECT_ROOT", "ProjectConfigs", "load_project_configs"]
