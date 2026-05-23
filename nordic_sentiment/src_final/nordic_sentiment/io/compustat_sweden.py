from __future__ import annotations

import pandas as pd

from nordic_sentiment.config import ProjectConfigs
from nordic_sentiment.io.storage import load_tabular_file


def _load_sweden_source(configs: ProjectConfigs, source_name: str) -> pd.DataFrame:
    source = configs.get_source("sweden", source_name)
    if not source:
        return pd.DataFrame()
    path = configs.resolve_path(source["path"])
    options = {key: value for key, value in source.items() if key not in {"path", "format"}}
    return load_tabular_file(path, source["format"], **options)


def load_sweden_daily_prices(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "daily_prices")


def load_sweden_monthly_prices(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "monthly_prices")


def load_sweden_economic_consumer_sentiment(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "economic_consumer_sentiment")


def load_sweden_quarterly_fundamentals(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "quarterly_fundamentals")


def load_sweden_ipo_offers(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "ipo_offers")


def load_sweden_interbank_rate_3m(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "interbank_rate_3m")


def load_sweden_bond_yield_10y(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "bond_yield_10y")


def load_sweden_macro_control_cpi(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "macro_control_cpi")


def load_sweden_macro_control_ppi(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "macro_control_ppi")


def load_sweden_macro_control_ip(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "macro_control_ip")


def load_sweden_macro_control_em(configs: ProjectConfigs) -> pd.DataFrame:
    return _load_sweden_source(configs, "macro_control_em")
