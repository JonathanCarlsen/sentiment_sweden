from __future__ import annotations

import pandas as pd

from nordic_sentiment.market.controls import build_sweden_rates_monthly


def stage_sweden_rates(
    interbank_rate_3m: pd.DataFrame,
    bond_yield_10y: pd.DataFrame,
    *,
    country_code: str = "SWE",
) -> pd.DataFrame:
    return build_sweden_rates_monthly(
        interbank_rate_3m,
        bond_yield_10y,
        country_code=country_code,
    )
