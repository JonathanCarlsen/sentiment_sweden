from __future__ import annotations

import numpy as np
import pandas as pd


BOOK_EQUITY_COMPONENT_COLUMNS = [
    "book_equity_base",
    "book_equity_base_source",
    "book_equity_preferred_stock_adjustment",
    "book_equity_nonpositive_flag",
    "book_equity_zero_ceq_replaced_flag",
    "BE",
    "book_equity",
]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def compute_book_equity_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Compustat Global book equity and source diagnostics.

    The final thesis extract uses `seqq` almost universally. The only observed
    fallback retained in `src_final` is `at - ltq`.
    """

    index = frame.index
    seqq = _numeric(frame, "seqq")
    ceq = _numeric(frame, "ceq")
    at = _numeric(frame, "at")
    ltq = _numeric(frame, "ltq")
    pstkq = _numeric(frame, "pstkq")

    at_minus_ltq = at - ltq
    at_minus_ltq = at_minus_ltq.where(at.notna() & ltq.notna())

    base = pd.Series(np.nan, index=index, dtype="float64")
    source = pd.Series(pd.NA, index=index, dtype="string")
    for candidate_source, candidate_values in [
        ("seqq", seqq),
        ("at_minus_ltq", at_minus_ltq),
    ]:
        use_candidate = base.isna() & candidate_values.notna() & candidate_values.ne(0)
        base.loc[use_candidate] = candidate_values.loc[use_candidate]
        source.loc[use_candidate] = candidate_source

    has_base = base.notna()
    preferred_stock_adjustment = pstkq.fillna(0.0).where(has_base)
    book_equity = (base - preferred_stock_adjustment).where(has_base)
    zero_ceq_replaced = ceq.eq(0) & source.isin(["seqq", "at_minus_ltq"])

    return pd.DataFrame(
        {
            "book_equity_base": base,
            "book_equity_base_source": source,
            "book_equity_preferred_stock_adjustment": preferred_stock_adjustment,
            "book_equity_nonpositive_flag": book_equity.notna() & book_equity.le(0),
            "book_equity_zero_ceq_replaced_flag": zero_ceq_replaced.fillna(False),
            "BE": book_equity,
            "book_equity": book_equity,
        },
        index=index,
    )
