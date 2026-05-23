"""Regression inference helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAP_REPETITIONS = 1_000
DEFAULT_BOOTSTRAP_SEED = 123
DEFAULT_MIN_BOOTSTRAP_BLOCK_LENGTH = 6


@dataclass(frozen=True)
class RegressionData:
    terms: list[str]
    y: np.ndarray
    x: np.ndarray
    sample: pd.DataFrame


def automatic_newey_west_lags(n_obs: int) -> int:
    """Automatic Newey-West lag rule for monthly predictive regressions."""
    if n_obs <= 1:
        return 0
    return int(np.floor(4 * (n_obs / 100.0) ** (2 / 9)))


def default_bootstrap_block_length(
    n_obs: int,
    *,
    return_horizon_months: int = 1,
    min_block_length: int = DEFAULT_MIN_BOOTSTRAP_BLOCK_LENGTH,
) -> int:
    """Default moving-block bootstrap length, capped to the sample size."""
    if n_obs <= 0:
        return 0
    horizon = max(int(return_horizon_months), 1)
    block_length = max(int(min_block_length), horizon)
    return min(block_length, int(n_obs))


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, max_lags: int) -> np.ndarray:
    """HAC covariance matrix using Bartlett weights."""
    n_obs, n_params = x.shape
    max_lags = min(max(int(max_lags), 0), max(n_obs - 1, 0))
    x_resid = x * residuals[:, None]
    meat = x_resid.T @ x_resid
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        gamma = x_resid[lag:].T @ x_resid[:-lag]
        meat += weight * (gamma + gamma.T)
    xtx_inv = np.linalg.pinv(x.T @ x)
    finite_sample_correction = n_obs / max(n_obs - n_params, 1)
    return finite_sample_correction * xtx_inv @ meat @ xtx_inv


def prepare_regression_data(
    frame: pd.DataFrame,
    y_col: str,
    x_cols: Iterable[str],
    *,
    date_column: str = "month_end_date",
) -> RegressionData:
    """Build aligned numeric arrays for a regression."""
    if date_column in frame.columns:
        frame = frame.sort_values(date_column)
    x_cols = list(x_cols)
    cols = [y_col, *x_cols]
    sample = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    terms = ["intercept", *x_cols]
    if sample.empty:
        return RegressionData(
            terms=terms,
            y=np.array([], dtype=float),
            x=np.empty((0, len(terms)), dtype=float),
            sample=sample,
        )
    y = sample[y_col].to_numpy(dtype=float)
    x = sample[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(sample)), x])
    return RegressionData(terms=terms, y=y, x=x, sample=sample)


def _fit_beta(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _derived_seed(seed: int, terms: list[str], n_obs: int) -> int:
    payload = "|".join([str(seed), str(n_obs), *terms]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def circular_moving_block_indices(
    n_obs: int,
    *,
    block_length: int,
    repetitions: int,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Generate circular moving-block bootstrap index arrays."""
    if n_obs <= 0:
        return np.empty((0, 0), dtype=int)
    if repetitions <= 0:
        return np.empty((0, n_obs), dtype=int)
    block_length = min(max(int(block_length), 1), int(n_obs))
    n_blocks = int(np.ceil(n_obs / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs, size=(int(repetitions), n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return indices.reshape(int(repetitions), n_blocks * block_length)[:, :n_obs]


def moving_block_bootstrap_regression(
    y: np.ndarray,
    x: np.ndarray,
    beta_original: np.ndarray,
    *,
    terms: list[str],
    block_length: int,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Pairs moving-block bootstrap inference for an OLS regression."""
    n_obs, n_params = x.shape
    block_length = min(max(int(block_length), 1), n_obs) if n_obs else 0
    if n_obs <= n_params or repetitions <= 1 or block_length <= 0:
        return pd.DataFrame(
            {
                "term": terms,
                "bootstrap_std_error": np.nan,
                "bootstrap_t_stat": np.nan,
                "bootstrap_p_value": np.nan,
                "bootstrap_repetitions": int(repetitions),
                "bootstrap_block_length": block_length,
                "bootstrap_seed": int(seed),
                "bootstrap_valid_repetitions": 0,
            }
        )

    indices = circular_moving_block_indices(
        n_obs,
        block_length=block_length,
        repetitions=repetitions,
        seed=_derived_seed(seed, terms, n_obs),
    )
    betas = np.full((int(repetitions), n_params), np.nan, dtype=float)
    for rep_idx, sample_idx in enumerate(indices):
        x_boot = x[sample_idx]
        y_boot = y[sample_idx]
        if np.linalg.matrix_rank(x_boot) < n_params:
            continue
        betas[rep_idx] = _fit_beta(y_boot, x_boot)

    valid = np.isfinite(betas).all(axis=1)
    valid_betas = betas[valid]
    valid_repetitions = len(valid_betas)
    if valid_repetitions <= 1:
        bootstrap_std_error = np.full(n_params, np.nan)
        bootstrap_t_stat = np.full(n_params, np.nan)
        bootstrap_p_value = np.full(n_params, np.nan)
    else:
        bootstrap_std_error = np.nanstd(valid_betas, axis=0, ddof=1)
        bootstrap_t_stat = np.divide(
            beta_original,
            bootstrap_std_error,
            out=np.full(n_params, np.nan, dtype=float),
            where=bootstrap_std_error > 0,
        )
        less_equal_zero = np.mean(valid_betas <= 0.0, axis=0)
        greater_equal_zero = np.mean(valid_betas >= 0.0, axis=0)
        bootstrap_p_value = np.minimum(1.0, 2.0 * np.minimum(less_equal_zero, greater_equal_zero))

    return pd.DataFrame(
        {
            "term": terms,
            "bootstrap_std_error": bootstrap_std_error,
            "bootstrap_t_stat": bootstrap_t_stat,
            "bootstrap_p_value": bootstrap_p_value,
            "bootstrap_repetitions": int(repetitions),
            "bootstrap_block_length": block_length,
            "bootstrap_seed": int(seed),
            "bootstrap_valid_repetitions": int(valid_repetitions),
        }
    )


def run_ols_with_hac_and_bootstrap(
    frame: pd.DataFrame,
    y_col: str,
    x_cols: Iterable[str],
    *,
    hac_lags: int | None = None,
    min_hac_lags: int = 0,
    return_horizon_months: int = 1,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_block_length: int | None = None,
) -> pd.DataFrame:
    """Run OLS with HAC inference and moving-block bootstrap robustness inference."""
    regression_data = prepare_regression_data(frame, y_col, x_cols)
    terms = regression_data.terms
    y = regression_data.y
    x = regression_data.x
    n_obs = len(y)
    if n_obs <= len(terms):
        block_length = default_bootstrap_block_length(
            n_obs,
            return_horizon_months=return_horizon_months,
        )
        return pd.DataFrame(
            [
                {
                    "term": "insufficient_observations",
                    "coef": np.nan,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "n_obs": n_obs,
                    "r2": np.nan,
                    "standard_error_type": "HAC_Newey_West",
                    "hac_lags": np.nan,
                    "bootstrap_std_error": np.nan,
                    "bootstrap_t_stat": np.nan,
                    "bootstrap_p_value": np.nan,
                    "bootstrap_repetitions": int(bootstrap_repetitions),
                    "bootstrap_block_length": block_length,
                    "bootstrap_seed": int(bootstrap_seed),
                    "bootstrap_valid_repetitions": 0,
                }
            ]
        )

    beta = _fit_beta(y, x)
    resid = y - x @ beta
    if hac_lags is None:
        hac_lags = max(automatic_newey_west_lags(n_obs), int(min_hac_lags))
    cov = newey_west_covariance(x, resid, hac_lags)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    t_stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    ss_total = float(((y - y.mean()) @ (y - y.mean())))
    ss_resid = float(resid @ resid)
    r2 = np.nan if ss_total == 0 else 1.0 - ss_resid / ss_total
    result = pd.DataFrame(
        {
            "term": terms,
            "coef": beta,
            "std_error": se,
            "t_stat": t_stat,
            "n_obs": n_obs,
            "r2": r2,
            "standard_error_type": "HAC_Newey_West",
            "hac_lags": int(hac_lags),
        }
    )
    block_length = (
        default_bootstrap_block_length(n_obs, return_horizon_months=return_horizon_months)
        if bootstrap_block_length is None
        else min(max(int(bootstrap_block_length), 1), n_obs)
    )
    bootstrap = moving_block_bootstrap_regression(
        y,
        x,
        beta,
        terms=terms,
        block_length=block_length,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    return result.merge(bootstrap, on="term", how="left")
