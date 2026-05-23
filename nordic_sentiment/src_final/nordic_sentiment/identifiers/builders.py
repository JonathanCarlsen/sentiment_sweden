from __future__ import annotations

import pandas as pd


def _clean_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def attach_company_security_ids(frame: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(frame).copy()
    out = frame.copy()
    if "country_code" not in out.columns:
        out["country_code"] = country_code
    if "gvkey" in out.columns:
        out["gvkey"] = _clean_key(out["gvkey"])
        out["company_id"] = out["country_code"].astype("string") + ":" + out["gvkey"]
    if "iid" in out.columns:
        out["iid"] = _clean_key(out["iid"])
        out["security_id"] = out["company_id"] + ":" + out["iid"]
    return out


def build_company_dimension(*frames: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    prepared: list[pd.DataFrame] = []
    for frame in frames:
        enriched = attach_company_security_ids(frame, country_code=country_code)
        if not enriched.empty and {"company_id", "gvkey"}.issubset(enriched.columns):
            cols = [col for col in ["company_id", "country_code", "gvkey", "conm", "fic"] if col in enriched.columns]
            prepared.append(enriched[cols])
    if not prepared:
        return pd.DataFrame(
            columns=["company_id", "country_code", "gvkey", "company_name", "valid_from", "valid_to"]
        )
    out = pd.concat(prepared, ignore_index=True).drop_duplicates("company_id")
    out = out.rename(columns={"conm": "company_name", "fic": "country_of_incorporation"})
    out["valid_from"] = pd.NaT
    out["valid_to"] = pd.NaT
    return out.sort_values("company_id").reset_index(drop=True)


def build_security_dimension(*frames: pd.DataFrame, country_code: str = "SWE") -> pd.DataFrame:
    prepared: list[pd.DataFrame] = []
    for frame in frames:
        enriched = attach_company_security_ids(frame, country_code=country_code)
        if not enriched.empty and {"security_id", "company_id", "iid"}.issubset(enriched.columns):
            cols = [
                col
                for col in [
                    "security_id",
                    "company_id",
                    "country_code",
                    "gvkey",
                    "iid",
                    "isin",
                    "exchg",
                    "tpci",
                    "trade_date",
                    "month_end_date",
                ]
                if col in enriched.columns
            ]
            prepared.append(enriched[cols])
    if not prepared:
        return pd.DataFrame(
            columns=[
                "security_id",
                "company_id",
                "country_code",
                "gvkey",
                "iid",
                "isin",
                "exchange_code",
                "issue_type_code",
                "listing_start_date",
                "listing_end_date",
            ]
        )
    stacked = pd.concat(prepared, ignore_index=True)
    date_candidates = [column for column in ["trade_date", "month_end_date"] if column in stacked.columns]
    if date_candidates:
        stacked["_obs_date"] = stacked[date_candidates].bfill(axis=1).iloc[:, 0]
        bounds = stacked.groupby("security_id")["_obs_date"].agg(["min", "max"]).rename(
            columns={"min": "listing_start_date", "max": "listing_end_date"}
        )
    else:
        bounds = pd.DataFrame(index=stacked["security_id"].drop_duplicates())
        bounds["listing_start_date"] = pd.NaT
        bounds["listing_end_date"] = pd.NaT
    out = stacked.drop(
        columns=[column for column in ["trade_date", "month_end_date", "_obs_date"] if column in stacked.columns]
    )
    out = out.sort_values("security_id").drop_duplicates("security_id")
    out = out.merge(bounds, left_on="security_id", right_index=True, how="left")
    out = out.rename(columns={"exchg": "exchange_code", "tpci": "issue_type_code"})
    return out.reset_index(drop=True)
