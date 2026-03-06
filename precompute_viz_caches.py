"""Pre-compute all runtime viz caches from lsdata.csv and save as gzipped JSON.

Run this locally whenever lsdata.csv is updated:
    python precompute_viz_caches.py

Output goes to precomputed_caches/ — commit that directory to git.
The server loads from these files in production (no CSV needed at runtime).
"""

import gzip
import json
import os

import numpy as np
import pandas as pd

CACHE_DIR = "precomputed_caches"
os.makedirs(CACHE_DIR, exist_ok=True)

_CURRENT_MAT_YEAR = 2026
_PACE_YEARS = [_CURRENT_MAT_YEAR - 3, _CURRENT_MAT_YEAR - 2, _CURRENT_MAT_YEAR - 1, _CURRENT_MAT_YEAR]


def save_cache(name: str, data: dict) -> None:
    path = os.path.join(CACHE_DIR, f"{name}.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {name}: {size_kb:.0f} KB")


# ─────────────────────────────────────────────
# Load CSV (terminal-decisions only, for most caches)
# ─────────────────────────────────────────────
print("Loading lsdata.csv (terminal decisions)...")
_viz_df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False)
_viz_df = _viz_df[_viz_df["result"].isin(["accepted", "waitlisted", "rejected"])].copy()
_viz_df = _viz_df.dropna(subset=["lsat", "gpa"])
for _col in ["sent_at", "complete_at", "decision_at"]:
    _viz_df[_col] = pd.to_datetime(_viz_df[_col], errors="coerce")
_viz_df["app_date"] = _viz_df["sent_at"].fillna(_viz_df["complete_at"])
_viz_df["cycle_start"] = pd.to_datetime((_viz_df["matriculating_year"] - 1).astype(str) + "-09-01")
_viz_df["day_of_cycle_decision"] = (_viz_df["decision_at"] - _viz_df["cycle_start"]).dt.days
_viz_df["day_of_cycle_sent"] = (_viz_df["app_date"] - _viz_df["cycle_start"]).dt.days
_viz_df["days_to_decision"] = (_viz_df["decision_at"] - _viz_df["app_date"]).dt.days
print(f"  {len(_viz_df):,} rows loaded.")

# ─────────────────────────────────────────────
# Load full CSV (all rows including pending) — for pace cache denominator
# ─────────────────────────────────────────────
print("Loading lsdata.csv (all rows for pace denominator)...")
_all_apps_df = pd.read_csv(
    "lsdata.csv", skiprows=1, low_memory=False,
    usecols=lambda c: c in {"school_name", "matriculating_year", "decision_at"},
)
_all_apps_df = _all_apps_df.dropna(subset=["school_name", "matriculating_year"])
_all_apps_df["matriculating_year"] = pd.to_numeric(_all_apps_df["matriculating_year"], errors="coerce")
_all_apps_df = _all_apps_df.dropna(subset=["matriculating_year"])
_all_apps_df["matriculating_year"] = _all_apps_df["matriculating_year"].astype(int)
_all_apps_df["decision_at"] = pd.to_datetime(_all_apps_df["decision_at"], errors="coerce")
_all_apps_df["cycle_start"] = pd.to_datetime(
    (_all_apps_df["matriculating_year"] - 1).astype(str) + "-09-01"
)
_all_apps_df["day_of_cycle_decision"] = (
    _all_apps_df["decision_at"] - _all_apps_df["cycle_start"]
).dt.days
print(f"  {len(_all_apps_df):,} total rows.")

# ─────────────────────────────────────────────
# Scatter cache
# ─────────────────────────────────────────────
print("Computing scatter cache...")
_scatter_cache = {}
for school_name, group in _viz_df.groupby("school_name"):
    sub = group[["lsat", "gpa", "result", "matriculating_year"]].dropna()
    _scatter_cache[school_name] = sub.to_dict(orient="records")
save_cache("scatter", _scatter_cache)

# ─────────────────────────────────────────────
# Drift cache
# ─────────────────────────────────────────────
print("Computing drift cache...")
_drift_cache = {}
for school_name, group in _viz_df.groupby("school_name"):
    acc = group[group["result"] == "accepted"]
    yearly = acc.groupby("matriculating_year").agg(
        median_lsat=("lsat", "median"),
        median_gpa=("gpa", "median"),
        p25_lsat=("lsat", lambda x: x.quantile(0.25)),
        p75_lsat=("lsat", lambda x: x.quantile(0.75)),
        p25_gpa=("gpa", lambda x: x.quantile(0.25)),
        p75_gpa=("gpa", lambda x: x.quantile(0.75)),
        count=("lsat", "count"),
    ).reset_index()
    yearly = yearly[yearly["count"] >= 10]
    _drift_cache[school_name] = yearly.to_dict(orient="records")
save_cache("drift", _drift_cache)

# ─────────────────────────────────────────────
# Heatmap cache
# ─────────────────────────────────────────────
print("Computing heatmap cache...")
_heatmap_cache = {}
for school_name, group in _viz_df.groupby("school_name"):
    valid = group[group["day_of_cycle_decision"].between(0, 400)].copy()
    if len(valid) < 20:
        continue
    valid["week"] = (valid["day_of_cycle_decision"] // 7).astype(int)
    weekly = valid.groupby("week").agg(
        total=("result", "count"),
        accepted=("result", lambda x: (x == "accepted").sum()),
        waitlisted=("result", lambda x: (x == "waitlisted").sum()),
        rejected=("result", lambda x: (x == "rejected").sum()),
    ).reset_index()
    weekly["day_start"] = weekly["week"] * 7
    _heatmap_cache[school_name] = weekly[["day_start", "total", "accepted", "waitlisted", "rejected"]].to_dict(orient="records")
save_cache("heatmap", _heatmap_cache)

# ─────────────────────────────────────────────
# Wait-time cache
# ─────────────────────────────────────────────
print("Computing waittime cache...")
_waittime_cache = {}
for school_name, group in _viz_df.groupby("school_name"):
    valid = group[group["days_to_decision"].between(1, 365)].copy()
    if len(valid) < 20:
        continue
    valid["bucket"] = ((valid["days_to_decision"] // 14) * 14).astype(int)
    buckets = valid.groupby(["bucket", "result"]).size().reset_index(name="count")
    _waittime_cache[school_name] = buckets.to_dict(orient="records")
save_cache("waittime", _waittime_cache)

# ─────────────────────────────────────────────
# Current-cycle applicants cache (includes pending)
# ─────────────────────────────────────────────
print("Computing current_cycle cache (includes pending applicants)...")
_cycle_df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False,
    usecols=lambda c: c in {"school_name", "matriculating_year", "lsat", "gpa", "result"})
_cycle_df = _cycle_df[_cycle_df["matriculating_year"] == _CURRENT_MAT_YEAR].copy()
_cycle_df = _cycle_df.dropna(subset=["lsat", "gpa", "school_name"])
_cycle_df["result"] = _cycle_df["result"].fillna("pending")
# Normalize results: anything not accepted/waitlisted/rejected/pending → pending
_cycle_df.loc[~_cycle_df["result"].isin(["accepted", "waitlisted", "rejected", "pending"]), "result"] = "pending"
_current_cycle_cache = {}
for school_name, group in _cycle_df.groupby("school_name"):
    sub = group[["lsat", "gpa", "result"]].copy()
    sub["lsat"] = sub["lsat"].astype(int)
    sub["gpa"] = sub["gpa"].round(2)
    _current_cycle_cache[school_name] = sub.to_dict(orient="records")
save_cache("current_cycle", _current_cycle_cache)

# ─────────────────────────────────────────────
# Pace cache
# ─────────────────────────────────────────────
print("Computing pace cache...")


def _build_pace_for_subset(decisions_sub: pd.DataFrame, all_apps_sub: pd.DataFrame) -> dict:
    result = {}
    for mat_year in _PACE_YEARS:
        total_apps = int((all_apps_sub["matriculating_year"] == mat_year).sum())
        if total_apps < 5:
            continue
        dec_sub = decisions_sub[decisions_sub["matriculating_year"] == mat_year]
        valid_days = dec_sub["day_of_cycle_decision"].dropna()
        valid_days = valid_days[(valid_days >= 0) & (valid_days <= 400)]
        sorted_days = np.sort(valid_days.values)
        decisions_total = int(len(sorted_days))
        curve = [
            {
                "day": int(d),
                "count": int(np.searchsorted(sorted_days, d, side="right")),
                "frac": round(
                    float(np.searchsorted(sorted_days, d, side="right")) / total_apps, 4
                ),
            }
            for d in range(0, 401, 3)
        ]
        result[mat_year] = {
            "total_apps": total_apps,
            "decisions_total": decisions_total,
            "curve": curve,
        }
    return result


_pace_cache: dict = {}
_all_apps_by_school = {s: g for s, g in _all_apps_df.groupby("school_name")}
_pace_cache["ALL"] = _build_pace_for_subset(_viz_df, _all_apps_df)
for _school, _grp in _viz_df.groupby("school_name"):
    _all_grp = _all_apps_by_school.get(_school, pd.DataFrame())
    _pr = _build_pace_for_subset(_grp, _all_grp)
    if _pr:
        _pace_cache[_school] = _pr
save_cache("pace", _pace_cache)

print(f"\nDone. All caches saved to {CACHE_DIR}/")
print("Commit the precomputed_caches/ directory to git.")
