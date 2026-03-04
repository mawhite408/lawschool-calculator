"""
Law School Admissions Calculator - Model Training Script (v3)
Uses LightGBM (GBDT) with wave-aware, survival-conditional features.

v3 changes:
- Recency-weighted school stats (exponential decay, recent cycles matter more)
- Per-cycle school medians as features (captures year-to-year target shifts)
- Normalized wave features: waves expressed as % of total decisions, not
  absolute calendar days — robust to year-to-year timing drift
- cycle_year as a feature for temporal trends
- Relative-to-CYCLE stats (lsat vs this cycle's emerging median)

Key design: each training row represents a decision, and we simulate
"current_date" = decision_date for that row. This teaches the model:
given your stats + sent date + where we are NOW in the cycle + which
waves have already passed without your name, what's the probability
of each outcome?

At inference, the user provides their actual current date, and the model
predicts conditional on having survived to that point.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = "model_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False)
print(f"  Raw rows: {len(df):,}")

# Keep only terminal decisions
df = df[df["result"].isin(["accepted", "waitlisted", "rejected"])].copy()
print(f"  Terminal decisions: {len(df):,}")

# Drop rows missing LSAT or GPA
df = df.dropna(subset=["lsat", "gpa"])
print(f"  After requiring LSAT+GPA: {len(df):,}")

# Parse date columns
for col in ["sent_at", "complete_at", "decision_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["app_date"] = df["sent_at"].fillna(df["complete_at"])
df["cycle_start"] = pd.to_datetime(
    (df["matriculating_year"] - 1).astype(str) + "-09-01"
)

df["day_of_cycle_sent"] = (df["app_date"] - df["cycle_start"]).dt.days
df.loc[df["day_of_cycle_sent"] < 0, "day_of_cycle_sent"] = np.nan
df.loc[df["day_of_cycle_sent"] > 450, "day_of_cycle_sent"] = np.nan

df["day_of_cycle_decision"] = (df["decision_at"] - df["cycle_start"]).dt.days
df.loc[df["day_of_cycle_decision"] < 0, "day_of_cycle_decision"] = np.nan
df.loc[df["day_of_cycle_decision"] > 450, "day_of_cycle_decision"] = np.nan

df["month_sent"] = df["app_date"].dt.month

# ─────────────────────────────────────────────
# 2. DETECT WAVES PER SCHOOL
# ─────────────────────────────────────────────

print("Detecting decision waves per school...")

def detect_waves(school_df, min_wave_size_mult=2.0, merge_gap=3):
    """
    Detect discrete decision-date wave clusters for a school.
    Returns list of dicts: [{center, start, end, count, results}, ...]
    """
    valid = school_df[school_df["day_of_cycle_decision"].between(0, 400)]
    if len(valid) < 20:
        return []

    day_counts = valid["day_of_cycle_decision"].value_counts().sort_index()
    mean_c = day_counts.mean()
    std_c = day_counts.std()
    if std_c == 0:
        return []

    threshold = mean_c + min_wave_size_mult * std_c
    wave_days = day_counts[day_counts >= threshold].sort_index()
    if len(wave_days) == 0:
        return []

    # Merge nearby days into clusters
    day_list = wave_days.index.tolist()
    clusters = []
    current = [day_list[0]]
    for d in day_list[1:]:
        if d - current[-1] <= merge_gap:
            current.append(d)
        else:
            clusters.append(current)
            current = [d]
    clusters.append(current)

    waves = []
    for c in clusters:
        wave_mask = valid["day_of_cycle_decision"].isin(c)
        wave_rows = valid[wave_mask]
        results = wave_rows["result"].value_counts().to_dict()
        waves.append({
            "start": int(min(c)),
            "end": int(max(c)),
            "center": int(np.mean(c)),
            "count": int(len(wave_rows)),
            "accepted": results.get("accepted", 0),
            "waitlisted": results.get("waitlisted", 0),
            "rejected": results.get("rejected", 0),
        })

    waves.sort(key=lambda w: w["center"])
    return waves

# Build wave table for every school
school_waves = {}
for school_name, group in df.groupby("school_name"):
    waves = detect_waves(group)
    school_waves[school_name] = waves

total_waves = sum(len(v) for v in school_waves.values())
schools_with_waves = sum(1 for v in school_waves.values() if len(v) > 0)
print(f"  Detected {total_waves} waves across {schools_with_waves} schools")

# ─────────────────────────────────────────────
# 3. SCHOOL-LEVEL AGGREGATE STATS (all-time + recency-weighted + per-cycle)
# ─────────────────────────────────────────────

print("Computing school-level aggregates...")

school_encoder = LabelEncoder()
df["school_encoded"] = school_encoder.fit_transform(df["school_name"])

# --- A. All-time school stats (same as before, for backward compat) ---
school_stats = df.groupby("school_name").agg(
    school_accept_rate=("result", lambda x: (x == "accepted").mean()),
    school_wl_rate=("result", lambda x: (x == "waitlisted").mean()),
    school_reject_rate=("result", lambda x: (x == "rejected").mean()),
    school_median_lsat=("lsat", "median"),
    school_median_gpa=("gpa", "median"),
    school_25_lsat=("lsat", lambda x: x.quantile(0.25)),
    school_75_lsat=("lsat", lambda x: x.quantile(0.75)),
    school_25_gpa=("gpa", lambda x: x.quantile(0.25)),
    school_75_gpa=("gpa", lambda x: x.quantile(0.75)),
    school_count=("result", "count"),
).reset_index()

df = df.merge(school_stats, on="school_name", how="left")

# --- B. Per-cycle school stats (accepted-only medians per year) ---
print("  Computing per-cycle school stats...")
cycle_stats = (
    df[df["result"] == "accepted"]
    .groupby(["school_name", "matriculating_year"])
    .agg(
        cycle_median_lsat=("lsat", "median"),
        cycle_median_gpa=("gpa", "median"),
        cycle_25_lsat=("lsat", lambda x: x.quantile(0.25)),
        cycle_75_lsat=("lsat", lambda x: x.quantile(0.75)),
        cycle_25_gpa=("gpa", lambda x: x.quantile(0.25)),
        cycle_75_gpa=("gpa", lambda x: x.quantile(0.75)),
        cycle_accept_count=("lsat", "count"),
    )
    .reset_index()
)
df = df.merge(cycle_stats, on=["school_name", "matriculating_year"], how="left")

# --- C. Recency-weighted school stats (exponential decay, half-life=2 years) ---
print("  Computing recency-weighted school stats...")
max_year = df["matriculating_year"].max()
HALF_LIFE = 2.0  # years
decay = np.log(2) / HALF_LIFE

def weighted_median(values, weights):
    """Weighted median using sorted values and cumulative weights."""
    idx = np.argsort(values)
    sv = values[idx]
    sw = weights[idx]
    cum = np.cumsum(sw)
    mid = cum[-1] / 2.0
    return float(sv[np.searchsorted(cum, mid)])

def weighted_quantile(values, weights, q):
    idx = np.argsort(values)
    sv = values[idx]
    sw = weights[idx]
    cum = np.cumsum(sw)
    target = cum[-1] * q
    return float(sv[np.searchsorted(cum, target)])

rw_stats_list = []
for school_name, group in df.groupby("school_name"):
    acc = group[group["result"] == "accepted"]
    if len(acc) < 5:
        rw_stats_list.append({
            "school_name": school_name,
            "rw_median_lsat": np.nan, "rw_median_gpa": np.nan,
            "rw_25_lsat": np.nan, "rw_75_lsat": np.nan,
            "rw_25_gpa": np.nan, "rw_75_gpa": np.nan,
        })
        continue
    years_ago = max_year - acc["matriculating_year"].values
    w = np.exp(-decay * years_ago)
    lsats = acc["lsat"].values.astype(float)
    gpas = acc["gpa"].values.astype(float)
    valid = ~(np.isnan(lsats) | np.isnan(gpas))
    lsats, gpas, w = lsats[valid], gpas[valid], w[valid]
    if len(lsats) < 5:
        rw_stats_list.append({
            "school_name": school_name,
            "rw_median_lsat": np.nan, "rw_median_gpa": np.nan,
            "rw_25_lsat": np.nan, "rw_75_lsat": np.nan,
            "rw_25_gpa": np.nan, "rw_75_gpa": np.nan,
        })
        continue
    rw_stats_list.append({
        "school_name": school_name,
        "rw_median_lsat": weighted_median(lsats, w),
        "rw_median_gpa": weighted_median(gpas, w),
        "rw_25_lsat": weighted_quantile(lsats, w, 0.25),
        "rw_75_lsat": weighted_quantile(lsats, w, 0.75),
        "rw_25_gpa": weighted_quantile(gpas, w, 0.25),
        "rw_75_gpa": weighted_quantile(gpas, w, 0.75),
    })

rw_stats_df = pd.DataFrame(rw_stats_list)
df = df.merge(rw_stats_df, on="school_name", how="left")

# --- D. Relative stats (vs all-time, vs recency-weighted, vs this cycle) ---
# All-time relative
df["lsat_above_median"] = df["lsat"] - df["school_median_lsat"]
df["gpa_above_median"] = df["gpa"] - df["school_median_gpa"]
df["lsat_above_75"] = df["lsat"] - df["school_75_lsat"]
df["gpa_above_75"] = df["gpa"] - df["school_75_gpa"]
df["lsat_above_25"] = df["lsat"] - df["school_25_lsat"]
df["gpa_above_25"] = df["gpa"] - df["school_25_gpa"]

# Recency-weighted relative
df["lsat_above_rw_median"] = df["lsat"] - df["rw_median_lsat"]
df["gpa_above_rw_median"] = df["gpa"] - df["rw_median_gpa"]
df["lsat_above_rw_75"] = df["lsat"] - df["rw_75_lsat"]
df["gpa_above_rw_75"] = df["gpa"] - df["rw_75_gpa"]

# Per-cycle relative (most important for current cycle)
df["lsat_above_cycle_median"] = df["lsat"] - df["cycle_median_lsat"]
df["gpa_above_cycle_median"] = df["gpa"] - df["cycle_median_gpa"]
df["lsat_above_cycle_75"] = df["lsat"] - df["cycle_75_lsat"]
df["gpa_above_cycle_75"] = df["gpa"] - df["cycle_75_gpa"]
df["lsat_above_cycle_25"] = df["lsat"] - df["cycle_25_lsat"]
df["gpa_above_cycle_25"] = df["gpa"] - df["cycle_25_gpa"]

# ─────────────────────────────────────────────
# 4. WAVE-AWARE SURVIVAL FEATURES
# ─────────────────────────────────────────────

print("Computing wave-aware survival features...")

def compute_wave_features(row_sent_day, row_current_day, waves):
    """
    Given an applicant's sent day, current day in cycle, and the school's
    wave table, compute survival features:
    - How many waves (total, accept, wl, reject) have passed since sent
      without the applicant being in them?
    - What's the next wave timing?
    - What fraction of the school's typical decisions have already occurred
      by current_day?
    - Accept/WL/reject rates in waves that passed vs upcoming waves
    """
    if not waves or pd.isna(row_current_day):
        return {
            "waves_passed_total": np.nan,
            "waves_passed_accept_heavy": np.nan,
            "waves_passed_wl_heavy": np.nan,
            "waves_passed_reject_heavy": np.nan,
            "decisions_in_passed_waves": np.nan,
            "accepts_in_passed_waves": np.nan,
            "wl_in_passed_waves": np.nan,
            "rejects_in_passed_waves": np.nan,
            "decisions_in_future_waves": np.nan,
            "accepts_in_future_waves": np.nan,
            "wl_in_future_waves": np.nan,
            "rejects_in_future_waves": np.nan,
            "days_to_next_wave": np.nan,
            "days_since_last_wave": np.nan,
            "next_wave_accept_pct": np.nan,
            "next_wave_wl_pct": np.nan,
            "next_wave_reject_pct": np.nan,
        }

    # Waves that the applicant could have been in (after sent, before current)
    applicable_sent = row_sent_day if not pd.isna(row_sent_day) else 0
    passed = [w for w in waves if w["end"] <= row_current_day and w["start"] >= applicable_sent]
    future = [w for w in waves if w["start"] > row_current_day]

    # Count passed waves by dominant outcome
    accept_heavy = sum(1 for w in passed if w["accepted"] >= w["waitlisted"] and w["accepted"] >= w["rejected"])
    wl_heavy = sum(1 for w in passed if w["waitlisted"] > w["accepted"] and w["waitlisted"] >= w["rejected"])
    reject_heavy = sum(1 for w in passed if w["rejected"] > w["accepted"] and w["rejected"] > w["waitlisted"])

    dec_passed = sum(w["count"] for w in passed)
    acc_passed = sum(w["accepted"] for w in passed)
    wl_passed = sum(w["waitlisted"] for w in passed)
    rej_passed = sum(w["rejected"] for w in passed)

    dec_future = sum(w["count"] for w in future)
    acc_future = sum(w["accepted"] for w in future)
    wl_future = sum(w["waitlisted"] for w in future)
    rej_future = sum(w["rejected"] for w in future)

    # Next wave info
    days_to_next = future[0]["start"] - row_current_day if future else np.nan
    days_since_last = row_current_day - passed[-1]["end"] if passed else np.nan

    if future:
        nw = future[0]
        nw_total = max(nw["count"], 1)
        next_acc_pct = nw["accepted"] / nw_total
        next_wl_pct = nw["waitlisted"] / nw_total
        next_rej_pct = nw["rejected"] / nw_total
    else:
        next_acc_pct = np.nan
        next_wl_pct = np.nan
        next_rej_pct = np.nan

    return {
        "waves_passed_total": len(passed),
        "waves_passed_accept_heavy": accept_heavy,
        "waves_passed_wl_heavy": wl_heavy,
        "waves_passed_reject_heavy": reject_heavy,
        "decisions_in_passed_waves": dec_passed,
        "accepts_in_passed_waves": acc_passed,
        "wl_in_passed_waves": wl_passed,
        "rejects_in_passed_waves": rej_passed,
        "decisions_in_future_waves": dec_future,
        "accepts_in_future_waves": acc_future,
        "wl_in_future_waves": wl_future,
        "rejects_in_future_waves": rej_future,
        "days_to_next_wave": days_to_next,
        "days_since_last_wave": days_since_last,
        "next_wave_accept_pct": next_acc_pct,
        "next_wave_wl_pct": next_wl_pct,
        "next_wave_reject_pct": next_rej_pct,
    }

# For training: "current_day" = day_of_cycle_decision (the day they got their answer)
# This teaches the model: given that we've reached this day without a decision,
# what's the outcome? At inference we substitute the user's actual today.
df["current_day"] = df["day_of_cycle_decision"]

# Precompute wave features - VECTORIZED by school for speed
print("  Building wave features for all rows (vectorized by school)...")

wave_feat_cols = [
    "waves_passed_total", "waves_passed_accept_heavy",
    "waves_passed_wl_heavy", "waves_passed_reject_heavy",
    "decisions_in_passed_waves", "accepts_in_passed_waves",
    "wl_in_passed_waves", "rejects_in_passed_waves",
    "decisions_in_future_waves", "accepts_in_future_waves",
    "wl_in_future_waves", "rejects_in_future_waves",
    "days_to_next_wave", "days_since_last_wave",
    "next_wave_accept_pct", "next_wave_wl_pct", "next_wave_reject_pct",
    # Normalized wave features (% of total school decisions, not absolute counts)
    "pct_decisions_in_passed_waves", "pct_decisions_in_future_waves",
    "pct_accepts_in_passed_waves", "pct_accepts_in_future_waves",
]

for col in wave_feat_cols:
    df[col] = np.nan

def compute_wave_features_vectorized(school_sub, waves, school_total_decisions):
    """Compute wave features for all rows of a school at once using numpy."""
    n = len(school_sub)
    result = {col: np.full(n, np.nan) for col in wave_feat_cols}

    if not waves:
        return result

    sent_days = school_sub["day_of_cycle_sent"].values.astype(float)
    current_days = school_sub["current_day"].values.astype(float)

    # Pre-extract wave arrays
    w_starts = np.array([w["start"] for w in waves], dtype=float)
    w_ends = np.array([w["end"] for w in waves], dtype=float)
    w_centers = np.array([w["center"] for w in waves], dtype=float)
    w_counts = np.array([w["count"] for w in waves], dtype=float)
    w_acc = np.array([w["accepted"] for w in waves], dtype=float)
    w_wl = np.array([w["waitlisted"] for w in waves], dtype=float)
    w_rej = np.array([w["rejected"] for w in waves], dtype=float)

    # Dominant outcome per wave
    w_is_acc_heavy = (w_acc >= w_wl) & (w_acc >= w_rej)
    w_is_wl_heavy = (w_wl > w_acc) & (w_wl >= w_rej)
    w_is_rej_heavy = (w_rej > w_acc) & (w_rej > w_wl)

    # Total for normalization
    total_dec = max(school_total_decisions, 1)
    total_acc_in_waves = w_acc.sum()
    total_acc_norm = max(total_acc_in_waves, 1)

    for i in range(n):
        cd = current_days[i]
        sd = sent_days[i]
        if np.isnan(cd):
            continue

        applicable_sent = sd if not np.isnan(sd) else 0.0

        # Passed waves: ended before current_day AND started after sent
        passed_mask = (w_ends <= cd) & (w_starts >= applicable_sent)
        future_mask = w_starts > cd

        result["waves_passed_total"][i] = passed_mask.sum()
        result["waves_passed_accept_heavy"][i] = (passed_mask & w_is_acc_heavy).sum()
        result["waves_passed_wl_heavy"][i] = (passed_mask & w_is_wl_heavy).sum()
        result["waves_passed_reject_heavy"][i] = (passed_mask & w_is_rej_heavy).sum()

        dec_passed = w_counts[passed_mask].sum()
        acc_passed = w_acc[passed_mask].sum()
        dec_future = w_counts[future_mask].sum()
        acc_future = w_acc[future_mask].sum()

        result["decisions_in_passed_waves"][i] = dec_passed
        result["accepts_in_passed_waves"][i] = acc_passed
        result["wl_in_passed_waves"][i] = w_wl[passed_mask].sum()
        result["rejects_in_passed_waves"][i] = w_rej[passed_mask].sum()

        result["decisions_in_future_waves"][i] = dec_future
        result["accepts_in_future_waves"][i] = acc_future
        result["wl_in_future_waves"][i] = w_wl[future_mask].sum()
        result["rejects_in_future_waves"][i] = w_rej[future_mask].sum()

        # Normalized: % of total school decisions in passed/future waves
        result["pct_decisions_in_passed_waves"][i] = dec_passed / total_dec
        result["pct_decisions_in_future_waves"][i] = dec_future / total_dec
        result["pct_accepts_in_passed_waves"][i] = acc_passed / total_acc_norm
        result["pct_accepts_in_future_waves"][i] = acc_future / total_acc_norm

        # Days to next wave
        future_starts = w_starts[future_mask]
        if len(future_starts) > 0:
            result["days_to_next_wave"][i] = future_starts.min() - cd
            # Next wave composition
            next_idx = np.where(future_mask)[0][0]
            nw_total = max(w_counts[next_idx], 1)
            result["next_wave_accept_pct"][i] = w_acc[next_idx] / nw_total
            result["next_wave_wl_pct"][i] = w_wl[next_idx] / nw_total
            result["next_wave_reject_pct"][i] = w_rej[next_idx] / nw_total

        # Days since last wave
        passed_ends = w_ends[passed_mask]
        if len(passed_ends) > 0:
            result["days_since_last_wave"][i] = cd - passed_ends.max()

    return result

processed = 0
for school_name, group_idx in df.groupby("school_name").groups.items():
    waves = school_waves.get(school_name, [])
    sub = df.loc[group_idx].reset_index(drop=False)
    school_total = len(sub)
    feats = compute_wave_features_vectorized(sub, waves, school_total)
    orig_idx = sub["index"].values
    for col in wave_feat_cols:
        df.loc[orig_idx, col] = feats[col]
    processed += len(group_idx)
    if processed % 100000 < len(group_idx):
        print(f"    Processed {processed:,} / {len(df):,} rows...")

print(f"  Done: {processed:,} rows processed")

# ─────────────────────────────────────────────
# 5. CUMULATIVE DECISION FRACTION
# ─────────────────────────────────────────────

# For each school, what fraction of all decisions typically occur by day X?
# This gives the model a sense of "how late in the cycle are we?"
print("Computing cumulative decision fractions per school...")

school_cum = {}
for school_name, group in df.groupby("school_name"):
    valid = group["day_of_cycle_decision"].dropna()
    if len(valid) < 10:
        school_cum[school_name] = {}
        continue
    sorted_days = np.sort(valid.values)
    total = len(sorted_days)
    # Build lookup: for any day, what fraction of decisions have been made?
    cum_dict = {}
    for d in range(0, 401):
        cum_dict[d] = float(np.searchsorted(sorted_days, d, side="right") / total)
    school_cum[school_name] = cum_dict

def get_cum_fraction(school, day):
    if pd.isna(day) or school not in school_cum or not school_cum[school]:
        return np.nan
    d = int(min(max(day, 0), 400))
    return school_cum[school].get(d, np.nan)

df["cum_decision_frac"] = df.apply(
    lambda r: get_cum_fraction(r["school_name"], r["current_day"]), axis=1
)

# Also compute per-outcome: what fraction of ACCEPTS have occurred by now?
print("Computing per-outcome cumulative fractions...")
for outcome in ["accepted", "waitlisted", "rejected"]:
    outcome_cum = {}
    for school_name, group in df.groupby("school_name"):
        valid = group[group["result"] == outcome]["day_of_cycle_decision"].dropna()
        if len(valid) < 5:
            outcome_cum[school_name] = {}
            continue
        sorted_days = np.sort(valid.values)
        total = len(sorted_days)
        cd = {}
        for d in range(0, 401):
            cd[d] = float(np.searchsorted(sorted_days, d, side="right") / total)
        outcome_cum[school_name] = cd

    col_name = f"cum_{outcome}_frac"
    df[col_name] = df.apply(
        lambda r, oc=outcome_cum: (
            np.nan if pd.isna(r["current_day"]) or r["school_name"] not in oc or not oc[r["school_name"]]
            else oc[r["school_name"]].get(int(min(max(r["current_day"], 0), 400)), np.nan)
        ),
        axis=1,
    )

# ─────────────────────────────────────────────
# 6. REMAINING FEATURES
# ─────────────────────────────────────────────

# Boolean features
df["urm_int"] = df["urm"].astype(int)
df["is_international_int"] = df["is_international"].astype(int)
df["non_trad_int"] = df["non_trad"].astype(int)
df["is_in_state_int"] = df["is_in_state"].astype(int)
df["is_fee_waived_int"] = df["is_fee_waived"].astype(int)
df["is_military_int"] = df["is_military"].astype(int)
df["c_and_f_int"] = df["is_character_and_fitness_issues"].astype(int)

# Softs
softs_map = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}
df["softs_encoded"] = df["softs"].str.lower().str.strip().map(softs_map)

# Days waiting (current_day - day_of_cycle_sent)
df["days_waiting"] = df["current_day"] - df["day_of_cycle_sent"]

# Cycle year (temporal trend feature)
df["cycle_year"] = df["matriculating_year"]

# ─────────────────────────────────────────────
# 7. DEFINE FEATURES & PREPARE DATA
# ─────────────────────────────────────────────

print("Preparing training data...")

feature_cols = [
    # Core stats
    "lsat", "gpa",
    # Timing
    "day_of_cycle_sent", "month_sent",
    "current_day", "days_waiting",
    "cycle_year",
    # School
    "school_encoded",
    # School aggregates (all-time)
    "school_accept_rate", "school_wl_rate", "school_reject_rate",
    "school_median_lsat", "school_median_gpa",
    "school_25_lsat", "school_75_lsat",
    "school_25_gpa", "school_75_gpa",
    "school_count",
    # Recency-weighted school stats
    "rw_median_lsat", "rw_median_gpa",
    "rw_25_lsat", "rw_75_lsat",
    "rw_25_gpa", "rw_75_gpa",
    # Per-cycle school stats (accepted medians for this specific cycle)
    "cycle_median_lsat", "cycle_median_gpa",
    "cycle_25_lsat", "cycle_75_lsat",
    "cycle_25_gpa", "cycle_75_gpa",
    "cycle_accept_count",
    # Relative to all-time
    "lsat_above_median", "gpa_above_median",
    "lsat_above_75", "gpa_above_75",
    "lsat_above_25", "gpa_above_25",
    # Relative to recency-weighted
    "lsat_above_rw_median", "gpa_above_rw_median",
    "lsat_above_rw_75", "gpa_above_rw_75",
    # Relative to THIS cycle
    "lsat_above_cycle_median", "gpa_above_cycle_median",
    "lsat_above_cycle_75", "gpa_above_cycle_75",
    "lsat_above_cycle_25", "gpa_above_cycle_25",
    # Wave survival features
    "waves_passed_total", "waves_passed_accept_heavy",
    "waves_passed_wl_heavy", "waves_passed_reject_heavy",
    "decisions_in_passed_waves", "accepts_in_passed_waves",
    "wl_in_passed_waves", "rejects_in_passed_waves",
    "decisions_in_future_waves", "accepts_in_future_waves",
    "wl_in_future_waves", "rejects_in_future_waves",
    "days_to_next_wave", "days_since_last_wave",
    "next_wave_accept_pct", "next_wave_wl_pct", "next_wave_reject_pct",
    # Normalized wave features (% of school total)
    "pct_decisions_in_passed_waves", "pct_decisions_in_future_waves",
    "pct_accepts_in_passed_waves", "pct_accepts_in_future_waves",
    # Cumulative cycle position
    "cum_decision_frac",
    "cum_accepted_frac", "cum_waitlisted_frac", "cum_rejected_frac",
    # Profile
    "urm_int", "is_international_int", "non_trad_int",
    "is_in_state_int", "is_fee_waived_int", "is_military_int", "c_and_f_int",
    "softs_encoded", "years_out",
]

# Encode target
label_map = {"accepted": 0, "waitlisted": 1, "rejected": 2}
label_names = ["accepted", "waitlisted", "rejected"]
df["label"] = df["result"].map(label_map)

X = df[feature_cols].copy()
y = df["label"].values

print(f"  Total samples: {len(X):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Label distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 8. TRAIN LIGHTGBM GBDT MODEL
# ─────────────────────────────────────────────

print("\nTraining LightGBM GBDT model...")

train_data = lgb.Dataset(
    X_train, label=y_train,
    feature_name=feature_cols,
    categorical_feature=["school_encoded"],
    free_raw_data=False,
)
test_data = lgb.Dataset(
    X_test, label=y_test,
    feature_name=feature_cols,
    categorical_feature=["school_encoded"],
    reference=train_data,
    free_raw_data=False,
)

params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[test_data],
    callbacks=[
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50),
    ],
)

# ─────────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────────

print("\n=== EVALUATION ===")
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

print(classification_report(y_test, y_pred, target_names=label_names))
print(f"Log loss: {log_loss(y_test, y_pred_proba):.4f}")

# Feature importance
importance = model.feature_importance(importance_type="gain")
feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
print("\nTop 25 features by gain:")
for name, imp in feat_imp[:25]:
    print(f"  {name:40s} {imp:>12,.0f}")

# ─────────────────────────────────────────────
# 10. SAVE ARTIFACTS
# ─────────────────────────────────────────────

print("\nSaving model artifacts...")

model.save_model(os.path.join(OUTPUT_DIR, "lgbm_model.txt"))
joblib.dump(school_encoder, os.path.join(OUTPUT_DIR, "school_encoder.joblib"))
school_stats.to_json(os.path.join(OUTPUT_DIR, "school_stats.json"), orient="records")

# Save recency-weighted stats
rw_stats_df.to_json(os.path.join(OUTPUT_DIR, "rw_stats.json"), orient="records")

# Save per-cycle stats
cycle_stats.to_json(os.path.join(OUTPUT_DIR, "cycle_stats.json"), orient="records")

# Save wave table
with open(os.path.join(OUTPUT_DIR, "school_waves.json"), "w") as f:
    json.dump(school_waves, f)

# Save cumulative decision curves per school (sampled at every 5 days)
cum_curves = {}
for school_name in school_cum:
    if school_cum[school_name]:
        cum_curves[school_name] = {
            str(d): school_cum[school_name].get(d, 0.0)
            for d in range(0, 401, 5)
        }
with open(os.path.join(OUTPUT_DIR, "cum_curves.json"), "w") as f:
    json.dump(cum_curves, f)

# Save per-outcome cumulative curves
# (recompute since we didn't save the dicts)
for outcome in ["accepted", "waitlisted", "rejected"]:
    oc = {}
    for school_name, group in df.groupby("school_name"):
        valid = group[group["result"] == outcome]["day_of_cycle_decision"].dropna()
        if len(valid) < 5:
            continue
        sorted_days = np.sort(valid.values)
        total = len(sorted_days)
        oc[school_name] = {
            str(d): float(np.searchsorted(sorted_days, d, side="right") / total)
            for d in range(0, 401, 5)
        }
    with open(os.path.join(OUTPUT_DIR, f"cum_{outcome}_curves.json"), "w") as f:
        json.dump(oc, f)

meta = {
    "feature_cols": feature_cols,
    "label_names": label_names,
    "label_map": label_map,
    "school_list": sorted(school_encoder.classes_.tolist()),
}
with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nAll artifacts saved to {OUTPUT_DIR}/")
print("Done!")
