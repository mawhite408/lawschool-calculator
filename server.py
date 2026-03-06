"""
Law School Admissions Calculator - FastAPI Backend (v3)
Wave-aware, survival-conditional predictions with cycle-year awareness.
Takes current_date as input to compute conditional probabilities
given that the applicant has survived to this point without a decision.
"""

import gzip
import json
import math
import os
import random
import urllib.request
from datetime import date, datetime
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────

MODEL_DIR = "model_artifacts"

# In production (Render), the large model file may need to be downloaded.
# Set MODEL_DOWNLOAD_URL env var to a direct download link (e.g. GitHub Release asset).
_model_path = os.path.join(MODEL_DIR, "lgbm_model.txt")
if not os.path.exists(_model_path):
    _url = os.environ.get("MODEL_DOWNLOAD_URL")
    if _url:
        print(f"Downloading model from {_url} ...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(_url, _model_path)
        print("Model downloaded.")
    else:
        raise FileNotFoundError(
            f"{_model_path} not found. Set MODEL_DOWNLOAD_URL env var or commit the model file."
        )

model = lgb.Booster(model_file=_model_path)
school_encoder = joblib.load(os.path.join(MODEL_DIR, "school_encoder.joblib"))

with open(os.path.join(MODEL_DIR, "school_stats.json")) as f:
    school_stats_list = json.load(f)
school_stats = {s["school_name"]: s for s in school_stats_list}

with open(os.path.join(MODEL_DIR, "school_waves.json")) as f:
    school_waves = json.load(f)

with open(os.path.join(MODEL_DIR, "cum_curves.json")) as f:
    cum_curves = json.load(f)

# Per-outcome cumulative curves
cum_outcome_curves = {}
for outcome in ["accepted", "waitlisted", "rejected"]:
    fpath = os.path.join(MODEL_DIR, f"cum_{outcome}_curves.json")
    if os.path.exists(fpath):
        with open(fpath) as f:
            cum_outcome_curves[outcome] = json.load(f)
    else:
        cum_outcome_curves[outcome] = {}

# Recency-weighted stats
with open(os.path.join(MODEL_DIR, "rw_stats.json")) as f:
    rw_stats_list = json.load(f)
rw_stats = {s["school_name"]: s for s in rw_stats_list}

# Per-cycle stats (school x year)
with open(os.path.join(MODEL_DIR, "cycle_stats.json")) as f:
    cycle_stats_list = json.load(f)
# Key by (school_name, matriculating_year)
cycle_stats = {}
for s in cycle_stats_list:
    key = (s["school_name"], int(s["matriculating_year"]))
    cycle_stats[key] = s

with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
    meta = json.load(f)

FEATURE_COLS = meta["feature_cols"]
LABEL_NAMES = meta["label_names"]
SCHOOL_LIST = meta["school_list"]

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Law School Admissions Calculator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    lsat: int
    gpa: float
    school_name: str
    sent_date: Optional[str] = None    # ISO YYYY-MM-DD
    current_date: Optional[str] = None  # ISO YYYY-MM-DD — "today"
    matriculating_year: int = 2026
    urm: bool = False
    is_international: bool = False
    non_trad: bool = False
    is_in_state: bool = False
    is_fee_waived: bool = False
    is_military: bool = False
    c_and_f: bool = False
    softs: Optional[str] = None
    years_out: Optional[int] = None


def interpolate_cum_curve(curve_dict: dict, day: float) -> float:
    """Interpolate a cumulative curve (keyed by string day every 5 days)."""
    if not curve_dict:
        return float("nan")
    d = int(min(max(day, 0), 400))
    # Snap to nearest 5-day bucket
    lo = (d // 5) * 5
    hi = lo + 5
    lo_val = curve_dict.get(str(lo), 0.0)
    hi_val = curve_dict.get(str(min(hi, 400)), lo_val)
    if hi == lo:
        return lo_val
    frac = (d - lo) / (hi - lo)
    return lo_val + frac * (hi_val - lo_val)


def compute_wave_features_for_request(sent_day, current_day, waves, school_total):
    """Compute wave survival features for a single prediction request."""
    NAN = float("nan")
    nan_result = {
        "waves_passed_total": NAN, "waves_passed_accept_heavy": NAN,
        "waves_passed_wl_heavy": NAN, "waves_passed_reject_heavy": NAN,
        "decisions_in_passed_waves": NAN, "accepts_in_passed_waves": NAN,
        "wl_in_passed_waves": NAN, "rejects_in_passed_waves": NAN,
        "decisions_in_future_waves": NAN, "accepts_in_future_waves": NAN,
        "wl_in_future_waves": NAN, "rejects_in_future_waves": NAN,
        "days_to_next_wave": NAN, "days_since_last_wave": NAN,
        "next_wave_accept_pct": NAN, "next_wave_wl_pct": NAN,
        "next_wave_reject_pct": NAN,
        "pct_decisions_in_passed_waves": NAN, "pct_decisions_in_future_waves": NAN,
        "pct_accepts_in_passed_waves": NAN, "pct_accepts_in_future_waves": NAN,
    }

    if not waves or current_day is None:
        return nan_result

    applicable_sent = sent_day if sent_day is not None else 0
    passed = [w for w in waves if w["end"] <= current_day and w["start"] >= applicable_sent]
    future = [w for w in waves if w["start"] > current_day]

    acc_heavy = sum(1 for w in passed if w["accepted"] >= w["waitlisted"] and w["accepted"] >= w["rejected"])
    wl_heavy = sum(1 for w in passed if w["waitlisted"] > w["accepted"] and w["waitlisted"] >= w["rejected"])
    rej_heavy = sum(1 for w in passed if w["rejected"] > w["accepted"] and w["rejected"] > w["waitlisted"])

    dec_passed = sum(w["count"] for w in passed)
    acc_passed = sum(w["accepted"] for w in passed)
    dec_future = sum(w["count"] for w in future)
    acc_future = sum(w["accepted"] for w in future)
    total_acc_in_waves = sum(w["accepted"] for w in waves)

    result = {}
    result["waves_passed_total"] = len(passed)
    result["waves_passed_accept_heavy"] = acc_heavy
    result["waves_passed_wl_heavy"] = wl_heavy
    result["waves_passed_reject_heavy"] = rej_heavy
    result["decisions_in_passed_waves"] = dec_passed
    result["accepts_in_passed_waves"] = acc_passed
    result["wl_in_passed_waves"] = sum(w["waitlisted"] for w in passed)
    result["rejects_in_passed_waves"] = sum(w["rejected"] for w in passed)
    result["decisions_in_future_waves"] = dec_future
    result["accepts_in_future_waves"] = acc_future
    result["wl_in_future_waves"] = sum(w["waitlisted"] for w in future)
    result["rejects_in_future_waves"] = sum(w["rejected"] for w in future)

    result["days_to_next_wave"] = (future[0]["start"] - current_day) if future else NAN
    result["days_since_last_wave"] = (current_day - passed[-1]["end"]) if passed else NAN

    if future:
        nw = future[0]
        nw_total = max(nw["count"], 1)
        result["next_wave_accept_pct"] = nw["accepted"] / nw_total
        result["next_wave_wl_pct"] = nw["waitlisted"] / nw_total
        result["next_wave_reject_pct"] = nw["rejected"] / nw_total
    else:
        result["next_wave_accept_pct"] = NAN
        result["next_wave_wl_pct"] = NAN
        result["next_wave_reject_pct"] = NAN

    # Normalized wave features
    total_dec = max(school_total, 1)
    total_acc_norm = max(total_acc_in_waves, 1)
    result["pct_decisions_in_passed_waves"] = dec_passed / total_dec
    result["pct_decisions_in_future_waves"] = dec_future / total_dec
    result["pct_accepts_in_passed_waves"] = acc_passed / total_acc_norm
    result["pct_accepts_in_future_waves"] = acc_future / total_acc_norm

    return result


def nan_safe(v):
    if v is None:
        return np.nan
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return np.nan
    return v


def build_features(req: PredictionRequest) -> np.ndarray:
    """Build the 81-feature vector matching v3 training."""

    cycle_start = datetime(req.matriculating_year - 1, 9, 1)

    # --- Sent date ---
    day_of_cycle_sent = None
    month_sent = None
    if req.sent_date:
        app_date = datetime.fromisoformat(req.sent_date)
        day_of_cycle_sent = (app_date - cycle_start).days
        if day_of_cycle_sent < 0 or day_of_cycle_sent > 450:
            day_of_cycle_sent = None
        month_sent = app_date.month

    # --- Current date (today) ---
    current_day = None
    if req.current_date:
        today = datetime.fromisoformat(req.current_date)
        current_day = (today - cycle_start).days
        if current_day < 0 or current_day > 450:
            current_day = None

    days_waiting = None
    if day_of_cycle_sent is not None and current_day is not None:
        days_waiting = current_day - day_of_cycle_sent

    # --- School stats ---
    ss = school_stats.get(req.school_name)
    if ss is None:
        raise ValueError(f"Unknown school: {req.school_name}")

    school_enc = int(school_encoder.transform([req.school_name])[0])

    # All-time
    s_ar = ss["school_accept_rate"]
    s_wr = ss["school_wl_rate"]
    s_rr = ss["school_reject_rate"]
    s_ml = ss["school_median_lsat"]
    s_mg = ss["school_median_gpa"]
    s_25l = ss["school_25_lsat"]
    s_75l = ss["school_75_lsat"]
    s_25g = ss["school_25_gpa"]
    s_75g = ss["school_75_gpa"]
    s_count = ss["school_count"]

    # Recency-weighted
    rw = rw_stats.get(req.school_name, {})
    rw_ml = rw.get("rw_median_lsat")
    rw_mg = rw.get("rw_median_gpa")
    rw_25l = rw.get("rw_25_lsat")
    rw_75l = rw.get("rw_75_lsat")
    rw_25g = rw.get("rw_25_gpa")
    rw_75g = rw.get("rw_75_gpa")

    # Per-cycle
    cs = cycle_stats.get((req.school_name, req.matriculating_year), {})
    c_ml = cs.get("cycle_median_lsat")
    c_mg = cs.get("cycle_median_gpa")
    c_25l = cs.get("cycle_25_lsat")
    c_75l = cs.get("cycle_75_lsat")
    c_25g = cs.get("cycle_25_gpa")
    c_75g = cs.get("cycle_75_gpa")
    c_acc_count = cs.get("cycle_accept_count")

    # Relative to all-time
    lsat_am = req.lsat - s_ml
    gpa_am = req.gpa - s_mg
    lsat_a75 = req.lsat - s_75l
    gpa_a75 = req.gpa - s_75g
    lsat_a25 = req.lsat - s_25l
    gpa_a25 = req.gpa - s_25g

    # Relative to recency-weighted
    lsat_a_rw_m = (req.lsat - rw_ml) if rw_ml is not None else None
    gpa_a_rw_m = (req.gpa - rw_mg) if rw_mg is not None else None
    lsat_a_rw_75 = (req.lsat - rw_75l) if rw_75l is not None else None
    gpa_a_rw_75 = (req.gpa - rw_75g) if rw_75g is not None else None

    # Relative to this cycle
    lsat_a_c_m = (req.lsat - c_ml) if c_ml is not None else None
    gpa_a_c_m = (req.gpa - c_mg) if c_mg is not None else None
    lsat_a_c_75 = (req.lsat - c_75l) if c_75l is not None else None
    gpa_a_c_75 = (req.gpa - c_75g) if c_75g is not None else None
    lsat_a_c_25 = (req.lsat - c_25l) if c_25l is not None else None
    gpa_a_c_25 = (req.gpa - c_25g) if c_25g is not None else None

    # --- Wave features ---
    waves = school_waves.get(req.school_name, [])
    wf = compute_wave_features_for_request(day_of_cycle_sent, current_day, waves, s_count)

    # --- Cumulative fractions ---
    cum_dec = float("nan")
    cum_acc = float("nan")
    cum_wl = float("nan")
    cum_rej = float("nan")
    if current_day is not None:
        school_curve = cum_curves.get(req.school_name, {})
        cum_dec = interpolate_cum_curve(school_curve, current_day)
        for outcome in ["accepted", "waitlisted", "rejected"]:
            oc = cum_outcome_curves.get(outcome, {}).get(req.school_name, {})
            val = interpolate_cum_curve(oc, current_day)
            if outcome == "accepted":
                cum_acc = val
            elif outcome == "waitlisted":
                cum_wl = val
            else:
                cum_rej = val

    # --- Softs ---
    softs_map = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}
    softs_encoded = None
    if req.softs:
        softs_encoded = softs_map.get(req.softs.lower().strip())

    # --- Build 81-feature vector in exact FEATURE_COLS order ---
    _ = nan_safe
    features = [
        req.lsat,                          # lsat
        req.gpa,                           # gpa
        _(day_of_cycle_sent),              # day_of_cycle_sent
        _(month_sent),                     # month_sent
        _(current_day),                    # current_day
        _(days_waiting),                   # days_waiting
        req.matriculating_year,            # cycle_year
        school_enc,                        # school_encoded
        # All-time school aggregates
        s_ar, s_wr, s_rr,
        s_ml, s_mg, s_25l, s_75l, s_25g, s_75g,
        s_count,
        # Recency-weighted
        _(rw_ml), _(rw_mg), _(rw_25l), _(rw_75l), _(rw_25g), _(rw_75g),
        # Per-cycle
        _(c_ml), _(c_mg), _(c_25l), _(c_75l), _(c_25g), _(c_75g), _(c_acc_count),
        # Relative to all-time
        lsat_am, gpa_am, lsat_a75, gpa_a75, lsat_a25, gpa_a25,
        # Relative to recency-weighted
        _(lsat_a_rw_m), _(gpa_a_rw_m), _(lsat_a_rw_75), _(gpa_a_rw_75),
        # Relative to this cycle
        _(lsat_a_c_m), _(gpa_a_c_m), _(lsat_a_c_75), _(gpa_a_c_75),
        _(lsat_a_c_25), _(gpa_a_c_25),
        # Wave survival features
        _(wf["waves_passed_total"]),
        _(wf["waves_passed_accept_heavy"]),
        _(wf["waves_passed_wl_heavy"]),
        _(wf["waves_passed_reject_heavy"]),
        _(wf["decisions_in_passed_waves"]),
        _(wf["accepts_in_passed_waves"]),
        _(wf["wl_in_passed_waves"]),
        _(wf["rejects_in_passed_waves"]),
        _(wf["decisions_in_future_waves"]),
        _(wf["accepts_in_future_waves"]),
        _(wf["wl_in_future_waves"]),
        _(wf["rejects_in_future_waves"]),
        _(wf["days_to_next_wave"]),
        _(wf["days_since_last_wave"]),
        _(wf["next_wave_accept_pct"]),
        _(wf["next_wave_wl_pct"]),
        _(wf["next_wave_reject_pct"]),
        # Normalized wave features
        _(wf["pct_decisions_in_passed_waves"]),
        _(wf["pct_decisions_in_future_waves"]),
        _(wf["pct_accepts_in_passed_waves"]),
        _(wf["pct_accepts_in_future_waves"]),
        # Cumulative cycle position
        _(cum_dec), _(cum_acc), _(cum_wl), _(cum_rej),
        # Profile
        int(req.urm), int(req.is_international), int(req.non_trad),
        int(req.is_in_state), int(req.is_fee_waived), int(req.is_military),
        int(req.c_and_f),
        _(softs_encoded), _(req.years_out),
    ]

    return np.array([features], dtype=np.float64)


@app.get("/api/schools")
def get_schools():
    """Return list of available schools with wave info."""
    return {"schools": SCHOOL_LIST}


@app.get("/api/school_waves/{school_name:path}")
def get_school_waves(school_name: str):
    """Return wave table for a specific school."""
    waves = school_waves.get(school_name, [])
    return {"school_name": school_name, "waves": waves}


def _anchored_proba(req) -> np.ndarray:
    """Return cum_frac-weighted probability vector for a prediction request.

    The raw model answers: "if you got a decision TODAY, what would it be?"
    But early in the cycle, the chance of getting a decision at all is tiny,
    and the few early decisions skew heavily toward accepts.  We blend:

        P = cum_frac * P(outcome | decision by now)      [raw model]
          + (1 - cum_frac) * P(outcome | decision later) [end-of-cycle model]

    cum_frac = fraction of this school's decisions typically made by current_day.
    The end-of-cycle prediction (day 365) gives a stats-anchored baseline that
    isn't inflated by early-cycle accept skew.
    """
    proba = model.predict(build_features(req))[0]

    # Compute cum_frac for current date
    school_cum_curve = cum_curves.get(req.school_name, {})
    cycle_start = datetime(req.matriculating_year - 1, 9, 1)
    if req.current_date:
        current_day = (datetime.fromisoformat(req.current_date) - cycle_start).days
    else:
        current_day = 0
    cum_frac = interpolate_cum_curve(school_cum_curve, current_day)

    if cum_frac >= 0.99:
        return proba  # late enough that virtually all decisions are in

    # End-of-cycle prediction: what happens if we wait until the cycle is over
    eoc_req = req.model_copy()
    eoc_date = cycle_start + pd.Timedelta(days=365)
    eoc_req.current_date = eoc_date.strftime("%Y-%m-%d")
    p_eoc = model.predict(build_features(eoc_req))[0]

    return cum_frac * proba + (1.0 - cum_frac) * p_eoc


@app.post("/api/predict")
def predict(req: PredictionRequest):
    """Return admission probabilities conditional on current date."""
    try:
        proba = _anchored_proba(req)
        result = {
            LABEL_NAMES[i]: round(float(proba[i]) * 100, 1)
            for i in range(len(LABEL_NAMES))
        }

        # School context
        ss = school_stats.get(req.school_name, {})
        context = {
            "school_median_lsat": ss.get("school_median_lsat"),
            "school_median_gpa": ss.get("school_median_gpa"),
            "school_25_lsat": ss.get("school_25_lsat"),
            "school_75_lsat": ss.get("school_75_lsat"),
            "school_25_gpa": ss.get("school_25_gpa"),
            "school_75_gpa": ss.get("school_75_gpa"),
            "school_accept_rate": round(ss.get("school_accept_rate", 0) * 100, 1),
            "school_count": ss.get("school_count"),
        }

        # Wave context for the UI
        waves = school_waves.get(req.school_name, [])
        wave_info = None
        if req.current_date and req.sent_date:
            cycle_start = datetime(req.matriculating_year - 1, 9, 1)
            current_day = (datetime.fromisoformat(req.current_date) - cycle_start).days
            sent_day = (datetime.fromisoformat(req.sent_date) - cycle_start).days

            passed = [w for w in waves if w["end"] <= current_day and w["start"] >= sent_day]
            future = [w for w in waves if w["start"] > current_day]
            wave_info = {
                "total_waves": len(waves),
                "waves_passed": len(passed),
                "waves_remaining": len(future),
                "passed_waves": passed,
                "upcoming_waves": future[:3],  # Next 3 waves
            }

        return {
            "probabilities": result,
            "school_context": context,
            "wave_info": wave_info,
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.post("/api/predict_timeline")
def predict_timeline(req: PredictionRequest):
    """Sweep current_day across the full cycle and return probabilities at each point.
    Used to generate the timeline chart showing how odds change as the cycle progresses."""
    try:
        cycle_start = datetime(req.matriculating_year - 1, 9, 1)

        # Determine sent_day
        sent_day = None
        if req.sent_date:
            sent_day = (datetime.fromisoformat(req.sent_date) - cycle_start).days

        # Determine the actual current_day for the marker
        actual_current_day = None
        if req.current_date:
            actual_current_day = (datetime.fromisoformat(req.current_date) - cycle_start).days

        # Sweep from sent_day (or cycle start) to day 365 (end of Aug)
        start_sweep = max(sent_day or 0, 0)
        end_sweep = 365  # ~Aug 31

        # Sample every 3 days for performance (model inference is fast but 365 calls adds up)
        timeline = []
        step = 3
        days_to_eval = list(range(start_sweep, end_sweep + 1, step))
        # Make sure we include the actual current day if provided
        if actual_current_day is not None and actual_current_day not in days_to_eval:
            days_to_eval.append(actual_current_day)
            days_to_eval.sort()

        # Preload cumulative curve for blending
        school_cum_curve = cum_curves.get(req.school_name, {})

        # End-of-cycle prediction: stats-anchored baseline without early-cycle
        # accept skew.  This is what the model predicts if we wait until all
        # decisions are in.
        eoc_req = req.model_copy()
        eoc_date = cycle_start + pd.Timedelta(days=365)
        eoc_req.current_date = eoc_date.strftime("%Y-%m-%d")
        p_eoc = model.predict(build_features(eoc_req))[0]

        # Build feature vectors for all days at once
        for day in days_to_eval:
            # Create a modified request with this day as current_date
            day_date = cycle_start + pd.Timedelta(days=int(day))
            modified = req.model_copy()
            modified.current_date = day_date.strftime("%Y-%m-%d")

            X = build_features(modified)
            proba = model.predict(X)[0]

            # Cumulative decision fraction = weight for blending
            cum_frac = interpolate_cum_curve(school_cum_curve, int(day))

            # Blend: cum_frac weight on survival-conditional prediction,
            # (1 - cum_frac) weight on end-of-cycle prediction.
            p_blended = cum_frac * proba + (1.0 - cum_frac) * p_eoc

            timeline.append({
                "day": int(day),
                "date": day_date.strftime("%Y-%m-%d"),
                "accepted": round(float(p_blended[0]) * 100, 1),
                "waitlisted": round(float(p_blended[1]) * 100, 1),
                "rejected": round(float(p_blended[2]) * 100, 1),
                "confidence": round(float(cum_frac), 3),
            })

        # Wave markers for the chart
        waves = school_waves.get(req.school_name, [])
        wave_markers = [
            {"day": w["center"], "date": (cycle_start + pd.Timedelta(days=w["center"])).strftime("%Y-%m-%d")}
            for w in waves
            if start_sweep <= w["center"] <= end_sweep
        ]

        return {
            "timeline": timeline,
            "actual_current_day": actual_current_day,
            "wave_markers": wave_markers,
            "cycle_start": cycle_start.strftime("%Y-%m-%d"),
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Timeline generation failed: {str(e)}"}


# ─────────────────────────────────────────────
# Visualization caches — load precomputed or build from CSV
# ─────────────────────────────────────────────

_CURRENT_MAT_YEAR = 2026
_PACE_YEARS = [_CURRENT_MAT_YEAR - 3, _CURRENT_MAT_YEAR - 2, _CURRENT_MAT_YEAR - 1, _CURRENT_MAT_YEAR]
_CACHE_DIR = "precomputed_caches"
_CACHE_NAMES = ["scatter", "drift", "heatmap", "waittime", "pace", "current_cycle"]


def _load_gz_cache(name: str) -> dict:
    path = os.path.join(_CACHE_DIR, f"{name}.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


_has_precomputed = all(
    os.path.exists(os.path.join(_CACHE_DIR, f"{n}.json.gz")) for n in _CACHE_NAMES
)

if _has_precomputed:
    print("Loading precomputed viz caches...")
    _scatter_cache = _load_gz_cache("scatter")
    _drift_cache   = _load_gz_cache("drift")
    _heatmap_cache = _load_gz_cache("heatmap")
    _waittime_cache = _load_gz_cache("waittime")
    _pace_cache    = _load_gz_cache("pace")
    _current_cycle_cache = _load_gz_cache("current_cycle")
    # Pace cache keys come back as strings from JSON; normalise int keys
    _pace_cache = {
        school: {int(yr): v for yr, v in yr_data.items()}
        for school, yr_data in _pace_cache.items()
    }
    print(f"  Precomputed caches loaded ({len(_scatter_cache)} schools).")
else:
    print("Precomputed caches not found — building from lsdata.csv...")
    print("Loading CSV for visualization endpoints...")
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

    print("  Pre-aggregating scatter data...")
    _scatter_cache = {}
    for school_name, group in _viz_df.groupby("school_name"):
        sub = group[["lsat", "gpa", "result", "matriculating_year"]].dropna()
        _scatter_cache[school_name] = sub.to_dict(orient="records")

    print("  Pre-aggregating median drift...")
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

    print("  Pre-aggregating wave heatmap data...")
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

    print("  Pre-aggregating wait time distributions...")
    _waittime_cache = {}
    for school_name, group in _viz_df.groupby("school_name"):
        valid = group[group["days_to_decision"].between(1, 365)].copy()
        if len(valid) < 20:
            continue
        valid["bucket"] = ((valid["days_to_decision"] // 14) * 14).astype(int)
        buckets = valid.groupby(["bucket", "result"]).size().reset_index(name="count")
        _waittime_cache[school_name] = buckets.to_dict(orient="records")

    print("  Visualization data ready.")
    print("  Pre-aggregating cycle pace data...")

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

    print("  Building ALL-schools pace cache...")
    _pace_cache: dict = {}
    _pace_cache["ALL"] = _build_pace_for_subset(_viz_df, _all_apps_df)

    print("  Building per-school pace cache...")
    _all_apps_by_school = {s: g for s, g in _all_apps_df.groupby("school_name")}
    for _school, _grp in _viz_df.groupby("school_name"):
        _all_grp = _all_apps_by_school.get(_school, pd.DataFrame())
        _pr = _build_pace_for_subset(_grp, _all_grp)
        if _pr:
            _pace_cache[_school] = _pr
    print(f"  Cycle pace data ready ({len(_pace_cache)} entries).")

    print("  Building current-cycle applicants cache...")
    _cycle_df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False,
        usecols=lambda c: c in {"school_name", "matriculating_year", "lsat", "gpa", "result"})
    _cycle_df = _cycle_df[_cycle_df["matriculating_year"] == _CURRENT_MAT_YEAR].copy()
    _cycle_df = _cycle_df.dropna(subset=["lsat", "gpa", "school_name"])
    _cycle_df["result"] = _cycle_df["result"].fillna("pending")
    _cycle_df.loc[~_cycle_df["result"].isin(["accepted", "waitlisted", "rejected", "pending"]), "result"] = "pending"
    _current_cycle_cache = {}
    for _school, _grp in _cycle_df.groupby("school_name"):
        _sub = _grp[["lsat", "gpa", "result"]].copy()
        _sub["lsat"] = _sub["lsat"].astype(int)
        _sub["gpa"] = _sub["gpa"].round(2)
        _current_cycle_cache[_school] = _sub.to_dict(orient="records")


@app.get("/api/viz/scatter/{school_name:path}")
def viz_scatter(school_name: str, year: Optional[int] = None):
    """Return LSAT/GPA scatter data for a school, optionally filtered by cycle year."""
    points = _scatter_cache.get(school_name, [])
    if year:
        # Return ALL points for a specific year (no sampling)
        points = [p for p in points if p.get("matriculating_year") == year]
    elif len(points) > 2000:
        # Only sample when showing all years combined
        rng = random.Random(42)
        points = rng.sample(points, 2000)
    return {"school_name": school_name, "points": points}


@app.get("/api/viz/median_drift/{school_name:path}")
def viz_median_drift(school_name: str):
    """Return year-over-year accepted median LSAT/GPA for a school."""
    data = _drift_cache.get(school_name, [])
    return {"school_name": school_name, "yearly": data}


@app.get("/api/viz/wave_heatmap/{school_name:path}")
def viz_wave_heatmap(school_name: str):
    """Return weekly decision density heatmap data for a school."""
    data = _heatmap_cache.get(school_name, [])
    return {"school_name": school_name, "weeks": data}


@app.get("/api/viz/similar_applicants/{school_name:path}")
def viz_similar_applicants(school_name: str, lsat: int, gpa: float, lsat_range: int = 2, gpa_range: float = 0.1):
    """Return outcome distribution for applicants with similar stats at this school."""
    points = _scatter_cache.get(school_name, [])
    similar = [
        p for p in points
        if abs(p["lsat"] - lsat) <= lsat_range and abs(p["gpa"] - gpa) <= gpa_range
    ]
    total = len(similar)
    if total == 0:
        return {"school_name": school_name, "total": 0, "accepted": 0, "waitlisted": 0, "rejected": 0, "applicants": []}
    outcomes = {"accepted": 0, "waitlisted": 0, "rejected": 0}
    for p in similar:
        outcomes[p["result"]] = outcomes.get(p["result"], 0) + 1
    return {
        "school_name": school_name,
        "total": total,
        "accepted": outcomes["accepted"],
        "waitlisted": outcomes["waitlisted"],
        "rejected": outcomes["rejected"],
        "applicants": similar[:50],  # Cap for response size
    }


@app.get("/api/viz/similar_applicants_cycle/{school_name:path}")
def viz_similar_applicants_cycle(school_name: str, lsat: int, gpa: float, lsat_range: int = 2, gpa_range: float = 0.1):
    """Return outcome distribution for current-cycle applicants with similar stats, including pending."""
    points = _current_cycle_cache.get(school_name, [])
    similar = [
        p for p in points
        if abs(p["lsat"] - lsat) <= lsat_range and abs(p["gpa"] - gpa) <= gpa_range
    ]
    total = len(similar)
    if total == 0:
        return {"school_name": school_name, "total": 0, "accepted": 0, "waitlisted": 0, "rejected": 0, "pending": 0}
    outcomes = {"accepted": 0, "waitlisted": 0, "rejected": 0, "pending": 0}
    for p in similar:
        outcomes[p["result"]] = outcomes.get(p["result"], 0) + 1
    return {
        "school_name": school_name,
        "total": total,
        "accepted": outcomes["accepted"],
        "waitlisted": outcomes["waitlisted"],
        "rejected": outcomes["rejected"],
        "pending": outcomes["pending"],
        "cycle_year": _CURRENT_MAT_YEAR,
    }


@app.get("/api/viz/wait_times/{school_name:path}")
def viz_wait_times(school_name: str):
    """Return days-to-decision distribution by outcome for a school."""
    data = _waittime_cache.get(school_name, [])
    return {"school_name": school_name, "buckets": data}


@app.get("/api/cycle_pace")
def cycle_pace(school_name: str = "ALL"):
    """Compare what fraction of applicants have a decision by today vs prior cycles.

    Uses frac = cumulative_decisions_by_day / total_applicants_that_cycle.
    This controls for LSD.law user growth: if the platform doubles in size,
    both numerator and denominator double, leaving the fraction unchanged.
    A genuinely slow cycle shows up as a lower fraction than prior years at
    the same day.
    """
    data = _pace_cache.get(school_name)
    if not data:
        return {"error": f"No pace data for '{school_name}'"}

    today = date.today()
    current_cycle_start = datetime(_CURRENT_MAT_YEAR - 1, 9, 1)
    today_day = max(0, min(
        (datetime.combine(today, datetime.min.time()) - current_cycle_start).days,
        400,
    ))

    past_years = sorted([y for y in data if y < _CURRENT_MAT_YEAR])[-3:]
    prior_year = past_years[-1] if past_years else None

    # Build per-year response: expose frac curves + metadata
    cycles: dict = {}
    for year, year_data in data.items():
        cycles[str(year)] = {
            "curve": [
                {"day": pt["day"], "frac": pt["frac"], "count": pt["count"]}
                for pt in year_data["curve"]
            ],
            "total_apps": year_data["total_apps"],
            "decisions_total": year_data["decisions_total"],
        }

    def _frac_at(year_key: str, day: int) -> float | None:
        if year_key not in cycles:
            return None
        pts = [p for p in cycles[year_key]["curve"] if p["day"] <= day]
        return pts[-1]["frac"] if pts else 0.0

    current_frac = _frac_at(str(_CURRENT_MAT_YEAR), today_day)
    prior_frac   = _frac_at(str(prior_year), today_day) if prior_year else None

    # Primary headline: fraction difference vs prior year (ppt + relative %)
    frac_diff_ppt: float | None = None       # percentage-point difference
    pct_vs_prior_year: float | None = None   # relative % change
    if current_frac is not None and prior_frac is not None and prior_frac > 0:
        frac_diff_ppt     = round((current_frac - prior_frac) * 100, 1)
        pct_vs_prior_year = round((current_frac / prior_frac - 1) * 100, 1)

    # 3-year average for reference
    past_fracs = [f for y in past_years if (f := _frac_at(str(y), today_day)) is not None]
    avg_past_frac = round(float(np.mean(past_fracs)), 4) if past_fracs else None
    pct_vs_3yr_avg: float | None = None
    if current_frac is not None and avg_past_frac and avg_past_frac > 0:
        pct_vs_3yr_avg = round((current_frac / avg_past_frac - 1) * 100, 1)

    return {
        "today_day": today_day,
        "today_date": today.isoformat(),
        "cycles": cycles,
        # Primary: fraction-based comparison vs most recent complete cycle
        "current_frac": round(current_frac * 100, 1) if current_frac is not None else None,
        "prior_frac": round(prior_frac * 100, 1) if prior_frac is not None else None,
        "frac_diff_ppt": frac_diff_ppt,
        "pct_vs_prior_year": pct_vs_prior_year,
        "prior_year": prior_year,
        # 3yr average reference
        "avg_past_frac": round(avg_past_frac * 100, 1) if avg_past_frac else None,
        "pct_vs_3yr_avg": pct_vs_3yr_avg,
        "current_mat_year": _CURRENT_MAT_YEAR,
        "past_mat_years": past_years,
    }


# Serve static frontend
if os.path.isdir("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = os.path.join("frontend/dist", full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse("frontend/dist/index.html")
