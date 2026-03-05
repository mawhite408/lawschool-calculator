# Law School Admissions Calculator

**Live**: [arewethereyetlaw.netlify.app](https://arewethereyetlaw.netlify.app)

Survival-conditional admissions probability model for U.S. law schools. Unlike static calculators that answer "what are my odds?", this answers **"given that I applied on date X and it's now date Y with no decision, what's the posterior?"** — i.e., P(outcome | survived to today).

## Architecture

```
Netlify (React/Vite)  →  Render (FastAPI)  →  LightGBM multiclass
                                            + precomputed viz caches
```

- **Frontend**: React + Vite + TailwindCSS + Recharts. SPA with `VITE_API_BASE` env var pointing to the API.
- **Backend**: FastAPI serving predictions + visualization endpoints. Loads precomputed gzipped JSON caches at startup (no CSV needed at runtime). Model file downloaded via `MODEL_DOWNLOAD_URL` env var if not present on disk.
- **Model**: LightGBM GBDT, multiclass (accepted/waitlisted/rejected), trained on ~700K terminal decisions from [LSD.law](https://lsd.law).

## Model Details

### Training paradigm

Each training row is a historical decision. We set `current_day = decision_day` to simulate the user's perspective at that point in the cycle. The model learns: *given your stats, your sent date, the current day, and which decision waves have passed without your name — what's the outcome distribution?*

### Feature vector (81 features)

| Category | Features | Notes |
|----------|----------|-------|
| **Core stats** | LSAT, GPA | Raw values |
| **Timing** | `day_of_cycle_sent`, `month_sent`, `current_day`, `days_waiting`, `cycle_year` | Cycle = Sep 1 → Aug 31 |
| **School aggregates** | Accept/WL/reject rates, median/25th/75th LSAT & GPA, total count | All-time, computed across full history |
| **Recency-weighted stats** | Same percentiles but with exponential decay (half-life = 2 years) | Captures median drift without discarding history |
| **Per-cycle stats** | Same percentiles for the current cycle only + accept count | **Temporal**: computed from decisions ≤ `current_day` only — no look-ahead bias |
| **Relative features** | Applicant LSAT/GPA minus school median/25th/75th | Computed against all-time, recency-weighted, AND current-cycle baselines |
| **Wave survival** | Waves passed (total, accept-heavy, WL-heavy, reject-heavy), decisions/accepts in passed & future waves, days to next wave, days since last wave, next wave composition | Waves detected via density clustering on historical decision dates |
| **Normalized wave** | Above counts as % of school's total historical decisions | Robust to year-over-year volume changes |
| **Cumulative position** | `cum_decision_frac`, `cum_accepted_frac`, `cum_waitlisted_frac`, `cum_rejected_frac` | What fraction of this school's typical decisions have occurred by `current_day` |
| **Profile flags** | URM, international, non-trad, in-state, fee-waived, military, C&F, softs tier, years out | Binary/ordinal |

### Key design decisions

1. **Temporal per-cycle stats (no look-ahead)**: During training, per-cycle medians/percentiles are computed using only accepted decisions *on or before* `current_day` for that row. This prevents a train/inference mismatch — the model learns that early-cycle stats are noisy and shouldn't be over-trusted. Binary search on precomputed expanding-window stats keeps this efficient.

2. **Base-rate anchoring**: Raw model output is blended toward a "base-rate" prediction (model run at `sent_date` with no temporal signal) weighted by `cum_frac / BLEND_THRESHOLD`. When few decisions have been made (pre-wave), this pulls predictions toward the applicant's raw stat profile, preventing wild swings from sparse temporal features.

3. **Wave detection**: Per-school density clustering on `day_of_cycle_decision` with a configurable z-score threshold (`mean + 2σ`) and 3-day merge gap. Waves are used as features but not surfaced in the UI (too noisy for user-facing display currently).

4. **Recency-weighted school stats**: Exponential decay with 2-year half-life. Weighted median/percentiles via sorted cumulative weight interpolation. This handles the systematic upward drift in law school medians without discarding older data entirely.

### Training config

```python
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
    "seed": 42,
}
# 1500 rounds, early stopping at 50
```

### Evaluation

| Metric | Value |
|--------|-------|
| Log loss | 0.466 |
| Accuracy | 81% |
| Accepted F1 | 0.88 |
| Waitlisted F1 | 0.68 |
| Rejected F1 | 0.79 |

Top features by gain: `school_encoded`, `lsat_above_25`, `cycle_year`, `lsat_above_cycle_25`, `gpa_above_25`, `gpa_above_cycle_25`, `school_accept_rate`, `cum_waitlisted_frac`, `cum_rejected_frac`.

### Known biases

- **LSD.law reporting skew**: Accepted applicants are more likely to report, inflating P(accepted) by an estimated 5–10 pp.
- **Waitlist→accept conflation**: LSD.law records waitlist-to-accepted movement as "accepted", which inflates late-cycle accept probabilities.
- **No interview signal**: Schools like Yale/Chicago where interview invites are near-deterministic are a blind spot — the model has no II feature.

## Visualization Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/viz/scatter/{school}` | LSAT/GPA scatter by outcome (sampled to 2000 for perf) |
| `/api/viz/median_drift/{school}` | Yearly accepted median LSAT/GPA with IQR |
| `/api/viz/wait_times/{school}` | Days-to-decision distribution by outcome |
| `/api/viz/similar_applicants/{school}` | Nearest-neighbor outcomes for given LSAT/GPA |
| `/api/cycle_pace` | Fraction of applicants with decisions by date, year-over-year |

All viz data is precomputed from the CSV and saved as gzipped JSON in `precomputed_caches/`. The 161MB CSV is not needed at runtime.

## Local Development

```bash
# Backend
pip install -r requirements.txt
python server.py              # needs lsdata.csv + model_artifacts/

# Frontend
cd frontend && npm install && npm run dev
# Vite proxies /api/* to localhost:8000

# Retrain model
python train_model.py         # outputs to model_artifacts/

# Regenerate viz caches (after CSV update)
python precompute_viz_caches.py
```

## Deployment

- **Frontend**: Netlify. Set `VITE_API_BASE` env var to the Render URL.
- **Backend**: Render (Python web service). Set `MODEL_DOWNLOAD_URL` env var to a direct download link for `lgbm_model.txt` (63MB, hosted on GitHub Releases). Python 3.11 via `runtime.txt`.
- **Data flow**: `lsdata.csv` (161MB, gitignored) → `precompute_viz_caches.py` → `precomputed_caches/*.json.gz` (committed, ~3.3MB) + `train_model.py` → `model_artifacts/` (JSONs committed, model binary on GitHub Releases).
