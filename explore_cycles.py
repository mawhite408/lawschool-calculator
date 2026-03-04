"""
Explore per-cycle median shifts and wave timing variance across years.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False)
df = df[df["result"].isin(["accepted", "waitlisted", "rejected"])].copy()
df = df.dropna(subset=["lsat", "gpa"])

for col in ["sent_at", "decision_at"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["cycle_start"] = pd.to_datetime((df["matriculating_year"] - 1).astype(str) + "-09-01")
df["day_of_cycle_decision"] = (df["decision_at"] - df["cycle_start"]).dt.days

# 1. Per-cycle median LSAT and GPA for WashU
print("=== WashU Median LSAT/GPA by Cycle Year ===")
washu = df[df["school_name"] == "Washington University in St. Louis"]
for year in sorted(washu["matriculating_year"].unique()):
    sub = washu[washu["matriculating_year"] == year]
    acc = sub[sub["result"] == "accepted"]
    if len(acc) < 10:
        continue
    print(f"  {year}: LSAT median={acc['lsat'].median():.0f}  GPA median={acc['gpa'].median():.2f}  (n={len(acc)})")

# 2. Same for Harvard, Georgetown, Michigan
for school in ["Harvard University", "Georgetown University", "University of Michigan"]:
    print(f"\n=== {school} Median LSAT/GPA by Cycle Year (accepted only) ===")
    sub = df[df["school_name"] == school]
    for year in sorted(sub["matriculating_year"].unique()):
        acc = sub[(sub["matriculating_year"] == year) & (sub["result"] == "accepted")]
        if len(acc) < 10:
            continue
        print(f"  {year}: LSAT median={acc['lsat'].median():.0f}  GPA median={acc['gpa'].median():.2f}  (n={len(acc)})")

# 3. How much do medians shift year-to-year?
print("\n\n=== Year-to-Year Median LSAT Shift (top 20 schools) ===")
top_schools = df["school_name"].value_counts().head(20).index
for school in top_schools:
    sub = df[(df["school_name"] == school) & (df["result"] == "accepted")]
    yearly = sub.groupby("matriculating_year")["lsat"].median()
    if len(yearly) < 3:
        continue
    shifts = yearly.diff().dropna()
    print(f"  {school[:40]:40s}  mean shift={shifts.mean():+.1f}  std={shifts.std():.1f}  range={shifts.min():+.0f} to {shifts.max():+.0f}")

# 4. Wave timing variance: for each school, when does the first big accept wave happen?
print("\n\n=== First Accept Wave Day by Cycle Year (top 10 schools) ===")
top10 = df["school_name"].value_counts().head(10).index
for school in top10:
    sub = df[(df["school_name"] == school) & (df["result"] == "accepted")]
    sub = sub[sub["day_of_cycle_decision"].between(0, 400)]
    print(f"\n  {school}:")
    for year in sorted(sub["matriculating_year"].unique()):
        ysub = sub[sub["matriculating_year"] == year]
        if len(ysub) < 10:
            continue
        # Find the peak day (biggest accept cluster)
        day_counts = ysub["day_of_cycle_decision"].value_counts()
        peak_day = day_counts.idxmax()
        peak_count = day_counts.max()
        median_day = ysub["day_of_cycle_decision"].median()
        print(f"    {year}: peak accept day={peak_day:.0f} ({peak_count} accepts)  median accept day={median_day:.0f}")

# 5. Cumulative % of decisions: is it more stable than absolute day?
print("\n\n=== Cumulative Decision % at Day 150 by Year (top 5 schools) ===")
top5 = df["school_name"].value_counts().head(5).index
for school in top5:
    sub = df[(df["school_name"] == school) & sub["day_of_cycle_decision"].between(0, 400)]
    print(f"\n  {school}:")
    for year in sorted(sub["matriculating_year"].unique()):
        ysub = sub[sub["matriculating_year"] == year]
        if len(ysub) < 20:
            continue
        total = len(ysub)
        by_150 = (ysub["day_of_cycle_decision"] <= 150).sum()
        by_200 = (ysub["day_of_cycle_decision"] <= 200).sum()
        print(f"    {year}: by day 150={by_150/total*100:.0f}%  by day 200={by_200/total*100:.0f}%  (n={total})")
