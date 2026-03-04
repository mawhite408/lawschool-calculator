"""
Deep exploration of wave structure in decision dates per school.
Goal: understand how decisions cluster into discrete waves, and how
to build features around 'surviving' past waves.
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
df["day_of_cycle_sent"] = (df["sent_at"] - df["cycle_start"]).dt.days

# Focus on WashU as an example (from the screenshot)
washu = df[df["school_name"] == "Washington University in St. Louis"].copy()
washu = washu[washu["decision_at"].notna() & washu["day_of_cycle_decision"].between(0, 400)]

print("=== WashU decision day distribution ===")
print(f"Total decisions: {len(washu)}")
print(f"\nDecision day stats:")
print(washu["day_of_cycle_decision"].describe())

# Look at decision dates - are they clustered on specific days?
print(f"\n=== Top 30 decision dates at WashU (all cycles) ===")
washu_dates = washu["decision_at"].dt.date.value_counts().head(30)
print(washu_dates)

# Look at decision day_of_cycle clustering
print(f"\n=== Top 30 decision day-of-cycle at WashU ===")
washu_days = washu["day_of_cycle_decision"].value_counts().head(30)
print(washu_days)

# Result breakdown by decision day-of-cycle (binned into weeks)
washu["week_of_cycle"] = (washu["day_of_cycle_decision"] // 7).astype(int)
print(f"\n=== WashU results by week of cycle ===")
ct = pd.crosstab(washu["week_of_cycle"], washu["result"])
print(ct.to_string())

# Now let's look at wave detection more generally
# For a few top schools, find the top decision days and see if they cluster
print("\n\n=== WAVE CLUSTERING FOR TOP 5 SCHOOLS ===")
top_schools = df["school_name"].value_counts().head(5).index.tolist()

for school in top_schools:
    sub = df[(df["school_name"] == school) & df["day_of_cycle_decision"].between(0, 400)].copy()
    if len(sub) < 100:
        continue
    
    print(f"\n--- {school} ({len(sub)} decisions) ---")
    
    # Find peaks: days where decision count is significantly above average
    day_counts = sub["day_of_cycle_decision"].value_counts().sort_index()
    mean_per_day = day_counts.mean()
    std_per_day = day_counts.std()
    threshold = mean_per_day + 2 * std_per_day
    
    wave_days = day_counts[day_counts >= threshold].sort_index()
    print(f"  Avg decisions/day: {mean_per_day:.1f}, threshold for wave: {threshold:.1f}")
    print(f"  Wave days (>{threshold:.0f} decisions): {len(wave_days)}")
    
    if len(wave_days) > 0:
        # Group nearby wave days into clusters (within 3 days = same wave)
        wave_day_list = wave_days.index.tolist()
        clusters = []
        current = [wave_day_list[0]]
        for d in wave_day_list[1:]:
            if d - current[-1] <= 3:
                current.append(d)
            else:
                clusters.append(current)
                current = [d]
        clusters.append(current)
        
        print(f"  Wave clusters: {len(clusters)}")
        for i, c in enumerate(clusters):
            center = int(np.mean(c))
            total_in_wave = day_counts[c].sum()
            # Get result breakdown for this wave
            wave_results = sub[sub["day_of_cycle_decision"].isin(c)]["result"].value_counts()
            print(f"    Wave {i+1}: days {c[0]}-{c[-1]} (center={center}), {total_in_wave} decisions | {wave_results.to_dict()}")


# Key question: for applicants who sent before a wave and WEREN'T in it,
# what happened to them later?
print("\n\n=== SURVIVAL ANALYSIS: WHAT HAPPENS AFTER MISSING A WAVE? ===")
school = "Washington University in St. Louis"
sub = df[(df["school_name"] == school) & df["day_of_cycle_decision"].between(0, 400)].copy()
sub = sub[sub["day_of_cycle_sent"].between(0, 400)]

# Find the biggest accept wave day
day_counts_accept = sub[sub["result"] == "accepted"]["day_of_cycle_decision"].value_counts()
if len(day_counts_accept) > 0:
    biggest_accept_day = day_counts_accept.idxmax()
    print(f"\nBiggest accept wave day at WashU: {biggest_accept_day} ({day_counts_accept.max()} accepts)")
    
    # People who sent BEFORE this wave day
    sent_before = sub[sub["day_of_cycle_sent"] < biggest_accept_day]
    
    # Of those, who got decided ON this wave vs AFTER?
    in_wave = sent_before[sent_before["day_of_cycle_decision"].between(biggest_accept_day - 2, biggest_accept_day + 2)]
    after_wave = sent_before[sent_before["day_of_cycle_decision"] > biggest_accept_day + 2]
    
    print(f"\nApplicants who sent before wave day {biggest_accept_day}:")
    print(f"  Total: {len(sent_before)}")
    print(f"  Decided during wave: {len(in_wave)}")
    print(f"    Results: {in_wave['result'].value_counts().to_dict()}")
    print(f"  Decided AFTER wave: {len(after_wave)}")
    print(f"    Results: {after_wave['result'].value_counts().to_dict()}")

# Also: what % of all decisions have occurred by each week?
print(f"\n=== CUMULATIVE % OF DECISIONS BY WEEK (WashU) ===")
sub_sorted = sub.sort_values("day_of_cycle_decision")
total = len(sub_sorted)
for week in range(0, 52, 4):
    day = week * 7
    cum = (sub_sorted["day_of_cycle_decision"] <= day).sum()
    pct = cum / total * 100
    if pct > 0:
        print(f"  Week {week:2d} (day {day:3d}): {pct:5.1f}% of decisions made")
