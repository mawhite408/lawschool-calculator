import pandas as pd
import numpy as np

df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False)
terminal = df[df["result"].isin(["accepted", "waitlisted", "rejected"])].copy()

# Fix: parse dates with errors='coerce' to handle bad dates
for col in ["sent_at", "complete_at", "decision_at"]:
    terminal[col] = pd.to_datetime(terminal[col], errors="coerce")

# sent_at vs complete_at diff where both present
both = terminal[terminal["sent_at"].notna() & terminal["complete_at"].notna()]
diff = (both["complete_at"] - both["sent_at"]).dt.days
print("=== sent_at vs complete_at diff (days) ===")
print(diff.describe())
print(f"Negative diffs (complete before sent): {(diff < 0).sum()}")
print(f"Diff > 90 days: {(diff > 90).sum()}")

# How many rows have sent_at only, complete_at only, both, neither?
has_s = terminal["sent_at"].notna()
has_c = terminal["complete_at"].notna()
print(f"\n=== Date availability in terminal decisions ===")
print(f"  sent only:      {(has_s & ~has_c).sum():,}")
print(f"  complete only:  {(~has_s & has_c).sum():,}")
print(f"  both:           {(has_s & has_c).sum():,}")
print(f"  neither:        {(~has_s & ~has_c).sum():,}")

# For rows with neither sent nor complete, what's the result distribution?
neither = terminal[~has_s & ~has_c]
print(f"\n=== Result dist for rows with NEITHER sent nor complete ===")
print(neither["result"].value_counts())
print(f"  Total: {len(neither):,}")
print(f"  Have decision_at: {neither['decision_at'].notna().sum():,}")

# Cycle year distribution for terminal
print(f"\n=== matriculating_year for terminal decisions ===")
print(terminal["matriculating_year"].value_counts().sort_index())

# Check: how many rows per user?
print(f"\n=== Applications per user ===")
user_counts = terminal["user_uuid"].value_counts()
print(user_counts.describe())
print(f"Users with 1 app: {(user_counts == 1).sum():,}")
print(f"Users with >10 apps: {(user_counts > 10).sum():,}")
print(f"Users with >20 apps: {(user_counts > 20).sum():,}")
print(f"Max apps per user: {user_counts.max()}")

# What does the sent_at distribution look like by month?
print(f"\n=== sent_at month distribution (terminal decisions) ===")
print(terminal["sent_at"].dt.month.value_counts().sort_index())

# What does decision_at month distribution look like?
print(f"\n=== decision_at month distribution (terminal decisions) ===")
print(terminal["decision_at"].dt.month.value_counts().sort_index())

# Result by month_sent
print(f"\n=== Result rate by month sent ===")
has_month = terminal[terminal["sent_at"].notna()].copy()
has_month["month_sent"] = has_month["sent_at"].dt.month
ct = pd.crosstab(has_month["month_sent"], has_month["result"], normalize="index")
print(ct.to_string())
