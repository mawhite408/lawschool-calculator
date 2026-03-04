import pandas as pd
import numpy as np

df = pd.read_csv("lsdata.csv", skiprows=1, low_memory=False)

print("=== DATE COLUMNS ===")
for col in ["sent_at", "received_at", "complete_at", "ur_at", "ur2_at", "interview_at", "decision_at"]:
    s = pd.to_datetime(df[col], errors="coerce")
    nn = s.notna().sum()
    print(f"  {col:20s}  non-null: {nn:>8,}  min: {s.min()}  max: {s.max()}")

print("\n=== international_gpa ===")
print(df["international_gpa"].value_counts(dropna=False).head(20))

print("\n=== school_name: count of unique ===")
print(df["school_name"].nunique())

print("\n=== school_name: bottom 20 by frequency ===")
print(df["school_name"].value_counts().tail(20))

print("\n=== matriculating_year ===")
print(df["matriculating_year"].value_counts().sort_index())

# Cross-check: result among rows WITH vs WITHOUT sent_at
print("\n=== result distribution among rows WITH sent_at ===")
has_sent = df["sent_at"].notna()
print(df.loc[has_sent, "result"].value_counts(dropna=False))

print("\n=== result distribution among rows WITHOUT sent_at ===")
print(df.loc[~has_sent, "result"].value_counts(dropna=False))

# How many terminal decisions have sent_at?
terminal = df[df["result"].isin(["accepted", "waitlisted", "rejected"])]
print(f"\n=== Terminal decisions: {len(terminal):,} ===")
print(f"  with sent_at:     {terminal['sent_at'].notna().sum():,}")
print(f"  with complete_at: {terminal['complete_at'].notna().sum():,}")
print(f"  with either:      {(terminal['sent_at'].notna() | terminal['complete_at'].notna()).sum():,}")
print(f"  with decision_at: {terminal['decision_at'].notna().sum():,}")
print(f"  with LSAT:        {terminal['lsat'].notna().sum():,}")
print(f"  with GPA:         {terminal['gpa'].notna().sum():,}")
print(f"  with LSAT+GPA:    {(terminal['lsat'].notna() & terminal['gpa'].notna()).sum():,}")

# What about having sent OR complete AND lsat AND gpa?
good = terminal[
    (terminal["sent_at"].notna() | terminal["complete_at"].notna()) &
    terminal["lsat"].notna() & terminal["gpa"].notna()
]
print(f"  with (sent|complete)+LSAT+GPA: {len(good):,}")

# Check: does sent_at ever differ hugely from complete_at?
both = terminal[terminal["sent_at"].notna() & terminal["complete_at"].notna()]
sent = pd.to_datetime(both["sent_at"])
comp = pd.to_datetime(both["complete_at"])
diff = (comp - sent).dt.days
print(f"\n=== sent_at vs complete_at diff (days) where both present ===")
print(diff.describe())
