"""
Step 3d — Build Firm-Patent Dataset
=====================================
Combines the eligibility results, crosswalk, and raw patent data to
produce the final dataset of patents linked to their BvD ID, deal date,
acquiror, and event time — restricted to DiD-eligible firms only.

Inputs:
  data/interim/patent_count_eligibility.csv  — output of step3c
  data/interim/crosswalk_final_clean.csv     — output of step2g
  data/raw/patents_main.csv                  — BigQuery main export

Output:
  data/interim/patents_eligible.csv          — one row per patent per
                                               eligible firm, with
                                               pre/post indicators,
                                               event quarter, and
                                               full_text for SBERT

Usage:
  python step3d_build_eligible_patents.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import os
import math

# =============================================================================
# CONFIGURATION
# =============================================================================
ELIGIBILITY_PATH = "data/interim/patent_count_eligibility.csv"
CROSSWALK_PATH   = "data/interim/crosswalk_final_clean.csv"
PATENTS_PATH     = "data/raw/patents_main.csv"
OUTPUT_PATH      = "data/interim/patents_eligible.csv"
PRE_POST_YEARS = 3  
EVENT_WINDOW = 12   # ±12 quarters = ±3 years 

# =============================================================================
# CHECKS
# =============================================================================
print("=" * 55, flush=True)
print("STEP 3d — BUILD FIRM-PATENT DATASET", flush=True)
print("=" * 55, flush=True)

for path, predecessor in [
    (ELIGIBILITY_PATH, "step3c_check_eligibility.py"),
    (CROSSWALK_PATH,   "step2d_clean_crosswalk.py"),
    (PATENTS_PATH,     "bigquery_main_query.sql (BigQuery export)"),
]:
    if not os.path.exists(path):
        print(f"ERROR: required input not found: {path}", flush=True)
        print(f"  Run {predecessor} first", flush=True)
        sys.exit(1)

os.makedirs("data/interim", exist_ok=True)

# =============================================================================
# LOAD INPUTS
# =============================================================================
print(f"\nLoading {ELIGIBILITY_PATH}...", flush=True)
eligibility = pd.read_csv(ELIGIBILITY_PATH)
print(f"  Rows: {len(eligibility):,}", flush=True)

print(f"Loading {CROSSWALK_PATH}...", flush=True)
crosswalk = pd.read_csv(CROSSWALK_PATH, low_memory=False)
crosswalk["uspto_assignee_name"] = crosswalk["uspto_assignee_name"].str.upper()
crosswalk["deal_date"]           = pd.to_datetime(
    crosswalk["deal_date"], errors="coerce"
)
print(f"  Rows: {len(crosswalk):,}  |  "
      f"Unique BvD IDs: {crosswalk['bvd_id'].nunique():,}", flush=True)

print(f"Loading {PATENTS_PATH}...", flush=True)
patents = pd.read_csv(PATENTS_PATH, low_memory=False)
patents["filing_date"] = pd.to_datetime(
    patents["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
print(f"  Rows: {len(patents):,}  |  "
      f"Unique assignees: {patents['assignee_name'].nunique():,}", flush=True)

# =============================================================================
# STEP 1 — Get eligible BvD IDs
# =============================================================================
print(f"\n--- Step 1: Get eligible BvD IDs ---", flush=True)

eligible_bvd_ids = eligibility[
    eligibility["eligible_for_did"] == True
]["bvd_id"].tolist()

print(f"  Eligible BvD IDs: {len(eligible_bvd_ids):,}", flush=True)

if len(eligible_bvd_ids) == 0:
    print("  ERROR: no eligible firms found — check step3c output", flush=True)
    sys.exit(1)

# =============================================================================
# STEP 2 — Get USPTO assignee names for eligible firms
# One BvD ID can map to multiple assignee names
# =============================================================================
print(f"\n--- Step 2: Get USPTO assignee names ---", flush=True)

crosswalk_eligible = crosswalk[
    crosswalk["bvd_id"].isin(eligible_bvd_ids)
].drop_duplicates(subset=["bvd_id", "uspto_assignee_name"])

eligible_assignees = (
    crosswalk_eligible["uspto_assignee_name"].dropna().unique().tolist()
)
print(f"  Eligible USPTO assignee names: {len(eligible_assignees):,}", flush=True)

# =============================================================================
# STEP 3 — Filter patents to eligible firms only
# =============================================================================
print(f"\n--- Step 3: Filter patents to eligible firms ---", flush=True)

patents_eligible = patents[
    patents["assignee_name"].isin(eligible_assignees)
].copy()

print(f"  Patents for eligible firms: {len(patents_eligible):,}", flush=True)

if len(patents_eligible) == 0:
    print("  ERROR: no patents found for eligible assignee names", flush=True)
    print("  Check that patents_main.csv assignee_name column matches "
          "crosswalk uspto_assignee_name (both should be uppercase)", flush=True)
    sys.exit(1)

# =============================================================================
# STEP 4 — Link each patent back to its BvD ID and deal metadata
# =============================================================================
print(f"\n--- Step 4: Merge deal metadata ---", flush=True)

patents_eligible = patents_eligible.merge(
    crosswalk_eligible[[
        "bvd_id",
        "uspto_assignee_name",
        "deal_date",
        "Acquiror name",
        "orbis_target_name",
    ]],
    left_on  = "assignee_name",
    right_on = "uspto_assignee_name",
    how      = "left"
)

# Drop the redundant column after merge
patents_eligible = patents_eligible.drop(columns=["uspto_assignee_name"])

n_missing_deal = patents_eligible["deal_date"].isna().sum()
if n_missing_deal > 0:
    print(f"  WARNING: {n_missing_deal:,} patents have no deal_date after merge",
          flush=True)
else:
    print(f"  All patents matched to a deal date", flush=True)

# =============================================================================
# STEP 5 — Add pre/post merger indicator and event time
# =============================================================================
print(f"\n--- Step 5: Add pre/post indicators and event time ---", flush=True)

patents_eligible["pre_merger"] = (
    (patents_eligible["filing_date"] < patents_eligible["deal_date"]) &
    (patents_eligible["filing_date"] >= patents_eligible["deal_date"]
     - pd.DateOffset(years=PRE_POST_YEARS))
)
patents_eligible["post_merger"] = (
    (patents_eligible["filing_date"] >= patents_eligible["deal_date"]) &
    (patents_eligible["filing_date"] <= patents_eligible["deal_date"]
     + pd.DateOffset(years=PRE_POST_YEARS))
)

# Days relative to deal date (negative = pre-merger, positive = post-merger)
patents_eligible["days_to_deal"] = (
    patents_eligible["filing_date"] - patents_eligible["deal_date"]
).dt.days

# Quarter relative to deal date (t=0 is the quarter the deal closed)
patents_eligible["event_quarter"] = (
    patents_eligible["days_to_deal"] / 91.25
).apply(lambda x: math.floor(x) if pd.notna(x) else None)

print(f"  Pre-merger patents:  {patents_eligible['pre_merger'].sum():,}",
      flush=True)
print(f"  Post-merger patents: {patents_eligible['post_merger'].sum():,}",
      flush=True)
print(f"  Event quarter range: {patents_eligible['event_quarter'].min()} "
      f"to {patents_eligible['event_quarter'].max()}", flush=True)

# =============================================================================
# STEP 6 — Build full_text column for SBERT encoding
# =============================================================================
print(f"\n--- Step 6: Build full_text column ---", flush=True)

patents_eligible["full_text"] = (
    patents_eligible["title_text"].fillna("") + " " +
    patents_eligible["abstract_text"].fillna("") + " " +
    patents_eligible["all_claims_text"].fillna("")
).str.strip()

n_empty = (patents_eligible["full_text"].str.strip() == "").sum()
if n_empty > 0:
    print(f"  WARNING: {n_empty:,} patents have empty full_text", flush=True)
else:
    print(f"  All patents have full_text", flush=True)

# =============================================================================
# STEP 7 — Sanity checks
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"FIRM-PATENT DATASET SUMMARY", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Total patent rows:      {len(patents_eligible):,}", flush=True)
print(f"  Unique publication nos: "
      f"{patents_eligible['publication_number'].nunique():,}", flush=True)
print(f"  Unique BvD IDs:         {patents_eligible['bvd_id'].nunique():,}",
      flush=True)
print(f"  Unique assignee names:  "
      f"{patents_eligible['assignee_name'].nunique():,}", flush=True)
print(f"  Filing date range:      "
      f"{patents_eligible['filing_date'].min().date()} "
      f"to {patents_eligible['filing_date'].max().date()}", flush=True)

print(f"\n  Missing values:", flush=True)
print(
    patents_eligible[["bvd_id", "deal_date", "full_text", "event_quarter"]]
    .isnull().sum().to_string(),
    flush=True
)

print(f"\n  Patents per firm (top 10):", flush=True)
print(
    patents_eligible.groupby("bvd_id")["publication_number"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .to_string(),
    flush=True
)

# Check for eligible BvD IDs that got no patents after the merge
bvd_in_patents = set(patents_eligible["bvd_id"].dropna().unique())
bvd_missing    = set(eligible_bvd_ids) - bvd_in_patents
if bvd_missing:
    print(f"\n  WARNING: {len(bvd_missing):,} eligible BvD IDs have no "
          f"patents after merge:", flush=True)
    for b in sorted(bvd_missing)[:5]:
        print(f"    {b}", flush=True)
    if len(bvd_missing) > 5:
        print(f"    ... and {len(bvd_missing) - 5} more", flush=True)
# =============================================================================
# STEP 8 — Apply ±12 quarter event window filter
# Drops patents outside the DiD analysis window before saving.
# Avoids encoding irrelevant patents in Step 4 (SBERT embeddings).
# Patents outside the window are retained in patents_main.csv for:
#   - patent_age computation  (step3e needs earliest filing date)
#   - log_target_pre_patents  (step3e uses 3yr pre-merger window)
# =============================================================================
print(f"\n--- Step 8: Apply event window filter (±{EVENT_WINDOW} quarters) ---",
      flush=True)

rows_before  = len(patents_eligible)
firms_before = set(patents_eligible["bvd_id"].unique())

window_mask = (
    patents_eligible["pre_merger"] | patents_eligible["post_merger"]
)
patents_eligible = patents_eligible[window_mask].copy()

rows_after  = len(patents_eligible)
firms_after = set(patents_eligible["bvd_id"].unique())
firms_lost  = firms_before - firms_after

print(f"  Rows before: {rows_before:,}  →  after: {rows_after:,}  "
      f"(removed {rows_before - rows_after:,})", flush=True)
print(f"  Event quarter range: {patents_eligible['event_quarter'].min()} "
      f"to {patents_eligible['event_quarter'].max()}", flush=True)

if firms_lost:
    print(f"\n  WARNING: {len(firms_lost):,} firms have no patents within "
          f"±{EVENT_WINDOW} quarters — excluded from DiD:", flush=True)
    for bvd in sorted(firms_lost):
        print(f"    {bvd}", flush=True)
else:
    print(f"  All {len(firms_after):,} firms retained", flush=True)

pre  = int((patents_eligible["event_quarter"] <  0).sum())
post = int((patents_eligible["event_quarter"] >= 0).sum())
print(f"  Pre-merger  (Q-{EVENT_WINDOW} to Q-1): {pre:,}", flush=True)
print(f"  Post-merger (Q0  to Q+{EVENT_WINDOW}): {post:,}", flush=True)

pre_counts = (
    patents_eligible[patents_eligible["pre_merger"]]
    .groupby("bvd_id")["publication_number"]
    .nunique()
)
failed = pre_counts[pre_counts < 3]
if len(failed) > 0:
    print(f"\n  WARNING: {len(failed)} firms have <3 pre-merger patents "
          f"after window filter — these slipped through step3c:")
    print(failed.to_string())
else:
    print(f"  All firms verified: >= 3 pre-merger patents")

# =============================================================================
# STEP 9 — Save
# =============================================================================
patents_eligible.to_csv(OUTPUT_PATH, index=False)
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

print(f"\n{'=' * 55}", flush=True)
print(f"SAVED", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Output:   {OUTPUT_PATH}", flush=True)
print(f"  Rows:     {len(patents_eligible):,}", flush=True)
print(f"  Size:     {size_mb:.1f} MB", flush=True)
print(f"  Columns:  {list(patents_eligible.columns)}", flush=True)

print(f"""
Next steps:
  1. step3e_map_acquirors.py      — verify acquiror names against USPTO
  2. step3f_generate_acquiror_sql.py — generate BigQuery SQL for acquiror patents
     (BigQuery manual) — run query, export data/raw/patents_acquiror.csv
  3. step3g_build_controls.py        — build all control variables
  Then: step4_compute_embeddings.py          — encode all five patent datasets with SBERT
""", flush=True)