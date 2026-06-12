"""
Step 2d — Final Crosswalk Cleaning
=====================================
Loads crosswalk_verified.csv, applies all manual review decisions,
aligns with Orbis deal data, applies patent count threshold, merges
deal dates, and saves the final clean crosswalk.

Inputs:
  data/interim/crosswalk_verified.csv          — output of post_spot_review.py
  data/manual/review_sub98_auto.csv           — manually filled in (yes/no)
  data/manual/review_duplicate_uspto.csv      — manually filled in (yes/no)
  data/raw/Matching_thesis.xlsx               — original Orbis export

Output:
  data/interim/crosswalk_final_clean.csv

Usage:
  python step2d_clean_crosswalk.py

Notes:
  Set REVIEWS_COMPLETE = False on first run to generate the two review
  files. Fill them in, then set REVIEWS_COMPLETE = True and re-run.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
CROSSWALK_PATH  = "data/interim/crosswalk_verified.csv"
ORBIS_PATH      = "data/raw/Matching_thesis.xlsx"
SUB98_PATH      = "data/manual/review_sub98_auto.csv"
DUPES_PATH      = "data/interim/crosswalk_duplicates.csv"
OUTPUT_PATH     = "data/interim/crosswalk_final_clean.csv"

MIN_PATENT_COUNT = 6     # minimum USPTO patent count to retain a firm

# Set to False on first run to generate the review files.
# Fill them in, then set to True and re-run.
REVIEWS_COMPLETE = True

# =============================================================================
# CHECKS
# =============================================================================
print("=" * 55, flush=True)
print("STEP 2d — FINAL CROSSWALK CLEANING", flush=True)
print("=" * 55, flush=True)

for path in [CROSSWALK_PATH, ORBIS_PATH]:
    if not os.path.exists(path):
        print(f"ERROR: required input not found: {path}", flush=True)
        sys.exit(1)

os.makedirs("data/manual", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)

# =============================================================================
# LOAD INPUTS
# =============================================================================
print(f"\nLoading {CROSSWALK_PATH}...", flush=True)
crosswalk = pd.read_csv(CROSSWALK_PATH, sep=None, engine="python")
print(f"  Starting rows: {len(crosswalk):,}", flush=True)

print(f"Loading {ORBIS_PATH}...", flush=True)
orbis = pd.read_excel(ORBIS_PATH, sheet_name="Results")
print(f"  Orbis rows: {len(orbis):,}", flush=True)

# =============================================================================
# STEP 1 — Basic cleaning
# =============================================================================
print(f"\n--- Step 1: Basic cleaning ---", flush=True)

crosswalk = crosswalk.dropna(subset=["bvd_id"])
print(f"  After dropping missing BvD IDs:  {len(crosswalk):,}", flush=True)

crosswalk = crosswalk[crosswalk["bvd_id"].astype(str).str.startswith("US")]
print(f"  After keeping US only:           {len(crosswalk):,}", flush=True)

crosswalk = crosswalk.drop_duplicates()
print(f"  After removing exact duplicates: {len(crosswalk):,}", flush=True)

# =============================================================================
# STEP 2 — Generate manual review files (only if they do not already exist)
# =============================================================================
print(f"\n--- Step 2: Generate review files ---", flush=True)

if not os.path.exists(SUB98_PATH):
    sub98 = crosswalk[
        (crosswalk["match_score"] < 98) &
        (crosswalk["verified"] == "auto")
    ].copy()
    sub98["correct"] = ""
    sub98.to_csv(SUB98_PATH, index=False)
    print(f"  Generated {SUB98_PATH} ({len(sub98):,} rows)", flush=True)
    print(f"  >>> Fill in the 'correct' column with yes/no", flush=True)
else:
    print(f"  {SUB98_PATH} already exists — skipping", flush=True)

if not os.path.exists(DUPES_PATH):
    dupes = crosswalk[
        crosswalk.duplicated(subset="uspto_assignee_name", keep=False)
    ].copy()
    dupes["correct"] = ""
    dupes.to_csv(DUPES_PATH, index=False)
    print(f"  Generated {DUPES_PATH} ({len(dupes):,} rows)", flush=True)
    print(f"  >>> Fill in the 'correct' column with yes/no", flush=True)
else:
    print(f"  {DUPES_PATH} already exists — skipping", flush=True)

# =============================================================================
# STEP 3 — Apply manual review decisions
# =============================================================================
print(f"\n--- Step 3: Apply manual review decisions ---", flush=True)

if not REVIEWS_COMPLETE:
    print(f"\n  REVIEWS_COMPLETE = False", flush=True)
    print(f"  Fill in both review files then set REVIEWS_COMPLETE = True and re-run:", flush=True)
    print(f"    {SUB98_PATH}", flush=True)
    print(f"    {DUPES_PATH}", flush=True)
    sys.exit(0)

for path in [SUB98_PATH, DUPES_PATH]:
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found — set REVIEWS_COMPLETE = False to regenerate",
              flush=True)
        sys.exit(1)

sub98_review = pd.read_csv(SUB98_PATH)
dupes_review = pd.read_csv(DUPES_PATH)

sub98_remove = sub98_review[
    sub98_review["correct"].fillna("").astype(str).str.strip().str.lower() == "no"
][["bvd_id", "uspto_assignee_name"]]

dupes_remove = dupes_review[
    dupes_review["correct"].fillna("").astype(str).str.strip().str.lower() == "no"
][["bvd_id", "uspto_assignee_name"]]

print(f"  Rows to remove from sub98 review: {len(sub98_remove):,}", flush=True)
print(f"  Rows to remove from dupes review: {len(dupes_remove):,}", flush=True)

to_remove   = pd.concat([sub98_remove, dupes_remove], ignore_index=True)
remove_keys = set(zip(to_remove["bvd_id"], to_remove["uspto_assignee_name"]))

before    = len(crosswalk)
crosswalk = crosswalk[~crosswalk.apply(
    lambda row: (row["bvd_id"], row["uspto_assignee_name"]) in remove_keys,
    axis=1
)]
print(f"  After removing failed reviews:   {len(crosswalk):,} "
      f"(removed {before - len(crosswalk):,})", flush=True)

# =============================================================================
# STEP 4 — Align with Orbis deal data
# Only keep firms that have a completed deal date in Orbis
# =============================================================================
print(f"\n--- Step 4: Align with Orbis ---", flush=True)

orbis = orbis.dropna(subset=["Completed date"])
orbis = orbis.drop_duplicates(
    subset=["Target name", "Completed date", "Acquiror name"], keep="first"
)
valid_bvd_ids = set(orbis["Target BvD ID number"].astype(str))
crosswalk     = crosswalk[crosswalk["bvd_id"].astype(str).isin(valid_bvd_ids)]
print(f"  After aligning with Orbis:       {len(crosswalk):,}", flush=True)

# =============================================================================
# STEP 5 — Apply minimum patent count threshold
# =============================================================================
print(f"\n--- Step 5: Patent count filter (>= {MIN_PATENT_COUNT}) ---", flush=True)

crosswalk = crosswalk[crosswalk["patent_count"] >= MIN_PATENT_COUNT]
print(f"  After patent count filter:       {len(crosswalk):,}", flush=True)

# =============================================================================
# STEP 6 — Merge deal dates and acquiror names from Orbis
# =============================================================================
print(f"\n--- Step 6: Merge deal dates from Orbis ---", flush=True)

orbis["deal_date"] = pd.to_datetime(orbis["Completed date"]).dt.date
orbis_slim = orbis[
    ["Target BvD ID number", "deal_date", "Acquiror name"]
].rename(columns={"Target BvD ID number": "bvd_id"})
orbis_slim["bvd_id"] = orbis_slim["bvd_id"].astype(str)

crosswalk = crosswalk.merge(orbis_slim, on="bvd_id", how="left")

n_missing_deal = crosswalk["deal_date"].isna().sum()
if n_missing_deal > 0:
    print(f"  WARNING: {n_missing_deal:,} rows missing deal_date after merge",
          flush=True)
else:
    print(f"  All rows have deal_date after merge", flush=True)

# =============================================================================
# STEP 7 — Uppercase assignee names, deduplicate, add FARADAY variant
# =============================================================================
print(f"\n--- Step 7: Standardise assignee names ---", flush=True)

crosswalk["uspto_assignee_name"] = crosswalk["uspto_assignee_name"].str.upper()
crosswalk = crosswalk.drop_duplicates()

if "FARADAY & FUTURE INC" not in crosswalk["uspto_assignee_name"].values:
    faraday_row = crosswalk[
        crosswalk["uspto_assignee_name"] == "FARADAY&FUTURE INC"
    ].copy()
    if len(faraday_row) > 0:
        faraday_row["uspto_assignee_name"] = "FARADAY & FUTURE INC"
        crosswalk = pd.concat([crosswalk, faraday_row], ignore_index=True)
        print(f"  Added FARADAY & FUTURE INC variant", flush=True)
    else:
        print(f"  FARADAY&FUTURE INC not found — skipping variant", flush=True)
else:
    print(f"  FARADAY & FUTURE INC already present", flush=True)

# =============================================================================
# SAVE
# =============================================================================
crosswalk.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'=' * 55}", flush=True)
print(f"FINAL CROSSWALK SUMMARY", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Total rows:             {len(crosswalk):,}", flush=True)
print(f"  Unique BvD IDs:         {crosswalk['bvd_id'].nunique():,}", flush=True)
print(f"  Unique USPTO assignees: {crosswalk['uspto_assignee_name'].nunique():,}", flush=True)
print(f"  Deal date range:        {crosswalk['deal_date'].min()} – "
      f"{crosswalk['deal_date'].max()}", flush=True)
print(f"\n  Saved to {OUTPUT_PATH}", flush=True)

print(f"""
Next step: step3a_generate_target_sql.py
  Reads {OUTPUT_PATH}
  Generates queries/bigquery_main_query.sql
  Generates queries/bigquery_validation_query.sql
""", flush=True)