"""
Step 3c — Validate Datasets and Check DiD Eligibility
=======================================================
Runs sanity checks on both the main and validation patent datasets,
then determines which firms have enough patents both before and after
the deal date to be eligible for the DiD analysis.

Eligibility requires >= MIN_PATENTS in both the 3-year pre-merger
window and the 3-year post-merger window.

Inputs:
  data/raw/patents_main.csv                 — target firm patents from BigQuery
  data/interim/patents_validation.csv       — validation dataset (2005-2015)
  data/interim/crosswalk_final_clean.csv    — clean crosswalk with deal dates

Output:
  data/interim/patent_count_eligibility.csv — one row per firm with
                                              pre/post patent counts
                                              and eligible_for_did flag

Usage:
  python step3c_check_eligibility.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PATENTS_MAIN_PATH  = "data/raw/patents_main.csv"
VALIDATION_PATH    = "data/interim/patents_validation.csv"
CROSSWALK_PATH     = "data/interim/crosswalk_final_clean.csv"
OUTPUT_PATH        = "data/interim/patent_count_eligibility.csv"

MIN_PATENTS        = 3    # minimum patents required in both pre and post window
PRE_POST_YEARS     = 3    # symmetric window around deal date in years

# =============================================================================
# CHECKS
# =============================================================================
print("=" * 55, flush=True)
print("STEP 3c — VALIDATE DATASETS AND DiD ELIGIBILITY CHECK", flush=True)
print("=" * 55, flush=True)

for path in [PATENTS_MAIN_PATH, VALIDATION_PATH, CROSSWALK_PATH]:
    if not os.path.exists(path):
        print(f"ERROR: required input not found: {path}", flush=True)
        if path == PATENTS_MAIN_PATH:
            print("  Run bigquery_main_query.sql in BigQuery and export "
                  "as data/raw/patents_main.csv", flush=True)
        elif path == VALIDATION_PATH:
            print("  Run step3b_combine_shards.py first", flush=True)
        elif path == CROSSWALK_PATH:
            print("  Run step2d_clean_crosswalk.py first", flush=True)
        sys.exit(1)

os.makedirs("data/interim", exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print(f"\nLoading {PATENTS_MAIN_PATH}...", flush=True)
patents_main = pd.read_csv(PATENTS_MAIN_PATH, low_memory=False)
patents_main["assignee_name"] = patents_main["assignee_name"].str.upper()
patents_main["filing_date"] = pd.to_datetime(
    patents_main["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
print(f"  Rows: {len(patents_main):,}", flush=True)

print(f"Loading {VALIDATION_PATH}...", flush=True)
patents_validation = pd.read_csv(VALIDATION_PATH, low_memory=False)
print(f"  Rows: {len(patents_validation):,}", flush=True)

print(f"Loading {CROSSWALK_PATH}...", flush=True)
crosswalk = pd.read_csv(CROSSWALK_PATH, low_memory=False)
crosswalk["uspto_assignee_name"] = crosswalk["uspto_assignee_name"].str.upper()
crosswalk["deal_date"]           = pd.to_datetime(
    crosswalk["deal_date"], errors="coerce"
)
print(f"  Rows: {len(crosswalk):,}  |  "
      f"Unique BvD IDs: {crosswalk['bvd_id'].nunique():,}", flush=True)

# =============================================================================
# SECTION 1 — MAIN DATASET CHECKS
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"SECTION 1 — MAIN DATASET (2000-2024)", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Total patent records:   {len(patents_main):,}", flush=True)
print(f"  Unique publication nos: {patents_main['publication_number'].nunique():,}",
      flush=True)
print(f"  Unique assignees:       {patents_main['assignee_name'].nunique():,}",
      flush=True)
print(f"  Filing date range:      {patents_main['filing_date'].min().date()} "
      f"– {patents_main['filing_date'].max().date()}", flush=True)
print(f"  Missing abstract text:  {patents_main['abstract_text'].isna().sum():,}",
      flush=True)
print(f"  Missing claim text:     {patents_main['all_claims_text'].isna().sum():,}",
      flush=True)

# Check which crosswalk firms are missing from BigQuery results
crosswalk_names = set(crosswalk["uspto_assignee_name"].dropna().unique())
patent_names    = set(patents_main["assignee_name"].dropna().unique())
missing_in_bq   = crosswalk_names - patent_names

print(f"\n  Crosswalk assignees not found in BigQuery: {len(missing_in_bq):,}",
      flush=True)
if missing_in_bq:
    print(f"  These firms may have no patents in 2000-2024:", flush=True)
    for name in sorted(missing_in_bq)[:10]:
        print(f"    {name}", flush=True)
    if len(missing_in_bq) > 10:
        print(f"    ... and {len(missing_in_bq) - 10} more", flush=True)

# =============================================================================
# SECTION 2 — VALIDATION DATASET CHECKS
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"SECTION 2 — VALIDATION DATASET (2005-2015)", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Total patent records:   {len(patents_validation):,}", flush=True)
print(f"  Unique publication nos: "
      f"{patents_validation['publication_number'].nunique():,}", flush=True)
print(f"  Filing date range:      {patents_validation['filing_date'].min()} "
      f"– {patents_validation['filing_date'].max()}", flush=True)

if "forward_citations_5yr" in patents_validation.columns:
    n_missing = patents_validation["forward_citations_5yr"].isna().sum()
    print(f"  Missing forward citations: {n_missing:,}", flush=True)
    print(f"\n  Citation count stats:", flush=True)
    print(patents_validation["forward_citations_5yr"].describe().round(2)
          .to_string(), flush=True)
else:
    print(f"  WARNING: forward_citations_5yr column not found — "
          f"check validation query output", flush=True)

# =============================================================================
# SECTION 3 — PRE/POST ACQUISITION PATENT COUNT
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"SECTION 3 — DiD ELIGIBILITY CHECK", flush=True)
print(f"  Window: {PRE_POST_YEARS} years pre and post deal date", flush=True)
print(f"  Threshold: >= {MIN_PATENTS} patents in each window", flush=True)
print(f"{'=' * 55}", flush=True)

crosswalk_for_merge = crosswalk.drop_duplicates(
    subset=["bvd_id", "uspto_assignee_name", "deal_date"]
)

patents_with_deals = patents_main.merge(
    crosswalk_for_merge[["uspto_assignee_name", "bvd_id", "deal_date"]],
    left_on  = "assignee_name",
    right_on = "uspto_assignee_name",
    how      = "inner"
)
print(f"\n  Patents matched to deal records: {len(patents_with_deals):,}",
      flush=True)

# Pre-merger: up to PRE_POST_YEARS before deal date
patents_with_deals["pre_merger"] = (
    (patents_with_deals["filing_date"] < patents_with_deals["deal_date"]) &
    (patents_with_deals["filing_date"] >= patents_with_deals["deal_date"]
     - pd.DateOffset(years=PRE_POST_YEARS))
)

# Post-merger: up to PRE_POST_YEARS after deal date
patents_with_deals["post_merger"] = (
    (patents_with_deals["filing_date"] >= patents_with_deals["deal_date"]) &
    (patents_with_deals["filing_date"] <= patents_with_deals["deal_date"]
     + pd.DateOffset(years=PRE_POST_YEARS))
)

pre_counts  = (
    patents_with_deals[patents_with_deals["pre_merger"]]
    .groupby("bvd_id")["publication_number"]
    .count()
    .rename("pre_patent_count")
)
post_counts = (
    patents_with_deals[patents_with_deals["post_merger"]]
    .groupby("bvd_id")["publication_number"]
    .count()
    .rename("post_patent_count")
)

patent_summary = pd.DataFrame(index=crosswalk["bvd_id"].unique())
patent_summary.index.name = "bvd_id"
patent_summary = patent_summary.join(pre_counts).join(post_counts)
patent_summary["pre_patent_count"]  = (
    patent_summary["pre_patent_count"].fillna(0).astype(int)
)
patent_summary["post_patent_count"] = (
    patent_summary["post_patent_count"].fillna(0).astype(int)
)

patent_summary["meets_pre_threshold"]  = (
    patent_summary["pre_patent_count"] >= MIN_PATENTS
)
patent_summary["meets_post_threshold"] = (
    patent_summary["post_patent_count"] >= MIN_PATENTS
)
patent_summary["eligible_for_did"] = (
    patent_summary["meets_pre_threshold"] &
    patent_summary["meets_post_threshold"]
)

total_firms     = len(patent_summary)
eligible        = int(patent_summary["eligible_for_did"].sum())
fails_pre_only  = int((~patent_summary["meets_pre_threshold"] &
                        patent_summary["meets_post_threshold"]).sum())
fails_post_only = int(( patent_summary["meets_pre_threshold"] &
                        ~patent_summary["meets_post_threshold"]).sum())
fails_both      = int((~patent_summary["meets_pre_threshold"] &
                        ~patent_summary["meets_post_threshold"]).sum())

print(f"\n  Total firms in crosswalk:              {total_firms:,}", flush=True)
print(f"  Firms with >= {MIN_PATENTS} pre-merger patents:   "
      f"{int(patent_summary['meets_pre_threshold'].sum()):,}", flush=True)
print(f"  Firms with >= {MIN_PATENTS} post-merger patents:  "
      f"{int(patent_summary['meets_post_threshold'].sum()):,}", flush=True)
print(f"  Firms eligible for DiD (both):         {eligible:,}", flush=True)
print(f"  Firms dropped (fail either threshold): {total_firms - eligible:,}",
      flush=True)
print(f"    Fail pre only:  {fails_pre_only:,}", flush=True)
print(f"    Fail post only: {fails_post_only:,}", flush=True)
print(f"    Fail both:      {fails_both:,}", flush=True)

# =============================================================================
# SECTION 4 — POST-MERGER FAILURE ANALYSIS
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"SECTION 4 — POST-MERGER FAILURE ANALYSIS", flush=True)
print(f"{'=' * 55}", flush=True)

fails_post = patent_summary[
    (patent_summary["pre_patent_count"]  >= MIN_PATENTS) &
    (patent_summary["post_patent_count"] <  MIN_PATENTS)
].copy()

fails_post = fails_post.reset_index().merge(
    crosswalk[["bvd_id", "deal_date"]].drop_duplicates(subset=["bvd_id"]),
    on  = "bvd_id",
    how = "left"
)
fails_post["deal_year"] = pd.to_datetime(
    fails_post["deal_date"], errors="coerce"
).dt.year

print(f"\n  Firms that pass pre but fail post: {len(fails_post):,}", flush=True)
if len(fails_post) > 0:
    print(f"\n  Post-merger failures by deal year:", flush=True)
    print(fails_post["deal_year"].value_counts().sort_index().to_string(),
          flush=True)
    n_zero = int((fails_post["post_patent_count"] == 0).sum())
    print(f"\n  Firms with exactly 0 post-merger patents: {n_zero:,}", flush=True)

# =============================================================================
# SAVE
# =============================================================================
patent_summary.to_csv(OUTPUT_PATH)
eligible_bvd_ids = patent_summary[
    patent_summary["eligible_for_did"]
].index.tolist()

print(f"\n{'=' * 55}", flush=True)
print(f"SAVED", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Output:       {OUTPUT_PATH}", flush=True)
print(f"  Total firms:  {total_firms:,}", flush=True)
print(f"  Eligible:     {eligible:,}", flush=True)
print(f"  Dropped:      {total_firms - eligible:,}", flush=True)

print(f"""
Next step: step3d_build_eligible_patents.py
  Reads  {OUTPUT_PATH}
  Reads  {CROSSWALK_PATH}
  Reads  {PATENTS_MAIN_PATH}
  Writes data/interim/patents_eligible.csv
""", flush=True)
