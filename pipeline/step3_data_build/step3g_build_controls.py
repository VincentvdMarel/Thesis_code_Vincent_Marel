"""
Step 3g — Build Control Variables
====================================
Computes all firm-level control variables and merges them into the
patent-level dataset. Integrates three separate sources:
  1. patents_eligible.csv  — target pre-merger patent volume
  2. patents_main.csv      — patent age (first filing to deal date)
  3. patents_acquiror.csv  — acquiror pre-merger patent count
  4. Matching_thesis.xlsx  — deal value, acquiror total assets

Control variables produced:
  log_target_pre_patents   — log(1 + patent count, 3yr pre-merger window)
  patent_age               — years from first patent filing to deal date
  log_deal_value           — log(1 + deal value USD thousands)
  deal_value_missing       — indicator: 1 if deal value was missing
  log_acquiror_assets      — log(1 + acquiror total assets)
  acquiror_assets_missing  — indicator: 1 if acquiror assets was missing
  log_acquiror_pre_patents — log(1 + acquiror patent count, 5yr pre-merger)

Inputs:
  data/interim/patents_eligible.csv        — output of step3d
  data/raw/patents_main.csv                — BigQuery main export (2000-2024)
  data/raw/patents_acquiror.csv            — BigQuery acquiror export
  data/interim/acquiror_name_map.csv       — output of step3e
  data/raw/Matching_thesis.xlsx            — original Orbis export

Output:
  data/interim/patents_eligible_controls.csv

Usage:
  python step3g_build_controls.py

Notes:
  Update ORBIS_COLUMN_MAP if any column names differ in your Orbis export.
  Run once — the script prints all available Orbis column names if any
  expected ones are missing.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PATENTS_ELIGIBLE_PATH  = "data/interim/patents_eligible.csv"
PATENTS_MAIN_PATH      = "data/raw/patents_main.csv"
ACQUIROR_PATENTS_PATH  = "data/raw/patents_acquiror.csv"
NAME_MAP_PATH          = "data/interim/acquiror_name_map.csv"
ORBIS_PATH             = "data/raw/Matching_thesis.xlsx"
OUTPUT_PATH            = "data/interim/patents_eligible_controls.csv"

TARGET_PRE_YEARS       = 3    # target pre-merger patent volume window
ACQUIROR_PRE_YEARS     = 5    # acquiror pre-merger patent volume window

# Update these to match your exact Orbis column names.
# Run once — missing columns will be printed so you can correct the map.
ORBIS_COLUMN_MAP = {
    "Target BvD ID number":          "bvd_id",
    "Target name":                   "target_name",
    "Acquiror name":                 "acquiror_name",
    "Target date of incorporation":  "incorporation_date",
    "Completed date":                "completed_date",
    "Deal value":                    "deal_value_usd",
    "acquiror total assets":         "acquiror_total_assets",
}

CONTROL_COLS = [
    "log_target_pre_patents",
    "patent_age",
    "log_deal_value",
    "deal_value_missing",
    "log_acquiror_assets",
    "acquiror_assets_missing",
    "log_acquiror_pre_patents",
]

# =============================================================================
# CHECKS
# =============================================================================
print("=" * 60, flush=True)
print("STEP 3g -- BUILD CONTROL VARIABLES", flush=True)
print("=" * 60, flush=True)

for path, predecessor in [
    (PATENTS_ELIGIBLE_PATH, "step3d_build_eligible_patents.py"),
    (PATENTS_MAIN_PATH,     "bigquery_main_query.sql (BigQuery export)"),
    (ACQUIROR_PATENTS_PATH, "step3f_generate_acquiror_sql.py + BigQuery export"),
    (NAME_MAP_PATH,         "step3e_map_acquirors.py"),  
    (ORBIS_PATH,            "Orbis manual export"),
]:
    if not os.path.exists(path):
        print(f"ERROR: required input not found: {path}", flush=True)
        print(f"  Source: {predecessor}", flush=True)
        sys.exit(1)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  Found: {os.path.basename(path):<50} {size_mb:>7.1f} MB",
          flush=True)

os.makedirs("data/interim", exist_ok=True)

# =============================================================================
# LOAD CORE INPUTS
# =============================================================================
print(f"\nLoading core inputs...", flush=True)

patents_eligible = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)
patents_eligible["filing_date"] = pd.to_datetime(
    patents_eligible["filing_date"], errors="coerce"
)
patents_eligible["deal_date"] = pd.to_datetime(
    patents_eligible["deal_date"], errors="coerce"
)
print(f"  patents_eligible:  {len(patents_eligible):,} rows  |  "
      f"{patents_eligible['bvd_id'].nunique():,} firms", flush=True)

# One acquiror name and deal date per target firm — built from patents_eligible
acquiror_lookup = (
    patents_eligible[["bvd_id", "Acquiror name", "deal_date"]]
    .drop_duplicates(subset="bvd_id")
    .copy()
)
acquiror_lookup["orbis_upper"] = (
    acquiror_lookup["Acquiror name"].str.upper().str.strip()
)

eligible_bvd_ids = patents_eligible["bvd_id"].dropna().unique().tolist()
n_firms          = len(eligible_bvd_ids)

# =============================================================================
# LOAD ORBIS
# =============================================================================
print(f"\nLoading {ORBIS_PATH}...", flush=True)
orbis_raw = pd.read_excel(ORBIS_PATH, sheet_name="Results")

found   = [k for k in ORBIS_COLUMN_MAP if k     in orbis_raw.columns]
missing = [k for k in ORBIS_COLUMN_MAP if k not in orbis_raw.columns]
print(f"  Orbis columns found:   {len(found)}", flush=True)
print(f"  Orbis columns missing: {len(missing)}", flush=True)

if missing:
    print(f"\n  Update ORBIS_COLUMN_MAP for these columns:", flush=True)
    for c in missing:
        print(f"    '{c}'", flush=True)
    print(f"\n  Available Orbis columns:", flush=True)
    for c in orbis_raw.columns:
        print(f"    '{c}'", flush=True)
    sys.exit(1)

orbis = orbis_raw.rename(
    columns={k: v for k, v in ORBIS_COLUMN_MAP.items()
             if k in orbis_raw.columns}
)
orbis["bvd_id"] = orbis["bvd_id"].astype(str)

# =============================================================================
# DATE PARSER -- handles MM/DD/YYYY, year-only, and Excel serial numbers
# =============================================================================
def parse_orbis_date(col):
    result = pd.Series([pd.NaT] * len(col), index=col.index)

    # Mask 1 -- year-only
    year_only = col.astype(str).str.strip().str.match(r"^\d{4}$")
    if year_only.any():
        result[year_only] = pd.to_datetime(
            col[year_only].astype(str).str.strip() + "-01-01",
            format="%Y-%m-%d", errors="coerce"
        )

    # Mask 2 -- MM/DD/YYYY strings
    not_yet = result.isna() & col.notna() & ~year_only
    if not_yet.any():
        result[not_yet] = pd.to_datetime(
            col[not_yet], format="%m/%d/%Y", errors="coerce"
        )

    # Mask 3 -- other date string variants
    still = result.isna() & col.notna()
    if still.any():
        result[still] = pd.to_datetime(
            col[still], dayfirst=False, errors="coerce"
        )

    # Mask 4 -- Excel serial numbers (last resort)
    still = result.isna() & col.notna()
    if still.any():
        numeric = pd.to_numeric(col[still], errors="coerce")
        valid   = numeric.notna() & (numeric > 1000) & (numeric < 100000)
        if valid.any():
            result.loc[numeric[valid].index] = pd.to_datetime(
                numeric[valid], unit="D", origin="1899-12-30", errors="coerce"
            )

    return result

orbis["deal_date_orbis"] = parse_orbis_date(orbis["completed_date"])

# =============================================================================
# CONTROL 1 -- Target pre-merger patent volume (3yr window, log)
# Source:    patents_eligible.csv
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"CONTROL 1 -- target pre-merger patent volume "
      f"({TARGET_PRE_YEARS}yr)", flush=True)
print(f"{'=' * 60}", flush=True)

pre_patents = patents_eligible[patents_eligible["pre_merger"]].copy()
pre_patents["years_before_deal"] = (
    pre_patents["deal_date"] - pre_patents["filing_date"]
).dt.days / 365.25

target_vol = (
    pre_patents[pre_patents["years_before_deal"] <= TARGET_PRE_YEARS]
    .groupby("bvd_id")["publication_number"]
    .count()
    .rename("target_pre_patent_volume")
    .reset_index()
)
target_vol["log_target_pre_patents"] = np.log1p(
    target_vol["target_pre_patent_volume"]
)

print(f"  Firms with data:  {len(target_vol):,}", flush=True)
print(f"  Mean volume:      "
      f"{target_vol['target_pre_patent_volume'].mean():.1f}", flush=True)
print(f"  Median volume:    "
      f"{target_vol['target_pre_patent_volume'].median():.1f}", flush=True)

# =============================================================================
# CONTROL 2 -- Patent age (years from first patent filing to deal date)
# Source:    patents_main.csv (2000-2024) -- wider window than patents_eligible
#            to capture true first filing date, not just post-2010 window
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"CONTROL 2 -- patent age (first filing to deal date)", flush=True)
print(f"{'=' * 60}", flush=True)

print(f"  Loading {PATENTS_MAIN_PATH}...", flush=True)
patents_main = pd.read_csv(PATENTS_MAIN_PATH, low_memory=False)
patents_main["filing_date"] = pd.to_datetime(
    patents_main["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
print(f"  Rows: {len(patents_main):,}  |  "
      f"Range: {patents_main['filing_date'].min().date()} -- "
      f"{patents_main['filing_date'].max().date()}", flush=True)

# Map assignee name to bvd_id using patents_eligible
# (avoids loading crosswalk_final_clean.csv separately)
name_to_bvd = (
    patents_eligible[["bvd_id", "assignee_name"]]
    .drop_duplicates()
    .assign(assignee_upper=lambda df: df["assignee_name"].str.upper())
    .set_index("assignee_upper")["bvd_id"]
    .to_dict()
)
patents_main["bvd_id"] = patents_main["assignee_name"].str.upper().map(name_to_bvd)

# Restrict to eligible firms only
patents_main_eligible = patents_main[
    patents_main["bvd_id"].isin(eligible_bvd_ids)
].copy()
print(f"  Patents for eligible firms: {len(patents_main_eligible):,}  |  "
      f"{patents_main_eligible['bvd_id'].nunique():,} firms", flush=True)

# First patent filing date per firm across full 2000-2024 window
first_patent = (
    patents_main_eligible
    .groupby("bvd_id")["filing_date"]
    .min()
    .reset_index()
    .rename(columns={"filing_date": "first_patent_date"})
)

# Flag possibly truncated firms
truncated = first_patent[first_patent["first_patent_date"].dt.year <= 2001]
if len(truncated) > 0:
    print(f"  WARNING: {len(truncated):,} firms have first patent in 2000-2001 "
          f"-- may be truncated", flush=True)

# Get one deal date per firm
deal_dates = (
    patents_eligible.groupby("bvd_id")["deal_date"]
    .first()
    .reset_index()
)

# Compute patent age
patent_age_df = first_patent.merge(deal_dates, on="bvd_id", how="left")
patent_age_df["patent_age"] = (
    patent_age_df["deal_date"] - patent_age_df["first_patent_date"]
).dt.days / 365.25

# Remove impossible values (first patent after deal date)
negative = patent_age_df[patent_age_df["patent_age"] < 0]
if len(negative) > 0:
    print(f"  WARNING: {len(negative):,} firms have negative patent_age "
          f"-- set to NaN", flush=True)
    patent_age_df.loc[patent_age_df["patent_age"] < 0, "patent_age"] = np.nan

print(f"  Firms with patent_age: "
      f"{patent_age_df['patent_age'].notna().sum():,}", flush=True)
print(f"  Mean:     {patent_age_df['patent_age'].mean():.2f} years", flush=True)
print(f"  Median:   {patent_age_df['patent_age'].median():.2f} years", flush=True)
print(f"  Range:    {patent_age_df['patent_age'].min():.2f} -- "
      f"{patent_age_df['patent_age'].max():.2f} years", flush=True)
print(f"  Skewness: {patent_age_df['patent_age'].skew():.2f}", flush=True)

patent_age_out = patent_age_df[["bvd_id", "patent_age"]]

# =============================================================================
# CONTROL 3 -- Log deal value + missingness indicator
# Source:    Orbis deal value
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"CONTROL 3 -- log deal value", flush=True)
print(f"{'=' * 60}", flush=True)

deal_val = orbis[["bvd_id", "deal_value_usd"]].drop_duplicates(
    subset="bvd_id"
).copy()
deal_val["deal_value_num"]     = pd.to_numeric(
    deal_val["deal_value_usd"], errors="coerce"
)
deal_val["deal_value_missing"] = deal_val["deal_value_num"].isna().astype(int)
deal_val["log_deal_value"]     = np.log1p(
    deal_val["deal_value_num"].fillna(0)
)
deal_val = deal_val[["bvd_id", "log_deal_value", "deal_value_missing"]]

n_present = int((deal_val["deal_value_missing"] == 0).sum())
n_missing = int((deal_val["deal_value_missing"] == 1).sum())
print(f"  Firms with deal value:  {n_present:,}", flush=True)
print(f"  Missing (set 0):        {n_missing:,}  (deal_value_missing = 1)",
      flush=True)

# =============================================================================
# CONTROL 4 -- Log acquiror total assets + missingness indicator
# Source:    Orbis acquiror total assets
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"CONTROL 4 -- log acquiror total assets", flush=True)
print(f"{'=' * 60}", flush=True)

acq_assets = orbis[["bvd_id", "acquiror_total_assets"]].drop_duplicates(
    subset="bvd_id"
).copy()
acq_assets["acquiror_assets_num"]     = pd.to_numeric(
    acq_assets["acquiror_total_assets"], errors="coerce"
)
acq_assets["acquiror_assets_missing"] = (
    acq_assets["acquiror_assets_num"].isna().astype(int)
)
acq_assets["log_acquiror_assets"]     = np.log1p(
    acq_assets["acquiror_assets_num"].fillna(0)
)
acq_assets = acq_assets[[
    "bvd_id", "log_acquiror_assets", "acquiror_assets_missing"
]]

n_present = int((acq_assets["acquiror_assets_missing"] == 0).sum())
n_missing = int((acq_assets["acquiror_assets_missing"] == 1).sum())
print(f"  Firms with assets data: {n_present:,}", flush=True)
print(f"  Missing (set 0):        {n_missing:,}  "
      f"(acquiror_assets_missing = 1)", flush=True)

# =============================================================================
# CONTROL 5 -- Acquiror pre-merger patent count (5yr window, log)
# Source:    patents_acquiror.csv + acquiror_name_map.csv
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"CONTROL 5 -- acquiror pre-merger patent count "
      f"({ACQUIROR_PRE_YEARS}yr)", flush=True)
print(f"{'=' * 60}", flush=True)

# Load acquiror name mapping
print(f"  Loading {NAME_MAP_PATH}...", flush=True)
name_mapping = pd.read_csv(NAME_MAP_PATH)
mapped       = name_mapping[name_mapping["status"] == "MAPPED"].copy()
mapped["bq_upper"]    = mapped["bq_acquiror_name"].str.upper().str.strip()
mapped["orbis_upper"] = mapped["orbis_acquiror_name"].str.upper().str.strip()
bq_to_orbis           = dict(zip(mapped["bq_upper"], mapped["orbis_upper"]))

n_needs_review = int((name_mapping["status"] == "NEEDS REVIEW").sum())
print(f"  MAPPED rows:             {len(mapped):,}", flush=True)
print(f"  NEEDS REVIEW (skipped):  {n_needs_review:,}", flush=True)
if n_needs_review > 0:
    print(f"  WARNING: {n_needs_review:,} acquirors still need review -- "
          f"their patents are excluded", flush=True)

# Load acquiror patents
print(f"\n  Loading {ACQUIROR_PATENTS_PATH}...", flush=True)
patents_acq = pd.read_csv(ACQUIROR_PATENTS_PATH, low_memory=False)
patents_acq["filing_date"] = pd.to_datetime(
    patents_acq["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
patents_acq["bq_upper"] = patents_acq["assignee_name"].str.upper().str.strip()
print(f"  Rows:  {len(patents_acq):,}  |  "
      f"Unique USPTO names: {patents_acq['bq_upper'].nunique():,}", flush=True)

# Map BigQuery names to Orbis names
patents_acq["orbis_upper"] = patents_acq["bq_upper"].map(bq_to_orbis)

unmapped_rows  = patents_acq["orbis_upper"].isna().sum()
unmapped_names = patents_acq[
    patents_acq["orbis_upper"].isna()
]["bq_upper"].unique()

if unmapped_rows > 0:
    print(f"\n  WARNING: {unmapped_rows:,} rows not mapped to an Orbis name",
          flush=True)
    print(f"  {len(unmapped_names):,} BigQuery names not in mapping file:",
          flush=True)
    for n in sorted(unmapped_names)[:10]:
        print(f"    {n}", flush=True)
    if len(unmapped_names) > 10:
        print(f"    ... and {len(unmapped_names) - 10} more", flush=True)
    print(f"  Add to {NAME_MAP_PATH} to include them", flush=True)
else:
    print(f"  All patent rows successfully mapped", flush=True)

patents_acq = patents_acq.dropna(subset=["orbis_upper"])
print(f"  Patents after mapping: {len(patents_acq):,}", flush=True)

# Match to deal records by Orbis acquiror name
matched = patents_acq.merge(
    acquiror_lookup[["bvd_id", "orbis_upper", "deal_date"]],
    on  = "orbis_upper",
    how = "inner"
)
print(f"  Acquiror patents matched to deals: {len(matched):,}  |  "
      f"{matched['bvd_id'].nunique():,} firms", flush=True)

# Count patents in 5yr pre-merger window
matched["years_before_deal"] = (
    matched["deal_date"] - matched["filing_date"]
).dt.days / 365.25

pre_window = matched[
    (matched["years_before_deal"] >= 0) &
    (matched["years_before_deal"] <= ACQUIROR_PRE_YEARS)
]
print(f"  Patents in {ACQUIROR_PRE_YEARS}yr pre-merger window: "
      f"{len(pre_window):,}", flush=True)

acquiror_counts = (
    pre_window
    .groupby("bvd_id")["publication_number"]
    .count()
    .rename("acquiror_pre_patent_count")
    .reset_index()
)
acquiror_counts["log_acquiror_pre_patents"] = np.log1p(
    acquiror_counts["acquiror_pre_patent_count"]
)

n_matched   = len(acquiror_counts)
n_unmatched = n_firms - n_matched
print(f"\n  Firms with acquiror patents: {n_matched:,}  "
      f"({n_matched / n_firms * 100:.1f}%)", flush=True)
print(f"  No acquiror patents (-> 0):  {n_unmatched:,}  "
      f"({n_unmatched / n_firms * 100:.1f}%)", flush=True)

# =============================================================================
# MERGE ALL CONTROLS INTO ONE FIRM-LEVEL TABLE
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"MERGING ALL CONTROLS", flush=True)
print(f"{'=' * 60}", flush=True)

controls = pd.DataFrame({"bvd_id": eligible_bvd_ids})
controls = (
    controls
    .merge(target_vol[["bvd_id", "log_target_pre_patents"]],
           on="bvd_id", how="left")
    .merge(patent_age_out,
           on="bvd_id", how="left")
    .merge(deal_val,
           on="bvd_id", how="left")
    .merge(acq_assets,
           on="bvd_id", how="left")
    .merge(acquiror_counts[["bvd_id", "log_acquiror_pre_patents"]],
           on="bvd_id", how="left")
)

# Patent count controls: fill NaN with 0 (no patents in window)
controls["log_target_pre_patents"]   = (
    controls["log_target_pre_patents"].fillna(0)
)
controls["log_acquiror_pre_patents"] = (
    controls["log_acquiror_pre_patents"].fillna(0)
)

print(f"  Control table: {len(controls):,} firms x "
      f"{len(controls.columns) - 1} variables", flush=True)

# =============================================================================
# MERGE CONTROLS INTO PATENT-LEVEL DATASET
# Drop any existing control columns first to avoid duplicates on re-run
# Also drops firm_age if present from earlier script versions
# =============================================================================
drop_cols = CONTROL_COLS + ["firm_age", "acquiror_pre_patent_count"]
patents_eligible = patents_eligible.drop(
    columns=[c for c in drop_cols if c in patents_eligible.columns],
    errors="ignore"
)

patents_eligible = patents_eligible.merge(controls, on="bvd_id", how="left")

# =============================================================================
# COVERAGE REPORT
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"CONTROL VARIABLE COVERAGE REPORT", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  {'Variable':<32} {'Firms':>8}  {'Coverage':>10}", flush=True)
print(f"  {'-' * 55}", flush=True)

total_firms = patents_eligible["bvd_id"].nunique()
all_ok      = True

for col in CONTROL_COLS:
    if col not in patents_eligible.columns:
        print(f"  {col:<32} {'NOT IN FILE':>20}", flush=True)
        all_ok = False
        continue
    firm_vals     = patents_eligible.groupby("bvd_id")[col].first()
    n_non_missing = int(firm_vals.notna().sum())
    pct           = n_non_missing / total_firms * 100
    flag          = "  WARNING low" if pct < 70 else ""
    if pct < 70:
        all_ok = False
    print(f"  {col:<32} {n_non_missing:>8,}  {pct:>9.1f}%{flag}", flush=True)

available = [c for c in CONTROL_COLS if c in patents_eligible.columns]
print(f"\n  Descriptive statistics (firm level, N={len(controls):,}):", flush=True)
print(controls[[c for c in CONTROL_COLS if c in controls.columns]].describe().round(3).to_string(), flush=True)

if not all_ok:
    print(f"\n  WARNING: one or more controls have low coverage or are missing.",
          flush=True)
    print(f"  Check ORBIS_COLUMN_MAP and data/interim/acquiror_name_map.csv.", flush=True)

# =============================================================================
# SAVE
# =============================================================================
patents_eligible.to_csv(OUTPUT_PATH, index=False)
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)

print(f"\n{'=' * 60}", flush=True)
print(f"SAVED", flush=True)
print(f"{'=' * 60}", flush=True)
print(f"  Output:   {OUTPUT_PATH}", flush=True)
print(f"  Rows:     {len(patents_eligible):,}", flush=True)
print(f"  Size:     {size_mb:.1f} MB", flush=True)
print(f"  Columns:  {list(patents_eligible.columns)}", flush=True)

print(f"""
Next step: step4_compute_embeddings.py
  Update DATASETS in step4_compute_embeddings.py to read:
    data/interim/patents_eligible_controls.csv
  instead of:
    data/interim/patents_eligible.csv
""", flush=True)