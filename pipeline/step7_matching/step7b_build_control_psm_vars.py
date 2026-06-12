"""
Step 7b — Build Control Firm PSM Variables
====================================================
Computes PSM matching variables for control group candidates.

Control firms have no actual deal date so PSM variables are computed
at multiple pseudo-deal-dates — one per calendar year of your deal
sample (2015-2021). Each control firm-year combination becomes one
candidate row. During matching, each treated firm is matched to a
control candidate from the same CPC class and same deal year.

PSM matching variables (same as treated firms):
  patent_age             — years from first filing to pseudo-deal-date
  log_target_pre_patents — log(1 + patent count in 3yr pre-window)
  pre_novelty_mean       — mean novelty_score in 2yr pre-window (Q-8 to Q-1 equiv)

Additional columns for matching:
  dominant_cpc           — most frequent 4-digit CPC class
  pseudo_deal_year       — calendar year used as pseudo deal date

Winsorisation thresholds from step7a_psm_diagnostics.py Section 9f:
  patent_age             clipped at 19.630
  log_target_pre_patents clipped at  6.905
  pre_novelty_mean       clipped at  0.583

Inputs:
  patents_control.csv    — control group patents with novelty_score
                           (must have been processed by step5_compute_novelty.py)

Output:
  control_psm_vars.csv   — one row per control firm per deal year
                           with all PSM matching variables
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
CONTROL_PATH = "data/interim/patents_control_novelty.csv"
OUTPUT_PATH  = "data/interim/control_psm_vars.csv"

# Deal years in your treated sample — pseudo-deal-dates assigned mid-year
DEAL_YEARS    = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

# Pre-merger windows
PRE_VOL_YEARS     = 3    # log_target_pre_patents: 3yr window
PRE_NOVELTY_YEARS = 2    # pre_novelty_mean: 2yr window (Q-8 to Q-1 equivalent)
MIN_PRE_PATENTS   = 3    # minimum patents in pre-window to be a valid candidate

# Winsorisation thresholds from step7a_psm_diagnostics.py Section 9f
WINSOR = {
    "patent_age":             19.630,
    "log_target_pre_patents":  6.905,
    "pre_novelty_mean":        0.583,
}

# =============================================================================
# LOAD CONTROL PATENTS
# =============================================================================
print("="*60, flush=True)
print("STEP 7b — BUILD CONTROL FIRM PSM VARIABLES", flush=True)
print("="*60, flush=True)

if not os.path.exists(CONTROL_PATH):
    print(f"ERROR: {CONTROL_PATH} not found", flush=True)
    print("Run step3b_combine_shards.py and step5_compute_novelty.py first",
          flush=True)
    exit()

print(f"\nLoading {CONTROL_PATH}...", flush=True)
df = pd.read_csv(CONTROL_PATH, low_memory=False)
df["filing_date"] = pd.to_datetime(
    df["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
df["cpc_4digit"] = df["cpc_code"].str[:4].str.upper()

print(f"  Total rows:        {len(df):,}", flush=True)
print(f"  Unique assignees:  {df['assignee_name'].nunique():,}", flush=True)
print(f"  Filing date range: {df['filing_date'].min().date()} "
      f"to {df['filing_date'].max().date()}", flush=True)

# Check novelty scores are present
if "novelty_score" not in df.columns:
    print(f"\nERROR: novelty_score column not found in {CONTROL_PATH}", flush=True)
    print("Run step5_compute_novelty.py on patents_control.csv first", flush=True)
    exit()

n_scored = df["novelty_score"].notna().sum()
print(f"  Novelty scores:    {n_scored:,} / {len(df):,} "
      f"({n_scored/len(df)*100:.1f}%)", flush=True)

if n_scored == 0:
    print(f"\nERROR: No novelty scores found — run step5_compute_novelty.py first",
          flush=True)
    exit()

# =============================================================================
# BUILD PSM VARIABLES PER FIRM PER DEAL YEAR
# =============================================================================
print(f"\nBuilding PSM variables for {len(DEAL_YEARS)} deal years: "
      f"{DEAL_YEARS}", flush=True)
print(f"Pre-volume window:  {PRE_VOL_YEARS} years", flush=True)
print(f"Pre-novelty window: {PRE_NOVELTY_YEARS} years", flush=True)
print(f"Min pre-patents:    {MIN_PRE_PATENTS}", flush=True)

assignees  = df["assignee_name"].unique()
n_total    = len(assignees)
rows       = []
start_time = time.time()

print(f"\nProcessing {n_total:,} control firms across "
      f"{len(DEAL_YEARS)} deal years...", flush=True)
print(f"(candidates with < {MIN_PRE_PATENTS} pre-window patents are excluded)\n",
      flush=True)

for i, assignee in enumerate(assignees):
    firm_df = df[df["assignee_name"] == assignee].copy()

    # Earliest filing date for patent_age
    earliest_filing = firm_df["filing_date"].min()

    # Dominant CPC class across all firm patents
    dominant_cpc = (
        firm_df["cpc_4digit"]
        .value_counts()
        .index[0]
        if len(firm_df) > 0 else None
    )

    for year in DEAL_YEARS:
        # Pseudo-deal-date: June 30 of the deal year
        # Using mid-year avoids edge effects at Jan 1 / Dec 31
        pseudo_deal = pd.Timestamp(f"{year}-06-30")

        # Pre-volume window: 3 years before pseudo-deal-date
        vol_start = pseudo_deal - pd.DateOffset(years=PRE_VOL_YEARS)
        vol_end   = pseudo_deal - pd.Timedelta(days=1)

        # Pre-novelty window: 2 years before pseudo-deal-date
        nov_start = pseudo_deal - pd.DateOffset(years=PRE_NOVELTY_YEARS)
        nov_end   = pseudo_deal - pd.Timedelta(days=1)

        # Count patents in pre-volume window
        pre_vol = firm_df[
            (firm_df["filing_date"] >= vol_start) &
            (firm_df["filing_date"] <= vol_end)
        ]

        # Skip if insufficient patent activity in this window
        if len(pre_vol) < MIN_PRE_PATENTS:
            continue

        # Novelty in pre-novelty window
        pre_nov = firm_df[
            (firm_df["filing_date"] >= nov_start) &
            (firm_df["filing_date"] <= nov_end) &
            (firm_df["novelty_score"].notna())
        ]

        # Patent age at pseudo-deal-date
        patent_age = (
            (pseudo_deal - earliest_filing).days / 365.25
            if pd.notna(earliest_filing) else np.nan
        )

        rows.append({
            "assignee_name":          assignee,
            "pseudo_deal_year":       year,
            "dominant_cpc":           dominant_cpc,
            "patent_age":             patent_age,
            "log_target_pre_patents": np.log1p(len(pre_vol)),
            "pre_novelty_mean":       (pre_nov["novelty_score"].mean()
                                       if len(pre_nov) >= 1 else np.nan),
            "pre_novelty_n":          len(pre_nov),
            "pre_vol_n":              len(pre_vol),
        })

    # Progress every 500 firms
    if (i + 1) % 500 == 0 or (i + 1) == n_total:
        elapsed = time.time() - start_time
        rate    = (i + 1) / elapsed * 60 if elapsed > 0 else 0
        eta     = (n_total - i - 1) / (rate / 60) / 60 if rate > 0 else 0
        print(
            f"  {i+1:>6,}/{n_total:,}  "
            f"({(i+1)/n_total*100:.1f}%)  "
            f"rows so far: {len(rows):,}  "
            f"ETA: {eta:.1f}min",
            flush=True
        )

# =============================================================================
# BUILD DATAFRAME AND APPLY WINSORISATION
# =============================================================================
control_psm = pd.DataFrame(rows)

print(f"\n{'='*55}", flush=True)
print(f"RAW RESULTS", flush=True)
print(f"{'='*55}", flush=True)
print(f"  Total candidate rows:     {len(control_psm):,}", flush=True)
print(f"  Unique control firms:     {control_psm['assignee_name'].nunique():,}",
      flush=True)
print(f"  Rows with novelty score:  "
      f"{control_psm['pre_novelty_mean'].notna().sum():,}", flush=True)

# Candidates per CPC class
print(f"\n  Candidates per CPC class:", flush=True)
for cpc, n in control_psm["dominant_cpc"].value_counts().items():
    print(f"    {cpc}: {n:,}", flush=True)

# Candidates per deal year
print(f"\n  Candidates per deal year:", flush=True)
for yr, n in control_psm["pseudo_deal_year"].value_counts().sort_index().items():
    print(f"    {yr}: {n:,}", flush=True)

# =============================================================================
# WINSORISE AT DIAGNOSTICS THRESHOLDS
# Same thresholds used for treated firms in step7a_psm_diagnostics.py
# =============================================================================
print(f"\n{'='*55}", flush=True)
print(f"APPLYING WINSORISATION", flush=True)
print(f"{'='*55}", flush=True)

for var, cap in WINSOR.items():
    if var not in control_psm.columns:
        continue
    n_clipped = (control_psm[var] > cap).sum()
    control_psm[var] = control_psm[var].clip(upper=cap)
    print(f"  {var:<32} clipped {n_clipped:,} rows at {cap}", flush=True)

# =============================================================================
# CHECK AVAILABILITY PER CPC CLASS AND DEAL YEAR
# Must have enough candidates to match all treated firms
# =============================================================================
print(f"\n{'='*55}", flush=True)
print(f"MATCHING AVAILABILITY CHECK", flush=True)
print(f"{'='*55}", flush=True)
print(f"  Required: enough candidates per CPC class per deal year", flush=True)
print(f"  for 1-to-1 nearest neighbour matching without replacement\n", flush=True)

# Load treated firm counts from psm_diagnostics.csv
if os.path.exists("data/interim/psm_diagnostics.csv"):
    treated = pd.read_csv("data/interim/psm_diagnostics.csv")
    treated["deal_year"] = pd.to_datetime(
        treated["deal_date"], errors="coerce"
    ).dt.year

    print(f"  {'CPC':<6}  {'Year':<6}  {'Treated':>9}  "
          f"{'Controls':>10}  {'Ratio':>7}  Status", flush=True)
    print(f"  {'-'*55}", flush=True)

    all_ok = True
    for cpc in ["G06F", "H04L", "H01L", "G06N"]:
        for year in DEAL_YEARS:
            n_treated = len(treated[
                (treated["dominant_cpc"] == cpc) &
                (treated["deal_year"] == year)
            ])
            n_control = len(control_psm[
                (control_psm["dominant_cpc"]     == cpc) &
                (control_psm["pseudo_deal_year"] == year) &
                (control_psm["pre_novelty_mean"].notna())
            ])

            if n_treated == 0:
                continue

            ratio  = n_control / n_treated if n_treated > 0 else 0
            status = "✅ OK" if n_control >= n_treated else "❌ INSUFFICIENT"
            if n_control < n_treated:
                all_ok = False

            print(
                f"  {cpc:<6}  {year:<6}  {n_treated:>9,}  "
                f"{n_control:>10,}  {ratio:>7.1f}x  {status}",
                flush=True
            )

    if all_ok:
        print(f"\n  All CPC/year combinations have sufficient candidates ✅",
              flush=True)
    else:
        print(f"\n  WARNING: Some combinations are insufficient", flush=True)
        print(f"  Options:", flush=True)
        print(f"    1. Relax MIN_PRE_PATENTS from {MIN_PRE_PATENTS} to 1",
              flush=True)
        print(f"    2. Expand pseudo-deal-year window to ±1 year during matching",
              flush=True)
        print(f"    3. Increase BigQuery control sample from 10% to 20%",
              flush=True)
else:
    print(f"  data/interim/psm_diagnostics.csv not found — skipping availability check",
          flush=True)
    print(f"  Run step7a_psm_diagnostics.py first", flush=True)

# =============================================================================
# DISTRIBUTION CHECK
# =============================================================================
print(f"\n{'='*55}", flush=True)
print(f"CONTROL PSM VARIABLE DISTRIBUTIONS", flush=True)
print(f"{'='*55}", flush=True)

for var in ["patent_age", "log_target_pre_patents", "pre_novelty_mean"]:
    if var in control_psm.columns:
        print(f"\n  {var}:", flush=True)
        print(control_psm[var].describe().round(4).to_string(), flush=True)

# =============================================================================
# SAVE
# =============================================================================
control_psm.to_csv(OUTPUT_PATH, index=False)
elapsed = time.time() - start_time

print(f"\n{'='*55}", flush=True)
print(f"SAVED {OUTPUT_PATH}", flush=True)
print(f"{'='*55}", flush=True)
print(f"  Rows:    {len(control_psm):,}", flush=True)
print(f"  Columns: {list(control_psm.columns)}", flush=True)
print(f"  Time:    {elapsed/60:.1f} minutes", flush=True)

print(f"""
Next step: step7c_psm_matching.py
  Loads psm_diagnostics.csv (treated PSM variables)
  Loads control_psm_vars.csv (control PSM variables)
  Runs 1-to-1 nearest neighbour matching per CPC class per deal year
  Outputs psm_matches.csv — matched pairs ready for panel construction
""", flush=True)
