"""
Step 7a Pre-check — PSM Diagnostics
=====================================
Examines patents_eligible.csv to assess what is available for PSM
matching in Step 7.

Per thesis proposal:
  1-to-1 nearest neighbour matching without replacement
  Same 4-digit CPC subclass
  Match on: firm age, pre-merger patent volume, pre-merger patent novelty
  Variables measured exactly 1 year before deal closing (event quarter -4)

This script reports:
  1. Variable coverage at firm level (188 firms total)
  2. Distribution of each PSM matching variable
  3. Novelty score coverage in PSM measurement window
  4. CPC subclass distribution (matching strata)
  5. Pre-merger / post-merger patent distribution
  6. Recommended PSM measurement window based on data availability

Output:
  Console report
  psm_diagnostics.csv — per-firm summary of PSM variables
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PATENTS_ELIGIBLE_PATH = "data/interim/patents_eligible_overlap.csv"
OUTPUT_PATH           = "data/interim/psm_diagnostics.csv"

# PSM measurement windows 
PSM_WINDOWS = {
    "Q-4 only":        (-4, -4),    # exactly one year before
    "Q-8 to Q-5":      (-8, -5),    # year prior to one-year-before
    "Q-8 to Q-1":      (-8, -1),    # 2 years pre-merger (broader)
    "Q-12 to Q-1":     (-12, -1),   # full 3yr pre-merger
}

# =============================================================================
# LOAD DATA
# =============================================================================
print("="*65, flush=True)
print("STEP 7a PRE-CHECK — PSM MATCHING DIAGNOSTICS", flush=True)
print("="*65, flush=True)

if not os.path.exists(PATENTS_ELIGIBLE_PATH):
    print(f"ERROR: {PATENTS_ELIGIBLE_PATH} not found", flush=True)
    exit()

print(f"\nLoading {PATENTS_ELIGIBLE_PATH}...", flush=True)
df = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)
df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
df["deal_date"]   = pd.to_datetime(df["deal_date"], errors="coerce")
df["cpc_4digit"]  = df["cpc_code"].str[:4].str.upper()

print(f"  Rows:           {len(df):,}", flush=True)
print(f"  Unique BvD IDs: {df['bvd_id'].nunique():,}", flush=True)
print(f"  Columns:        {len(df.columns)}", flush=True)

# Show available columns
print(f"\nColumns in patents_eligible.csv:", flush=True)
for col in df.columns:
    print(f"  {col}", flush=True)

# =============================================================================
# SECTION 1 — FIRM-LEVEL VARIABLE COVERAGE
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 1 — FIRM-LEVEL VARIABLE COVERAGE", flush=True)
print(f"{'='*65}", flush=True)

total_firms = df["bvd_id"].nunique()

# Get one row per firm for firm-level vars
firm_vars = (
    df.groupby("bvd_id")
    .agg({
        "log_target_pre_patents":   "first",
        "log_acquiror_pre_patents": "first",
        "patent_age":               "first",
        "log_deal_value":           "first",
        "log_acquiror_assets":      "first",
        "knowledge_overlap":        "first",
        "novelty_score":            "first",
    })
)

print(f"\n  {'Variable':<32} {'Firms with data':>16}  Coverage", flush=True)
print(f"  {'-'*60}", flush=True)
for col in firm_vars.columns:
    n_valid = firm_vars[col].notna().sum()
    pct     = n_valid / total_firms * 100
    flag    = " <-- LOW" if pct < 70 else ""
    print(f"  {col:<32} {n_valid:>10,}/{total_firms}  {pct:>6.1f}%{flag}",
          flush=True)

# =============================================================================
# SECTION 2 — DISTRIBUTION OF PSM MATCHING VARIABLES
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 2 — PSM MATCHING VARIABLE DISTRIBUTIONS", flush=True)
print(f"{'='*65}", flush=True)

print(f"\nlog_target_pre_patents (3yr pre-merger patent volume):", flush=True)
print(firm_vars["log_target_pre_patents"].describe().round(3).to_string(), flush=True)

print(f"\npatent_age (years since first patent at deal date):", flush=True)
print(firm_vars["patent_age"].describe().round(2).to_string(), flush=True)

# Check skewness
for col in ["log_target_pre_patents", "patent_age"]:
    v = firm_vars[col].dropna()
    if len(v) > 0:
        print(f"\n  {col} skewness: {v.skew():.3f}", flush=True)

# =============================================================================
# SECTION 3 — NOVELTY SCORE COVERAGE IN PSM WINDOWS
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 3 — PRE-MERGER NOVELTY COVERAGE BY WINDOW", flush=True)
print(f"{'='*65}", flush=True)
print(f"  Goal: find a measurement window where most firms have", flush=True)
print(f"  enough novelty_score values to compute a stable mean", flush=True)

if "novelty_score" not in df.columns:
    print(f"\n  ERROR: novelty_score column not in patents_eligible.csv", flush=True)
    print(f"  Run step5_compute_novelty.py first", flush=True)
else:
    print(f"\n  {'Window':<15} {'Mean N/firm':>13} "
          f"{'Firms >=1':>12} {'Firms >=3':>12}", flush=True)
    print(f"  {'-'*55}", flush=True)
    for label, (qmin, qmax) in PSM_WINDOWS.items():
        win = df[
            (df["event_quarter"] >= qmin) &
            (df["event_quarter"] <= qmax) &
            (df["novelty_score"].notna())
        ]
        per_firm = win.groupby("bvd_id")["novelty_score"].count()
        n1 = (per_firm >= 1).sum()
        n3 = (per_firm >= 3).sum()
        mn = per_firm.mean() if len(per_firm) > 0 else 0
        print(f"  {label:<15} {mn:>13.1f} "
              f"{n1:>9,}/{total_firms} {n3:>9,}/{total_firms}",
              flush=True)

    # Pick the best window — recommend tightest viable
    print(f"\n  Recommendation: pick the tightest window where", flush=True)
    print(f"  >= 80% of firms have at least 3 novelty observations", flush=True)
    print(f"  for stable mean estimation", flush=True)

# =============================================================================
# SECTION 4 — CPC SUBCLASS DISTRIBUTION (PSM matching strata)
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 4 — CPC SUBCLASS DISTRIBUTION", flush=True)
print(f"{'='*65}", flush=True)
print(f"  PSM requires matching within same 4-digit CPC subclass", flush=True)
print(f"  Each treated firm needs at least 1 control candidate in same CPC", flush=True)

# Dominant CPC per firm = most frequent 4-digit CPC across all their patents
dominant_cpc = (
    df.groupby("bvd_id")["cpc_4digit"]
    .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else None)
    .rename("dominant_cpc")
)

print(f"\n  Treated firms by dominant CPC subclass:", flush=True)
cpc_counts = dominant_cpc.value_counts()
for cpc, n in cpc_counts.items():
    print(f"    {cpc}: {n:,} firms", flush=True)

print(f"\n  Total CPC subclasses with treated firms: {len(cpc_counts):,}", flush=True)

# =============================================================================
# SECTION 5 — PRE/POST PATENT DISTRIBUTION
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 5 — PRE/POST PATENT DISTRIBUTION", flush=True)
print(f"{'='*65}", flush=True)

pre_post = (
    df.groupby("bvd_id")
    .agg(
        n_pre  = ("event_quarter", lambda x: (x < 0).sum()),
        n_post = ("event_quarter", lambda x: (x >= 0).sum()),
    )
)

print(f"\n  Pre-merger patents per firm (quarters -12 to -1):", flush=True)
print(pre_post["n_pre"].describe().round(1).to_string(), flush=True)
print(f"\n  Post-merger patents per firm (quarters 0 to 12):", flush=True)
print(pre_post["n_post"].describe().round(1).to_string(), flush=True)

# Coverage in panel
firms_with_pre  = (pre_post["n_pre"]  > 0).sum()
firms_with_post = (pre_post["n_post"] > 0).sum()
firms_with_both = ((pre_post["n_pre"] > 0) & (pre_post["n_post"] > 0)).sum()

print(f"\n  Firms with pre-merger patents:    {firms_with_pre:,}/{total_firms} "
      f"({firms_with_pre/total_firms*100:.1f}%)", flush=True)
print(f"  Firms with post-merger patents:   {firms_with_post:,}/{total_firms} "
      f"({firms_with_post/total_firms*100:.1f}%)", flush=True)
print(f"  Firms with BOTH (DiD eligible):   {firms_with_both:,}/{total_firms} "
      f"({firms_with_both/total_firms*100:.1f}%)", flush=True)

# =============================================================================
# SECTION 6 — EVENT QUARTER COVERAGE
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 6 — EVENT QUARTER COVERAGE", flush=True)
print(f"{'='*65}", flush=True)
print(f"  How many firms have at least 1 patent in each event quarter:", flush=True)

quarter_coverage = (
    df.groupby("event_quarter")["bvd_id"]
    .nunique()
    .rename("n_firms")
)

print(f"\n  {'Quarter':>8} {'Firms':>8} {'Pct':>6}", flush=True)
print(f"  {'-'*30}", flush=True)
for q in range(-12, 13):
    n = quarter_coverage.get(q, 0)
    pct = n / total_firms * 100
    bar = "#" * int(pct / 2)
    print(f"  {q:>8} {n:>8,} {pct:>5.1f}% {bar}", flush=True)

# =============================================================================
# SECTION 7 — DEAL DATE TIME RANGE
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 7 — DEAL DATE TIME RANGE", flush=True)
print(f"{'='*65}", flush=True)

deal_dates = (
    df.groupby("bvd_id")["deal_date"].first().dropna()
)

print(f"\n  Earliest deal:  {deal_dates.min().date()}", flush=True)
print(f"  Latest deal:    {deal_dates.max().date()}", flush=True)
print(f"\n  Deals per year:", flush=True)
print(deal_dates.dt.year.value_counts().sort_index().to_string(), flush=True)

# =============================================================================
# SECTION 8 — PER-FIRM PSM SUMMARY EXPORT
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 8 — EXPORT PER-FIRM PSM SUMMARY", flush=True)
print(f"{'='*65}", flush=True)

PSM_BEST_WINDOW = (-12, -1)

novelty_pre = (
    df[
        (df["event_quarter"] >= PSM_BEST_WINDOW[0]) &
        (df["event_quarter"] <= PSM_BEST_WINDOW[1]) &
        (df["novelty_score"].notna())
    ]
    .groupby("bvd_id")["novelty_score"]
    .agg(
        pre_novelty_mean = "mean",
        pre_novelty_n    = "count",
    )
)

# Build per-firm summary
psm_summary = (
    df.groupby("bvd_id")
    .agg(
        target_name        = ("orbis_target_name", "first"),
        deal_date          = ("deal_date", "first"),
        dominant_cpc       = ("cpc_4digit",
                              lambda x: x.value_counts().index[0] if len(x) > 0 else None),
        n_patents_total    = ("publication_number", "count"),
        n_patents_pre      = ("event_quarter", lambda x: (x < 0).sum()),
        n_patents_post     = ("event_quarter", lambda x: (x >= 0).sum()),
        log_target_pre_patents   = ("log_target_pre_patents", "first"),
        log_acquiror_pre_patents = ("log_acquiror_pre_patents", "first"),
        patent_age         = ("patent_age", "first"),
        knowledge_overlap  = ("knowledge_overlap", "first"),
    )
    .merge(novelty_pre, left_index=True, right_index=True, how="left")
    .reset_index()
)

# Coverage of PSM-required variables
psm_required = ["patent_age", "log_target_pre_patents", "pre_novelty_mean"]
psm_complete = psm_summary[psm_required].notna().all(axis=1).sum()

print(f"\n  PSM matching variables required:", flush=True)
for v in psm_required:
    n = psm_summary[v].notna().sum()
    print(f"    {v:<32} {n:>4,}/{total_firms}  "
          f"({n/total_firms*100:.1f}%)", flush=True)

print(f"\n  Firms with ALL PSM variables non-null: "
      f"{psm_complete:,}/{total_firms} "
      f"({psm_complete/total_firms*100:.1f}%)", flush=True)

# Save
psm_summary.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {OUTPUT_PATH}: {len(psm_summary):,} rows", flush=True)
print(f"  One row per firm with all candidate PSM variables", flush=True)

# =============================================================================
# RECOMMENDATION
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"RECOMMENDATIONS FOR STEP 7", flush=True)
print(f"{'='*65}", flush=True)

# PSM viability
if psm_complete >= 0.8 * total_firms:
    print(f"\n  PSM is viable on {psm_complete} treated firms", flush=True)
    print(f"  Approach: 1-to-1 nearest neighbour without replacement", flush=True)
    print(f"  Strata: 4-digit CPC subclass", flush=True)
    print(f"  Match on: patent_age, log_target_pre_patents, pre_novelty_mean", flush=True)
else:
    print(f"\n  WARNING: only {psm_complete} firms have all PSM variables", flush=True)
    print(f"  Consider relaxing matching to 2 variables", flush=True)
    print(f"  or using broader pre-merger novelty window", flush=True)

# Control group source
print(f"\n  Control group candidates needed:", flush=True)
print(f"  Firms in same CPC subclass that were NOT acquired", flush=True)
print(f"  Source options:", flush=True)
print(f"    Option A: patents_validation.csv (full universe 2005-2015)", flush=True)
print(f"    Option B: patents_main.csv non-eligible firms (failed thresholds)", flush=True)
print(f"    Option C: pull new BigQuery dataset of non-acquired US tech firms", flush=True)

print(f"\n  Top CPC subclasses by treated firm count:", flush=True)
for cpc, n in dominant_cpc.value_counts().head(5).items():
    print(f"    {cpc}: need {n:,} matched controls", flush=True)

# =============================================================================
# SECTION 9 — PSM VARIABLE VALIDATION
# Checks needed before running actual PSM matching:
#   9a. Correlation matrix — variables should not be highly collinear
#   9b. Outlier detection — extreme firms distort nearest-neighbour matching
#   9c. Pre-novelty distribution — check it is meaningful as a match variable
#   9d. Standardisation check — confirm z-scores work correctly
#   9e. Per-CPC strata sample sizes — minimum controls needed per class
#   9f. Winsorisation thresholds — recommended clipping values
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"SECTION 9 — PSM VARIABLE VALIDATION", flush=True)
print(f"{'='*65}", flush=True)

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn", "--quiet"])
    from sklearn.preprocessing import StandardScaler

psm_vars_list = ["patent_age", "log_target_pre_patents", "pre_novelty_mean"]

psm_df = psm_summary[
    ["bvd_id", "dominant_cpc"] + psm_vars_list
].dropna(subset=psm_vars_list).copy()

print(f"\n  Firms available for PSM (all 3 variables non-null): "
      f"{len(psm_df):,}", flush=True)

# -------------------------------------------------------------------------
# 9a — Correlation matrix
# -------------------------------------------------------------------------
print(f"\n  9a. Correlation matrix:", flush=True)
corr = psm_df[psm_vars_list].corr().round(3)
print(corr.to_string(), flush=True)

high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r = abs(corr.iloc[i, j])
        if r > 0.5:
            high_corr.append((corr.columns[i], corr.columns[j], r))

if high_corr:
    print(f"\n  WARNING: high correlations detected (|r| > 0.5):", flush=True)
    for v1, v2, r in high_corr:
        print(f"    {v1} vs {v2}:  r = {r:.3f}", flush=True)
    print(f"  Consider dropping the weaker variable from matching", flush=True)
else:
    print(f"\n  No high correlations (|r| > 0.5) — all three variables",
          flush=True)
    print(f"  contribute independent information to the match", flush=True)

# -------------------------------------------------------------------------
# 9b — Outlier detection (> 3 std from mean)
# -------------------------------------------------------------------------
print(f"\n  9b. Outlier detection (> 3 std from mean):", flush=True)
outlier_firms = set()
for var in psm_vars_list:
    mean = psm_df[var].mean()
    std  = psm_df[var].std()
    out  = psm_df[
        (psm_df[var] < mean - 3*std) |
        (psm_df[var] > mean + 3*std)
    ]
    flag = " <-- WINSORISE" if len(out) > 0 else ""
    print(f"    {var:<32} {len(out):>3} outliers{flag}", flush=True)
    if len(out) > 0:
        for _, row in out.iterrows():
            tname = psm_summary[
                psm_summary["bvd_id"] == row["bvd_id"]
            ]["target_name"].values
            tname = tname[0] if len(tname) > 0 else "unknown"
            print(f"      {row['bvd_id']:<20} {var} = {row[var]:.3f}  "
                  f"({tname[:35]})", flush=True)
        outlier_firms.update(out["bvd_id"].tolist())

print(f"\n  Total unique firms flagged: {len(outlier_firms):,}", flush=True)
if outlier_firms:
    print(f"  Recommendation: winsorise at 1st/99th percentile",
          flush=True)
    print(f"  before computing match distances", flush=True)
else:
    print(f"  No outliers — matching variables are well-behaved", flush=True)

# -------------------------------------------------------------------------
# 9c — Pre-novelty mean distribution
# -------------------------------------------------------------------------
print(f"\n  9c. pre_novelty_mean distribution:", flush=True)
print(psm_df["pre_novelty_mean"].describe().round(4).to_string(), flush=True)
skew_nov = psm_df["pre_novelty_mean"].skew()
iqr_nov  = (psm_df["pre_novelty_mean"].quantile(0.75) -
            psm_df["pre_novelty_mean"].quantile(0.25))
print(f"\n  Skewness:  {skew_nov:.3f}", flush=True)
print(f"  IQR:       {iqr_nov:.4f}", flush=True)

if iqr_nov < 0.05:
    print(f"\n  WARNING: IQR is very small ({iqr_nov:.4f}) — pre_novelty_mean",
          flush=True)
    print(f"  has little variation and may not discriminate well between",
          flush=True)
    print(f"  treated firms in the matching algorithm", flush=True)
    print(f"  Consider dropping it from PSM and using it as a balance check",
          flush=True)
elif iqr_nov < 0.10:
    print(f"\n  NOTE: moderate IQR ({iqr_nov:.4f}) — pre_novelty_mean adds",
          flush=True)
    print(f"  some discriminating power but patent_age and",
          flush=True)
    print(f"  log_target_pre_patents will dominate the match", flush=True)
else:
    print(f"\n  IQR looks healthy — pre_novelty_mean is a useful",
          flush=True)
    print(f"  discriminating variable for PSM matching", flush=True)

# -------------------------------------------------------------------------
# 9d — Standardisation check
# -------------------------------------------------------------------------
print(f"\n  9d. Standardised variable scales (z-scores):", flush=True)
print(f"  All should have mean ~0 and std ~1 after scaling", flush=True)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(psm_df[psm_vars_list])
scaled_df = pd.DataFrame(X_scaled, columns=psm_vars_list)

print(f"\n  {'Variable':<32} {'Mean':>8}  {'Std':>8}", flush=True)
print(f"  {'-'*52}", flush=True)
for col in psm_vars_list:
    print(f"  {col:<32} {scaled_df[col].mean():>8.4f}  "
          f"{scaled_df[col].std():>8.4f}", flush=True)

print(f"\n  Scaling confirmed — use StandardScaler before matching",
      flush=True)

# -------------------------------------------------------------------------
# 9e — Per-CPC strata requirements
# -------------------------------------------------------------------------
print(f"\n  9e. Control candidates needed per CPC stratum:", flush=True)
print(f"  {'CPC':>6}  {'Treated':>9}  {'Min controls':>14}  "
      f"{'Suggested pool':>16}", flush=True)
print(f"  {'-'*52}", flush=True)
for cpc, n in psm_df["dominant_cpc"].value_counts().items():
    # Suggest 10x pool so 1-to-1 matching has good quality options
    suggested = n * 10
    print(f"  {cpc:>6}  {n:>9,}  {n:>14,}  {suggested:>16,}", flush=True)

print(f"\n  Suggested pool = 10x treated count per CPC class", flush=True)
print(f"  Larger pool improves match quality for nearest-neighbour",
      flush=True)

# -------------------------------------------------------------------------
# 9f — Winsorisation thresholds
# -------------------------------------------------------------------------
print(f"\n  9f. Winsorisation thresholds (1st / 99th percentile):", flush=True)
print(f"  {'Variable':<32}  {'P01':>8}  {'P99':>8}  "
      f"{'Below':>7}  {'Above':>7}", flush=True)
print(f"  {'-'*68}", flush=True)
for var in psm_vars_list:
    p01 = psm_df[var].quantile(0.01)
    p99 = psm_df[var].quantile(0.99)
    n_below = int((psm_df[var] < p01).sum())
    n_above = int((psm_df[var] > p99).sum())
    print(f"  {var:<32}  {p01:>8.3f}  {p99:>8.3f}  "
          f"{n_below:>7}  {n_above:>7}", flush=True)

# -------------------------------------------------------------------------
# SECTION 9 SUMMARY
# -------------------------------------------------------------------------
print(f"\n  {'='*55}", flush=True)
print(f"  SECTION 9 SUMMARY", flush=True)
print(f"  {'='*55}", flush=True)

issues = []
if high_corr:
    issues.append(f"High correlation between {high_corr[0][0]} and "
                  f"{high_corr[0][1]} (r={high_corr[0][2]:.2f})")
if outlier_firms:
    issues.append(f"{len(outlier_firms)} firms with outlier values — "
                  f"winsorise before matching")
if iqr_nov < 0.05:
    issues.append("pre_novelty_mean IQR too small — consider dropping "
                  "from PSM variables")

if not issues:
    print(f"\n  All PSM validation checks passed:", flush=True)
    print(f"  - No high correlations between matching variables", flush=True)
    print(f"  - No extreme outliers", flush=True)
    print(f"  - pre_novelty_mean has sufficient variation", flush=True)
    print(f"  - Variables scale correctly with StandardScaler", flush=True)
    print(f"\n  Ready to proceed to Step 7c PSM matching", flush=True)
    print(f"  once control group candidates are available", flush=True)
else:
    print(f"\n  Issues to resolve before PSM matching:", flush=True)
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}", flush=True)
    print(f"\n  Address these before building the matched panel", flush=True)