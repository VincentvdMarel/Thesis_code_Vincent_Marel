"""
Step 7c — PSM Matching
========================
Performs 1-to-1 nearest neighbour PSM matching without replacement
between treated firms and control candidates.

Matching specification (thesis proposal section 3.2):
  - 1-to-1 nearest neighbour without replacement
  - Same 4-digit CPC subclass (G06F, H04L, H01L, G06N)
  - Match on: patent_age, log_target_pre_patents, pre_novelty_mean
  - Variables standardised to z-scores before distance computation
  - Winsorisation at P99 thresholds from step7a_psm_diagnostics.py

Post-match balance check:
  - Standardised Mean Difference (SMD) < 0.1 per variable = good balance

Inputs:
  psm_diagnostics.csv    — treated firm PSM variables (185 firms)
  control_psm_vars.csv   — control firm PSM variables (firm x deal year)

Outputs:
  psm_matches.csv        — matched pairs with balance diagnostics
  psm_balance.csv        — SMD balance table for thesis reporting
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# CONFIGURATION
# =============================================================================
TREATED_PATH   = "data/interim/psm_diagnostics.csv"
CONTROL_PATH   = "data/interim/control_psm_vars.csv"
MATCHES_OUTPUT = "data/interim/psm_matches.csv"
BALANCE_OUTPUT = "data/interim/psm_balance.csv"

PSM_VARS         = ["log_target_pre_patents", "pre_novelty_mean"]
CPC_CLASSES      = ["G06F", "H04L", "H01L", "G06N"]


CALIPER          = None    

RELAX_YEAR       = True

WINSOR = {
    "patent_age":             19.630,   # clipped for balance check only
    "log_target_pre_patents": 6.905,   # PSM matching variable
    "pre_novelty_mean":       0.583,   # PSM matching variable
}

# =============================================================================
# SETUP
# =============================================================================
print("="*60, flush=True)
print("STEP 7c — PSM MATCHING", flush=True)
print("="*60, flush=True)
print(f"Method:    1-to-1 nearest neighbour without replacement", flush=True)
print(f"Strata:    4-digit CPC subclass", flush=True)
print(f"Variables: {PSM_VARS}", flush=True)
print(f"Note:      patent_age excluded from matching — controlled", flush=True)
print(f"           parametrically as covariate in Step 9b DiD", flush=True)
print(f"Caliper:   {CALIPER} (standardised units)", flush=True)
print(f"Relax year: {RELAX_YEAR}", flush=True)

for path in [TREATED_PATH, CONTROL_PATH]:
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found", flush=True)
        exit()

# =============================================================================
# LOAD TREATED AND CONTROL PSM VARIABLES
# =============================================================================
print(f"\nLoading treated firms: {TREATED_PATH}...", flush=True)
treated = pd.read_csv(TREATED_PATH)
treated["deal_year"] = pd.to_datetime(
    treated["deal_date"], errors="coerce"
).dt.year

# Drop firms missing any PSM variable
treated_clean = treated.dropna(
    subset=PSM_VARS + ["dominant_cpc", "deal_year"]
).copy()

print(f"  Total treated firms:    {len(treated):,}", flush=True)
print(f"  With all PSM vars:      {len(treated_clean):,}", flush=True)
dropped = len(treated) - len(treated_clean)
if dropped > 0:
    print(f"  Dropped (missing vars): {dropped:,}", flush=True)

print(f"\nLoading control candidates: {CONTROL_PATH}...", flush=True)
control = pd.read_csv(CONTROL_PATH)
control_clean = control.dropna(
    subset=PSM_VARS + ["dominant_cpc", "pseudo_deal_year"]
).copy()
control_clean = control_clean.reset_index(drop=True)

print(f"  Total control rows:     {len(control):,}", flush=True)
print(f"  With all PSM vars:      {len(control_clean):,}", flush=True)
print(f"  Unique control firms:   {control_clean['assignee_name'].nunique():,}",
      flush=True)

# =============================================================================
# APPLY WINSORISATION
# =============================================================================
print(f"\nApplying winsorisation (same thresholds as treated firms)...",
      flush=True)
for var, cap in WINSOR.items():
    if var in treated_clean.columns:
        n = (treated_clean[var] > cap).sum()
        treated_clean[var] = treated_clean[var].clip(upper=cap)
        if n > 0:
            print(f"  Treated  {var:<32} {n} rows clipped at {cap}",
                  flush=True)
    if var in control_clean.columns:
        n = (control_clean[var] > cap).sum()
        control_clean[var] = control_clean[var].clip(upper=cap)
        if n > 0:
            print(f"  Control  {var:<32} {n} rows clipped at {cap}",
                  flush=True)

# Snapshot of the full control pool (pre-matching) for the balance table.
# One row per unique firm; multiple pseudo-years would otherwise inflate N.
control_unmatched = control_clean.drop_duplicates(
    subset=["assignee_name"]
).copy()

# =============================================================================
# RUN PSM MATCHING
# Loop over CPC classes then deal years
# For each treated firm find nearest control in same strata
# =============================================================================
print(f"\n{'='*60}", flush=True)
print(f"RUNNING 1-TO-1 NEAREST NEIGHBOUR MATCHING", flush=True)
print(f"{'='*60}", flush=True)

matched_pairs = []
used_controls = set()   # track used control assignee names — no replacement
unmatched     = []

print(f"\n  {'CPC':<6}  {'Year':<6}  {'Treated':>9}  "
      f"{'Matched':>9}  {'Caliper fail':>13}  {'No pool':>8}",
      flush=True)
print(f"  {'-'*60}", flush=True)

for cpc in CPC_CLASSES:
    treated_cpc = treated_clean[
        treated_clean["dominant_cpc"] == cpc
    ].copy()

    if len(treated_cpc) == 0:
        continue

    deal_years = sorted(treated_cpc["deal_year"].dropna().unique())

    for year in deal_years:
        year = int(year)
        treated_yr = treated_cpc[treated_cpc["deal_year"] == year].copy()

        if len(treated_yr) == 0:
            continue

        # Control pool: same CPC, same year, not yet used
        def get_pool(yr_range):
            return control_clean[
                (control_clean["dominant_cpc"]       == cpc) &
                (control_clean["pseudo_deal_year"].isin(yr_range)) &
                (~control_clean["assignee_name"].isin(used_controls))
            ].copy()

        pool = get_pool([year])

        # Relax to ±1 year if pool is smaller than treated count
        if RELAX_YEAR and len(pool) < len(treated_yr):
            pool_relaxed = get_pool([year-1, year, year+1])
            if len(pool_relaxed) > len(pool):
                pool = pool_relaxed

        n_treated_yr  = len(treated_yr)
        n_caliper_fail = 0
        n_no_pool      = 0

        if len(pool) == 0:
            for _, t_row in treated_yr.iterrows():
                unmatched.append({
                    "bvd_id":  t_row["bvd_id"],
                    "reason":  f"no control pool for {cpc} {year}",
                })
            n_no_pool = n_treated_yr
            print(
                f"  {cpc:<6}  {year:<6}  {n_treated_yr:>9,}  "
                f"{'0':>9}  {'0':>13}  {n_no_pool:>8}",
                flush=True
            )
            continue

        # Match one treated firm at a time so used_controls is updated
        # between each match — prevents multiple treated firms from
        # being matched to the same control in a single batch
        n_matched_yr = 0
        for j, (t_idx, t_row) in enumerate(treated_yr.iterrows()):

            # Re-filter pool after every match to exclude used controls
            current_pool = get_pool(
                [year] if not RELAX_YEAR else [year-1, year, year+1]
            )

            if len(current_pool) == 0:
                unmatched.append({
                    "bvd_id":  t_row["bvd_id"],
                    "reason":  f"pool exhausted for {cpc} {year}",
                })
                n_no_pool += 1
                continue

            # Fit scaler on this treated firm + current pool
            one_treated = pd.DataFrame([t_row[PSM_VARS]])
            all_data    = pd.concat(
                [one_treated, current_pool[PSM_VARS]], ignore_index=True
            )
            scaler = StandardScaler()
            scaler.fit(all_data)

            t_scaled = scaler.transform(one_treated)
            c_scaled = scaler.transform(current_pool[PSM_VARS])

            # Find nearest neighbour
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(c_scaled)
            distances, indices = nn.kneighbors(t_scaled)

            dist     = float(distances[0][0])
            pool_pos = int(indices[0][0])
            ctrl_row = current_pool.iloc[pool_pos]

            # Apply caliper
            if CALIPER is not None and dist > CALIPER:
                unmatched.append({
                    "bvd_id":  t_row["bvd_id"],
                    "reason":  f"caliper exceeded ({dist:.3f} > {CALIPER})",
                })
                n_caliper_fail += 1
                continue

            # Mark control as used — immediately so next iteration excludes it
            used_controls.add(ctrl_row["assignee_name"])
            n_matched_yr += 1

            matched_pairs.append({
                # Identifiers
                "bvd_id":              t_row["bvd_id"],
                "target_name":         t_row.get("target_name", ""),
                "control_assignee":    ctrl_row["assignee_name"],
                "cpc":                 cpc,
                "deal_year":           year,
                "match_distance":      round(dist, 4),
                # Treated vars (patent_age included for balance check only)
                "t_patent_age":        round(t_row["patent_age"], 3),
                "t_log_pre_patents":   round(t_row["log_target_pre_patents"], 3),
                "t_pre_novelty":       round(t_row["pre_novelty_mean"], 4),
                # Control vars
                "c_patent_age":        round(ctrl_row["patent_age"], 3),
                "c_log_pre_patents":   round(ctrl_row["log_target_pre_patents"], 3),
                "c_pre_novelty":       round(ctrl_row["pre_novelty_mean"], 4),
                # Pseudo deal year for control
                "control_deal_year":   int(ctrl_row["pseudo_deal_year"]),
            })

        print(
            f"  {cpc:<6}  {year:<6}  {n_treated_yr:>9,}  "
            f"{n_matched_yr:>9,}  {n_caliper_fail:>13,}  "
            f"{n_no_pool:>8,}",
            flush=True
        )

# =============================================================================
# MATCHING SUMMARY
# =============================================================================
matches_df  = pd.DataFrame(matched_pairs)
n_treated   = len(treated_clean)
n_matched   = len(matches_df)
n_unmatched = len(unmatched)

print(f"\n{'='*60}", flush=True)
print(f"MATCHING SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Treated firms available:  {n_treated:,}", flush=True)
print(f"  Matched pairs:            {n_matched:,}  "
      f"({n_matched/n_treated*100:.1f}%)", flush=True)
print(f"  Unmatched treated:        {n_unmatched:,}", flush=True)

if n_unmatched > 0:
    print(f"\n  Unmatched firms and reasons:", flush=True)
    unmatched_df = pd.DataFrame(unmatched)
    print(unmatched_df["reason"].value_counts().to_string(), flush=True)
    print(f"\n  Unmatched BvD IDs:", flush=True)
    for row in unmatched:
        print(f"    {row['bvd_id']}  — {row['reason']}", flush=True)

if n_matched > 0:
    print(f"\n  Match distance distribution:", flush=True)
    print(matches_df["match_distance"].describe().round(4).to_string(),
          flush=True)

    good_matches = (matches_df["match_distance"] <= 0.1).sum()
    ok_matches   = (matches_df["match_distance"] <= 0.25).sum()
    poor_matches = (matches_df["match_distance"] >  0.25).sum()
    print(f"\n  Match quality:", flush=True)
    print(f"    Excellent (dist <= 0.10): {good_matches:,}  "
          f"({good_matches/n_matched*100:.1f}%)", flush=True)
    print(f"    Good      (dist <= 0.25): {ok_matches:,}  "
          f"({ok_matches/n_matched*100:.1f}%)", flush=True)
    print(f"    Poor      (dist >  0.25): {poor_matches:,}  "
          f"({poor_matches/n_matched*100:.1f}%)", flush=True)

# =============================================================================
# BALANCE TABLE  (standard PSM paper format)
# -----------------------------------------------------------------------
# Columns: Treated mean | Unmatched control mean | Matched control mean
#          | SMD before matching | SMD after matching
# SMD = |mean_T - mean_C| / pooled_std
# SMD < 0.10 = good balance;  SMD < 0.25 = acceptable
# -----------------------------------------------------------------------
# patent_age is included even though it was not a matching variable —
# this shows whether the 2-variable match incidentally improved age balance.
# =============================================================================
print(f"\n{'='*78}", flush=True)
print(f"BALANCE TABLE — TREATED vs UNMATCHED vs MATCHED CONTROL", flush=True)
print(f"{'='*78}", flush=True)
print(f"  SMD < 0.10 = good  |  SMD < 0.25 = acceptable  |  SMD > 0.25 = poor\n",
      flush=True)

# (label, col_in_treated_clean, col_in_control_unmatched,
#         col_in_matches_treated, col_in_matches_control)
VAR_SPECS = [
    ("patent_age (covariate only)",
     "patent_age",             "patent_age",
     "t_patent_age",           "c_patent_age"),
    ("log_target_pre_patents",
     "log_target_pre_patents", "log_target_pre_patents",
     "t_log_pre_patents",      "c_log_pre_patents"),
    ("pre_novelty_mean",
     "pre_novelty_mean",       "pre_novelty_mean",
     "t_pre_novelty",          "c_pre_novelty"),
]

def _smd(t_vals: pd.Series, c_vals: pd.Series) -> float:
    pooled_std = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
    return abs(t_vals.mean() - c_vals.mean()) / pooled_std if pooled_std > 0 else 0.0

hdr = (
    f"  {'Variable':<30}  {'Treated':>8}  {'Unmatched':>10}  "
    f"{'Matched':>8}  {'SMD_pre':>8}  {'SMD_post':>9}  Status"
)
print(hdr, flush=True)
print(f"  {'-'*len(hdr)}", flush=True)

balance_rows = []

for label, t_col, cu_col, tm_col, cm_col in VAR_SPECS:
    if tm_col not in matches_df.columns:
        continue

    t_all  = treated_clean[t_col].dropna()
    cu_all = control_unmatched[cu_col].dropna()
    t_mat  = matches_df[tm_col].dropna()
    c_mat  = matches_df[cm_col].dropna()

    t_mean  = t_all.mean()
    cu_mean = cu_all.mean()
    cm_mean = c_mat.mean()
    smd_pre  = _smd(t_all, cu_all)
    smd_post = _smd(t_mat, c_mat)

    if smd_post < 0.10:
        status = "Good"
    elif smd_post < 0.25:
        status = "Acceptable"
    else:
        status = "Poor"

    print(
        f"  {label:<30}  {t_mean:>8.3f}  {cu_mean:>10.3f}  "
        f"{cm_mean:>8.3f}  {smd_pre:>8.4f}  {smd_post:>9.4f}  {status}",
        flush=True
    )

    balance_rows.append({
        "variable":            label,
        "t_mean":              round(t_mean, 4),
        "ctrl_unmatched_mean": round(cu_mean, 4),
        "ctrl_matched_mean":   round(cm_mean, 4),
        "smd_before":          round(smd_pre, 4),
        "smd_after":           round(smd_post, 4),
        "balanced":            smd_post < 0.10,
    })

balance_df = pd.DataFrame(balance_rows)

# =============================================================================
# SAVE OUTPUTS
# =============================================================================
matches_df.to_csv(MATCHES_OUTPUT, index=False)
balance_df.to_csv(BALANCE_OUTPUT, index=False)

print(f"\n{'='*60}", flush=True)
print(f"SAVED", flush=True)
print(f"{'='*60}", flush=True)
print(f"  {MATCHES_OUTPUT:<40} {len(matches_df):,} matched pairs",
      flush=True)
print(f"  {BALANCE_OUTPUT:<40} {len(balance_df):,} variables",
      flush=True)

print(f"""
psm_balance.csv columns:
  variable            — covariate name
  t_mean              — treated-firm mean (all matchable treated firms)
  ctrl_unmatched_mean — control mean BEFORE matching (full candidate pool)
  ctrl_matched_mean   — control mean AFTER matching (matched controls only)
  smd_before          — SMD between treated and unmatched control pool
  smd_after           — SMD between treated and matched control (post-match)
  balanced            — True if smd_after < 0.10

This is the standard three-column / two-SMD format used in PSM papers.
Report as Table A1 — Covariate Balance Before and After Matching.

Note on patent_age: matching used log_target_pre_patents and pre_novelty_mean
only. patent_age is shown as a balance diagnostic and enters Step 9b DiD as
a covariate (double-robust control; Imbens and Rubin 2015).

Next step: step7d_build_panel.py
""", flush=True)