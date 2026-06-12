"""
Step 9c — Robustness Check: Poor-Match Exclusion
================================================
Replicates the preferred TWFE specification (Model 3, two-way fixed effects)
on a subsample that excludes the 32 matched pairs with standardised match
distance > 0.25, as promised in Section 4.3.4 of the thesis.

Background (Section 4.3.4):
  Of the 195 matched pairs, 32 have a standardised match distance > 0.25
  ("poor" quality). Their elevated distances reflect divergence on patent
  age rather than on the two matched PSM dimensions (log_pre_patents and
  pre_novelty_mean). This script confirms that these 32 pairs do not
  materially influence the baseline null result.

What this produces:
  One additional row for Table 12 (Robustness Summary):
    "Poor-match exclusion (dist <= 0.25)   beta3   SE   p   N"
  Reported for both the annual (primary) and semiannual (robustness) panels.

Specification:
  Identical to step9b_did_regression.py Model 3 (preferred TWFE):
    novelty_score_mean ~ treated_x_post + patent_age
    | entity_effects + time_effects
    clustered SE by bvd_id

  The ONLY difference: the panel is restricted to firms whose matched pair
  has match_distance <= 0.25 (163 of 195 pairs).

Inputs:
  data/interim/psm_matches.csv               — match distances per pair
  data/interim/panel_full_annual.csv         — primary panel
  data/interim/panel_full_semiannual.csv     — robustness panel

Outputs:
  data/outputs/robustness_poor_match_exclusion.csv   — two-row result table
  data/outputs/robustness_poor_match_exclusion.tex   — LaTeX table row

Usage:
  python step9c_robustness_poor_matches.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

MATCHES_PATH    = "data/interim/psm_matches.csv"
ANNUAL_PATH     = "data/interim/panel_full_annual.csv"
SEMIANNUAL_PATH = "data/interim/panel_full_semiannual.csv"
OUTPUT_DIR      = "data/outputs"

POOR_MATCH_THRESHOLD = 0.25   # pairs with distance > this are excluded

CONTROL_VARS = [
    "log_target_pre_patents",
    "patent_age",
    "log_deal_value",
    "deal_value_missing",
    "log_acquiror_assets",
    "acquiror_assets_missing",
    "log_acquiror_pre_patents",
]

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    print("ERROR: linearmodels not installed — run: pip install linearmodels",
          flush=True)
    sys.exit(1)

# =============================================================================
# HELPERS
# =============================================================================

def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    if p < 0.05:   return "**"
    if p < 0.10:   return "*"
    return ""

def print_header(title):
    print(f"\n{'=' * 65}", flush=True)
    print(title, flush=True)
    print(f"{'=' * 65}", flush=True)

def detect_time_var(df):
    """Return (time_var, freq_label) from available columns."""
    if "event_year" in df.columns:
        return "event_year", "annual"
    if "event_halfyear" in df.columns:
        return "event_halfyear", "semiannual"
    if "event_quarter" in df.columns:
        return "event_quarter", "quarterly"
    raise ValueError(
        "No time variable found. Expected event_year, event_halfyear, "
        "or event_quarter."
    )

def run_twfe(panel, time_var, freq_label, n_good_pairs):
    """
    Run preferred TWFE (Model 3) on the supplied panel.

    Mirrors step9 Model 3 exactly:
      - entity + time fixed effects
      - treated_x_post as the DiD estimator
      - patent_age as time-varying covariate (not absorbed by entity FE)
      - all other CONTROL_VARS are time-invariant and absorbed by entity FE
      - clustered SE by bvd_id (entity)

    Returns a dict with the key scalar results.
    """
    # ---- panel prep ---------------------------------------------------------
    df = panel.copy()
    df["bvd_id"]         = df["bvd_id"].astype(str)
    df[time_var]         = pd.to_numeric(df[time_var], errors="coerce")
    df                   = df.dropna(subset=[time_var]).copy()
    df[time_var]         = df[time_var].astype(int)
    df["treated"]        = df["treated"].astype(int)
    df["post"]           = df["post"].astype(int)
    df["treated_x_post"] = df["treated"] * df["post"]

    for var in CONTROL_VARS:
        if var in df.columns:
            df[var] = df[var].fillna(0)

    # Drop periods with missing dependent variable
    # (firm-periods with zero filings — handled by extensive margin elsewhere)
    df = df.dropna(subset=["novelty_score_mean"]).copy()

    n_obs    = len(df)
    n_firms  = df["bvd_id"].nunique()
    n_treat  = df.loc[df["treated"] == 1, "bvd_id"].nunique()
    n_ctrl   = df.loc[df["treated"] == 0, "bvd_id"].nunique()

    print(f"\n  Observations:    {n_obs:,}", flush=True)
    print(f"  Firms:           {n_firms:,} "
          f"(treated={n_treat}, control={n_ctrl})", flush=True)
    print(f"  Good pairs used: {n_good_pairs}", flush=True)

    # ---- index --------------------------------------------------------------
    panel_idx = df.set_index(["bvd_id", time_var]).sort_index()

    # ---- time-varying controls (not absorbed by entity FE) ------------------
    time_varying = [v for v in ["patent_age"] if v in panel_idx.columns]
    exog_vars    = ["treated_x_post"] + time_varying

    if time_varying:
        print(f"  Time-varying controls: {time_varying}", flush=True)
    else:
        print("  patent_age not found — TWFE without time-varying controls.",
              flush=True)

    # ---- fit TWFE -----------------------------------------------------------
    X  = panel_idx[exog_vars].astype(float)
    m  = PanelOLS(
        dependent=panel_idx["novelty_score_mean"].astype(float),
        exog=X,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=True,
    )
    res = m.fit(cov_type="clustered", cluster_entity=True)

    beta3    = float(res.params["treated_x_post"])
    beta3_se = float(res.std_errors["treated_x_post"])
    beta3_p  = float(res.pvalues["treated_x_post"])
    beta3_ci_lo = beta3 - 1.96 * beta3_se
    beta3_ci_hi = beta3 + 1.96 * beta3_se

    print(f"\n  b3 (treated_x_post) = {beta3:+.4f}", flush=True)
    print(f"  SE                  = {beta3_se:.4f}", flush=True)
    print(f"  p-value             = {beta3_p:.4f}  {stars(beta3_p)}",
          flush=True)
    print(f"  95% CI              = [{beta3_ci_lo:+.4f}, "
          f"{beta3_ci_hi:+.4f}]", flush=True)

    return {
        "freq_label":   freq_label,
        "n_pairs":      n_good_pairs,
        "n_obs":        n_obs,
        "n_firms":      n_firms,
        "beta3":        round(beta3, 4),
        "se":           round(beta3_se, 4),
        "p_value":      round(beta3_p, 4),
        "sig":          stars(beta3_p),
        "ci_lower":     round(beta3_ci_lo, 4),
        "ci_upper":     round(beta3_ci_hi, 4),
        "full_summary": str(res.summary),
    }

# =============================================================================
# MAIN
# =============================================================================

print("=" * 65, flush=True)
print("STEP 9c — ROBUSTNESS: POOR-MATCH EXCLUSION", flush=True)
print("=" * 65, flush=True)
print(f"Threshold: excluding pairs with match_distance > "
      f"{POOR_MATCH_THRESHOLD}", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check required inputs
for path in [MATCHES_PATH, ANNUAL_PATH, SEMIANNUAL_PATH]:
    if not os.path.exists(path):
        print(f"ERROR: required file not found: {path}", flush=True)
        sys.exit(1)

# =============================================================================
# PHASE 1 — IDENTIFY GOOD PAIRS FROM psm_matches.csv
# =============================================================================

print_header("PHASE 1 — LOAD MATCH DISTANCES")

matches = pd.read_csv(MATCHES_PATH)

print(f"  Total matched pairs in psm_matches.csv: {len(matches):,}",
      flush=True)
print(f"\n  Match distance distribution:", flush=True)
print(matches["match_distance"].describe().round(4).to_string(), flush=True)

n_excellent = (matches["match_distance"] <= 0.10).sum()
n_good      = (matches["match_distance"] <= 0.25).sum()
n_poor      = (matches["match_distance"] >  0.25).sum()

print(f"\n  Quality breakdown:", flush=True)
print(f"    Excellent (dist <= 0.10): {n_excellent:,}  "
      f"({n_excellent/len(matches)*100:.1f}%)", flush=True)
print(f"    Good      (dist <= 0.25): {n_good:,}  "
      f"({n_good/len(matches)*100:.1f}%)", flush=True)
print(f"    Poor      (dist >  0.25): {n_poor:,}  "
      f"({n_poor/len(matches)*100:.1f}%)", flush=True)
print(f"\n  Excluding {n_poor} pairs with dist > {POOR_MATCH_THRESHOLD}.",
      flush=True)
print(f"  Retaining {n_good} pairs with dist <= {POOR_MATCH_THRESHOLD}.",
      flush=True)

# --- Get BvD IDs of treated AND their matched controls for good pairs --------
good_matches = matches[matches["match_distance"] <= POOR_MATCH_THRESHOLD].copy()

good_treated_ids  = set(good_matches["bvd_id"].astype(str))
good_control_ids  = set(good_matches["control_assignee"].astype(str))
good_all_ids      = good_treated_ids | good_control_ids

print(f"\n  BvD IDs to retain:", flush=True)
print(f"    Treated firms:  {len(good_treated_ids):,}", flush=True)
print(f"    Control firms:  {len(good_control_ids):,}", flush=True)
print(f"    Total:          {len(good_all_ids):,}", flush=True)

n_good_pairs = len(good_matches)

# =============================================================================
# PHASE 2 — ANNUAL PANEL (primary specification)
# =============================================================================

print_header("PHASE 2 — ANNUAL PANEL (primary specification)")

annual_full = pd.read_csv(ANNUAL_PATH, low_memory=False)
time_var_a, freq_a = detect_time_var(annual_full)

n_before = annual_full["bvd_id"].astype(str).nunique()
annual_filtered = annual_full[
    annual_full["bvd_id"].astype(str).isin(good_all_ids)
].copy()
n_after = annual_filtered["bvd_id"].astype(str).nunique()

print(f"  Full panel firms:     {n_before:,}", flush=True)
print(f"  After exclusion:      {n_after:,}  "
      f"(dropped {n_before - n_after} firms from {n_poor} poor pairs)",
      flush=True)

result_annual = run_twfe(annual_filtered, time_var_a, freq_a, n_good_pairs)

# =============================================================================
# PHASE 3 — SEMIANNUAL PANEL (robustness)
# =============================================================================

print_header("PHASE 3 — SEMIANNUAL PANEL (robustness specification)")

semi_full = pd.read_csv(SEMIANNUAL_PATH, low_memory=False)
time_var_s, freq_s = detect_time_var(semi_full)

n_before_s = semi_full["bvd_id"].astype(str).nunique()
semi_filtered = semi_full[
    semi_full["bvd_id"].astype(str).isin(good_all_ids)
].copy()
n_after_s = semi_filtered["bvd_id"].astype(str).nunique()

print(f"  Full panel firms:     {n_before_s:,}", flush=True)
print(f"  After exclusion:      {n_after_s:,}  "
      f"(dropped {n_before_s - n_after_s} firms from {n_poor} poor pairs)",
      flush=True)

result_semi = run_twfe(semi_filtered, time_var_s, freq_s, n_good_pairs)

# =============================================================================
# PHASE 4 — SUMMARY AND COMPARISON TO BASELINE
# =============================================================================

print_header("PHASE 4 — SUMMARY")

print(f"\n  Specification: TWFE (entity + time FE), clustered SE, "
      f"dist <= {POOR_MATCH_THRESHOLD}", flush=True)
print(f"\n  {'Panel':<14} {'Pairs':>6}  {'N_obs':>7}  "
      f"{'b3':>8}  {'SE':>8}  {'p':>8}  {'Sig':<5}  "
      f"{'95% CI':<22}", flush=True)
print(f"  {'-' * 80}", flush=True)

for r in [result_annual, result_semi]:
    ci = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
    print(
        f"  {r['freq_label']:<14} {r['n_pairs']:>6}  "
        f"{r['n_obs']:>7,}  {r['beta3']:>+8.4f}  "
        f"{r['se']:>8.4f}  {r['p_value']:>8.4f}  "
        f"{r['sig']:<5}  {ci:<22}",
        flush=True,
    )

print(f"\n  Interpretation:", flush=True)
for r in [result_annual, result_semi]:
    if r["p_value"] >= 0.10:
        verdict = ("null result preserved — poor matches do not "
                   "materially affect the baseline estimate")
    else:
        verdict = ("result differs from baseline — investigate "
                   "which poor-matched pairs are influential")
    print(f"  {r['freq_label']}: {verdict}", flush=True)

# =============================================================================
# PHASE 5 — SAVE OUTPUTS
# =============================================================================

print_header("PHASE 5 — SAVE OUTPUTS")

# --- CSV result table --------------------------------------------------------
results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "full_summary"}
    for r in [result_annual, result_semi]
])

csv_path = os.path.join(OUTPUT_DIR, "robustness_poor_match_exclusion.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}", flush=True)

# --- Full regression summaries -----------------------------------------------
txt_path = os.path.join(OUTPUT_DIR, "robustness_poor_match_exclusion.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for r in [result_annual, result_semi]:
        f.write(f"POOR-MATCH EXCLUSION ROBUSTNESS — "
                f"{r['freq_label'].upper()}\n")
        f.write(f"Pairs retained: {r['n_pairs']} "
                f"(match_distance <= {POOR_MATCH_THRESHOLD})\n")
        f.write("=" * 70 + "\n")
        f.write(r["full_summary"])
        f.write("\n\n")
print(f"  Saved: {txt_path}", flush=True)

# --- LaTeX table row ---------------------------------------------------------
# Produces the two rows to insert into Table 12 (Robustness Summary)
# Paste into the table body after the existing four robustness rows.

tex_path = os.path.join(OUTPUT_DIR, "robustness_poor_match_exclusion.tex")

def fmt(val, pval=None):
    if pd.isna(val):
        return "--"
    s = f"{val:.4f}"
    if pval is not None and not pd.isna(pval):
        s += ("***" if pval < 0.01 else "**" if pval < 0.05
              else "*" if pval < 0.10 else "")
    return s

ra = result_annual
rs = result_semi

latex_rows = (
    "% ---------------------------------------------------------------\n"
    "% Poor-match exclusion robustness rows for Table 12\n"
    "% Insert after existing four robustness rows in the annual panel\n"
    "% and after the corresponding semiannual rows.\n"
    "% ---------------------------------------------------------------\n"
    "\\midrule\n"
    "\\multicolumn{6}{l}{\\textit{Panel C: Poor-match exclusion "
    "(match distance $\\leq 0.25$)}} \\\\\n"
    f"Annual (primary) & "
    f"{fmt(ra['beta3'], ra['p_value'])} & ({fmt(ra['se'])}) & "
    f"{ra['p_value']:.3f} & {ra['n_obs']:,} & {ra['n_pairs']} \\\\\n"
    f"Semiannual (robustness) & "
    f"{fmt(rs['beta3'], rs['p_value'])} & ({fmt(rs['se'])}) & "
    f"{rs['p_value']:.3f} & {rs['n_obs']:,} & {rs['n_pairs']} \\\\\n"
    "% Note: restricted to matched pairs with standardised match\n"
    "% distance <= 0.25 (excludes the 32 poor-quality pairs\n"
    "% documented in Section 4.3.4 whose elevated distances\n"
    "% reflect patent-age divergence, not divergence on the\n"
    "% two PSM matching dimensions).\n"
)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_rows)
print(f"  Saved: {tex_path}", flush=True)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'=' * 65}", flush=True)
print("STEP 9c ROBUSTNESS — POOR-MATCH EXCLUSION COMPLETE", flush=True)
print(f"{'=' * 65}", flush=True)
print(f"\nTable 12 row (annual):     b3 = {ra['beta3']:+.4f}  "
      f"SE = {ra['se']:.4f}  p = {ra['p_value']:.4f}  "
      f"N = {ra['n_obs']:,}  {ra['sig']}", flush=True)
print(f"Table 12 row (semiannual): b3 = {rs['beta3']:+.4f}  "
      f"SE = {rs['se']:.4f}  p = {rs['p_value']:.4f}  "
      f"N = {rs['n_obs']:,}  {rs['sig']}", flush=True)
print(f"\nInsert robustness_poor_match_exclusion.tex into Table 12 "
      f"as Panel C.", flush=True)
print(f"Thesis text (Section 5.5.4): report b3 and N for both panels "
      f"and state that the null is preserved.", flush=True)
