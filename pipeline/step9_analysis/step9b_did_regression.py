"""
Step 9b — DiD Regression Analysis (revised, v2)

Nine fixes applied — three from the original revision plus six additional
validity fixes addressing issues 11-16 raised during external review.

ORIGINAL FIXES (v1):
  FIX 1 — Missing Overlap x Post term (H2 moderation)
  FIX 2 — Zero-patent quarter selection (extensive margin)
  FIX 3 — Staggered DiD without heterogeneity-robust check

Inputs:
  data/interim/panel_full_semiannual.csv   (or quarterly/annual variant)

Outputs (frequency-labelled):
  did_h1_summary_{freq}.txt
  did_h2_summary_{freq}.txt
  event_study_coefs_{freq}.csv
  figure_event_study_{freq}.png
  cohort_ftest_{freq}.txt
  sunab_coefs_{freq}.csv
  extensive_margin_{freq}.txt
  caltime_fe_{freq}.txt              
  weighted_{freq}.txt                
  table_did_main_{freq}.tex
  table_moderation_{freq}.tex
  robustness_summary_{freq}.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PANEL_PATH          = "data/interim/panel_full_semiannual.csv"
OUTPUT_DIR          = "data/outputs"
REFERENCE_PERIOD    = -1      
REFERENCE_PERIOD_ALT = -2     
CLUSTER_VAR         = "bvd_id"
ALPHA               = 0.05

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
    import statsmodels.api as sm
except ImportError:
    print("ERROR: statsmodels not installed")
    sys.exit(1)

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    print("ERROR: linearmodels not installed — run: pip install linearmodels")
    sys.exit(1)

try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("WARNING: pyfixest not installed — Sun-Abraham will be skipped")
    print("  Run: pip install pyfixest")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# =============================================================================
# HELPERS
# =============================================================================

def detect_time_spec(df):
    if "event_quarter" in df.columns:
        return {"time_var": "event_quarter",  "freq_label": "quarterly",  "period_word": "quarter",    "multiplier": 4}
    elif "event_halfyear" in df.columns:
        return {"time_var": "event_halfyear", "freq_label": "semiannual", "period_word": "half-year",  "multiplier": 2}
    elif "event_year" in df.columns:
        return {"time_var": "event_year",     "freq_label": "annual",     "period_word": "year",       "multiplier": 1}
    else:
        raise ValueError("Could not detect time variable. Expected: event_quarter, event_halfyear, or event_year")

def make_output_paths(output_dir, freq_label):
    return {
        "h1_txt":          os.path.join(output_dir, f"did_h1_summary_{freq_label}.txt"),
        "h2_txt":          os.path.join(output_dir, f"did_h2_summary_{freq_label}.txt"),
        "event_csv":       os.path.join(output_dir, f"event_study_coefs_{freq_label}.csv"),
        "table_h1_tex":    os.path.join(output_dir, f"table_did_main_{freq_label}.tex"),
        "table_h2_tex":    os.path.join(output_dir, f"table_moderation_{freq_label}.tex"),
        "event_plot_png":  os.path.join(output_dir, f"figure_event_study_{freq_label}.png"),
        "robust_csv":      os.path.join(output_dir, f"robustness_summary_{freq_label}.csv"),
        "cohort_txt":      os.path.join(output_dir, f"cohort_ftest_{freq_label}.txt"),
        "sunab_csv":       os.path.join(output_dir, f"sunab_coefs_{freq_label}.csv"),
        "extensive_txt":   os.path.join(output_dir, f"extensive_margin_{freq_label}.txt"),
        "caltime_txt":     os.path.join(output_dir, f"caltime_fe_{freq_label}.txt"),
        "weighted_txt":    os.path.join(output_dir, f"weighted_{freq_label}.txt"),
    }

def stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""

def fmt_coef(val, pval):
    if pd.isna(val): return "--"
    return f"{val:.4f}{stars(pval)}"

def fmt_se(val):
    if pd.isna(val): return ""
    return f"({val:.4f})"

def print_header(title):
    print(f"\n{'=' * 70}", flush=True)
    print(title, flush=True)
    print(f"{'=' * 70}", flush=True)

def safe_get(result_obj, var_name, statsmodels_result=False):
    try:
        if statsmodels_result:
            if var_name in result_obj.params.index:
                return (float(result_obj.params[var_name]),
                        float(result_obj.bse[var_name]),
                        float(result_obj.pvalues[var_name]))
        else:
            if var_name in result_obj.params.index:
                return (float(result_obj.params[var_name]),
                        float(result_obj.std_errors[var_name]),
                        float(result_obj.pvalues[var_name]))
    except Exception:
        pass
    return (np.nan, np.nan, np.nan)


def extract_pyfixest_event_coefs(fit_es, time_var, periods_no_ref,
                                  reference_period=REFERENCE_PERIOD):
    """Extract event-study coefficients from a pyfixest feols result.

    Tries multiple naming conventions across pyfixest versions.
    Raises a printed warning if fewer than 50 % of expected periods match.
    """
    coef_series  = fit_es.coef()
    se_series    = fit_es.se()
    pval_series  = fit_es.pvalue()
    tstat_series = fit_es.tstat()

    rows = []
    matched = 0
    unmatched_periods = []

    for p in periods_no_ref:
        candidate_names = [
            f"C({time_var}, {p}):treated",
            f"C({time_var},  {p}):treated",
            f"{time_var}::{p}:treated",
            f"{time_var}::[{p}]:treated",
            f"{time_var}::{float(p)}:treated",
            f"i({time_var},treated,ref={reference_period})::{p}",
            f"i({time_var}, treated, ref={reference_period})::{p}",
        ]

        found = None
        for nm in candidate_names:
            if nm in coef_series.index:
                found = nm
                break

        # Substring fallback
        if found is None:
            matches = [idx for idx in coef_series.index
                       if (time_var in idx) and ("treated" in idx.lower())
                       and (str(p) in idx)]
            if len(matches) == 1:
                found = matches[0]
            elif len(matches) > 1:
                found = sorted(matches, key=len)[0]

        if found is not None:
            coef  = float(coef_series[found])
            se    = float(se_series[found])
            pval  = float(pval_series[found])
            tstat = float(tstat_series[found])
            matched += 1
        else:
            coef, se, pval, tstat = np.nan, np.nan, np.nan, np.nan
            unmatched_periods.append(p)

        rows.append({
            time_var: p,
            "coefficient": coef,
            "std_error": se,
            "t_stat": tstat,
            "p_value": pval,
            "ci_lower": coef - 1.96 * se if pd.notna(se) else np.nan,
            "ci_upper": coef + 1.96 * se if pd.notna(se) else np.nan,
            "period": "pre" if p < 0 else "post",
            "coef_name_found": found,
        })

    n_expected  = len(periods_no_ref)
    match_rate  = matched / n_expected if n_expected > 0 else 0

    if match_rate < 0.5:
        print(f"\n  WARNING: only {matched}/{n_expected} event-study "
              f"periods matched ({match_rate:.0%}).", flush=True)
        print(f"    This likely indicates a pyfixest version change.", flush=True)
        print(f"    Unmatched periods: {unmatched_periods}", flush=True)
        print(f"    Available coefficient names (first 10):", flush=True)
        for name in list(coef_series.index[:10]):
            print(f"      '{name}'", flush=True)
        print(f"    EVENT-STUDY PLOT MAY BE EMPTY OR INCORRECT.", flush=True)
    elif match_rate < 1.0:
        print(f"  Note: {len(unmatched_periods)} period(s) unmatched: "
              f"{unmatched_periods}", flush=True)
    else:
        print(f"  All {matched} event-study periods matched.", flush=True)

    return pd.DataFrame(rows)


# =============================================================================
# SETUP
# =============================================================================

print("=" * 70, flush=True)
print("STEP 9b — DiD REGRESSION ANALYSIS (REVISED v2)", flush=True)
print("=" * 70, flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(PANEL_PATH):
    print(f"ERROR: panel file not found: {PANEL_PATH}", flush=True)
    sys.exit(1)

start_time = time.time()

# =============================================================================
# PHASE 1 — LOAD PANEL
# =============================================================================

print_header("PHASE 1 — LOAD PANEL")

panel = pd.read_csv(PANEL_PATH, low_memory=False)

spec        = detect_time_spec(panel)
TIME_VAR    = spec["time_var"]
FREQ_LABEL  = spec["freq_label"]
PERIOD_WORD = spec["period_word"]
MULTIPLIER  = spec["multiplier"]
OUT         = make_output_paths(OUTPUT_DIR, FREQ_LABEL)

print(f"Detected frequency: {FREQ_LABEL}", flush=True)
print(f"Time variable:      {TIME_VAR}", flush=True)
print(f"Input panel:        {PANEL_PATH}", flush=True)

panel["bvd_id"]        = panel["bvd_id"].astype(str)
panel[TIME_VAR]        = pd.to_numeric(panel[TIME_VAR], errors="coerce")
panel                  = panel.dropna(subset=[TIME_VAR]).copy()
panel[TIME_VAR]        = panel[TIME_VAR].astype(int)
panel["treated"]       = panel["treated"].astype(int)
panel["post"]          = panel["post"].astype(int)
panel["treated_x_post"] = panel["treated"] * panel["post"]

for var in CONTROL_VARS:
    if var in panel.columns:
        panel[var] = panel[var].fillna(0)

# -------------------------------------------------------------------------
# Build calendar time + cohort variables for Sun-Abraham
# -------------------------------------------------------------------------
deal_dates = pd.to_datetime(panel["deal_date"], errors="coerce")
panel["deal_year_cal"] = deal_dates.dt.year.fillna(0).astype(int)

if TIME_VAR == "event_quarter":
    panel["deal_sub"] = (deal_dates.dt.quarter.fillna(1) - 1).astype(int)
elif TIME_VAR == "event_halfyear":
    panel["deal_sub"] = (deal_dates.dt.month.fillna(1) > 6).astype(int)
else:
    panel["deal_sub"] = 0

panel["calendar_time"] = (
    panel["deal_year_cal"] * MULTIPLIER
    + panel["deal_sub"]
    + panel[TIME_VAR]
)

panel["sunab_cohort"] = np.where(
    panel["treated"] == 1,
    panel["deal_year_cal"] * MULTIPLIER + panel["deal_sub"],
    0
)

# -------------------------------------------------------------------------
# Build deal cohort groups for heterogeneity F-test
# -------------------------------------------------------------------------
panel["deal_year_cal"] = panel["deal_year_cal"].replace(0, np.nan)
panel["cohort_group"] = pd.cut(
    panel["deal_year_cal"],
    bins=[2014, 2017, 2019, 2022],
    labels=["early_15_17", "mid_18_19", "late_20_21"],
)
panel["cohort_mid"]  = (panel["cohort_group"] == "mid_18_19").astype(int)
panel["cohort_late"] = (panel["cohort_group"] == "late_20_21").astype(int)
panel["cohort_mid_x_treated_post"]  = panel["cohort_mid"]  * panel["treated_x_post"]
panel["cohort_late_x_treated_post"] = panel["cohort_late"] * panel["treated_x_post"]

# -------------------------------------------------------------------------
# Keep full panel before dropping missing novelty scores
# -------------------------------------------------------------------------
panel_full = panel.copy()
n_full = len(panel_full)

panel = panel.dropna(subset=["novelty_score_mean"]).copy()
n_dropped = n_full - len(panel)

print(f"\nFull panel (all periods): {n_full:,} observations", flush=True)
if n_dropped > 0:
    print(f"Dropped {n_dropped:,} rows with missing novelty_score_mean "
          f"({n_dropped/n_full*100:.1f}%)", flush=True)
    print(f"  These are periods where the firm filed zero patents.", flush=True)
    print(f"  Extensive margin regression in Phase 7 checks whether", flush=True)
    print(f"  acquisition affects filing rates (attrition mechanism).", flush=True)

print(f"\nNovelty regression sample: {len(panel):,} observations", flush=True)
print(f"Firms:   {panel['bvd_id'].nunique():,}", flush=True)
print(f"Treated: {panel.loc[panel['treated']==1, 'bvd_id'].nunique():,}", flush=True)
print(f"Control: {panel.loc[panel['treated']==0, 'bvd_id'].nunique():,}", flush=True)

# =============================================================================
# CHECK — TIME-INVARIANT CONTROL CONSISTENCY
# =============================================================================

print_header("CHECK — TIME-INVARIANT CONTROL CONSISTENCY")

for var in CONTROL_VARS:
    if var in panel.columns:
        n_varying = int((panel.groupby("bvd_id")[var].nunique(dropna=False) > 1).sum())
        print(f"{var:<28} varying within firm for {n_varying:>4,} entities",
              flush=True)

# =============================================================================
# PHASE 2 — DESCRIPTIVE STATISTICS
# =============================================================================

print_header(f"PHASE 2 — DESCRIPTIVE STATISTICS ({FREQ_LABEL.upper()})")

desc = (panel.groupby(["treated", "post"])["novelty_score_mean"]
        .agg(["mean", "std", "count"]).round(4))
print("\nnovelty_score_mean by treated x post:", flush=True)
print(desc.to_string(), flush=True)

mean_t_pre  = panel.loc[(panel["treated"]==1) & (panel["post"]==0), "novelty_score_mean"].mean()
mean_t_post = panel.loc[(panel["treated"]==1) & (panel["post"]==1), "novelty_score_mean"].mean()
mean_c_pre  = panel.loc[(panel["treated"]==0) & (panel["post"]==0), "novelty_score_mean"].mean()
mean_c_post = panel.loc[(panel["treated"]==0) & (panel["post"]==1), "novelty_score_mean"].mean()
raw_did     = (mean_t_post - mean_t_pre) - (mean_c_post - mean_c_pre)

print(f"\nRaw DiD (unconditional): {raw_did:+.4f}", flush=True)
print(f"  Treated delta: {mean_t_post - mean_t_pre:+.4f}", flush=True)
print(f"  Control delta: {mean_c_post - mean_c_pre:+.4f}", flush=True)

print(f"\nPanel balance (firms per event {PERIOD_WORD}):", flush=True)
print(f"{'Period':>8} {'Treated':>9} {'Control':>9} {'Total':>9}", flush=True)
print(f"{'-' * 42}", flush=True)
event_window = int(max(abs(panel[TIME_VAR].min()), abs(panel[TIME_VAR].max())))
for p in range(-event_window, event_window + 1):
    p_df = panel[panel[TIME_VAR] == p]
    nt = p_df.loc[p_df["treated"]==1, "bvd_id"].nunique()
    nc = p_df.loc[p_df["treated"]==0, "bvd_id"].nunique()
    print(f"{p:>8} {nt:>9,} {nc:>9,} {p_df['bvd_id'].nunique():>9,}", flush=True)

# =============================================================================
# PANEL INDEX (novelty regression sample)
# =============================================================================

panel_idx = panel.set_index(["bvd_id", TIME_VAR]).sort_index()

# =============================================================================
# PHASE 3 — H1 DIFFERENCE-IN-DIFFERENCES
# =============================================================================

print_header(f"PHASE 3 — H1 DiD ({FREQ_LABEL.upper()})")

# --- Model 1: Pooled OLS ---
print("\n--- Model 1: Pooled OLS ---", flush=True)

exog_m1 = ["treated", "post", "treated_x_post"] + [v for v in CONTROL_VARS if v in panel_idx.columns]
X_m1    = sm.add_constant(panel_idx[exog_m1].astype(float))
y_m1    = panel_idx["novelty_score_mean"].astype(float)

ols_m1 = sm.OLS(y_m1, X_m1).fit(
    cov_type="cluster",
    cov_kwds={"groups": panel_idx.index.get_level_values(0)}
)
b1, se1, p1 = ols_m1.params["treated_x_post"], ols_m1.bse["treated_x_post"], ols_m1.pvalues["treated_x_post"]
print(f"b3 = {b1:+.4f}  SE = {se1:.4f}  p = {p1:.4f}", flush=True)

# --- Model 2: Entity FE only ---
print("\n--- Model 2: Entity FE only ---", flush=True)

X_m2 = sm.add_constant(panel_idx[["post", "treated_x_post"]].astype(float))
m2   = PanelOLS(dependent=panel_idx["novelty_score_mean"].astype(float),
                exog=X_m2, entity_effects=True, time_effects=False,
                drop_absorbed=True, check_rank=True)
res_m2 = m2.fit(cov_type="clustered", cluster_entity=True)
b2, se2, p2 = (res_m2.params["treated_x_post"],
                res_m2.std_errors["treated_x_post"],
                res_m2.pvalues["treated_x_post"])
print(f"b3 = {b2:+.4f}  SE = {se2:.4f}  p = {p2:.4f}", flush=True)

# -------------------------------------------------------------------------
# --- Model 3: Two-Way FE — main specification ---
# -------------------------------------------------------------------------
print("\n--- Model 3: Two-Way FE (MAIN SPECIFICATION) ---", flush=True)

time_varying_controls = [v for v in ["patent_age"] if v in panel_idx.columns]
m3_vars = ["treated_x_post"] + time_varying_controls

if time_varying_controls:
    print(f"  Time-varying controls included: {time_varying_controls}", flush=True)
    print(f"  Time-invariant controls absorbed by entity FE.", flush=True)
else:
    print(f"  patent_age not found — proceeding without time-varying controls.", flush=True)

X_m3 = panel_idx[m3_vars].astype(float)
m3   = PanelOLS(dependent=panel_idx["novelty_score_mean"].astype(float),
                exog=X_m3, entity_effects=True, time_effects=True,
                drop_absorbed=True, check_rank=True)
res_m3 = m3.fit(cov_type="clustered", cluster_entity=True)

beta3      = float(res_m3.params["treated_x_post"])
beta3_se   = float(res_m3.std_errors["treated_x_post"])
beta3_t    = float(res_m3.tstats["treated_x_post"])
beta3_p    = float(res_m3.pvalues["treated_x_post"])
beta3_ci_lo = beta3 - 1.96 * beta3_se
beta3_ci_hi = beta3 + 1.96 * beta3_se

print(f"\n  b3 (treated_x_post) = {beta3:+.4f}", flush=True)
print(f"  SE                  = {beta3_se:.4f}", flush=True)
print(f"  t-stat              = {beta3_t:.4f}", flush=True)
print(f"  p-value             = {beta3_p:.4f}", flush=True)
print(f"  95% CI              = [{beta3_ci_lo:+.4f}, {beta3_ci_hi:+.4f}]", flush=True)

if beta3 < 0 and beta3_p < ALPHA:
    print("\nH1 SUPPORTED: acquisition significantly decreases novelty.", flush=True)
elif beta3_p >= ALPHA:
    print("\nH1 NOT SUPPORTED: no statistically significant effect.", flush=True)
else:
    print("\nUnexpected direction: acquisition significantly increases novelty.", flush=True)

with open(OUT["h1_txt"], "w", encoding="utf-8") as f:
    f.write(f"H1 — DiD REGRESSION RESULTS ({FREQ_LABEL})\n{'='*70}\n\n")

    f.write("COEFFICIENT SUMMARY — treated_x_post\n")
    f.write(f"{'-'*70}\n")
    f.write(f"{'Model':<30} {'Coef':>10} {'SE':>10} {'t/z':>10} "
            f"{'p-value':>10} {'Sig':>5}\n")
    f.write(f"{'-'*70}\n")

    def _sig(p):
        if pd.isna(p): return ""
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    # Model 1: Pooled OLS (statsmodels result)
    t1 = ols_m1.tvalues["treated_x_post"] if "treated_x_post" in ols_m1.tvalues.index else float("nan")
    f.write(f"{'(1) Pooled OLS':<30} {b1:>+10.4f} {se1:>10.4f} {t1:>10.4f} "
            f"{p1:>10.4f} {_sig(p1):>5}\n")

    # Model 2: Entity FE (linearmodels result)
    t2 = float(res_m2.tstats["treated_x_post"])
    f.write(f"{'(2) Entity FE':<30} {b2:>+10.4f} {se2:>10.4f} {t2:>10.4f} "
            f"{p2:>10.4f} {_sig(p2):>5}\n")

    # Model 3: Two-Way FE — main specification
    f.write(f"{'(3) Two-Way FE [MAIN]':<30} {beta3:>+10.4f} {beta3_se:>10.4f} "
            f"{beta3_t:>10.4f} {beta3_p:>10.4f} {_sig(beta3_p):>5}\n")

    f.write(f"{'-'*70}\n")
    f.write("Significance: * p<0.10  ** p<0.05  *** p<0.01\n")
    f.write("Clustered standard errors (firm level) for all models.\n")
    f.write(f"95% CI (Two-Way FE): [{beta3_ci_lo:+.4f}, {beta3_ci_hi:+.4f}]\n\n")

    f.write(f"{'='*70}\n")
    f.write("FULL SUMMARY — (1) Pooled OLS\n")
    f.write(f"{'='*70}\n")
    f.write(str(ols_m1.summary()))
    f.write(f"\n\n{'='*70}\n")
    f.write("FULL SUMMARY — (2) Entity FE\n")
    f.write(f"{'='*70}\n")
    f.write(str(res_m2.summary))
    f.write(f"\n\n{'='*70}\n")
    f.write("FULL SUMMARY — (3) Two-Way FE [MAIN SPECIFICATION]\n")
    f.write(f"{'='*70}\n")
    f.write(str(res_m3.summary))
print(f"Saved: {OUT['h1_txt']}", flush=True)

# =============================================================================
# PHASE 4 — EVENT STUDY
# =============================================================================

print_header(f"PHASE 4 — EVENT STUDY ({FREQ_LABEL.upper()})")

periods        = sorted(panel[TIME_VAR].dropna().astype(int).unique().tolist())
periods_no_ref = [p for p in periods if p != REFERENCE_PERIOD]
es_df          = pd.DataFrame()

if len(periods_no_ref) == 0:
    print("No event-study periods after excluding reference period.", flush=True)
else:
    formula_es = (
        f"novelty_score_mean ~ i({TIME_VAR}, treated, ref={REFERENCE_PERIOD}) "
        f"| bvd_id + {TIME_VAR}"
    )
    print(f"pyfixest formula:\n  {formula_es}", flush=True)

    if HAS_PYFIXEST:
        try:
            fit_es = pf.feols(formula_es, data=panel, vcov={"CRV1": "bvd_id"})

            es_df = extract_pyfixest_event_coefs(
                fit_es=fit_es, time_var=TIME_VAR, periods_no_ref=periods_no_ref,
                reference_period=REFERENCE_PERIOD
            )

            es_ref = pd.DataFrame([{
                TIME_VAR: REFERENCE_PERIOD, "coefficient": 0.0,
                "std_error": 0.0, "t_stat": 0.0, "p_value": 1.0,
                "ci_lower": 0.0, "ci_upper": 0.0,
                "period": "pre" if REFERENCE_PERIOD < 0 else "post",
                "coef_name_found": "REFERENCE_PERIOD",
            }])
            es_df = pd.concat([es_df, es_ref], ignore_index=True)
            es_df = es_df.sort_values(TIME_VAR).reset_index(drop=True)

            es_df.to_csv(OUT["event_csv"], index=False)
            print(f"Saved: {OUT['event_csv']}", flush=True)

            print(f"\n{'Period':>8} {'Coef':>10} {'SE':>10} {'p':>10} {'95% CI':>24}",
                  flush=True)
            print(f"{'-' * 80}", flush=True)
            for _, row in es_df.iterrows():
                p   = int(row[TIME_VAR])
                sig = stars(row["p_value"])
                coef_s = "nan" if pd.isna(row["coefficient"]) else f"{row['coefficient']:+10.4f}"
                se_s   = "nan" if pd.isna(row["std_error"])   else f"{row['std_error']:10.4f}"
                p_s    = "nan" if pd.isna(row["p_value"])     else f"{row['p_value']:10.4f}"
                cil    = "nan" if pd.isna(row["ci_lower"])    else f"{row['ci_lower']:+7.4f}"
                cih    = "nan" if pd.isna(row["ci_upper"])    else f"{row['ci_upper']:+7.4f}"
                print(f"{p:>8} {coef_s} {se_s} {p_s} [{cil}, {cih}] {sig}", flush=True)

            pre_df    = es_df[(es_df[TIME_VAR] < 0) & (es_df[TIME_VAR] != REFERENCE_PERIOD)]
            n_sig_pre = int((pre_df["p_value"] < ALPHA).sum())
            print(f"\nParallel-trends: {n_sig_pre}/{len(pre_df)} pre-period "
                  f"coefficients significant at alpha={ALPHA}.", flush=True)

            # ---------------------------------------------------------------
            # Formal joint pre-trends test
            # ---------------------------------------------------------------
            print(f"\n--- Formal pre-trends test (FIX 13) ---", flush=True)
            print(f"  H0: all pre-treatment event-time coefficients "
                  f"are jointly zero.", flush=True)

            try:
                pre_valid = pre_df.dropna(subset=["coefficient", "std_error"])
                pre_valid = pre_valid[pre_valid["std_error"] > 0]
                n_pre_valid = len(pre_valid)

                if n_pre_valid > 0:
                    from scipy import stats as sp_stats
                    wald_stats = (pre_valid["coefficient"].values
                                  / pre_valid["std_error"].values) ** 2
                    chi2_stat = float(np.sum(wald_stats))
                    chi2_pval = float(1 - sp_stats.chi2.cdf(chi2_stat,
                                                             df=n_pre_valid))
                    print(f"  chi2 = {chi2_stat:.3f}  (df = {n_pre_valid})  "
                          f"p = {chi2_pval:.4f}", flush=True)
                    if chi2_pval > 0.10:
                        print(f"  -> Cannot reject parallel trends "
                              f"(p > 0.10).", flush=True)
                    else:
                        print(f"  -> Pre-trends jointly significant — "
                              f"parallel trends assumption may be violated.",
                              flush=True)
                else:
                    print(f"  No valid pre-period coefficients for "
                          f"joint test.", flush=True)
            except Exception as e_pretest:
                print(f"  Pre-trends test failed: {e_pretest}", flush=True)

            print(f"\n  Note for thesis: cite Roth (2022, AER: Insights) "
                  f"for discussion of", flush=True)
            print(f"  pre-testing pitfalls and Rambachan & Roth (2023, "
                  f"ReStud) for", flush=True)
            print(f"  sensitivity analysis under violations of parallel "
                  f"trends.", flush=True)

            # ---------------------------------------------------------------
            # Event study robustness with reference at t = -2
            # ---------------------------------------------------------------
            print(f"\n--- Event study robustness: reference at "
                  f"t = {REFERENCE_PERIOD_ALT} (FIX 12) ---", flush=True)
            print(f"  Rationale: if M&A announcements precede closing, "
                  f"period t=-1 may", flush=True)
            print(f"  already reflect anticipation effects.  Setting "
                  f"ref = t=-2 allows", flush=True)
            print(f"  the t=-1 coefficient to be estimated freely.",
                  flush=True)

            try:
                periods_no_ref_alt = [p for p in periods
                                      if p != REFERENCE_PERIOD_ALT]
                formula_es_alt = (
                    f"novelty_score_mean ~ "
                    f"i({TIME_VAR}, treated, ref={REFERENCE_PERIOD_ALT}) "
                    f"| bvd_id + {TIME_VAR}"
                )

                fit_es_alt = pf.feols(
                    formula_es_alt, data=panel,
                    vcov={"CRV1": "bvd_id"},
                )

                es_alt = extract_pyfixest_event_coefs(
                    fit_es_alt, TIME_VAR, periods_no_ref_alt,
                    reference_period=REFERENCE_PERIOD_ALT,
                )

                if es_alt is not None and len(es_alt) > 0:
                    t_minus1 = es_alt[es_alt[TIME_VAR] == -1]
                    if len(t_minus1) > 0:
                        b_ant = float(t_minus1["coefficient"].values[0])
                        p_ant = float(t_minus1["p_value"].values[0])
                        print(f"\n  t=-1 coefficient: b = {b_ant:+.4f}  "
                              f"p = {p_ant:.4f} {stars(p_ant)}", flush=True)
                        if p_ant < 0.10:
                            print(f"  -> t=-1 is significant — possible "
                                  f"anticipation effect.", flush=True)
                        else:
                            print(f"  -> t=-1 is not significant — no "
                                  f"evidence of anticipation.", flush=True)
                    else:
                        print(f"  Could not extract t=-1 coefficient.",
                              flush=True)

            except Exception as e_ref:
                print(f"  Reference-period robustness failed: {e_ref}",
                      flush=True)

            # --- Event study plot ---
            if HAS_MPL and len(es_df) > 0:
                plot_df = es_df.sort_values(TIME_VAR)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.errorbar(plot_df[TIME_VAR], plot_df["coefficient"],
                            yerr=1.96 * plot_df["std_error"],
                            fmt="o-", capsize=3)
                ax.axhline(0, color="black", linewidth=1)
                ax.axvline(REFERENCE_PERIOD, color="gray",
                           linestyle="--", linewidth=1)
                ax.set_title(f"Event Study ({FREQ_LABEL})")
                ax.set_xlabel(TIME_VAR)
                ax.set_ylabel("Treatment effect on novelty score")
                fig.tight_layout()
                fig.savefig(OUT["event_plot_png"], dpi=300)
                plt.close(fig)
                print(f"Saved: {OUT['event_plot_png']}", flush=True)

        except Exception as e:
            print(f"Event study failed: {e}", flush=True)
    else:
        print("pyfixest not available — event study skipped.", flush=True)

# =============================================================================
# PHASE 5 — STAGGERED DiD ROBUSTNESS
# =============================================================================

print_header(f"PHASE 5 — STAGGERED DiD ROBUSTNESS ({FREQ_LABEL.upper()})")

# Initialise results that will be referenced in Phase 8
result_caltime  = None   # FIX 11
result_weighted = None   # FIX 14

# --- 5a: Cohort F-test ---
print("\n--- 5a: Cohort heterogeneity F-test ---", flush=True)
print("  H0: treatment effect is homogeneous across deal cohorts", flush=True)
print("  Cohorts: early (2015-2017), mid (2018-2019), late (2020-2021)",
      flush=True)

result_cohort = None

try:
    cohort_exog = ["treated_x_post", "cohort_mid_x_treated_post",
                    "cohort_late_x_treated_post"]
    cohort_panel = panel.copy()
    cohort_idx   = cohort_panel.set_index(["bvd_id", TIME_VAR]).sort_index()

    for col in cohort_exog:
        if col not in cohort_idx.columns:
            raise ValueError(f"Column {col} not found — check "
                             f"deal_date availability")

    m_cohort = PanelOLS(
        dependent=cohort_idx["novelty_score_mean"].astype(float),
        exog=cohort_idx[cohort_exog].astype(float),
        entity_effects=True, time_effects=True,
        drop_absorbed=True, check_rank=True,
    )
    result_cohort = m_cohort.fit(cov_type="clustered", cluster_entity=True)

    b_base = safe_get(result_cohort, "treated_x_post")
    b_mid  = safe_get(result_cohort, "cohort_mid_x_treated_post")
    b_late = safe_get(result_cohort, "cohort_late_x_treated_post")

    print(f"\n  Baseline (early 2015-17):  b = {b_base[0]:+.4f}  "
          f"SE = {b_base[1]:.4f}  p = {b_base[2]:.4f}", flush=True)
    print(f"  Mid cohort (2018-19):      b = {b_mid[0]:+.4f}  "
          f"SE = {b_mid[1]:.4f}  p = {b_mid[2]:.4f}", flush=True)
    print(f"  Late cohort (2020-21):     b = {b_late[0]:+.4f}  "
          f"SE = {b_late[1]:.4f}  p = {b_late[2]:.4f}", flush=True)

    try:
        f_stat = result_cohort.f_statistic.stat
        f_pval = result_cohort.f_statistic.pval
        print(f"\n  Joint F-test on cohort interactions: "
              f"F = {f_stat:.4f}  p = {f_pval:.4f}", flush=True)
        if f_pval > 0.10:
            print("  -> Cohort effects not jointly significant. "
                  "TWFE pooling defensible.", flush=True)
        else:
            print("  -> Significant cohort heterogeneity. "
                  "Interpret TWFE estimate cautiously.", flush=True)
    except Exception:
        print("  (F-test on cohort interactions not available "
              "from this fit object)", flush=True)

    with open(OUT["cohort_txt"], "w", encoding="utf-8") as f:
        f.write(f"COHORT HETEROGENEITY F-TEST ({FREQ_LABEL})\n"
                f"{'='*70}\n")
        f.write(str(result_cohort.summary))
    print(f"Saved: {OUT['cohort_txt']}", flush=True)

except Exception as e:
    print(f"  Cohort F-test failed: {e}", flush=True)

# --- 5b: Sun-Abraham estimator ---
print("\n--- 5b: Sun-Abraham estimator (pyfixest sunab) ---", flush=True)

result_sunab = None

if not HAS_PYFIXEST:
    print("  pyfixest not installed — Sun-Abraham skipped.", flush=True)
else:
    try:
        sa_panel = panel.dropna(
            subset=["calendar_time", "sunab_cohort"]
        ).copy()
        sa_panel = sa_panel[sa_panel["calendar_time"] > 0].copy()
        sa_panel["calendar_time"] = sa_panel["calendar_time"].astype(int)
        sa_panel["sunab_cohort"]  = sa_panel["sunab_cohort"].astype(int)

        if sa_panel["sunab_cohort"].nunique() < 3:
            raise ValueError("Too few cohorts for Sun-Abraham — "
                             "check deal_date column")

        formula_sa = (
            f"novelty_score_mean ~ sunab(sunab_cohort, calendar_time) "
            f"| bvd_id + calendar_time"
        )
        print(f"  Formula: {formula_sa}", flush=True)

        result_sunab = pf.feols(
            formula_sa, data=sa_panel, vcov={"CRV1": "bvd_id"},
        )

        sa_coefs = pd.DataFrame({
            "coef_name":   result_sunab.coef().index,
            "coefficient": result_sunab.coef().values,
            "std_error":   result_sunab.se().values,
            "p_value":     result_sunab.pvalue().values,
        })
        sa_coefs.to_csv(OUT["sunab_csv"], index=False)
        print(f"  Sun-Abraham coefficients: {len(sa_coefs)} terms",
              flush=True)
        print(f"Saved: {OUT['sunab_csv']}", flush=True)

        post_coefs = result_sunab.coef()[
            [c for c in result_sunab.coef().index
             if "post" in c.lower()
             or any(str(p) in c for p in range(0, event_window + 1))]
        ]
        if len(post_coefs) > 0:
            att_sunab = float(post_coefs.mean())
            print(f"  Aggregated ATT (mean of post-period coefs): "
                  f"{att_sunab:+.4f}", flush=True)
            print(f"  Compare to TWFE b3: {beta3:+.4f}", flush=True)
            if abs(att_sunab - beta3) < 0.5 * abs(beta3):
                print("  -> Sun-Abraham ATT close to TWFE. "
                      "Staggered bias likely minor.", flush=True)
            else:
                print("  -> Meaningful difference. Report Sun-Abraham "
                      "as primary robustness.", flush=True)

    except Exception as e:
        print(f"  Sun-Abraham failed: {e}", flush=True)
        print("  Cohort F-test (5a) remains the staggered DiD "
              "robustness check.", flush=True)

# -------------------------------------------------------------------------
# 5c — Calendar-time FE robustness 
# -------------------------------------------------------------------------
print("\n--- 5c: Calendar-time FE robustness (FIX 11) ---", flush=True)
print("  Event-time FEs do not absorb calendar-time shocks.", flush=True)
print("  This model adds calendar-year FEs alongside entity FEs.",
      flush=True)

try:
    cal_panel = panel.copy()
    # Reconstruct calendar year from deal_year_cal + event time
    # deal_year_cal may be NaN for controls; use deal_date instead
    _dd = pd.to_datetime(cal_panel["deal_date"], errors="coerce")
    _dy = _dd.dt.year.fillna(0).astype(int)
    cal_panel["calendar_year"] = (
        _dy + (cal_panel[TIME_VAR] / MULTIPLIER)
    ).round(0).astype(int)

    # Filter out rows where calendar_year could not be computed
    cal_panel = cal_panel[cal_panel["calendar_year"] > 0].copy()

    # Create calendar-year dummies (drop earliest as reference)
    cal_years = sorted(cal_panel["calendar_year"].unique())
    cal_ref = cal_years[0]
    for yr in cal_years:
        if yr != cal_ref:
            cal_panel[f"cal_yr_{yr}"] = (
                cal_panel["calendar_year"] == yr
            ).astype(int)

    cal_yr_dummies = [f"cal_yr_{yr}" for yr in cal_years if yr != cal_ref]

    cal_idx = cal_panel.set_index(["bvd_id", TIME_VAR]).sort_index()

    X_cal = cal_idx[["treated_x_post"] + cal_yr_dummies].astype(float)
    m_cal = PanelOLS(
        dependent=cal_idx["novelty_score_mean"].astype(float),
        exog=X_cal,
        entity_effects=True,
        time_effects=False,       # replaced by calendar-year dummies
        drop_absorbed=True,
        check_rank=True,
    )
    result_caltime = m_cal.fit(cov_type="clustered", cluster_entity=True)

    b_cal  = float(result_caltime.params["treated_x_post"])
    se_cal = float(result_caltime.std_errors["treated_x_post"])
    p_cal  = float(result_caltime.pvalues["treated_x_post"])

    print(f"\n  b3 (treated_x_post) = {b_cal:+.4f}", flush=True)
    print(f"  SE = {se_cal:.4f}   p = {p_cal:.4f} {stars(p_cal)}",
          flush=True)
    print(f"  Calendar-year dummies: {len(cal_yr_dummies)}", flush=True)
    print(f"  N = {int(result_caltime.nobs):,}", flush=True)

    pct_change = (
        ((b_cal - beta3) / abs(beta3) * 100)
        if beta3 != 0 else float("nan")
    )
    print(f"\n  Main TWFE b3:          {beta3:+.4f}", flush=True)
    print(f"  Calendar-time FE b3:   {b_cal:+.4f}  "
          f"(delta = {pct_change:+.1f}%)", flush=True)

    if abs(pct_change) < 15:
        print("  -> Calendar-time confounders do not materially "
              "alter the estimate.", flush=True)
    else:
        print("  -> Estimate shifts >15% — calendar-time shocks "
              "may matter.", flush=True)

    with open(OUT["caltime_txt"], "w", encoding="utf-8") as f:
        f.write(f"CALENDAR-TIME FE ROBUSTNESS ({FREQ_LABEL})\n"
                f"{'='*70}\n")
        f.write(str(result_caltime.summary))
    print(f"Saved: {OUT['caltime_txt']}", flush=True)

except Exception as e:
    print(f"  Calendar-time FE robustness failed: {e}", flush=True)

# -------------------------------------------------------------------------
# 5d — Precision-weighted novelty robustness 
# -------------------------------------------------------------------------
print("\n--- 5d: Precision-weighted novelty robustness (FIX 14) ---",
      flush=True)
print("  novelty_score_mean treats all firm-periods equally "
      "regardless of", flush=True)
print("  patent count.  Two approaches tested below.", flush=True)

try:
    wt_panel = panel.copy()

    if "patent_count" in wt_panel.columns:
        wt_panel["patent_count"] = (
            wt_panel["patent_count"].fillna(1).clip(lower=1)
        )
        wt_panel["log_patent_count"] = np.log(
            wt_panel["patent_count"] + 1
        )

        wt_idx = wt_panel.set_index(["bvd_id", TIME_VAR]).sort_index()

        # --- (a) WLS with patent_count weights ---
        print("\n  (a) WLS with patent_count weights...", flush=True)

        X_wt = wt_idx[["treated_x_post"]].astype(float)
        m_wt = PanelOLS(
            dependent=wt_idx["novelty_score_mean"].astype(float),
            exog=X_wt,
            entity_effects=True,
            time_effects=True,
            weights=wt_idx["patent_count"].astype(float),
            drop_absorbed=True,
            check_rank=True,
        )
        res_wt = m_wt.fit(cov_type="clustered", cluster_entity=True)

        b_wt  = float(res_wt.params["treated_x_post"])
        se_wt = float(res_wt.std_errors["treated_x_post"])
        p_wt  = float(res_wt.pvalues["treated_x_post"])

        print(f"  b3 = {b_wt:+.4f}  SE = {se_wt:.4f}  "
              f"p = {p_wt:.4f} {stars(p_wt)}", flush=True)

        result_weighted = res_wt

        # --- (b) TWFE with log(patent_count+1) covariate ---
        print("\n  (b) TWFE with log(patent_count+1) as time-varying "
              "covariate...", flush=True)

        X_lpc = wt_idx[
            ["treated_x_post", "log_patent_count"]
        ].astype(float)
        m_lpc = PanelOLS(
            dependent=wt_idx["novelty_score_mean"].astype(float),
            exog=X_lpc,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True,
            check_rank=True,
        )
        res_lpc = m_lpc.fit(cov_type="clustered", cluster_entity=True)

        b_lpc  = float(res_lpc.params["treated_x_post"])
        se_lpc = float(res_lpc.std_errors["treated_x_post"])
        p_lpc  = float(res_lpc.pvalues["treated_x_post"])

        print(f"  b3 = {b_lpc:+.4f}  SE = {se_lpc:.4f}  "
              f"p = {p_lpc:.4f} {stars(p_lpc)}", flush=True)

        # Comparison
        print(f"\n  Comparison:", flush=True)
        print(f"    Main TWFE (unweighted):    b3 = {beta3:+.4f}",
              flush=True)
        print(f"    WLS (patent-count wt):     b3 = {b_wt:+.4f}",
              flush=True)
        print(f"    TWFE + log(patent_count):  b3 = {b_lpc:+.4f}",
              flush=True)

        with open(OUT["weighted_txt"], "w", encoding="utf-8") as f:
            f.write(f"PRECISION-WEIGHTED ROBUSTNESS ({FREQ_LABEL})\n"
                    f"{'='*70}\n\n")
            f.write("(a) WLS — patent_count as analytic weights\n")
            f.write(str(res_wt.summary))
            f.write("\n\n" + "=" * 70 + "\n\n")
            f.write("(b) TWFE — log(patent_count+1) as covariate\n")
            f.write(str(res_lpc.summary))
        print(f"Saved: {OUT['weighted_txt']}", flush=True)

    else:
        print("  patent_count column not found — skipping.", flush=True)
        print("  Ensure step7c outputs patent_count alongside "
              "novelty_score_mean.", flush=True)

except Exception as e:
    print(f"  Precision-weighted robustness failed: {e}", flush=True)

# =============================================================================
# PHASE 6 — H2 KNOWLEDGE OVERLAP MODERATION 
# =============================================================================

print_header(f"PHASE 6 — H2 KNOWLEDGE OVERLAP MODERATION ({FREQ_LABEL.upper()})")

result_m4 = None

if "knowledge_overlap" not in panel.columns:
    print("knowledge_overlap not found — skipping H2.", flush=True)
else:
    mod_panel = panel.dropna(subset=["knowledge_overlap"]).copy()

    if len(mod_panel) == 0:
        print("No rows with non-missing knowledge_overlap — skipping H2.",
              flush=True)
    else:
        # Build all three interaction terms
        mod_panel["overlap_x_post"] = (
            mod_panel["knowledge_overlap"] * mod_panel["post"]
        )
        mod_panel["overlap_x_treated_post"] = (
            mod_panel["knowledge_overlap"] * mod_panel["treated_x_post"]
        )

        mod_idx = mod_panel.set_index(["bvd_id", TIME_VAR]).sort_index()

        # FIX 1: include overlap_x_post to avoid omitted-variable bias
        exog_m4 = ["treated_x_post", "overlap_x_post",
                    "overlap_x_treated_post"]

        m4 = PanelOLS(
            dependent=mod_idx["novelty_score_mean"].astype(float),
            exog=mod_idx[exog_m4].astype(float),
            entity_effects=True, time_effects=True,
            drop_absorbed=True, check_rank=True,
        )
        result_m4 = m4.fit(cov_type="clustered", cluster_entity=True)

        b3_h2  = safe_get(result_m4, "treated_x_post")
        gamma3 = safe_get(result_m4, "overlap_x_post")
        gamma4 = safe_get(result_m4, "overlap_x_treated_post")

        print(f"\n  b3  (treated_x_post)         = {b3_h2[0]:+.4f}  "
              f"SE = {b3_h2[1]:.4f}  p = {b3_h2[2]:.4f}", flush=True)
        print(f"  g3  (overlap_x_post)         = {gamma3[0]:+.4f}  "
              f"SE = {gamma3[1]:.4f}  p = {gamma3[2]:.4f}", flush=True)
        print(f"  g4  (overlap_x_treated_post) = {gamma4[0]:+.4f}  "
              f"SE = {gamma4[1]:.4f}  p = {gamma4[2]:.4f}", flush=True)

        with open(OUT["h2_txt"], "w", encoding="utf-8") as f:
            f.write(f"H2 — MODERATION SPECIFICATION ({FREQ_LABEL})\n"
                    f"{'='*70}\n")
            f.write(str(result_m4.summary))
        print(f"Saved: {OUT['h2_txt']}", flush=True)

# =============================================================================
# PHASE 7 — EXTENSIVE MARGIN: PATENT COUNT REGRESSION 
# =============================================================================

print_header(f"PHASE 7 — EXTENSIVE MARGIN: PATENT COUNT ({FREQ_LABEL.upper()})")

result_extensive = None

if "patent_count" not in panel_full.columns:
    print("patent_count column not found in panel — skipping extensive "
          "margin.", flush=True)
    print("  Check that step7d_build_panel.py outputs patent_count.",
          flush=True)
else:
    ext_panel = panel_full.copy()
    ext_panel["patent_count"] = (
        ext_panel["patent_count"].fillna(0).astype(float)
    )

    print(f"  Full panel observations: {len(ext_panel):,}", flush=True)
    print(f"  Zero-patent periods:     "
          f"{(ext_panel['patent_count']==0).sum():,} "
          f"({(ext_panel['patent_count']==0).mean()*100:.1f}%)", flush=True)
    print(f"  Mean patents per period: "
          f"{ext_panel['patent_count'].mean():.2f}", flush=True)

    desc_ext = (ext_panel.groupby(["treated", "post"])["patent_count"]
                .agg(["mean", "std", "count"]).round(3))
    print("\n  patent_count by treated x post:", flush=True)
    print(desc_ext.to_string(), flush=True)

    ext_idx = ext_panel.set_index(["bvd_id", TIME_VAR]).sort_index()

    m_ext = PanelOLS(
        dependent=ext_idx["patent_count"].astype(float),
        exog=ext_idx[["treated_x_post"]].astype(float),
        entity_effects=True, time_effects=True,
        drop_absorbed=True, check_rank=True,
    )
    result_extensive = m_ext.fit(cov_type="clustered", cluster_entity=True)

    ext_b3 = safe_get(result_extensive, "treated_x_post")
    print(f"\n  b3 (treated_x_post) = {ext_b3[0]:+.4f}  "
          f"SE = {ext_b3[1]:.4f}  p = {ext_b3[2]:.4f}  "
          f"{stars(ext_b3[2])}", flush=True)

    if ext_b3[2] < ALPHA and ext_b3[0] < 0:
        print("\n  WARNING: Acquisition significantly reduces patent "
              "filing rate.", flush=True)
        print("  The novelty regression (Phase 3) is estimated on a "
              "selected", flush=True)
        print("  subsample of firms that continued filing "
              "post-acquisition.", flush=True)
        print("  Acknowledge this selection in the paper and interpret "
              "the", flush=True)
        print("  main DiD estimate as conditional on continued filing.",
              flush=True)
    elif ext_b3[2] >= ALPHA:
        print("\n  No significant effect on patent filing rate.",
              flush=True)
        print("  Selection concern is empirically small — novelty "
              "regression", flush=True)
        print("  not meaningfully affected by extensive margin "
              "attrition.", flush=True)

    with open(OUT["extensive_txt"], "w", encoding="utf-8") as f:
        f.write(f"EXTENSIVE MARGIN — PATENT COUNT REGRESSION "
                f"({FREQ_LABEL})\n")
        f.write("=" * 70 + "\n")
        f.write(f"Full panel: {len(ext_panel):,} observations\n")
        f.write(f"DV: patent_count (0 for zero-patent periods)\n")
        f.write(f"Specification: TWFE, entity + time FEs, "
                f"clustered SE\n\n")
        f.write(str(result_extensive.summary))
    print(f"Saved: {OUT['extensive_txt']}", flush=True)

# =============================================================================
# PHASE 8 — ROBUSTNESS SUMMARY
# =============================================================================

print_header(f"PHASE 8 — ROBUSTNESS SUMMARY ({FREQ_LABEL.upper()})")

robust_rows = [
    {"specification": f"Main TWFE ({FREQ_LABEL})",
     "beta3": beta3, "std_error": beta3_se,
     "p_value": beta3_p, "n_obs": int(res_m3.nobs)},
]

if result_m4 is not None:
    b3_h2_val = safe_get(result_m4, "treated_x_post")
    robust_rows.append({
        "specification": (f"Moderation TWFE — with overlap_x_post "
                          f"({FREQ_LABEL})"),
        "beta3": b3_h2_val[0], "std_error": b3_h2_val[1],
        "p_value": b3_h2_val[2], "n_obs": int(result_m4.nobs),
    })

if result_cohort is not None:
    b_base_val = safe_get(result_cohort, "treated_x_post")
    robust_rows.append({
        "specification": (f"Cohort het. test — early cohort baseline "
                          f"({FREQ_LABEL})"),
        "beta3": b_base_val[0], "std_error": b_base_val[1],
        "p_value": b_base_val[2], "n_obs": int(result_cohort.nobs),
    })

if result_extensive is not None:
    ext_val = safe_get(result_extensive, "treated_x_post")
    robust_rows.append({
        "specification": (f"Extensive margin — patent count "
                          f"({FREQ_LABEL})"),
        "beta3": ext_val[0], "std_error": ext_val[1],
        "p_value": ext_val[2], "n_obs": int(result_extensive.nobs),
    })

# Calendar-time FE
if result_caltime is not None:
    b_cal_val = safe_get(result_caltime, "treated_x_post")
    robust_rows.append({
        "specification": f"Calendar-year FE ({FREQ_LABEL})",
        "beta3": b_cal_val[0], "std_error": b_cal_val[1],
        "p_value": b_cal_val[2], "n_obs": int(result_caltime.nobs),
    })

# Precision-weighted
if result_weighted is not None:
    b_wt_val = safe_get(result_weighted, "treated_x_post")
    robust_rows.append({
        "specification": f"WLS patent-count weighted ({FREQ_LABEL})",
        "beta3": b_wt_val[0], "std_error": b_wt_val[1],
        "p_value": b_wt_val[2], "n_obs": int(result_weighted.nobs),
    })

robust_df = pd.DataFrame(robust_rows)
robust_df.to_csv(OUT["robust_csv"], index=False)
print(f"Saved: {OUT['robust_csv']}", flush=True)
print(robust_df.to_string(index=False), flush=True)

# =============================================================================
# PHASE 9 — LATEX TABLES
# =============================================================================

print_header(f"PHASE 9 — LATEX TABLES ({FREQ_LABEL.upper()})")

c1 = safe_get(ols_m1, "treated_x_post", statsmodels_result=True)
c2 = safe_get(res_m2, "treated_x_post")
c3 = safe_get(res_m3, "treated_x_post")

latex_h1 = (
    "\\begin{table}[htbp]\n"
    "\\centering\n"
    f"\\caption{{Difference-in-Differences Regression Results ({FREQ_LABEL})}}\n"
    f"\\label{{tab:did_main_{FREQ_LABEL}}}\n"
    "\\begin{tabular}{lccc}\n"
    "\\hline\n"
    "& (1) Pooled OLS & (2) Entity FE & (3) Two-Way FE \\\\\n"
    "\\hline\n"
    f"Treated $\\times$ Post & {fmt_coef(c1[0], c1[2])} & "
    f"{fmt_coef(c2[0], c2[2])} & {fmt_coef(c3[0], c3[2])} \\\\\n"
    f"& {fmt_se(c1[1])} & {fmt_se(c2[1])} & {fmt_se(c3[1])} \\\\\n"
    "\\hline\n"
    "Firm FE        & No  & Yes & Yes \\\\\n"
    "Time FE        & No  & No  & Yes \\\\\n"
    "Clustered SE   & Yes & Yes & Yes \\\\\n"
    f"Observations   & {int(ols_m1.nobs):,} & {int(res_m2.nobs):,} & "
    f"{int(res_m3.nobs):,} \\\\\n"
    "\\hline\n"
    "\\multicolumn{4}{p{12cm}}{\\footnotesize Dependent variable: firm-period "
    "mean patent novelty score (conditional on filing). Standard errors "
    "clustered at the firm level. Column (3) is the main specification with "
    "entity and time fixed effects. Time-invariant controls "
    "(log\\_target\\_pre\\_patents, log\\_deal\\_value, etc.) are absorbed by "
    "entity FEs; patent\\_age is retained as a time-varying covariate that "
    "increments each period and is therefore not absorbed (FIX~16). "
    "Significance: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.} \\\\\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open(OUT["table_h1_tex"], "w", encoding="utf-8") as f:
    f.write(latex_h1)
print(f"Saved: {OUT['table_h1_tex']}", flush=True)

if result_m4 is not None:
    c3_main = safe_get(res_m3,    "treated_x_post")
    c4_b3   = safe_get(result_m4, "treated_x_post")
    c4_g3   = safe_get(result_m4, "overlap_x_post")
    c4_g4   = safe_get(result_m4, "overlap_x_treated_post")

    latex_h2 = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{Knowledge Overlap Moderation Results ({FREQ_LABEL})}}\n"
        f"\\label{{tab:moderation_{FREQ_LABEL}}}\n"
        "\\begin{tabular}{lcc}\n"
        "\\hline\n"
        "& (1) Main DiD & (2) Moderation \\\\\n"
        "\\hline\n"
        f"Treated $\\times$ Post & "
        f"{fmt_coef(c3_main[0], c3_main[2])} & "
        f"{fmt_coef(c4_b3[0], c4_b3[2])} \\\\\n"
        f"& {fmt_se(c3_main[1])} & {fmt_se(c4_b3[1])} \\\\\n"
        f"Overlap $\\times$ Post & -- & "
        f"{fmt_coef(c4_g3[0], c4_g3[2])} \\\\\n"
        f"&  & {fmt_se(c4_g3[1])} \\\\\n"
        f"Overlap $\\times$ Treated $\\times$ Post & -- & "
        f"{fmt_coef(c4_g4[0], c4_g4[2])} \\\\\n"
        f"&  & {fmt_se(c4_g4[1])} \\\\\n"
        "\\hline\n"
        "Firm FE       & Yes & Yes \\\\\n"
        "Time FE       & Yes & Yes \\\\\n"
        "Clustered SE  & Yes & Yes \\\\\n"
        f"Observations  & {int(res_m3.nobs):,} & "
        f"{int(result_m4.nobs):,} \\\\\n"
        "\\hline\n"
        "\\multicolumn{3}{p{11cm}}{\\footnotesize Column (2) includes all "
        "lower-order interaction terms. Overlap $\\times$ Post captures "
        "differential novelty trends for firms with higher pre-merger "
        "knowledge overlap that are not absorbed by entity or time fixed "
        "effects. $\\gamma_4$ = Overlap $\\times$ Treated $\\times$ Post is "
        "the moderation coefficient of interest. Standard errors clustered "
        "at the firm level. "
        "Significance: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.} \\\\\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(OUT["table_h2_tex"], "w", encoding="utf-8") as f:
        f.write(latex_h2)
    print(f"Saved: {OUT['table_h2_tex']}", flush=True)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

elapsed = time.time() - start_time

print(f"\n{'=' * 70}", flush=True)
print("STEP 9b COMPLETE — DiD ANALYSIS (REVISED v2)", flush=True)
print(f"{'=' * 70}", flush=True)
print(f"Frequency:    {FREQ_LABEL}", flush=True)
print(f"Elapsed time: {elapsed/60:.1f} minutes", flush=True)

print(f"\nH1 main result:", flush=True)
print(f"  b3 (treated_x_post) = {beta3:+.4f}  "
      f"(p = {beta3_p:.4f}  {stars(beta3_p)})", flush=True)

if result_m4 is not None:
    g4_val = safe_get(result_m4, "overlap_x_treated_post")
    print(f"\nH2 main result:", flush=True)
    print(f"  g4 (overlap_x_treated_post) = {g4_val[0]:+.4f}  "
          f"(p = {g4_val[2]:.4f}  {stars(g4_val[2])})", flush=True)

if result_extensive is not None:
    ext_val = safe_get(result_extensive, "treated_x_post")
    print(f"\nExtensive margin:", flush=True)
    print(f"  b3 on patent_count = {ext_val[0]:+.4f}  "
          f"(p = {ext_val[2]:.4f}  {stars(ext_val[2])})", flush=True)

if result_caltime is not None:
    cal_val = safe_get(result_caltime, "treated_x_post")
    print(f"\nCalendar-time FE (FIX 11):", flush=True)
    print(f"  b3 = {cal_val[0]:+.4f}  "
          f"(p = {cal_val[2]:.4f}  {stars(cal_val[2])})", flush=True)

if result_weighted is not None:
    wt_val = safe_get(result_weighted, "treated_x_post")
    print(f"\nPrecision-weighted (FIX 14):", flush=True)
    print(f"  b3 = {wt_val[0]:+.4f}  "
          f"(p = {wt_val[2]:.4f}  {stars(wt_val[2])})", flush=True)

print("\nOutput files:", flush=True)
for key, path in OUT.items():
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {os.path.basename(path):<50} {size_kb:>8.1f} KB",
              flush=True)
