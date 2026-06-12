"""
Step 9a — EDA / Descriptive Statistics (multi-frequency aware)

Builds descriptive statistics, diagnostics, LaTeX tables, and figures for:
- quarterly panels
- semiannual panels
- annual panels

This script assumes Step 7d has produced:
- data/interim/panel_full.csv
- data/interim/panel_full_quarterly.csv
- data/interim/panel_full_semiannual.csv
- data/interim/panel_full_annual.csv

Core non-panel inputs:
- data/interim/patents_eligible_overlap.csv
- data/interim/psm_diagnostics.csv
- data/interim/control_psm_vars.csv
- data/interim/psm_matches.csv

Outputs:
- data/outputs/eda_summary_<freq>.txt
- data/outputs/table_descriptive_stats_<freq>.tex
- data/outputs/table_balance_<freq>.tex
- data/outputs/figure_novelty_trajectory_<freq>.png
- data/outputs/figure_patent_counts_<freq>.png
- data/outputs/figure_deal_year_dist_<freq>.png
- data/outputs/figure_overlap_dist_<freq>.png
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "data/outputs"

# Common inputs
PATENTS_ELIGIBLE_OVERLAP_PATH = "data/interim/patents_eligible_overlap.csv"
PSM_DIAGNOSTICS_PATH = "data/interim/psm_diagnostics.csv"
CONTROL_PSM_PATH = "data/interim/control_psm_vars.csv"
PSM_MATCHES_PATH = "data/interim/psm_matches.csv"

# Panel inputs
PANEL_PATHS = {
    "quarterly": [
        "data/interim/panel_full_quarterly.csv",
        "data/interim/panel_full.csv",  # fallback / backward compatibility
    ],
    "semiannual": [
        "data/interim/panel_full_semiannual.csv",
    ],
    "annual": [
        "data/interim/panel_full_annual.csv",
    ],
}

ALPHA = 0.05
CPC_CLASSES = ["G06F", "H04L", "H01L", "G06N"]

# =============================================================================
# HELPERS
# =============================================================================

def print_line(char="=", n=70):
    print(char * n, flush=True)

def detect_time_spec(df):
    if "event_quarter" in df.columns:
        return {
            "time_var": "event_quarter",
            "freq_label": "quarterly",
            "period_word": "quarter",
        }
    elif "event_halfyear" in df.columns:
        return {
            "time_var": "event_halfyear",
            "freq_label": "semiannual",
            "period_word": "half-year",
        }
    elif "event_year" in df.columns:
        return {
            "time_var": "event_year",
            "freq_label": "annual",
            "period_word": "year",
        }
    else:
        raise ValueError(
            "Could not detect time variable. Expected one of: "
            "event_quarter, event_halfyear, event_year"
        )

def mode_or_nan(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().index[0]

def safe_read_csv(path, **kwargs):
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    return None

def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def latex_escape(text):
    if pd.isna(text):
        return ""
    text = str(text)
    reps = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in reps.items():
        text = text.replace(k, v)
    return text

def std_mean_diff(treated_vals, control_vals):
    """
    Standardized mean difference.
    """
    treated_vals = pd.Series(treated_vals).dropna().astype(float)
    control_vals = pd.Series(control_vals).dropna().astype(float)

    if len(treated_vals) == 0 or len(control_vals) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mt = treated_vals.mean()
    mc = control_vals.mean()
    vt = treated_vals.var(ddof=1) if len(treated_vals) > 1 else 0
    vc = control_vals.var(ddof=1) if len(control_vals) > 1 else 0
    pooled_sd = np.sqrt((vt + vc) / 2) if (vt + vc) > 0 else np.nan
    smd = (mt - mc) / pooled_sd if pd.notna(pooled_sd) and pooled_sd > 0 else np.nan

    return mt, mc, mt - mc, smd

def describe_series(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
            "skew": np.nan,
            "n": 0,
        }
    return {
        "mean": x.mean(),
        "median": x.median(),
        "std": x.std(ddof=1),
        "p10": x.quantile(0.10),
        "p25": x.quantile(0.25),
        "p75": x.quantile(0.75),
        "p90": x.quantile(0.90),
        "min": x.min(),
        "max": x.max(),
        "skew": x.skew(),
        "n": len(x),
    }

def format_money_millions(m):
    if pd.isna(m):
        return "NA"
    return f"${m:,.0f}M"

def save_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def build_output_paths(freq_label):
    return {
        "summary_txt": os.path.join(OUTPUT_DIR, f"eda_summary_{freq_label}.txt"),
        "table_desc_tex": os.path.join(OUTPUT_DIR, f"table_descriptive_stats_{freq_label}.tex"),
        "table_balance_tex": os.path.join(OUTPUT_DIR, f"table_balance_{freq_label}.tex"),
        "fig_novelty": os.path.join(OUTPUT_DIR, f"figure_novelty_trajectory_{freq_label}.png"),
        "fig_counts": os.path.join(OUTPUT_DIR, f"figure_patent_counts_{freq_label}.png"),
        "fig_deal_year": os.path.join(OUTPUT_DIR, f"figure_deal_year_dist_{freq_label}.png"),
        "fig_overlap": os.path.join(OUTPUT_DIR, f"figure_overlap_dist_{freq_label}.png"),
    }

def save_latex_table_desc(path, freq_label, sample_counts, novelty_table):
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Descriptive statistics ({freq_label})}}")
    latex.append(rf"\label{{tab:descriptive_{freq_label}}}")
    latex.append(r"\begin{tabular}{lrr}")
    latex.append(r"\hline")
    latex.append(r"Statistic & Value \\")
    latex.append(r"\hline")
    latex.append(rf"Eligible treated firms & {sample_counts['treated_firms']:,} \\")
    latex.append(rf"Matched pairs & {sample_counts['matched_pairs']:,} \\")
    latex.append(rf"Panel firms & {sample_counts['panel_firms']:,} \\")
    latex.append(rf"Panel observations & {sample_counts['panel_obs']:,} \\")
    latex.append(rf"Overall mean novelty & {novelty_table['overall_mean']:.4f} \\")
    latex.append(rf"Treated pre-merger novelty & {novelty_table['treated_pre']:.4f} \\")
    latex.append(rf"Treated post-merger novelty & {novelty_table['treated_post']:.4f} \\")
    latex.append(rf"Control pre-merger novelty & {novelty_table['control_pre']:.4f} \\")
    latex.append(rf"Control post-merger novelty & {novelty_table['control_post']:.4f} \\")
    latex.append(rf"Raw DiD & {novelty_table['raw_did']:+.4f} \\")
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

def save_latex_table_balance(path, balance_df, freq_label):
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{Post-match balance ({freq_label})}}")
    latex.append(rf"\label{{tab:balance_{freq_label}}}")
    latex.append(r"\begin{tabular}{lrrrr}")
    latex.append(r"\hline")
    latex.append(r"Variable & Treated & Control & SMD & Status \\")
    latex.append(r"\hline")

    if len(balance_df) == 0:
        latex.append(r"No post-match balance rows available & & & & \\")
    else:
        for _, row in balance_df.iterrows():
            label = latex_escape(row["variable"])
            t = "" if pd.isna(row["treated_mean"]) else f"{row['treated_mean']:.4f}"
            c = "" if pd.isna(row["control_mean"]) else f"{row['control_mean']:.4f}"
            smd = "" if pd.isna(row["smd"]) else f"{row['smd']:.4f}"
            status = latex_escape(row["status"])
            latex.append(rf"{label} & {t} & {c} & {smd} & {status} \\")
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

def ci_bound_diff(t, c):
    """
    Returns (diff, 95% CI half-width, n_t, n_c)
    """
    t = pd.Series(t).dropna().astype(float)
    c = pd.Series(c).dropna().astype(float)
    if len(t) == 0 or len(c) == 0:
        return np.nan, np.nan, len(t), len(c)
    diff = t.mean() - c.mean()
    se = np.sqrt((t.var(ddof=1) / len(t) if len(t) > 1 else 0) + (c.var(ddof=1) / len(c) if len(c) > 1 else 0))
    ci = 1.96 * se
    return diff, ci, len(t), len(c)

# =============================================================================
# LOAD COMMON INPUTS
# =============================================================================

print_line()
print("STEP 9a — EDA / DESCRIPTIVE STATISTICS (MULTI-FREQUENCY)")
print_line()

os.makedirs(OUTPUT_DIR, exist_ok=True)

required_common = [
    PATENTS_ELIGIBLE_OVERLAP_PATH,
    CONTROL_PSM_PATH,
    PSM_MATCHES_PATH,
]

for path in required_common:
    if not os.path.exists(path):
        print(f"ERROR: missing required common input: {path}", flush=True)
        sys.exit(1)

eligible = pd.read_csv(PATENTS_ELIGIBLE_OVERLAP_PATH, low_memory=False)
eligible["bvd_id"] = eligible["bvd_id"].astype(str)
eligible["filing_date"] = pd.to_datetime(eligible["filing_date"], errors="coerce")
eligible["deal_date"] = pd.to_datetime(eligible["deal_date"], errors="coerce")

if "event_quarter" in eligible.columns:
    eligible["event_quarter"] = pd.to_numeric(eligible["event_quarter"], errors="coerce")

if "cpc_4digit" not in eligible.columns and "cpc_code" in eligible.columns:
    eligible["cpc_4digit"] = eligible["cpc_code"].astype(str).str[:4].str.upper()

control_psm = pd.read_csv(CONTROL_PSM_PATH, low_memory=False)
psm_matches = pd.read_csv(PSM_MATCHES_PATH, low_memory=False)
psm_diag = safe_read_csv(PSM_DIAGNOSTICS_PATH, low_memory=False)

# =============================================================================
# COMMON (NON-PANEL) SUMMARIES
# =============================================================================

# One row per treated firm for firm-level summaries
firm_info_cols = [
    "bvd_id", "deal_date", "knowledge_overlap",
    "log_target_pre_patents", "patent_age",
    "log_deal_value", "deal_value_missing",
    "log_acquiror_assets", "acquiror_assets_missing",
    "log_acquiror_pre_patents", "cpc_4digit"
]
firm_info_cols = [c for c in firm_info_cols if c in eligible.columns]

firm_info = eligible[firm_info_cols].drop_duplicates(subset=["bvd_id"]).copy()
firm_info["deal_year"] = pd.to_datetime(firm_info["deal_date"], errors="coerce").dt.year

# Sample overview
treated_firms_n = firm_info["bvd_id"].nunique()
total_patents_n = len(eligible)
patents_per_firm = eligible.groupby("bvd_id")["publication_number"].count()

pre_patents_n = int(eligible.get("pre_merger", pd.Series(dtype=float)).fillna(False).sum()) if "pre_merger" in eligible.columns else int((eligible["event_quarter"] < 0).sum())
post_patents_n = int(eligible.get("post_merger", pd.Series(dtype=float)).fillna(False).sum()) if "post_merger" in eligible.columns else int((eligible["event_quarter"] >= 0).sum())

# CPC distribution (dominant CPC per firm)
if "cpc_4digit" in eligible.columns:
    dominant_cpc_firm = (
        eligible.groupby("bvd_id")["cpc_4digit"]
        .agg(mode_or_nan)
        .rename("dominant_cpc")
        .reset_index()
    )
    cpc_dist = dominant_cpc_firm["dominant_cpc"].value_counts(dropna=True)
else:
    dominant_cpc_firm = pd.DataFrame(columns=["bvd_id", "dominant_cpc"])
    cpc_dist = pd.Series(dtype=int)

# Patent novelty distributions
if "novelty_score" in eligible.columns:
    novelty_all = describe_series(eligible["novelty_score"])
    novelty_pre = describe_series(eligible.loc[eligible["event_quarter"] < 0, "novelty_score"]) if "event_quarter" in eligible.columns else describe_series([])
    novelty_post = describe_series(eligible.loc[eligible["event_quarter"] >= 0, "novelty_score"]) if "event_quarter" in eligible.columns else describe_series([])

    novelty_by_cpc = []
    for cpc in CPC_CLASSES:
        vals = eligible.loc[eligible["cpc_4digit"] == cpc, "novelty_score"] if "cpc_4digit" in eligible.columns else pd.Series(dtype=float)
        d = describe_series(vals)
        novelty_by_cpc.append({
            "cpc": cpc,
            "mean": d["mean"],
            "median": d["median"],
            "std": d["std"],
            "n": d["n"],
        })
    novelty_by_cpc_df = pd.DataFrame(novelty_by_cpc)
else:
    novelty_all = novelty_pre = novelty_post = describe_series([])
    novelty_by_cpc_df = pd.DataFrame(columns=["cpc", "mean", "median", "std", "n"])

# Deal characteristics
deal_year_counts = firm_info["deal_year"].value_counts(dropna=True).sort_index()

# Deal value / acquiror assets
deal_val = firm_info["log_deal_value"] if "log_deal_value" in firm_info.columns else pd.Series(dtype=float)
acq_assets = firm_info["log_acquiror_assets"] if "log_acquiror_assets" in firm_info.columns else pd.Series(dtype=float)

# Overlap
overlap_vals = firm_info["knowledge_overlap"] if "knowledge_overlap" in firm_info.columns else pd.Series(dtype=float)
overlap_valid = overlap_vals.dropna()

# Pre-match diagnostics
pre_match_rows = []
if psm_diag is not None:
    # expected treated columns from your diagnostics pipeline
    treated_vars = ["log_target_pre_patents", "pre_novelty_mean", "patent_age"]
    for var in treated_vars:
        if var in psm_diag.columns and var in control_psm.columns:
            mt, mc, diff, smd = std_mean_diff(psm_diag[var], control_psm[var])
            pre_match_rows.append({
                "variable": var,
                "treated_mean": mt,
                "control_mean": mc,
                "diff": diff,
                "smd": smd
            })
pre_match_df = pd.DataFrame(pre_match_rows)

# Post-match balance
post_match_rows = []
match_var_map = [
    ("patent_age (not matched — covariate)", "t_patent_age", "c_patent_age"),
    ("log_target_pre_patents", "t_log_pre_patents", "c_log_pre_patents"),
    ("pre_novelty_mean", "t_pre_novelty", "c_pre_novelty"),
]
for label, tcol, ccol in match_var_map:
    if tcol in psm_matches.columns and ccol in psm_matches.columns:
        mt, mc, diff, smd = std_mean_diff(psm_matches[tcol], psm_matches[ccol])

        if pd.isna(smd):
            status = "NA"
        elif abs(smd) < 0.10:
            status = "Good"
        elif abs(smd) < 0.25:
            status = "Acceptable"
        else:
            status = "Poor"

        post_match_rows.append({
            "variable": label,
            "treated_mean": mt,
            "control_mean": mc,
            "smd": smd,
            "status": status
        })
post_match_df = pd.DataFrame(post_match_rows)

# =============================================================================
# FIGURES COMMON ACROSS FREQUENCIES
# =============================================================================

# Save deal year figure once per frequency label later (same content, different filename)
def plot_deal_year_dist(path):
    plt.figure(figsize=(8, 4.5))
    deal_year_counts.plot(kind="bar")
    plt.title("Deals per year")
    plt.xlabel("Deal year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_overlap_dist(path):
    plt.figure(figsize=(8, 4.5))
    if len(overlap_valid) > 0:
        plt.hist(overlap_valid, bins=15)
    plt.title("Knowledge overlap distribution")
    plt.xlabel("Knowledge overlap")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# =============================================================================
# MAIN LOOP BY PANEL FREQUENCY
# =============================================================================

for requested_freq, candidate_paths in PANEL_PATHS.items():
    panel_path = find_existing_path(candidate_paths)
    if panel_path is None:
        print(f"\nSkipping {requested_freq}: no panel file found.", flush=True)
        continue

    print_line()
    print(f"RUNNING EDA FOR PANEL: {panel_path}")
    print_line()

    panel = pd.read_csv(panel_path, low_memory=False)
    panel["bvd_id"] = panel["bvd_id"].astype(str)

    spec = detect_time_spec(panel)
    TIME_VAR = spec["time_var"]
    FREQ_LABEL = spec["freq_label"]
    PERIOD_WORD = spec["period_word"]
    OUT = build_output_paths(FREQ_LABEL)

    panel[TIME_VAR] = pd.to_numeric(panel[TIME_VAR], errors="coerce")
    panel = panel.dropna(subset=[TIME_VAR]).copy()
    panel[TIME_VAR] = panel[TIME_VAR].astype(int)

    panel["treated"] = panel["treated"].astype(int)
    panel["post"] = panel["post"].astype(int)
    panel["treated_x_post"] = panel["treated"] * panel["post"]

    # =====================================================================
    # PANEL-LEVEL SUMMARIES
    # =====================================================================

    panel_obs = len(panel)
    panel_firms = panel["bvd_id"].nunique()
    treated_panel_firms = panel.loc[panel["treated"] == 1, "bvd_id"].nunique()
    control_panel_firms = panel.loc[panel["treated"] == 0, "bvd_id"].nunique()

    periods_per_firm = panel.groupby("bvd_id")[TIME_VAR].nunique()

    coverage_rows = []
    event_window = int(max(abs(panel[TIME_VAR].min()), abs(panel[TIME_VAR].max())))
    for p in range(-event_window, event_window + 1):
        tmp = panel[panel[TIME_VAR] == p]
        nt = tmp.loc[tmp["treated"] == 1, "bvd_id"].nunique()
        nc = tmp.loc[tmp["treated"] == 0, "bvd_id"].nunique()
        coverage_rows.append({
            "period": p,
            "treated": nt,
            "control": nc,
            "total": tmp["bvd_id"].nunique(),
            "coverage_pct": tmp["bvd_id"].nunique() / panel["bvd_id"].nunique() * 100
        })
    coverage_df = pd.DataFrame(coverage_rows)

    # Panel novelty
    panel_novelty = describe_series(panel["novelty_score_mean"])

    grouped_panel = (
        panel.groupby(["treated", "post"])["novelty_score_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    def get_group_mean(treated, post):
        tmp = grouped_panel[(grouped_panel["treated"] == treated) & (grouped_panel["post"] == post)]
        return float(tmp["mean"].iloc[0]) if len(tmp) else np.nan

    treated_pre = get_group_mean(1, 0)
    treated_post = get_group_mean(1, 1)
    control_pre = get_group_mean(0, 0)
    control_post = get_group_mean(0, 1)
    raw_did = (treated_post - treated_pre) - (control_post - control_pre)

    novelty_by_cpc_panel_rows = []
    if "dominant_cpc" in panel.columns:
        for cpc in CPC_CLASSES:
            sub = panel[panel["dominant_cpc"] == cpc]
            if len(sub) > 0:
                novelty_by_cpc_panel_rows.append({
                    "cpc": cpc,
                    "mean": sub["novelty_score_mean"].mean(),
                    "std": sub["novelty_score_mean"].std(),
                    "firms": sub["bvd_id"].nunique()
                })
    novelty_by_cpc_panel_df = pd.DataFrame(novelty_by_cpc_panel_rows)

    # Filing activity by event period
    filing_rows = []
    for p in range(-event_window, event_window + 1):
        tmp = panel[panel[TIME_VAR] == p]
        ttmp = tmp[tmp["treated"] == 1]
        ctmp = tmp[tmp["treated"] == 0]
        filing_rows.append({
            "period": p,
            "t_mean": ttmp["patent_count"].mean(),
            "c_mean": ctmp["patent_count"].mean(),
            "t_total": ttmp["patent_count"].sum(),
            "c_total": ctmp["patent_count"].sum(),
        })
    filing_df = pd.DataFrame(filing_rows)

    # Event-period novelty differences
    trend_rows = []
    for p in range(-event_window, event_window + 1):
        tmp = panel[panel[TIME_VAR] == p]
        tv = tmp.loc[tmp["treated"] == 1, "novelty_score_mean"]
        cv = tmp.loc[tmp["treated"] == 0, "novelty_score_mean"]

        t_mean = tv.mean() if len(tv) else np.nan
        c_mean = cv.mean() if len(cv) else np.nan
        diff, ci, n_t, n_c = ci_bound_diff(tv, cv)

        trend_rows.append({
            "period": p,
            "t_mean": t_mean,
            "c_mean": c_mean,
            "diff": diff,
            "ci_bound": ci,
            "n_t": n_t,
            "n_c": n_c,
        })
    trend_df = pd.DataFrame(trend_rows)

    # Deal cohort analysis (treated firms only)
    treated_firms_set = set(panel.loc[panel["treated"] == 1, "bvd_id"].unique())
    cohort_info = firm_info[firm_info["bvd_id"].isin(treated_firms_set)].copy()

    cohort_panel = panel[panel["treated"] == 1].merge(
        cohort_info[["bvd_id", "deal_year"]],
        on="bvd_id",
        how="left"
    )

    cohort_rows = []
    for year, sub in cohort_panel.groupby("deal_year", dropna=True):
        cohort_rows.append({
            "year": int(year),
            "firms": sub["bvd_id"].nunique(),
            "mean_novelty": sub["novelty_score_mean"].mean(),
            "pre_obs": int((sub["post"] == 0).sum()),
            "post_obs": int((sub["post"] == 1).sum()),
            "note": "right-censored" if year >= 2020 else ""
        })
    cohort_df = pd.DataFrame(cohort_rows).sort_values("year") if len(cohort_rows) else pd.DataFrame()

    # =====================================================================
    # FIGURES (FREQUENCY-SPECIFIC FILENAMES)
    # =====================================================================

    # Figure: novelty trajectory
    plt.figure(figsize=(9, 5))
    for treated_val, label in [(1, "Treated"), (0, "Control")]:
        tmp = panel[panel["treated"] == treated_val].groupby(TIME_VAR)["novelty_score_mean"].mean().sort_index()
        if len(tmp) > 0:
            plt.plot(tmp.index, tmp.values, marker="o", label=label)
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(f"Mean novelty by event {PERIOD_WORD} ({FREQ_LABEL})")
    plt.xlabel(TIME_VAR)
    plt.ylabel("Mean novelty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT["fig_novelty"], dpi=300)
    plt.close()

    # Figure: patent counts
    plt.figure(figsize=(9, 5))
    for treated_val, label in [(1, "Treated"), (0, "Control")]:
        tmp = panel[panel["treated"] == treated_val].groupby(TIME_VAR)["patent_count"].mean().sort_index()
        if len(tmp) > 0:
            plt.plot(tmp.index, tmp.values, marker="o", label=label)
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(f"Mean patent count by event {PERIOD_WORD} ({FREQ_LABEL})")
    plt.xlabel(TIME_VAR)
    plt.ylabel("Mean patent count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT["fig_counts"], dpi=300)
    plt.close()

    # Same content, but frequency-labeled paths
    plot_deal_year_dist(OUT["fig_deal_year"])
    plot_overlap_dist(OUT["fig_overlap"])

    # =====================================================================
    # LATEX TABLES
    # =====================================================================

    sample_counts = {
        "treated_firms": treated_firms_n,
        "matched_pairs": int(len(psm_matches)),
        "panel_firms": panel_firms,
        "panel_obs": panel_obs,
    }
    novelty_table = {
        "overall_mean": panel_novelty["mean"],
        "treated_pre": treated_pre,
        "treated_post": treated_post,
        "control_pre": control_pre,
        "control_post": control_post,
        "raw_did": raw_did,
    }

    save_latex_table_desc(OUT["table_desc_tex"], FREQ_LABEL, sample_counts, novelty_table)
    save_latex_table_balance(OUT["table_balance_tex"], post_match_df, FREQ_LABEL)

    # =====================================================================
    # TXT SUMMARY
    # =====================================================================

    lines = []
    lines.append("=" * 70)
    lines.append(f"EDA SUMMARY — {FREQ_LABEL.upper()}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("SECTION 1 — SAMPLE OVERVIEW")
    lines.append(f"Eligible treated firms:        {treated_firms_n:,}")
    lines.append(f"Total patents (eligible):      {total_patents_n:,}")
    lines.append(
        "Patents per firm: "
        f"Mean={patents_per_firm.mean():.1f} | Median={patents_per_firm.median():.0f} | "
        f"Min={patents_per_firm.min():.0f} | Max={patents_per_firm.max():.0f}"
    )
    if total_patents_n > 0:
        lines.append(f"Pre-merger patents:            {pre_patents_n:,} ({pre_patents_n / total_patents_n * 100:.1f}%)")
        lines.append(f"Post-merger patents:           {post_patents_n:,} ({post_patents_n / total_patents_n * 100:.1f}%)")
    lines.append("")

    lines.append("Dominant CPC distribution:")
    for cpc, n in cpc_dist.items():
        lines.append(f"  {cpc}: {n:,} firms ({n / treated_firms_n * 100:.1f}%)" if treated_firms_n > 0 else f"  {cpc}: {n:,}")

    lines.append("")
    lines.append("SECTION 2 — DEAL CHARACTERISTICS")
    if len(firm_info) > 0 and "deal_date" in firm_info.columns:
        lines.append(f"Deal date range: {firm_info['deal_date'].min().date()} – {firm_info['deal_date'].max().date()}")
    for yr, n in deal_year_counts.items():
        lines.append(f"  {int(yr)}: {int(n):,}")

    if "deal_value_missing" in firm_info.columns and "log_deal_value" in firm_info.columns:
        n_present = int((firm_info["deal_value_missing"] == 0).sum())
        n_missing = int((firm_info["deal_value_missing"] == 1).sum())
        lines.append(f"Deal value available: {n_present}/{treated_firms_n} ({n_present/treated_firms_n*100:.1f}%)" if treated_firms_n > 0 else "")
        lines.append(f"Deal value missing:   {n_missing}/{treated_firms_n} ({n_missing/treated_firms_n*100:.1f}%)" if treated_firms_n > 0 else "")
        lines.append(
            f"Log deal value: mean={firm_info['log_deal_value'].mean():.3f}, "
            f"median={firm_info['log_deal_value'].median():.3f}, "
            f"p75={firm_info['log_deal_value'].quantile(0.75):.3f}"
        )

    lines.append("")
    lines.append("SECTION 3 — PATENT-LEVEL NOVELTY DISTRIBUTIONS")
    lines.append(f"All patents:    mean={novelty_all['mean']:.4f}, median={novelty_all['median']:.4f}, std={novelty_all['std']:.4f}, skew={novelty_all['skew']:.4f}, N={novelty_all['n']:,}")
    lines.append(f"Pre-merger:     mean={novelty_pre['mean']:.4f}, median={novelty_pre['median']:.4f}, std={novelty_pre['std']:.4f}, skew={novelty_pre['skew']:.4f}, N={novelty_pre['n']:,}")
    lines.append(f"Post-merger:    mean={novelty_post['mean']:.4f}, median={novelty_post['median']:.4f}, std={novelty_post['std']:.4f}, skew={novelty_post['skew']:.4f}, N={novelty_post['n']:,}")
    lines.append("")
    for _, row in novelty_by_cpc_df.iterrows():
        lines.append(f"  {row['cpc']}: mean={row['mean']:.4f}, median={row['median']:.4f}, std={row['std']:.4f}, N={int(row['n']):,}")

    lines.append("")
    lines.append("SECTION 4 — CONTROL VARIABLE DISTRIBUTIONS (FIRM LEVEL)")
    ctrl_vars = [
        "log_target_pre_patents",
        "patent_age",
        "log_deal_value",
        "log_acquiror_assets",
        "log_acquiror_pre_patents",
    ]
    for var in ctrl_vars:
        if var in firm_info.columns:
            d = describe_series(firm_info[var])
            lines.append(
                f"{var}: mean={d['mean']:.3f}, median={d['median']:.3f}, std={d['std']:.3f}, "
                f"p10={d['p10']:.3f}, p90={d['p90']:.3f}, N={d['n']:,}"
            )

    if "deal_value_missing" in firm_info.columns:
        lines.append(f"deal_value_missing: {(firm_info['deal_value_missing'] == 1).sum():,}/{treated_firms_n:,}")
    if "acquiror_assets_missing" in firm_info.columns:
        lines.append(f"acquiror_assets_missing: {(firm_info['acquiror_assets_missing'] == 1).sum():,}/{treated_firms_n:,}")

    lines.append("")
    lines.append("SECTION 5 — KNOWLEDGE OVERLAP")
    lines.append(f"Firms with overlap score: {overlap_valid.shape[0]:,}/{treated_firms_n:,} ({overlap_valid.shape[0] / treated_firms_n * 100:.1f}%)" if treated_firms_n > 0 else "Firms with overlap score: 0")
    lines.append(f"Firms without overlap:    {treated_firms_n - overlap_valid.shape[0]:,}/{treated_firms_n:,} ({(treated_firms_n - overlap_valid.shape[0]) / treated_firms_n * 100:.1f}%)" if treated_firms_n > 0 else "")
    if len(overlap_valid) > 0:
        d = describe_series(overlap_valid)
        lines.append(
            f"Overlap distribution: mean={d['mean']:.4f}, median={d['median']:.4f}, "
            f"std={d['std']:.4f}, min={d['min']:.4f}, p25={d['p25']:.4f}, "
            f"p75={d['p75']:.4f}, max={d['max']:.4f}"
        )
        low = int((overlap_valid < 0.3).sum())
        mid = int(((overlap_valid >= 0.3) & (overlap_valid < 0.6)).sum())
        high = int((overlap_valid >= 0.6).sum())
        lines.append(f"Overlap bands: low={low}, mid={mid}, high={high}")

    lines.append("")
    lines.append("SECTION 6 — PSM DIAGNOSTICS")
    if psm_diag is not None:
        lines.append(f"N treated firms in psm_diagnostics: {len(psm_diag):,}")
        for var in ["log_target_pre_patents", "pre_novelty_mean", "patent_age"]:
            if var in psm_diag.columns:
                d = describe_series(psm_diag[var])
                lines.append(
                    f"{var}: mean={d['mean']:.3f}, std={d['std']:.3f}, "
                    f"p25={d['p25']:.3f}, p50={d['median']:.3f}, p75={d['p75']:.3f}, skew={d['skew']:.3f}"
                )

    lines.append("")
    lines.append("SECTION 7 — PRE-MATCH COMPARISON")
    if len(pre_match_df) > 0:
        for _, row in pre_match_df.iterrows():
            flag = "HIGH" if pd.notna(row["smd"]) and abs(row["smd"]) >= 0.25 else ""
            lines.append(
                f"{row['variable']}: treated={row['treated_mean']:.4f}, control={row['control_mean']:.4f}, "
                f"diff={row['diff']:+.4f}, SMD={row['smd']:.3f} {flag}"
            )

    lines.append("")
    lines.append("SECTION 8 — POST-MATCH BALANCE")
    lines.append(f"Matched pairs: {len(psm_matches):,}")
    if len(post_match_df) > 0:
        for _, row in post_match_df.iterrows():
            lines.append(
                f"{row['variable']}: treated={row['treated_mean']:.4f}, control={row['control_mean']:.4f}, "
                f"SMD={row['smd']:.4f}, status={row['status']}"
            )

    if "match_distance" in psm_matches.columns:
        d = describe_series(psm_matches["match_distance"])
        lines.append("")
        lines.append("SECTION 9 — MATCH QUALITY")
        lines.append(
            f"Propensity score distance: mean={d['mean']:.4f}, median={d['median']:.4f}, "
            f"p75={d['p75']:.4f}, p90={d['p90']:.4f}, max={d['max']:.4f}"
        )
        excellent = int((psm_matches["match_distance"] <= 0.10).sum())
        good = int((psm_matches["match_distance"] <= 0.25).sum())
        poor = int((psm_matches["match_distance"] > 0.25).sum())
        lines.append(f"Excellent (<=0.10): {excellent:,}")
        lines.append(f"Good (<=0.25):      {good:,}")
        lines.append(f"Poor (>0.25):       {poor:,}")

    lines.append("")
    lines.append(f"SECTION 10 — PANEL STRUCTURE ({FREQ_LABEL.upper()})")
    lines.append(f"Firm-period observations: {panel_obs:,}")
    lines.append(f"Firms:                    {panel_firms:,} ({treated_panel_firms:,} treated / {control_panel_firms:,} control)")
    lines.append(
        f"Periods per firm: mean={periods_per_firm.mean():.1f}, median={periods_per_firm.median():.1f}, "
        f"min={periods_per_firm.min():.0f}, max={periods_per_firm.max():.0f}"
    )
    lines.append("")
    lines.append(f"Coverage by event {PERIOD_WORD}:")
    for _, row in coverage_df.iterrows():
        lines.append(
            f"  {int(row['period']):>3}: treated={int(row['treated']):>3}, control={int(row['control']):>3}, "
            f"total={int(row['total']):>3}, coverage={row['coverage_pct']:.1f}%"
        )

    lines.append("")
    lines.append("SECTION 11 — NOVELTY SCORE IN PANEL")
    lines.append(
        f"Overall: mean={panel_novelty['mean']:.4f}, median={panel_novelty['median']:.4f}, "
        f"std={panel_novelty['std']:.4f}, min={panel_novelty['min']:.4f}, "
        f"max={panel_novelty['max']:.4f}, skew={panel_novelty['skew']:.4f}, N={panel_novelty['n']:,}"
    )
    lines.append(f"Treated pre:    {treated_pre:.4f}")
    lines.append(f"Treated post:   {treated_post:.4f}")
    lines.append(f"Control pre:    {control_pre:.4f}")
    lines.append(f"Control post:   {control_post:.4f}")
    lines.append(f"Raw DiD:        {raw_did:+.4f}")

    if len(novelty_by_cpc_panel_df) > 0:
        lines.append("")
        lines.append("Mean novelty by dominant CPC (panel):")
        for _, row in novelty_by_cpc_panel_df.iterrows():
            lines.append(
                f"  {row['cpc']}: mean={row['mean']:.4f}, std={row['std']:.4f}, firms={int(row['firms']):,}"
            )

    lines.append("")
    lines.append("SECTION 12 — PATENT FILING ACTIVITY")
    for _, row in filing_df.iterrows():
        lines.append(
            f"  {int(row['period']):>3}: T_mean={row['t_mean']:.2f}, C_mean={row['c_mean']:.2f}, "
            f"T_total={int(row['t_total']):,}, C_total={int(row['c_total']):,}"
        )

    lines.append("")
    lines.append("SECTION 13 — EVENT-TIME NOVELTY TRENDS")
    for _, row in trend_df.iterrows():
        star = " *" if pd.notna(row["diff"]) and pd.notna(row["ci_bound"]) and abs(row["diff"]) > row["ci_bound"] else ""
        lines.append(
            f"  {int(row['period']):>3}: T_mean={row['t_mean']:.4f}, C_mean={row['c_mean']:.4f}, "
            f"Diff={row['diff']:+.4f}, 95% CI bound=±{row['ci_bound']:.4f}{star}"
        )

    lines.append("")
    lines.append("SECTION 14 — DEAL COHORT ANALYSIS")
    if len(cohort_df) > 0:
        for _, row in cohort_df.iterrows():
            note = f" ({row['note']})" if str(row["note"]).strip() != "" else ""
            lines.append(
                f"  {int(row['year'])}: firms={int(row['firms']):,}, mean novelty={row['mean_novelty']:.4f}, "
                f"pre obs={int(row['pre_obs']):,}, post obs={int(row['post_obs']):,}{note}"
            )

    lines.append("")
    lines.append("SECTION 15 — SAVED OUTPUTS")
    for key, path in OUT.items():
        lines.append(f"  {os.path.basename(path)}")

    save_text(OUT["summary_txt"], lines)

    print(f"Saved summary: {OUT['summary_txt']}", flush=True)
    print(f"Saved LaTeX table: {OUT['table_desc_tex']}", flush=True)
    print(f"Saved LaTeX table: {OUT['table_balance_tex']}", flush=True)
    print(f"Saved figure: {OUT['fig_novelty']}", flush=True)
    print(f"Saved figure: {OUT['fig_counts']}", flush=True)
    print(f"Saved figure: {OUT['fig_deal_year']}", flush=True)
    print(f"Saved figure: {OUT['fig_overlap']}", flush=True)

print_line()
print("STEP 9a COMPLETE — MULTI-FREQUENCY EDA")
print_line()
print("Check data/outputs/ for frequency-labeled summaries, tables, and figures.", flush=True)
