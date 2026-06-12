"""
Step 7d — Panel Construction (Quarterly + Semi-Annual + Annual)

Builds firm-event-time panels for DiD regression at multiple temporal
aggregations from the same matched sample:
- quarterly   (baseline)
- semiannual  (robustness)
- annual      (robustness)

Outputs:
- data/interim/panel_full.csv                -> quarterly baseline
- data/interim/panel_full_quarterly.csv
- data/interim/panel_full_semiannual.csv
- data/interim/panel_full_annual.csv
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

MATCHES_PATH = "data/interim/psm_matches.csv"
ELIGIBLE_PATH = "data/interim/patents_eligible_overlap.csv"
CONTROL_PATH = "data/interim/patents_control_novelty.csv"

OUTPUT_QUARTERLY_MAIN = "data/interim/panel_full.csv"  # backward compatibility
OUTPUT_PATHS = {
    "quarterly":   "data/interim/panel_full_quarterly.csv",
    "semiannual":  "data/interim/panel_full_semiannual.csv",
    "annual":      "data/interim/panel_full_annual.csv",
}

EVENT_WINDOW_QUARTERS = 12   # ±12 quarters = ±3 years
MIN_PERIOD_COVERAGE = 0      # set >0 if you want to drop sparse firms

# Time aggregation mapping:
# quarterly  : 1 quarter  per period
# semiannual : 2 quarters per period
# annual     : 4 quarters per period
AGG_CONFIG = {
    "quarterly": {
        "divisor": 1,
        "period_col": "event_quarter",
        "window": 12,
        "label": "quarter"
    },
    "semiannual": {
        "divisor": 2,
        "period_col": "event_halfyear",
        "window": 6,
        "label": "half-year"
    },
    "annual": {
        "divisor": 4,
        "period_col": "event_year",
        "window": 3,
        "label": "year"
    },
}

# =============================================================================
# SETUP
# =============================================================================

print("=" * 70, flush=True)
print("STEP 7d — PANEL CONSTRUCTION (MULTI-AGGREGATION)", flush=True)
print("=" * 70, flush=True)
print("Outputs: quarterly + semiannual + annual", flush=True)

for path in [MATCHES_PATH, ELIGIBLE_PATH, CONTROL_PATH]:
    if not os.path.exists(path):
        print(f"\nERROR: required input not found: {path}", flush=True)
        sys.exit(1)

os.makedirs("data/interim", exist_ok=True)
start_time = time.time()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def mode_or_nan(series):
    """Return modal value, else NaN."""
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.value_counts().index[0]

def assign_event_quarter(filing_date, pseudo_deal_date):
    """
    Compute event quarter relative to pseudo-deal-date.
    Quarter 0 = quarter containing pseudo-deal-date.
    """
    if pd.isna(filing_date) or pd.isna(pseudo_deal_date):
        return np.nan
    delta_days = (filing_date - pseudo_deal_date).days
    return int(delta_days // 91)

def map_period_from_quarter(event_quarter, agg_name):
    """
    Map event_quarter to aggregated event period.
    Uses floor division so:
    semiannual:
      [-2, -1] -> -1
      [0, 1]   -> 0
      [2, 3]   -> 1
    annual:
      [-4,-3,-2,-1] -> -1
      [0,1,2,3]     -> 0
      [4,5,6,7]     -> 1
    """
    divisor = AGG_CONFIG[agg_name]["divisor"]
    if divisor == 1:
        return event_quarter
    return np.floor_divide(event_quarter, divisor)

def summarise_panel(panel, agg_name, period_col, window):
    """Print diagnostics for one panel."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"PANEL DIAGNOSTICS — {agg_name.upper()}", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"Rows: {len(panel):,}", flush=True)
    print(f"Firms: {panel['bvd_id'].nunique():,}", flush=True)
    print(f"Treated firms: {panel.loc[panel['treated'] == 1, 'bvd_id'].nunique():,}", flush=True)
    print(f"Control firms: {panel.loc[panel['treated'] == 0, 'bvd_id'].nunique():,}", flush=True)

    print(f"\nFirm coverage per event {AGG_CONFIG[agg_name]['label']}:", flush=True)
    print(f"{'Period':>8} {'Treated':>9} {'Control':>9} {'Total':>9}", flush=True)
    print(f"{'-' * 42}", flush=True)

    for p in range(-window, window + 1):
        tmp = panel[panel[period_col] == p]
        nt = tmp.loc[tmp["treated"] == 1, "bvd_id"].nunique()
        nc = tmp.loc[tmp["treated"] == 0, "bvd_id"].nunique()
        print(f"{p:>8} {nt:>9,} {nc:>9,} {tmp['bvd_id'].nunique():>9,}", flush=True)

    n_novelty = panel["novelty_score_mean"].notna().sum()
    print(f"\nObservations with novelty score: {n_novelty:,}/{len(panel):,} "
          f"({n_novelty / len(panel) * 100:.1f}%)", flush=True)

    print(f"\nMean novelty by treated/post:", flush=True)
    grp = (
        panel.groupby(["treated", "post"])["novelty_score_mean"]
        .agg(mean="mean", count="count")
        .round(4)
    )
    print(grp.to_string(), flush=True)

    ctrl_vars = [
        "log_target_pre_patents",
        "log_acquiror_pre_patents",
        "log_deal_value",
        "log_acquiror_assets",
        "patent_age",
        "knowledge_overlap",
    ]
    total_firms = panel["bvd_id"].nunique()

    print(f"\nControl variable coverage:", flush=True)
    for var in ctrl_vars:
        if var in panel.columns:
            n = panel.groupby("bvd_id")[var].first().notna().sum()
            print(f"  {var:<32} {n:>4,}/{total_firms:<4,} ({n/total_firms*100:.1f}%)", flush=True)

def build_period_column(df, agg_name):
    """
    Add the relevant event-period column to a patent-level dataframe.
    """
    period_col = AGG_CONFIG[agg_name]["period_col"]
    if period_col not in df.columns:
        df[period_col] = df["event_quarter"].apply(lambda x: map_period_from_quarter(x, agg_name))
    return df

# =============================================================================
# LOAD MATCHES
# =============================================================================

print(f"\nLoading matched pairs: {MATCHES_PATH}", flush=True)
matches = pd.read_csv(MATCHES_PATH)

print(f"Matched pairs: {len(matches):,}", flush=True)
print(f"Unique treated firms: {matches['bvd_id'].nunique():,}", flush=True)
print(f"Unique control firms: {matches['control_assignee'].nunique():,}", flush=True)

treated_bvd_ids = set(matches["bvd_id"].astype(str).unique())
control_assignees = set(matches["control_assignee"].astype(str).unique())

# control assignee -> matched pseudo deal year
control_year_map = dict(zip(
    matches["control_assignee"].astype(str),
    matches["control_deal_year"]
))

# control assignee -> matched treated bvd_id
control_to_treated = dict(zip(
    matches["control_assignee"].astype(str),
    matches["bvd_id"].astype(str)
))

# =============================================================================
# LOAD TREATED PATENT-LEVEL DATA
# =============================================================================

print(f"\nLoading treated patent data: {ELIGIBLE_PATH}", flush=True)
eligible = pd.read_csv(ELIGIBLE_PATH, low_memory=False)

eligible["bvd_id"] = eligible["bvd_id"].astype(str)
eligible["filing_date"] = pd.to_datetime(eligible["filing_date"], errors="coerce")
eligible["deal_date"] = pd.to_datetime(eligible["deal_date"], errors="coerce")

if "event_quarter" not in eligible.columns:
    print("ERROR: event_quarter column not found in patents_eligible_overlap.csv", flush=True)
    sys.exit(1)

eligible["event_quarter"] = pd.to_numeric(eligible["event_quarter"], errors="coerce")
eligible = eligible.dropna(subset=["event_quarter"]).copy()
eligible["event_quarter"] = eligible["event_quarter"].astype(int)

# Keep matched treated firms only
eligible = eligible[eligible["bvd_id"].isin(treated_bvd_ids)].copy()

# Restrict to ±12 quarters before creating alternative aggregations
eligible = eligible[eligible["event_quarter"].between(-EVENT_WINDOW_QUARTERS, EVENT_WINDOW_QUARTERS)].copy()

if "cpc_4digit" not in eligible.columns:
    eligible["cpc_4digit"] = eligible["cpc_code"].astype(str).str[:4].str.upper()

print(f"Treated patent rows after match + event-window filter: {len(eligible):,}", flush=True)
print(f"Unique treated firms retained: {eligible['bvd_id'].nunique():,}", flush=True)

# =============================================================================
# LOAD CONTROL PATENT-LEVEL DATA
# =============================================================================

print(f"\nLoading control patent data: {CONTROL_PATH}", flush=True)
control = pd.read_csv(CONTROL_PATH, low_memory=False)

control["assignee_name"] = control["assignee_name"].astype(str)
control["filing_date"] = pd.to_datetime(
    control["filing_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# Keep matched control firms only
control = control[control["assignee_name"].isin(control_assignees)].copy()

print(f"Control patent rows (matched controls only): {len(control):,}", flush=True)
print(f"Unique matched control firms retained: {control['assignee_name'].nunique():,}", flush=True)

if "cpc_4digit" not in control.columns:
    control["cpc_4digit"] = control["cpc_code"].astype(str).str[:4].str.upper()

# Assign pseudo-deal-date and event quarter at patent level
control["pseudo_deal_year"] = control["assignee_name"].map(control_year_map)
control["pseudo_deal_date"] = pd.to_datetime(
    control["pseudo_deal_year"].astype("Int64").astype(str) + "-06-30",
    errors="coerce"
)

control["event_quarter"] = control.apply(
    lambda r: assign_event_quarter(r["filing_date"], r["pseudo_deal_date"]),
    axis=1
)

control = control.dropna(subset=["event_quarter"]).copy()
control["event_quarter"] = control["event_quarter"].astype(int)

# Restrict to same base quarter window
control = control[control["event_quarter"].between(-EVENT_WINDOW_QUARTERS, EVENT_WINDOW_QUARTERS)].copy()

print(f"Control patent rows after event-window filter: {len(control):,}", flush=True)

# =============================================================================
# BUILD FIRM-LEVEL TREATED VARIABLES FOR CONTROL MERGE
# =============================================================================

treated_firm_vars = (
    eligible.groupby("bvd_id")
    .agg(
        orbis_target_name=("orbis_target_name", "first"),
        log_target_pre_patents=("log_target_pre_patents", "first"),
        log_acquiror_pre_patents=("log_acquiror_pre_patents", "first"),
        log_deal_value=("log_deal_value", "first"),
        log_acquiror_assets=("log_acquiror_assets", "first"),
        patent_age=("patent_age", "first"),
        deal_value_missing=("deal_value_missing", "first"),
        acquiror_assets_missing=("acquiror_assets_missing", "first"),
        knowledge_overlap=("knowledge_overlap", "first"),
        deal_date=("deal_date", "first"),
    )
    .reset_index()
)

# =============================================================================
# MAIN LOOP — BUILD ALL TEMPORAL AGGREGATIONS
# =============================================================================

built_panels = {}

for agg_name, cfg in AGG_CONFIG.items():
    print(f"\n{'#' * 70}", flush=True)
    print(f"BUILDING {agg_name.upper()} PANEL", flush=True)
    print(f"{'#' * 70}", flush=True)

    period_col = cfg["period_col"]
    window = cfg["window"]

    # -------------------------------------------------------------------------
    # TREATED PANEL
    # -------------------------------------------------------------------------
    treated_tmp = eligible.copy()
    treated_tmp = build_period_column(treated_tmp, agg_name)

    treated_panel = (
        treated_tmp
        .groupby(["bvd_id", period_col])
        .agg(
            novelty_score_mean=("novelty_score", "mean"),
            novelty_score_median=("novelty_score", "median"),
            novelty_score_std=("novelty_score", "std"),
            patent_count=("publication_number", "count"),

            # Firm-level / constant vars
            orbis_target_name=("orbis_target_name", "first"),
            deal_date=("deal_date", "first"),
            log_target_pre_patents=("log_target_pre_patents", "first"),
            log_acquiror_pre_patents=("log_acquiror_pre_patents", "first"),
            log_deal_value=("log_deal_value", "first"),
            log_acquiror_assets=("log_acquiror_assets", "first"),
            patent_age=("patent_age", "first"),
            deal_value_missing=("deal_value_missing", "first"),
            acquiror_assets_missing=("acquiror_assets_missing", "first"),
            knowledge_overlap=("knowledge_overlap", "first"),
            dominant_cpc=("cpc_4digit", mode_or_nan),
        )
        .reset_index()
    )

    treated_panel["treated"] = 1
    treated_panel["post"] = (treated_panel[period_col] >= 0).astype(int)

    print(f"Treated firm-period rows ({agg_name}): {len(treated_panel):,}", flush=True)

    # -------------------------------------------------------------------------
    # CONTROL PANEL
    # -------------------------------------------------------------------------
    control_tmp = control.copy()
    control_tmp = build_period_column(control_tmp, agg_name)

    control_panel = (
        control_tmp
        .groupby(["assignee_name", period_col])
        .agg(
            novelty_score_mean=("novelty_score", "mean"),
            novelty_score_median=("novelty_score", "median"),
            novelty_score_std=("novelty_score", "std"),
            patent_count=("publication_number", "count"),
            dominant_cpc=("cpc_4digit", mode_or_nan),
            pseudo_deal_date=("pseudo_deal_date", "first"),
        )
        .reset_index()
        .rename(columns={"assignee_name": "bvd_id"})
    )

    control_panel["treated"] = 0
    control_panel["post"] = (control_panel[period_col] >= 0).astype(int)

    # Bring in treated firm-level controls from the matched treated partner
    control_panel["matched_bvd_id"] = control_panel["bvd_id"].map(control_to_treated)

    control_panel = control_panel.merge(
        treated_firm_vars.rename(columns={"bvd_id": "matched_bvd_id"}),
        on="matched_bvd_id",
        how="left"
    )

    # Overwrite deal_date for control firms with pseudo-deal-date
    control_panel["deal_date"] = control_panel["pseudo_deal_date"]

    control_panel = control_panel.drop(
        columns=["matched_bvd_id", "pseudo_deal_date"],
        errors="ignore"
    )

    print(f"Control firm-period rows ({agg_name}): {len(control_panel):,}", flush=True)

    # -------------------------------------------------------------------------
    # COMBINE
    # -------------------------------------------------------------------------
    panel = pd.concat([treated_panel, control_panel], ignore_index=True)

    panel = panel.sort_values(["bvd_id", period_col]).reset_index(drop=True)
    panel["treated_x_post"] = panel["treated"] * panel["post"]

    # Optional sparse-firm filter
    if MIN_PERIOD_COVERAGE > 0:
        coverage = panel.groupby("bvd_id")[period_col].nunique()
        keep_firms = coverage[coverage >= MIN_PERIOD_COVERAGE].index
        before = len(panel)
        panel = panel[panel["bvd_id"].isin(keep_firms)].copy()
        print(f"Applied MIN_PERIOD_COVERAGE={MIN_PERIOD_COVERAGE}: "
              f"{before:,} -> {len(panel):,} rows", flush=True)

    # Save output
    out_path = OUTPUT_PATHS[agg_name]
    panel.to_csv(out_path, index=False)
    built_panels[agg_name] = panel

    print(f"Saved: {out_path}", flush=True)

    # Keep quarterly path for backward compatibility with Step 9
    if agg_name == "quarterly":
        panel.to_csv(OUTPUT_QUARTERLY_MAIN, index=False)
        print(f"Saved baseline quarterly copy: {OUTPUT_QUARTERLY_MAIN}", flush=True)

    summarise_panel(panel, agg_name, period_col, window)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

elapsed = time.time() - start_time

print(f"\n{'=' * 70}", flush=True)
print("STEP 7d COMPLETE — MULTI-AGGREGATION PANEL CONSTRUCTION", flush=True)
print(f"{'=' * 70}", flush=True)
print(f"Elapsed time: {elapsed/60:.1f} minutes", flush=True)

for agg_name, path in OUTPUT_PATHS.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        panel = built_panels[agg_name]
        period_col = AGG_CONFIG[agg_name]["period_col"]
        print(f"\n{agg_name.upper()}:", flush=True)
        print(f"  Path: {path}", flush=True)
        print(f"  Rows: {len(panel):,}", flush=True)
        print(f"  Firms: {panel['bvd_id'].nunique():,}", flush=True)
        print(f"  Time variable: {period_col}", flush=True)
        print(f"  Size: {size_mb:.1f} MB", flush=True)

print(f"""
Next step:
- Quarterly baseline:
    set PANEL_PATH = "data/interim/panel_full.csv"
- Semi-annual robustness:
    set PANEL_PATH = "data/interim/panel_full_semiannual.csv"
- Annual robustness:
    set PANEL_PATH = "data/interim/panel_full_annual.csv"

Then run Step 9b DiD regression on the desired panel.
""", flush=True)