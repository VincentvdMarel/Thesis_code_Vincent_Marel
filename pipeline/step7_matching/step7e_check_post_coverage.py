"""
Step 7e — Post-Patent Coverage Check
======================================
Checks whether every firm in panel_full.csv has sufficient patent activity
in the post-merger (post == 1) window. This is the control-side analogue of
the treated-firm eligibility filter applied in step3c, which required >= 3
patents in both the 3-year pre- and post-merger windows.

Because step7b_build_control_psm_vars.py only filters on pre-window activity
(MIN_PRE_PATENTS = 3), a matched control firm can enter the panel with zero
post-period patents, which would mechanically suppress its mean novelty in
post quarters and bias the DiD estimate.

Checks performed
-----------------
  1. Per-firm post-quarter coverage   — quarters with >= 1 patent filed (post == 1)
  2. Per-firm post-patent total       — total patents filed in post window
  3. Zero-post-patent firms           — firms with no patents at all after deal/pseudo-date
  4. Sparse-post-patent firms         — firms below MIN_POST_PATENTS threshold
  5. Matched-pair asymmetry           — pairs where one arm fails but the other passes
  6. Novelty score availability       — post quarters with a valid novelty_score_mean

Any firm (or matched pair) flagged here should be reviewed before running
the DiD regressions in step 9b. Dropping flagged pairs is conservative and
removes the pair from both treated and control arms simultaneously.

Inputs
------
  panel_full.csv      — output of step7d_build_panel.py
  psm_matches.csv     — matched pairs from step7c_psm_matching.py

Outputs
-------
  post_patent_check.csv        — one row per firm with coverage diagnostics
  post_patent_flagged_pairs.csv — matched pairs where >= 1 arm fails threshold

Usage
-----
  python step7e_check_post_coverage.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# Match these to step3c (MIN_PATENTS=3, PRE_POST_YEARS=3) and
# step7c (EVENT_WINDOW=12 quarters = 3 years)
# =============================================================================
PANEL_PATH        = "data/interim/panel_full.csv"
MATCHES_PATH      = "data/interim/psm_matches.csv"
OUTPUT_FIRM_PATH  = "post_patent_check.csv"
OUTPUT_PAIRS_PATH = "post_patent_flagged_pairs.csv"

MIN_POST_PATENTS  = 3    # minimum total patents in post window (mirrors step3c)
MIN_POST_QUARTERS = 1    # minimum quarters with >= 1 patent filed in post window
POST_WINDOW_Q     = 12   # quarters considered post (event_quarter 0 to +12)

# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 65, flush=True)
print("STEP 7e — POST-PATENT COVERAGE CHECK", flush=True)
print("=" * 65, flush=True)

for path in [PANEL_PATH, MATCHES_PATH]:
    if not os.path.exists(path):
        print(f"\nERROR: {path} not found", flush=True)
        print("  Run step7d_build_panel.py first", flush=True)
        sys.exit(1)

print(f"\nLoading {PANEL_PATH}...", flush=True)
panel = pd.read_csv(PANEL_PATH, low_memory=False)
print(f"  Total firm-quarters: {len(panel):,}", flush=True)
print(f"  Unique firms:        {panel['bvd_id'].nunique():,}", flush=True)
print(f"    Treated:           "
      f"{panel[panel['treated']==1]['bvd_id'].nunique():,}", flush=True)
print(f"    Control:           "
      f"{panel[panel['treated']==0]['bvd_id'].nunique():,}", flush=True)

print(f"\nLoading {MATCHES_PATH}...", flush=True)
matches = pd.read_csv(MATCHES_PATH)
print(f"  Matched pairs: {len(matches):,}", flush=True)

# =============================================================================
# ISOLATE POST-PERIOD ROWS
# post == 1  ↔  event_quarter >= 0  (quarters 0 through +POST_WINDOW_Q)
# =============================================================================
post = panel[panel["post"] == 1].copy()
print(f"\nPost-period firm-quarters: {len(post):,}", flush=True)

# =============================================================================
# PER-FIRM POST-PERIOD DIAGNOSTICS
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"COMPUTING PER-FIRM POST-PERIOD COVERAGE", flush=True)
print(f"{'='*65}", flush=True)

firm_stats = (
    post.groupby("bvd_id")
    .agg(
        treated               = ("treated",              "first"),
        n_post_quarters       = ("event_quarter",        "count"),     # rows present
        total_post_patents    = ("patent_count",         "sum"),       # total patents
        quarters_with_patents = ("patent_count",         lambda x:
                                 (x > 0).sum()),                       # quarters with >= 1 patent
        quarters_with_novelty = ("novelty_score_mean",   lambda x:
                                 x.notna().sum()),                     # quarters with novelty score
        mean_post_novelty     = ("novelty_score_mean",   "mean"),
        max_post_patents_q    = ("patent_count",         "max"),       # busiest post quarter
    )
    .reset_index()
)

# Firms that appear in the panel but have NO post-period rows at all
all_firms = panel[["bvd_id", "treated"]].drop_duplicates()
firms_with_post = set(post["bvd_id"].unique())
firms_no_post = all_firms[~all_firms["bvd_id"].isin(firms_with_post)].copy()
firms_no_post["n_post_quarters"]       = 0
firms_no_post["total_post_patents"]    = 0
firms_no_post["quarters_with_patents"] = 0
firms_no_post["quarters_with_novelty"] = 0
firms_no_post["mean_post_novelty"]     = np.nan
firms_no_post["max_post_patents_q"]    = 0

firm_stats = pd.concat([firm_stats, firms_no_post], ignore_index=True)

# Flag firms below threshold
firm_stats["flag_zero_post_patents"]   = (
    firm_stats["total_post_patents"] == 0
).astype(int)
firm_stats["flag_low_post_patents"]    = (
    firm_stats["total_post_patents"] < MIN_POST_PATENTS
).astype(int)
firm_stats["flag_zero_post_quarters"]  = (
    firm_stats["quarters_with_patents"] == 0
).astype(int)
firm_stats["flag_low_post_quarters"]   = (
    firm_stats["quarters_with_patents"] < MIN_POST_QUARTERS
).astype(int)
firm_stats["flag_no_novelty"]          = (
    firm_stats["quarters_with_novelty"] == 0
).astype(int)

# Overall flag: fails on patents OR novelty
firm_stats["flag_any"] = (
    (firm_stats["flag_low_post_patents"] == 1) |
    (firm_stats["flag_no_novelty"]       == 1)
).astype(int)

# =============================================================================
# SUMMARY BY GROUP
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"POST-PERIOD COVERAGE SUMMARY", flush=True)
print(f"{'='*65}", flush=True)
print(f"  Thresholds: MIN_POST_PATENTS={MIN_POST_PATENTS}, "
      f"MIN_POST_QUARTERS={MIN_POST_QUARTERS}", flush=True)

for group_label, group_val in [("ALL FIRMS", None),
                                 ("TREATED",  1),
                                 ("CONTROL",  0)]:
    if group_val is None:
        g = firm_stats
    else:
        g = firm_stats[firm_stats["treated"] == group_val]

    n = len(g)
    if n == 0:
        continue

    print(f"\n  {group_label} (n={n:,}):", flush=True)
    print(f"    Post-patents  — mean: {g['total_post_patents'].mean():.1f}, "
          f"median: {g['total_post_patents'].median():.0f}, "
          f"min: {g['total_post_patents'].min():.0f}, "
          f"max: {g['total_post_patents'].max():.0f}", flush=True)
    print(f"    Quarters w/ patents — mean: {g['quarters_with_patents'].mean():.1f}, "
          f"min: {g['quarters_with_patents'].min():.0f}", flush=True)
    print(f"    Zero post-patents:          "
          f"{g['flag_zero_post_patents'].sum():>4,} / {n}  "
          f"({g['flag_zero_post_patents'].mean()*100:.1f}%)", flush=True)
    print(f"    Below MIN_POST_PATENTS ({MIN_POST_PATENTS}): "
          f"{g['flag_low_post_patents'].sum():>4,} / {n}  "
          f"({g['flag_low_post_patents'].mean()*100:.1f}%)", flush=True)
    print(f"    No post novelty score:      "
          f"{g['flag_no_novelty'].sum():>4,} / {n}  "
          f"({g['flag_no_novelty'].mean()*100:.1f}%)", flush=True)
    print(f"    Flagged (any):              "
          f"{g['flag_any'].sum():>4,} / {n}  "
          f"({g['flag_any'].mean()*100:.1f}%)", flush=True)

# =============================================================================
# DISTRIBUTION OF POST-PATENT COUNTS
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"DISTRIBUTION OF TOTAL POST-PERIOD PATENTS PER FIRM", flush=True)
print(f"{'='*65}", flush=True)

bins = [0, 1, 3, 6, 10, 20, 50, 9999]
labels = ["0", "1–2", "3–5", "6–9", "10–19", "20–49", "50+"]

for group_label, group_val in [("Treated", 1), ("Control", 0)]:
    g = firm_stats[firm_stats["treated"] == group_val]["total_post_patents"]
    counts = pd.cut(g, bins=bins, labels=labels, right=False).value_counts().sort_index()
    print(f"\n  {group_label}:", flush=True)
    for label, count in counts.items():
        bar = "█" * int(count / max(counts) * 30)
        flag = "  ← BELOW THRESHOLD" if label in ["0", "1–2"] else ""
        print(f"    {label:>6}  {count:>4,}  {bar}{flag}", flush=True)

# =============================================================================
# MATCHED PAIR ANALYSIS
# Flag any pair where at least one arm fails the threshold
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"MATCHED PAIR ASYMMETRY CHECK", flush=True)
print(f"{'='*65}", flush=True)

# Build a flag lookup: bvd_id → flag_any
flag_map = dict(zip(firm_stats["bvd_id"], firm_stats["flag_any"]))

# Map flags onto matched pairs
matches["treated_flag"] = matches["bvd_id"].map(flag_map).fillna(0).astype(int)
matches["control_flag"] = matches["control_assignee"].map(flag_map).fillna(0).astype(int)
matches["pair_flag"]    = ((matches["treated_flag"] == 1) |
                           (matches["control_flag"] == 1)).astype(int)

# Asymmetric pairs: one arm fails, the other passes
matches["pair_asymmetric"] = (
    matches["treated_flag"] != matches["control_flag"]
).astype(int)

n_both_pass  = (matches["pair_flag"]    == 0).sum()
n_either_fail = matches["pair_flag"].sum()
n_both_fail  = ((matches["treated_flag"] == 1) &
                (matches["control_flag"] == 1)).sum()
n_asymmetric  = matches["pair_asymmetric"].sum()

print(f"\n  Total matched pairs:          {len(matches):,}", flush=True)
print(f"  Both arms pass:               {n_both_pass:,}  "
      f"({n_both_pass/len(matches)*100:.1f}%)", flush=True)
print(f"  At least one arm fails:       {n_either_fail:,}  "
      f"({n_either_fail/len(matches)*100:.1f}%)", flush=True)
print(f"    Both arms fail:             {n_both_fail:,}", flush=True)
print(f"    One arm fails (asymmetric): {n_asymmetric:,}", flush=True)

if n_either_fail > 0:
    print(f"\n  Breakdown of flagged pairs:", flush=True)
    print(f"    {'BvD ID':<20}  {'Control assignee':<35}  "
          f"{'T-flag':>7}  {'C-flag':>7}  {'Asymmetric':>10}", flush=True)
    print(f"    {'-'*83}", flush=True)
    flagged_pairs = matches[matches["pair_flag"] == 1].copy()
    for _, row in flagged_pairs.iterrows():
        print(
            f"    {str(row['bvd_id']):<20}  "
            f"{str(row['control_assignee'])[:35]:<35}  "
            f"{'FAIL' if row['treated_flag'] else 'pass':>7}  "
            f"{'FAIL' if row['control_flag'] else 'pass':>7}  "
            f"{'YES' if row['pair_asymmetric'] else 'no':>10}",
            flush=True
        )

# =============================================================================
# NOVELTY SCORE AVAILABILITY IN POST QUARTERS
# Separate concern: firm has post patents but they lack novelty scores
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"POST-PERIOD NOVELTY SCORE AVAILABILITY BY QUARTER", flush=True)
print(f"{'='*65}", flush=True)

print(f"\n  {'Quarter':>8}  {'Treated w/ novelty':>20}  "
      f"{'Control w/ novelty':>20}  {'T-coverage':>12}  {'C-coverage':>12}",
      flush=True)
print(f"  {'-'*78}", flush=True)

for q in range(0, POST_WINDOW_Q + 1):
    q_rows    = panel[panel["event_quarter"] == q]
    t_rows    = q_rows[q_rows["treated"] == 1]
    c_rows    = q_rows[q_rows["treated"] == 0]
    t_n       = len(t_rows)
    c_n       = len(c_rows)
    t_novelty = t_rows["novelty_score_mean"].notna().sum()
    c_novelty = c_rows["novelty_score_mean"].notna().sum()
    t_cov     = f"{t_novelty/t_n*100:.1f}%" if t_n > 0 else "—"
    c_cov     = f"{c_novelty/c_n*100:.1f}%" if c_n > 0 else "—"
    flag      = "  ← low" if (t_n > 0 and t_novelty/t_n < 0.8) or \
                             (c_n > 0 and c_novelty/c_n < 0.8) else ""
    print(f"  {q:>8}  {t_novelty:>10,} / {t_n:<7,}  "
          f"{c_novelty:>10,} / {c_n:<7,}  "
          f"{t_cov:>12}  {c_cov:>12}{flag}", flush=True)

# =============================================================================
# RECOMMENDATION
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"RECOMMENDATION", flush=True)
print(f"{'='*65}", flush=True)

n_flagged_firms  = firm_stats["flag_any"].sum()
n_flagged_pairs  = matches["pair_flag"].sum()
n_clean_pairs    = len(matches) - n_flagged_pairs

if n_flagged_pairs == 0:
    print(f"\n  ✅ All {len(matches):,} matched pairs meet the post-patent threshold.",
          flush=True)
    print(f"  No pairs need to be dropped before step 8.", flush=True)
else:
    print(f"\n  ⚠️  {n_flagged_pairs:,} of {len(matches):,} matched pairs "
          f"have at least one arm below threshold.", flush=True)
    print(f"\n  Options:", flush=True)
    print(f"    A) Drop flagged pairs (conservative, symmetric):", flush=True)
    print(f"       Retains {n_clean_pairs:,} pairs for DiD regression.", flush=True)
    print(f"       Do this by filtering panel_full.csv to bvd_ids NOT in "
          f"post_patent_flagged_pairs.csv.", flush=True)
    print(f"    B) Keep all pairs, add a post_patent_sparse indicator:", flush=True)
    print(f"       Include firm_stats['flag_any'] as a robustness control.", flush=True)
    print(f"    C) Revisit step7b_build_control_psm_vars.py:", flush=True)
    print(f"       Add a symmetric MIN_POST_PATENTS={MIN_POST_PATENTS} filter "
          f"before building the candidate pool, then re-run step7b and step7c.",
          flush=True)
    print(f"\n  Option C is the cleanest fix: it prevents under-active control firms",
          flush=True)
    print(f"  from ever entering the matching pool, exactly mirroring the treated-side",
          flush=True)
    print(f"  eligibility logic in step3c.", flush=True)

# =============================================================================
# SAVE OUTPUTS
# =============================================================================
firm_stats.to_csv(OUTPUT_FIRM_PATH, index=False)
flagged_pairs = matches[matches["pair_flag"] == 1].copy()
flagged_pairs.to_csv(OUTPUT_PAIRS_PATH, index=False)

print(f"\n{'='*65}", flush=True)
print(f"SAVED", flush=True)
print(f"{'='*65}", flush=True)
print(f"  {OUTPUT_FIRM_PATH:<40} — per-firm coverage diagnostics "
      f"({len(firm_stats):,} rows)", flush=True)
print(f"  {OUTPUT_PAIRS_PATH:<40} — flagged matched pairs "
      f"({len(flagged_pairs):,} rows)", flush=True)

print(f"""
Next step: step8_did_analysis.ipynb
  If flagged pairs exist, filter panel_full.csv before running regressions:

    import pandas as pd
    panel     = pd.read_csv("panel_full.csv")
    flagged   = pd.read_csv("post_patent_flagged_pairs.csv")
    bad_ids   = set(flagged["bvd_id"]) | set(flagged["control_assignee"])
    panel_clean = panel[~panel["bvd_id"].isin(bad_ids)]
    panel_clean.to_csv("panel_clean.csv", index=False)

  Then run the DiD on panel_clean.csv instead of panel_full.csv.
""", flush=True)
