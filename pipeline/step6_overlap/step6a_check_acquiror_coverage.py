"""
Step 6a Pre-check — Verify patents_acquiror.csv Coverage
=========================================================
Before running SBERT embeddings on patents_acquiror.csv, this script
checks whether all acquirors from patents_eligible.csv are present
in patents_acquiror.csv and have sufficient pre-merger patents for
knowledge overlap computation.

Inputs:
  patents_eligible.csv   — contains Acquiror name and deal_date per BvD ID
  patents_acquiror.csv   — acquiror patent data from BigQuery
  acquiror_name_map.csv  — mapping between Orbis and BigQuery names

Output:
  Console report showing which acquirors are found, missing, or low-coverage
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PATENTS_ELIGIBLE_PATH = "data/interim/patents_eligible_novelty.csv"
PATENTS_ACQUIROR_PATH = "data/raw/patents_acquiror.csv"
NAME_MAP_PATH         = "data/interim/acquiror_name_map.csv"
PRE_MERGER_YEARS      = 5     # 5-year pre-merger window
MIN_PATENTS           = 3     # minimum pre-merger patents needed for centroid

# =============================================================================
# LOAD
# =============================================================================
print("="*65, flush=True)
print("STEP 6a PRE-CHECK — ACQUIROR COVERAGE VERIFICATION", flush=True)
print("="*65, flush=True)

for path in [PATENTS_ELIGIBLE_PATH, PATENTS_ACQUIROR_PATH, NAME_MAP_PATH]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", flush=True)
        sys.exit(1)

print(f"\nLoading {PATENTS_ELIGIBLE_PATH}...", flush=True)
eligible = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)
eligible["filing_date"] = pd.to_datetime(
    eligible["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
eligible["deal_date"] = pd.to_datetime(eligible["deal_date"], errors="coerce")
print(f"  Rows:          {len(eligible):,}", flush=True)
print(f"  Unique firms:  {eligible['bvd_id'].nunique():,}", flush=True)

print(f"\nLoading {PATENTS_ACQUIROR_PATH}...", flush=True)
acquiror = pd.read_csv(PATENTS_ACQUIROR_PATH, low_memory=False)
acquiror["filing_date"] = pd.to_datetime(
    acquiror["filing_date"].astype(str), format="%Y%m%d", errors="coerce"
)
print(f"  Rows:          {len(acquiror):,}", flush=True)
print(f"  Unique names:  {acquiror['assignee_name'].nunique():,}", flush=True)

print(f"\nLoading {NAME_MAP_PATH}...", flush=True)
name_map = pd.read_csv(NAME_MAP_PATH, low_memory=False)
bq_names_per_orbis = name_map.groupby("orbis_acquiror_name")["bq_acquiror_name"].apply(list).to_dict()

# =============================================================================
# EXTRACT ONE ACQUIROR NAME + DEAL DATE PER FIRM
# =============================================================================
deal_info = (
    eligible[["bvd_id", "orbis_target_name", "deal_date", "Acquiror name"]]
    .drop_duplicates(subset="bvd_id")
    .copy()
)

# Standardise names for matching
deal_info["acq_upper"] = (
    deal_info["Acquiror name"].str.upper().str.strip()
)
acquiror["assignee_upper"] = (
    acquiror["assignee_name"].str.upper().str.strip()
)

file_names = set(acquiror["assignee_upper"].dropna().unique())
cw_names   = set(deal_info["acq_upper"].dropna().unique())

print(f"\nUnique acquiror names in patents_eligible:  {len(cw_names):,}")
print(f"Unique assignee names in patents_acquiror:  {len(file_names):,}", flush=True)

# =============================================================================
# EXACT NAME MATCH (VIA BQ NAME MAP)
# =============================================================================
exact_found = set()
exact_missing_upper = set()  # Used for fuzzy matching below

# Check if at least one BigQuery mapped name exists for each Orbis acquiror
for acq_orig in deal_info["Acquiror name"].dropna().unique():
    acq_upper = str(acq_orig).upper().strip()
    bq_variants = bq_names_per_orbis.get(acq_orig, [acq_orig])
    
    # Uppercase mapped variants for safe comparison against file_names
    bq_variants_upper = [str(v).upper().strip() for v in bq_variants]
    
    if any(var in file_names for var in bq_variants_upper):
        exact_found.add(acq_orig)
    else:
        exact_missing_upper.add(acq_upper)

cw_names_total = len(deal_info["Acquiror name"].dropna().unique())

print(f"\n{'='*65}", flush=True)
print(f"EXACT NAME MATCH RESULTS", flush=True)
print(f"{'='*65}", flush=True)
print(f"  Exact match found:   {len(exact_found):,}/{cw_names_total:,}", flush=True)
print(f"  Not found exactly:   {len(exact_missing_upper):,}/{cw_names_total:,}", flush=True)

# =============================================================================
# FUZZY MATCH FOR MISSING NAMES
# Try to find close matches for names not found exactly
# =============================================================================
if exact_missing_upper:
    print(f"\n{'='*65}", flush=True)
    print(f"FUZZY MATCH FOR {len(exact_missing_upper)} MISSING ACQUIRORS", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Checking if missing names appear as substrings...\n", flush=True)

    fuzzy_found   = {}
    still_missing = []

    for acq_name_upper in sorted(exact_missing_upper):
        # Check if name is a substring of any file name or vice versa
        candidates = [
            fn for fn in file_names
            if acq_name_upper in fn or fn in acq_name_upper or
            # Check first 10 characters match (handles suffix differences)
            acq_name_upper[:10] == fn[:10]
        ]
        if candidates:
            fuzzy_found[acq_name_upper] = candidates
        else:
            still_missing.append(acq_name_upper)

    if fuzzy_found:
        print(f"  Likely matches found ({len(fuzzy_found)}):", flush=True)
        print(f"  {'Crosswalk name':<45} {'Best candidate in file'}", flush=True)
        print(f"  {'-'*80}", flush=True)
        for cw_name, candidates in fuzzy_found.items():
            # Show first candidate as best match
            print(f"  {cw_name:<45} {candidates[0]}", flush=True)

    if still_missing:
        print(f"\n  No match found at all ({len(still_missing)}):", flush=True)
        for name in sorted(still_missing):
            # Find which target firm this acquiror belongs to
            targets = deal_info[deal_info["acq_upper"] == name]["orbis_target_name"].tolist()
            print(f"  {name:<50} → acquired: {', '.join(targets[:2])}", flush=True)

# =============================================================================
# PRE-MERGER PATENT COVERAGE CHECK
# For each deal, count acquiror patents in the 5yr pre-merger window
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"PRE-MERGER PATENT COVERAGE (5yr window per deal)", flush=True)
print(f"{'='*65}", flush=True)

results = []
for _, deal in deal_info.iterrows():
    bvd_id     = deal["bvd_id"]
    target     = deal["orbis_target_name"]
    deal_date  = deal["deal_date"]
    acq_name   = deal["acq_upper"]
    acq_orig   = deal["Acquiror name"]

    if pd.isna(deal_date) or pd.isna(acq_name):
        results.append({
            "bvd_id":          bvd_id,
            "target_name":     target,
            "acquiror_name":   acq_orig,
            "deal_date":       deal_date,
            "pre_patents":     0,
            "status":          "MISSING DEAL DATE OR ACQUIROR NAME",
        })
        continue

    window_start = deal_date - pd.DateOffset(years=PRE_MERGER_YEARS)
    window_end   = deal_date - pd.Timedelta(days=1)

    # For each deal, look up all BQ variants and count pre-merger patents across all of them
    bq_variants = bq_names_per_orbis.get(acq_orig, [acq_orig])
    
    # Standardize casing for robust matching
    bq_variants_upper = [str(v).upper().strip() for v in bq_variants]

    pre = acquiror[
        (acquiror["assignee_upper"].isin(bq_variants_upper)) &
        (acquiror["filing_date"] >= window_start) &
        (acquiror["filing_date"] <= window_end)
    ]

    # If no exact match, try partial name match based on the uppercase string
    if len(pre) == 0 and acq_name in exact_missing_upper:
        # Try first-word match
        first_word = acq_name.split()[0] if acq_name else ""
        candidates = [fn for fn in file_names if fn.startswith(first_word)]
        for candidate in candidates:
            pre = acquiror[
                (acquiror["assignee_upper"] == candidate) &
                (acquiror["filing_date"]    >= window_start) &
                (acquiror["filing_date"]    <= window_end)
            ]
            if len(pre) > 0:
                break

    n_pre = len(pre)

    if n_pre == 0:
        status = "❌ NO PATENTS IN WINDOW"
    elif n_pre < MIN_PATENTS:
        status = f"⚠️  LOW ({n_pre} patents — below MIN={MIN_PATENTS})"
    else:
        status = f"✅ OK ({n_pre} patents)"

    results.append({
        "bvd_id":        bvd_id,
        "target_name":   target,
        "acquiror_name": acq_orig,
        "deal_date":     deal_date,
        "pre_patents":   n_pre,
        "status":        status,
    })

results_df = pd.DataFrame(results)

# Summary counts
ok       = results_df[results_df["pre_patents"] >= MIN_PATENTS]
low      = results_df[(results_df["pre_patents"] > 0) &
                      (results_df["pre_patents"] < MIN_PATENTS)]
zero     = results_df[results_df["pre_patents"] == 0]

print(f"\n  ✅ Sufficient patents (>= {MIN_PATENTS}):  {len(ok):,}/{len(results_df):,}", flush=True)
print(f"  ⚠️  Low patents (1-{MIN_PATENTS-1}):         {len(low):,}/{len(results_df):,}", flush=True)
print(f"  ❌ No patents in window:          {len(zero):,}/{len(results_df):,}", flush=True)

# Show firms with zero patents
if len(zero) > 0:
    print(f"\n  Firms with NO acquiror patents in 5yr pre-merger window:", flush=True)
    print(f"  {'Target':<40} {'Acquiror':<40} {'Deal date'}", flush=True)
    print(f"  {'-'*95}", flush=True)
    for _, row in zero.iterrows():
        print(
            f"  {str(row['target_name'])[:38]:<40} "
            f"{str(row['acquiror_name'])[:38]:<40} "
            f"{str(row['deal_date'])[:10]}",
            flush=True
        )

# Show firms with low patents
if len(low) > 0:
    print(f"\n  Firms with LOW acquiror patents (<{MIN_PATENTS}) in window:", flush=True)
    print(f"  {'Target':<40} {'Acquiror':<35} {'Patents':>8}", flush=True)
    print(f"  {'-'*87}", flush=True)
    for _, row in low.sort_values("pre_patents").iterrows():
        print(
            f"  {str(row['target_name'])[:38]:<40} "
            f"{str(row['acquiror_name'])[:33]:<35} "
            f"{int(row['pre_patents']):>8,}",
            flush=True
        )

# =============================================================================
# DISTRIBUTION OF PRE-MERGER PATENT COUNTS
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"PRE-MERGER PATENT COUNT DISTRIBUTION", flush=True)
print(f"{'='*65}", flush=True)
print(results_df["pre_patents"].describe().round(1), flush=True)

print(f"\n  Top 10 acquirors by pre-merger patent count:", flush=True)
print(f"  {'Acquiror':<45} {'Target':<35} {'Patents':>8}", flush=True)
print(f"  {'-'*92}", flush=True)
for _, row in results_df.sort_values("pre_patents", ascending=False).head(10).iterrows():
    print(
        f"  {str(row['acquiror_name'])[:43]:<45} "
        f"{str(row['target_name'])[:33]:<35} "
        f"{int(row['pre_patents']):>8,}",
        flush=True
    )

# =============================================================================
# FINAL VERDICT
# =============================================================================
print(f"\n{'='*65}", flush=True)
print(f"VERDICT", flush=True)
print(f"{'='*65}", flush=True)

n_overlap_computable = len(ok) + len(low)
pct = n_overlap_computable / len(results_df) * 100

print(f"\n  Firms where overlap CAN be computed:     "
      f"{n_overlap_computable:,}/{len(results_df):,}  ({pct:.1f}%)", flush=True)
print(f"  Firms where overlap CANNOT be computed:  "
      f"{len(zero):,}/{len(results_df):,}", flush=True)

if len(exact_missing_upper) > 0:
    print(f"\n  ACTION REQUIRED — {len(exact_missing_upper)} acquiror names not found exactly:", flush=True)
    print(f"  Check fuzzy matches above and update acquiror_name_map.csv", flush=True)
    print(f"  or add name variants to the BigQuery acquiror query", flush=True)

if len(zero) > 0:
    print(f"\n  ACTION REQUIRED — {len(zero)} firms have no acquiror patents:", flush=True)
    print(f"  These firms will get knowledge_overlap = NaN in Step 6b", flush=True)
    print(f"  and will be excluded from the H2 moderation regression", flush=True)
    print(f"  Consider re-running BigQuery acquiror query with", flush=True)
    print(f"  broader name variants for these acquirors", flush=True)

if len(exact_missing_upper) == 0 and len(zero) == 0:
    print(f"\n  ✅ All acquirors found and have sufficient pre-merger patents", flush=True)
    print(f"  Ready to run step6b_compute_overlap.py", flush=True)

print(f"""
Next step: step6b_compute_overlap.py
  Reads  data/interim/patents_eligible_novelty.csv
  Reads  data/raw/patents_acquiror.csv
  Reads  data/interim/embeddings_acquiror.npy
  Writes data/interim/knowledge_overlap.csv
  Writes data/interim/patents_eligible_overlap.csv
""", flush=True)