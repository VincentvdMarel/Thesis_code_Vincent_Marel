"""
Step 6b — Compute Knowledge Overlap
=====================================
Computes pre-merger knowledge overlap between acquiror and target
patent portfolio centroids, following Han, Jo and Kang (2018) as
specified in the thesis proposal (section 3.2).

For each deal:
  1. Compute target centroid  VT = mean of target SBERT embeddings
                                   for patents filed in 5yr pre-merger window
  2. Compute acquiror centroid VA = mean of acquiror SBERT embeddings
                                   for patents filed in 5yr pre-merger window
  3. Overlap = cosine similarity(VA, VT) = (VA · VT) / (||VA|| × ||VT||)

This is the moderator variable (Overlap_i) used in the H2 moderation:
  gamma4 * (Overlap x Treated x Post)

Inputs:
  patents_eligible.csv         — target firm patents with bvd_id, filing_date
  embeddings_eligible.npy      — target firm patent embeddings (normalised)
  patents_acquiror.csv         — acquiror patents with assignee_name, filing_date
  embeddings_acquiror.npy      — acquiror patent embeddings (normalised)
  acquiror_name_mapping.csv    — Orbis acquiror name -> USPTO assignee name

Outputs:
  knowledge_overlap.csv                       — one row per bvd_id with overlap score
  data/interim/knowledge_overlap.csv          — one row per firm with overlap score
  data/interim/patents_eligible_overlap.csv   — patents_eligible with knowledge_overlap added
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
PRE_MERGER_YEARS      = 5

PATENTS_ELIGIBLE_PATH = "data/interim/patents_eligible_novelty.csv"
EMBEDDINGS_ELIGIBLE   = "data/interim/embeddings_eligible.npy"
PATENTS_ACQUIROR_PATH = "data/raw/patents_acquiror.csv"
EMBEDDINGS_ACQUIROR   = "data/interim/embeddings_acquiror.npy"
NAME_MAP_PATH         = "data/interim/acquiror_name_map.csv"
OUTPUT_PATH           = "data/interim/knowledge_overlap.csv"
OVERLAP_OUTPUT_PATH   = "data/interim/patents_eligible_overlap.csv"

# =============================================================================
# SETUP
# =============================================================================
print("="*60, flush=True)
print("STEP 6b — KNOWLEDGE OVERLAP", flush=True)
print("="*60, flush=True)
print(f"Method:            Centroid cosine similarity (Han et al. 2018)", flush=True)
print(f"Formula:           Overlap = (VA·VT) / (||VA|| x ||VT||)", flush=True)
print(f"Pre-merger window: {PRE_MERGER_YEARS} years before deal date", flush=True)

# =============================================================================
# DATE PARSING
# =============================================================================
def parse_filing_date(col):
    """Handles YYYYMMDD integers (BigQuery) and ISO date strings."""
    parsed = pd.to_datetime(
        col.astype(str).str.strip(), format="%Y%m%d", errors="coerce"
    )
    failed = parsed.isna() & col.notna()
    if failed.any():
        parsed[failed] = pd.to_datetime(col[failed], errors="coerce")
    return parsed

# =============================================================================
# CHECK ALL INPUT FILES EXIST
# =============================================================================
required = [PATENTS_ELIGIBLE_PATH, EMBEDDINGS_ELIGIBLE,
            PATENTS_ACQUIROR_PATH, EMBEDDINGS_ACQUIROR,
            NAME_MAP_PATH]
missing  = [p for p in required if not os.path.exists(p)]
if missing:
    print(f"ERROR: Missing files: {missing}", flush=True)
    sys.exit(1)

# =============================================================================
# LOAD NAME MAPPING
# Maps Orbis acquiror names -> USPTO assignee name variants
# =============================================================================
print(f"\nLoading {NAME_MAP_PATH}...", flush=True)
name_mapping = pd.read_csv(NAME_MAP_PATH)
mapped = name_mapping[name_mapping["status"] == "MAPPED"].copy()
mapped["orbis_upper"] = mapped["orbis_acquiror_name"].str.upper().str.strip()
mapped["bq_upper"]    = mapped["bq_acquiror_name"].str.upper().str.strip()

# Build lookup: Orbis name (upper) -> list of USPTO names
# One acquiror can have multiple USPTO name variants
orbis_to_uspto = (
    mapped.groupby("orbis_upper")["bq_upper"]
    .apply(list)
    .to_dict()
)

print(f"  Total rows:              {len(name_mapping):,}", flush=True)
print(f"  MAPPED rows:             {len(mapped):,}", flush=True)
print(f"  NEEDS REVIEW (skipped):  {(name_mapping['status'] == 'NEEDS REVIEW').sum():,}", flush=True)
print(f"  Unique Orbis acquirors:  {len(orbis_to_uspto):,}", flush=True)
# =============================================================================
# LOAD TARGET PATENTS AND EMBEDDINGS
# =============================================================================
print(f"\nLoading {PATENTS_ELIGIBLE_PATH}...", flush=True)
target_df = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)
target_df["bvd_id"]      = target_df["bvd_id"].astype(str).str.strip()
target_df["filing_date"] = parse_filing_date(target_df["filing_date"])
target_df["deal_date"]   = pd.to_datetime(target_df["deal_date"], errors="coerce")
target_df = target_df.reset_index(drop=True)
print(f"  Rows: {len(target_df):,}  BvD IDs: {target_df['bvd_id'].nunique():,}", flush=True)

# Build deal_info — one row per firm with deal_date and acquiror name
deal_info = (
    target_df[["bvd_id", "deal_date", "Acquiror name"]]
    .drop_duplicates(subset="bvd_id")
    .copy()
    .reset_index(drop=True)
)
deal_info["orbis_upper"] = deal_info["Acquiror name"].str.upper().str.strip()
print(f"  Unique BvD IDs (deals): {len(deal_info):,}", flush=True)

print(f"Loading {EMBEDDINGS_ELIGIBLE}...", flush=True)
target_embs = np.load(EMBEDDINGS_ELIGIBLE).astype("float32")
print(f"  Shape: {target_embs.shape}", flush=True)

if len(target_df) != target_embs.shape[0]:
    print(f"ERROR: target CSV {len(target_df):,} rows != "
          f"embeddings {target_embs.shape[0]:,}", flush=True)
    print("Re-run Step 4 alignment fix", flush=True)
    sys.exit(1)

# =============================================================================
# LOAD ACQUIROR PATENTS AND EMBEDDINGS
# =============================================================================
print(f"\nLoading {PATENTS_ACQUIROR_PATH}...", flush=True)
acquiror_df = pd.read_csv(PATENTS_ACQUIROR_PATH, low_memory=False)
acquiror_df["filing_date"]    = parse_filing_date(acquiror_df["filing_date"])
acquiror_df["assignee_upper"] = acquiror_df["assignee_name"].str.upper().str.strip()
acquiror_df = acquiror_df.reset_index(drop=True)
acquiror_df["emb_row"] = acquiror_df.index
print(f"  Rows: {len(acquiror_df):,}  "
      f"Assignees: {acquiror_df['assignee_upper'].nunique():,}", flush=True)

print(f"Loading {EMBEDDINGS_ACQUIROR}...", flush=True)
acquiror_embs = np.load(EMBEDDINGS_ACQUIROR).astype("float32")
print(f"  Shape: {acquiror_embs.shape}", flush=True)

if len(acquiror_df) != acquiror_embs.shape[0]:
    print(f"ERROR: acquiror CSV {len(acquiror_df):,} rows != "
          f"embeddings {acquiror_embs.shape[0]:,}", flush=True)
    sys.exit(1)

# Verify normalisation
for label, embs in [("target", target_embs), ("acquiror", acquiror_embs)]:
    norms = np.linalg.norm(embs, axis=1)
    print(f"  {label} embedding norms: "
          f"min={norms.min():.4f}  max={norms.max():.4f}", flush=True)
    if norms.min() < 0.98:
        print(f"  WARNING: {label} embeddings not normalised — "
              f"re-run Step 4 with normalize_embeddings=True", flush=True)

# =============================================================================
# COMPUTE KNOWLEDGE OVERLAP PER FIRM
# =============================================================================
print(f"\nComputing knowledge overlap for {len(deal_info):,} firms...", flush=True)
print(f"Pre-merger window: {PRE_MERGER_YEARS} years before deal date\n", flush=True)
print(f"  {'BvD ID':<20} {'Overlap':>8}  {'Tgt pts':>8}  {'Acq pts':>8}  Acquiror",
      flush=True)
print(f"  {'-'*70}", flush=True)

results    = []
start_time = time.time()

for _, deal in deal_info.iterrows():
    bvd_id      = deal["bvd_id"]
    deal_date   = deal["deal_date"]
    orbis_name  = deal["orbis_upper"]
    acq_display = str(deal["Acquiror name"])[:30]

    if pd.isna(deal_date):
        results.append({
            "bvd_id":            bvd_id,
            "knowledge_overlap": np.nan,
            "target_patents":    0,
            "acquiror_patents":  0,
            "note":              "missing deal date",
        })
        continue

    window_start = deal_date - pd.DateOffset(years=PRE_MERGER_YEARS)
    window_end   = deal_date - pd.Timedelta(days=1)

    # -------------------------------------------------------------------------
    # TARGET CENTROID
    # -------------------------------------------------------------------------
    target_rows = target_df[
        (target_df["bvd_id"]      == bvd_id) &
        (target_df["filing_date"] >= window_start) &
        (target_df["filing_date"] <= window_end)
    ].index.tolist()

    if len(target_rows) == 0:
        results.append({
            "bvd_id":            bvd_id,
            "knowledge_overlap": np.nan,
            "target_patents":    0,
            "acquiror_patents":  0,
            "note":              "no target patents in pre-merger window",
        })
        continue

    target_centroid = target_embs[target_rows].mean(axis=0)
    target_norm     = np.linalg.norm(target_centroid)
    if target_norm > 0:
        target_centroid = target_centroid / target_norm

    # -------------------------------------------------------------------------
    # ACQUIROR CENTROID
    # Use all USPTO name variants from acquiror_name_mapping.csv
    # -------------------------------------------------------------------------
    uspto_names = orbis_to_uspto.get(orbis_name, [])

    if not uspto_names:
        results.append({
            "bvd_id":            bvd_id,
            "knowledge_overlap": np.nan,
            "target_patents":    len(target_rows),
            "acquiror_patents":  0,
            "note":              "acquiror name not in mapping file",
        })
        continue

    acquiror_rows = acquiror_df[
        (acquiror_df["assignee_upper"].isin(uspto_names)) &
        (acquiror_df["filing_date"]   >= window_start) &
        (acquiror_df["filing_date"]   <= window_end)
    ]["emb_row"].tolist()

    if len(acquiror_rows) == 0:
        results.append({
            "bvd_id":            bvd_id,
            "knowledge_overlap": np.nan,
            "target_patents":    len(target_rows),
            "acquiror_patents":  0,
            "note":              "no acquiror patents in pre-merger window",
        })
        continue

    acquiror_centroid = acquiror_embs[acquiror_rows].mean(axis=0)
    acquiror_norm     = np.linalg.norm(acquiror_centroid)
    if acquiror_norm > 0:
        acquiror_centroid = acquiror_centroid / acquiror_norm

    # -------------------------------------------------------------------------
    # COSINE SIMILARITY
    # Both centroids are re-normalised so dot product = cosine similarity
    # -------------------------------------------------------------------------
    overlap = float(np.clip(np.dot(target_centroid, acquiror_centroid), -1.0, 1.0))

    results.append({
        "bvd_id":            bvd_id,
        "knowledge_overlap": overlap,
        "target_patents":    len(target_rows),
        "acquiror_patents":  len(acquiror_rows),
        "note":              "",
    })

    print(
        f"  {bvd_id:<20} {overlap:>8.4f}  "
        f"{len(target_rows):>8,}  {len(acquiror_rows):>8,}  {acq_display}",
        flush=True
    )

elapsed = time.time() - start_time

# =============================================================================
# RESULTS AND DIAGNOSTICS
# =============================================================================
overlap_df = pd.DataFrame(results)
valid      = overlap_df["knowledge_overlap"].notna()
missing    = ~valid

print(f"\n{'='*60}", flush=True)
print(f"KNOWLEDGE OVERLAP RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Total firms:              {len(overlap_df):,}", flush=True)
print(f"  With overlap score:       {valid.sum():,}  ({valid.mean()*100:.1f}%)", flush=True)
print(f"  Missing overlap:          {missing.sum():,}", flush=True)
print(f"  Computation time:         {elapsed:.1f}s", flush=True)

if valid.sum() > 0:
    print(f"\n  Distribution of knowledge_overlap:", flush=True)
    print(overlap_df["knowledge_overlap"].describe().round(4), flush=True)

    m = float(overlap_df["knowledge_overlap"].mean())
    if m > 0.9:
        print("  WARNING: mean very high — check acquiror/target are different",
              flush=True)
    else:
        print(f"  Mean {m:.3f} looks plausible", flush=True)

    lo  = (overlap_df["knowledge_overlap"] < 0.3).sum()
    mid = ((overlap_df["knowledge_overlap"] >= 0.3) &
           (overlap_df["knowledge_overlap"] < 0.6)).sum()
    hi  = (overlap_df["knowledge_overlap"] >= 0.6).sum()
    print(f"\n  Low overlap  (< 0.3):    {lo:,}  — diversifying acquisitions",
          flush=True)
    print(f"  Mid overlap  (0.3-0.6):  {mid:,}  — related acquisitions",
          flush=True)
    print(f"  High overlap (>= 0.6):   {hi:,}  — closely related acquisitions",
          flush=True)

if missing.sum() > 0:
    print(f"\n  Missing overlap breakdown:", flush=True)
    print(overlap_df[missing]["note"].value_counts().to_string(), flush=True)
    print(f"\n  Firms with missing overlap excluded from H2 moderation regression",
          flush=True)

# =============================================================================
# SAVE knowledge_overlap.csv
# =============================================================================
overlap_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}", flush=True)
print(f"  Rows:    {len(overlap_df):,}", flush=True)
print(f"  Columns: {list(overlap_df.columns)}", flush=True)

# =============================================================================
# MERGE knowledge_overlap INTO patents_eligible.csv
# Each patent row for a firm gets the same knowledge_overlap value
# Firms with NaN overlap get NaN — excluded from H2 moderation regression
# =============================================================================
print(f"\nMerging knowledge_overlap into {PATENTS_ELIGIBLE_PATH}...", flush=True)

patents_eligible = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)
patents_eligible["bvd_id"] = patents_eligible["bvd_id"].astype(str).str.strip()
print(f"  Loaded: {len(patents_eligible):,} rows", flush=True)

# Drop existing column to avoid duplicates on re-run
patents_eligible = patents_eligible.drop(
    columns=["knowledge_overlap"],
    errors="ignore"
)

# Merge — one overlap value per bvd_id broadcast to all patent rows
patents_eligible = patents_eligible.merge(
    overlap_df[["bvd_id", "knowledge_overlap"]],
    on="bvd_id",
    how="left"
)

# Coverage report
total_firms   = patents_eligible["bvd_id"].nunique()
firms_with    = (
    patents_eligible.groupby("bvd_id")["knowledge_overlap"]
    .first().notna().sum()
)
firms_without = total_firms - firms_with

print(f"\n  knowledge_overlap coverage:", flush=True)
print(f"    Firms with score:  {firms_with:,}/{total_firms:,}  "
      f"({firms_with/total_firms*100:.1f}%)", flush=True)
print(f"    Firms with NaN:    {firms_without:,}/{total_firms:,}  "
      f"({firms_without/total_firms*100:.1f}%)", flush=True)

if firms_with > 0:
    valid_vals = (
        patents_eligible.groupby("bvd_id")["knowledge_overlap"]
        .first().dropna()
    )
    print(f"\n  Distribution (firms with score):", flush=True)
    print(f"    Mean:    {valid_vals.mean():.4f}", flush=True)
    print(f"    Median:  {valid_vals.median():.4f}", flush=True)
    print(f"    Std:     {valid_vals.std():.4f}", flush=True)
    print(f"    Min:     {valid_vals.min():.4f}", flush=True)
    print(f"    Max:     {valid_vals.max():.4f}", flush=True)

patents_eligible.to_csv(OVERLAP_OUTPUT_PATH, index=False)
print(f"\nSaved updated {OVERLAP_OUTPUT_PATH}", flush=True)
print(f"  Total rows:    {len(patents_eligible):,}", flush=True)
print(f"  Total columns: {len(patents_eligible.columns)}", flush=True)

print(f"""
{'='*60}
STEP 6b COMPLETE
{'='*60}

knowledge_overlap.csv  — one row per bvd_id (standalone file)
patents_eligible.csv   — updated with knowledge_overlap column

knowledge_overlap column:
  float  cosine similarity between acquiror and target centroids
  NaN    no acquiror patents found / name not in mapping file
         these firms are excluded from H2 moderation regression

This is the moderator variable for H2:
  gamma4 * (Overlap x Treated x Post)

Next step: step7a_psm_diagnostics.py
  Reads  data/interim/patents_eligible_overlap.csv
  Builds PSM matching variables for treated firms
  Output: psm_diagnostics.csv — ready for PSM matching
""", flush=True)