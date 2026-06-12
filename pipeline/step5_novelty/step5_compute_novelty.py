"""
Step 5 — Compute Patent Novelty Scores
========================================
PRIOR ART SOURCE:
  patents_eligible  → patents_prior_art.csv  (full universe 2010-2024)
  patents_validation → patents_validation.csv (2005-2009 as prior art
                       for 2010-2015 focal patents — self-referential)

Inputs:
  data/interim/patents_eligible_controls.csv + data/interim/embeddings_eligible.npy
  data/interim/patents_validation.csv        + data/interim/embeddings_validation.npy
  data/interim/patents_prior_art.csv         + data/interim/embeddings_prior_art.npy
  data/interim/patents_control.csv           + data/interim/embeddings_control.npy

Outputs:
  data/interim/patents_eligible_novelty.csv   — updated with novelty_score
  data/interim/patents_validation_novelty.csv — updated with novelty_score
  data/interim/patents_control_novelty.csv    — updated with novelty_score

Install:
  pip install faiss-cpu numpy pandas
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
PRIOR_ART_YEARS          = 5
TOP_K                    = 2000                           
MIN_PRIOR_ART            = 1    
CPC_CHARS                = 4    
VALIDATION_NOVELTY_START = pd.Timestamp("2010-01-01")


# =============================================================================
# SETUP
# =============================================================================
print("="*60, flush=True)
print("STEP 5 — PATENT NOVELTY SCORES (SINGLE INDEX PER SUBCLASS)", flush=True)
print("="*60, flush=True)
print(f"Method:           Backward Similarity", flush=True)
print(f"Formula:          novelty_it = 1 - B_it", flush=True)
print(f"Prior art window: {PRIOR_ART_YEARS} years, same {CPC_CHARS}-char CPC subgroup", flush=True)
print(f"FAISS strategy:   one index per subclass, date filter after search", flush=True)
print(f"TOP_K candidates: {TOP_K}", flush=True)

try:
    import faiss
    print(f"\nFAISS: OK", flush=True)
except ImportError:
    import subprocess
    print("faiss not found — installing...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "faiss-cpu", "--quiet"])
    import faiss
    print("FAISS installed: OK", flush=True)

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
# LOAD PRIOR ART
# =============================================================================
def load_prior_art(csv_path, emb_path):
    """
    Loads the prior art reference dataset and embeddings.
    Returns DataFrame with filing_date and cpc_subgroup columns,
    and the corresponding embedding array.
    """
    print(f"\nLoading prior art: {csv_path}...", flush=True)
    df = pd.read_csv(csv_path, low_memory=False,
                     usecols=lambda c: c in [
                         "publication_number", "filing_date", "cpc_code"
                     ])
    df["filing_date"]  = parse_filing_date(df["filing_date"])
    df["cpc_subgroup"] = df["cpc_code"].str[:CPC_CHARS].str.upper()
    df = df.reset_index(drop=True)

    print(f"  Rows:       {len(df):,}", flush=True)
    print(f"  Date range: {df['filing_date'].min().date()} "
          f"to {df['filing_date'].max().date()}", flush=True)
    print(f"  Subgroups:  {df['cpc_subgroup'].nunique():,}", flush=True)

    print(f"Loading embeddings: {emb_path}...", flush=True)
    embs = np.load(emb_path).astype("float32")
    print(f"  Shape:      {embs.shape}", flush=True)

    if len(df) != embs.shape[0]:
        raise ValueError(
            f"Row count mismatch: CSV {len(df):,} vs "
            f"embeddings {embs.shape[0]:,}\n"
            f"Re-run Step 4 if {csv_path} was modified after embedding."
        )

    norms = np.linalg.norm(embs, axis=1)
    print(f"  Norms:      min={norms.min():.4f}  max={norms.max():.4f} "
          f"(should be ~1.0)", flush=True)
    if norms.min() < 0.98:
        print("  WARNING: embeddings not normalised — re-run Step 4 with "
              "normalize_embeddings=True", flush=True)

    return df, embs

# =============================================================================
# BUILD FAISS INDICES — ONE PER CPC SUBGROUP
# =============================================================================
def build_subclass_indices(prior_df, prior_embs):
    """
    Builds one FAISS IndexFlatIP per CPC subgroup.

    Returns dict:
      subgroup_string → {
        "index":    faiss.IndexFlatIP   pre-built index for this subgroup
        "dates":    numpy array         filing dates of patents in this subgroup
                                        index position matches FAISS result index
      }

    The dates array is used to filter FAISS results to the 5-year window
    after search, without needing to rebuild the index per patent.
    """
    print(f"\nBuilding FAISS indices (one per CPC subgroup)...", flush=True)

    subclasses  = sorted(prior_df["cpc_subgroup"].dropna().unique())
    indices     = {}
    build_start = time.time()

    for i, subclass in enumerate(subclasses):
        mask    = prior_df["cpc_subgroup"] == subclass
        row_pos = np.where(mask)[0]

        if len(row_pos) == 0:
            continue

        embs  = prior_embs[row_pos].astype("float32")
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        # Store dates in the same order as rows added to the index
        # so FAISS result index i → dates[i] → filing date
        indices[subclass] = {
            "index": index,
            "dates": prior_df.loc[row_pos, "filing_date"].values,
        }

        elapsed = time.time() - build_start
        print(
            f"  [{i+1:>4}/{len(subclasses)}]  {subclass:<10}  "
            f"{len(row_pos):>8,} patents  "
            f"{elapsed:.1f}s",
            flush=True
        )

    elapsed = time.time() - build_start
    print(f"\n  All {len(indices):,} indices built in {elapsed:.1f}s", flush=True)

    return indices

# =============================================================================
# COMPUTE NOVELTY SCORES
# =============================================================================
def compute_novelty_scores(focal_df, focal_embs, subclass_indices,
                           novelty_start_date=None):
    """
    Queries pre-built FAISS indices to compute backward similarity novelty.

    For each focal patent:
      1. Look up the pre-built FAISS index for its CPC subgroup
      2. Retrieve top-K candidates from the full subgroup index
      3. Filter candidates to those within the 5-year prior art window
         using the stored dates array
      4. Average similarities of filtered results → backward similarity
      5. novelty = 1 - backward_similarity

    This avoids rebuilding the index for every patent — queries are
    microseconds each once the index is built.
    """
    novelty_scores = np.full(len(focal_df), np.nan)

    focal_df = focal_df.reset_index(drop=True).copy()
    focal_df["filing_date"]  = parse_filing_date(focal_df["filing_date"])
    focal_df["cpc_subgroup"] = focal_df["cpc_code"].str[:CPC_CHARS].str.upper()

    # Determine focal patents
    if novelty_start_date is not None:
        focal_mask = focal_df["filing_date"] >= novelty_start_date
        print(f"  Focal (>= {novelty_start_date.date()}): "
              f"{focal_mask.sum():,}", flush=True)
        print(f"  Prior art only:           "
              f"{(~focal_mask).sum():,}", flush=True)
    else:
        focal_mask = pd.Series(True, index=focal_df.index)
        print(f"  All {len(focal_df):,} patents are focal", flush=True)

    total_focal = int(focal_mask.sum())
    processed   = 0
    no_prior    = 0
    no_index    = 0
    start_time  = time.time()

    print(f"\n  {'Done':>8}  {'Pct':>6}  {'Rate/m':>8}  "
          f"{'ETA':>7}  Last subgroup", flush=True)
    print(f"  {'-'*52}", flush=True)

    for idx in focal_df[focal_mask].index:
        row         = focal_df.loc[idx]
        filing_date = row["filing_date"]
        subclass    = row["cpc_subgroup"]

        if pd.isna(filing_date) or pd.isna(subclass):
            processed += 1
            continue

        # Subclass not in prior art → assign max novelty
        if subclass not in subclass_indices:
            novelty_scores[idx] = 1.0
            no_index  += 1
            processed += 1
            continue

        entry       = subclass_indices[subclass]
        index       = entry["index"]
        prior_dates = entry["dates"]

        # 5-year prior art window
        window_start = filing_date - pd.DateOffset(years=PRIOR_ART_YEARS)
        window_end   = filing_date - pd.Timedelta(days=1)

        # Query pre-built index — top-K candidates from full subgroup
        query = focal_embs[idx:idx+1].astype("float32")
        k     = min(TOP_K, index.ntotal)
        D, I  = index.search(query, k)   # D=(1,k) similarities, I=(1,k) indices

        # Filter to prior art window using stored dates
        valid_sims = []
        for rank, faiss_idx in enumerate(I[0]):
            if faiss_idx < 0:
                continue
            date = pd.Timestamp(prior_dates[faiss_idx])
            if window_start <= date <= window_end:
                valid_sims.append(D[0][rank])

        if len(valid_sims) < MIN_PRIOR_ART:
            novelty_scores[idx] = 1.0
            no_prior  += 1
        else:
            backward_sim        = float(np.clip(np.mean(valid_sims), 0.0, 1.0))
            novelty_scores[idx] = 1.0 - backward_sim

        processed += 1

        # Progress every 500 patents
        if processed % 500 == 0 or processed == total_focal:
            elapsed = time.time() - start_time
            rate    = processed / elapsed * 60 if elapsed > 0 else 0
            eta     = (total_focal - processed) / (rate / 60) / 60 \
                      if rate > 0 else 0
            print(
                f"  {processed:>8,}  "
                f"{processed/total_focal*100:>5.1f}%  "
                f"{rate:>7.0f}/m  "
                f"{eta:>5.1f}min  "
                f"{subclass}",
                flush=True
            )

    elapsed  = time.time() - start_time
    n_scored = int(np.sum(~np.isnan(novelty_scores)))

    print(f"\n  Complete in {elapsed/60:.1f} minutes", flush=True)
    print(f"  Scored:          {n_scored:,}", flush=True)
    print(f"  No prior art:    {no_prior:,}  (assigned novelty = 1.0)", flush=True)
    print(f"  No subclass idx: {no_index:,}  (subclass not in prior art)", flush=True)

    return novelty_scores

# =============================================================================
# DATASETS TO PROCESS
# =============================================================================
DATASETS = [
    {
        "name":               "eligible",
        "focal_csv":          "data/interim/patents_eligible_controls.csv",
        "focal_emb":          "data/interim/embeddings_eligible.npy",
        "prior_csv":          "data/interim/patents_prior_art.csv",
        "prior_emb":          "data/interim/embeddings_prior_art.npy",
        "output_csv":         "data/interim/patents_eligible_novelty.csv",
        "novelty_start_date": None,
    },
    {
        "name":               "validation",
        "focal_csv":          "data/interim/patents_validation.csv",
        "focal_emb":          "data/interim/embeddings_validation.npy",
        "prior_csv":          "data/interim/patents_validation.csv",
        "prior_emb":          "data/interim/embeddings_validation.npy",
        "output_csv":         "data/interim/patents_validation_novelty.csv",
        "novelty_start_date": VALIDATION_NOVELTY_START,
    },
    {
        "name":               "control",
        "focal_csv":          "data/interim/patents_control.csv",
        "focal_emb":          "data/interim/embeddings_control.npy",
        "prior_csv":          "data/interim/patents_prior_art.csv",
        "prior_emb":          "data/interim/embeddings_prior_art.npy",
        "output_csv":         "data/interim/patents_control_novelty.csv",
        "novelty_start_date": None,
    },
]

# =============================================================================
# MAIN LOOP
# =============================================================================
for dataset in DATASETS:
    name               = dataset["name"]
    focal_csv          = dataset["focal_csv"]
    focal_emb_path     = dataset["focal_emb"]
    prior_csv          = dataset["prior_csv"]
    prior_emb_path     = dataset["prior_emb"]
    output_csv         = dataset["output_csv"]
    novelty_start_date = dataset["novelty_start_date"]

    print(f"\n{'='*60}", flush=True)
    print(f"DATASET: {name.upper()}", flush=True)
    print(f"  Focal:     {focal_csv}", flush=True)
    print(f"  Prior art: {prior_csv}", flush=True)
    print(f"{'='*60}", flush=True)

    # Check files exist
    missing = [p for p in [focal_csv, focal_emb_path, prior_csv, prior_emb_path]
               if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Missing files: {missing}", flush=True)
        print("Run Step 4 first to generate all embedding files", flush=True)
        continue

    # Load focal dataset
    print(f"\nLoading focal: {focal_csv}...", flush=True)
    focal_df = pd.read_csv(focal_csv, low_memory=False)
    print(f"  Rows: {len(focal_df):,}", flush=True)

    focal_embs = np.load(focal_emb_path).astype("float32")
    print(f"  Embeddings: {focal_embs.shape}", flush=True)

    if len(focal_df) != focal_embs.shape[0]:
        print(f"ERROR: {focal_csv} {len(focal_df):,} rows != "
              f"embeddings {focal_embs.shape[0]:,}", flush=True)
        print("Re-run Step 4", flush=True)
        continue

    # Check existing novelty scores
    if "novelty_score" in focal_df.columns:
        n_exist = focal_df["novelty_score"].notna().sum()
        print(f"\n  Existing novelty_score: {n_exist:,} computed", flush=True)
        ans = input("  Re-compute from scratch? (yes/no): ").strip().lower()
        if ans != "yes":
            print("  Skipping", flush=True)
            continue
        focal_df["novelty_score"] = np.nan
        print("  Cleared — recomputing...", flush=True)

    # Load prior art
    if prior_csv == focal_csv:
        # Validation: use same dataset as prior art reference
        print(f"\nUsing {focal_csv} as its own prior art reference", flush=True)
        print(f"  2005-2009 = prior art pool", flush=True)
        print(f"  2010-2015 = focal patents", flush=True)
        prior_df   = focal_df.copy()
        prior_df["filing_date"]  = parse_filing_date(prior_df["filing_date"])
        prior_df["cpc_subgroup"] = prior_df["cpc_code"].str[:CPC_CHARS].str.upper()
        prior_df = prior_df.reset_index(drop=True)
        prior_embs = focal_embs.copy()
        print(f"  Prior art rows:  {len(prior_df):,}", flush=True)
        print(f"  Prior art shape: {prior_embs.shape}", flush=True)
    else:
        prior_df, prior_embs = load_prior_art(prior_csv, prior_emb_path)

    # Build one FAISS index per subclass
    subclass_indices = build_subclass_indices(prior_df, prior_embs)

    # Compute novelty
    print(f"\nComputing novelty scores...", flush=True)
    scores = compute_novelty_scores(
        focal_df, focal_embs, subclass_indices, novelty_start_date
    )

    # Save
    focal_df["novelty_score"] = scores
    focal_df.to_csv(output_csv, index=False)

    valid = focal_df["novelty_score"].notna()
    print(f"\n{'='*50}", flush=True)
    print(f"RESULTS — {name.upper()}", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Scored:  {valid.sum():,}  ({valid.mean()*100:.1f}%)", flush=True)
    print(f"  Missing: {(~valid).sum():,}", flush=True)

    if valid.sum() > 0:
        print(f"\n{focal_df['novelty_score'].describe().round(4)}", flush=True)
        m = float(focal_df["novelty_score"].mean())
        if   m < 0.10:
            print("  WARNING: mean very low — check normalisation", flush=True)
        elif m > 0.95:
            print("  WARNING: mean very high — prior art too sparse", flush=True)
        else:
            print(f"  Mean {m:.3f} looks correct (expected 0.2-0.7)", flush=True)

        print(f"\n  Mean novelty by top-level CPC class:", flush=True)
        print(
            focal_df[valid]
            .groupby(focal_df["cpc_code"].str[:4])["novelty_score"]
            .agg(mean="mean", count="count")
            .sort_values("count", ascending=False)
            .round(4)
            .to_string(),
            flush=True
        )

    print(f"\nSaved {output_csv}", flush=True)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*60}", flush=True)
print("STEP 5 COMPLETE", flush=True)
print(f"{'='*60}", flush=True)

for d in DATASETS:
    if os.path.exists(d["output_csv"]):
        tmp = pd.read_csv(d["output_csv"], usecols=["novelty_score"])
        n   = tmp["novelty_score"].notna().sum()
        print(
            f"\n  {d['name']:12}  scored={n:,}/{len(tmp):,}  "
            f"mean={tmp['novelty_score'].mean():.4f}  "
            f"std={tmp['novelty_score'].std():.4f}",
            flush=True
        )

print(f"""
Architecture:
  FAISS indices built ONCE per {CPC_CHARS}-char CPC subgroup.
  Top-{TOP_K} candidates retrieved per focal patent, filtered to {PRIOR_ART_YEARS}-year window.
  patents_eligible prior art: data/interim/patents_prior_art.csv (full universe)
  patents_validation prior art: self (2005-2009 pool for 2010-2015 focal)

Next steps:
  1. Mean novelty should be 0.2-0.7 for all datasets
  2. Run step6a_check_acquiror_coverage.py  — verify acquiror coverage before overlap computation
  3. Run step6b_compute_overlap.py   — knowledge overlap (H2 moderator)
  4. Run step7a_psm_diagnostics.py — begin PSM matching and panel construction
""", flush=True)