"""
Embedding Validity Check
=========================
Verifies that existing .npy embedding files are still valid for their
corresponding CSV input files by checking:

  1. Row count match      — embeddings must have same number of rows as CSV
  2. Publication number   — spot-checks that IDs in *_ids.csv still match CSV
  3. Text hash sample     — re-encodes a small random sample and compares
                            cosine similarity to stored embeddings (should be ~1.0)
  4. Norm check           — all embedding vectors should have norm ~1.0

A result of VALID means you do NOT need to rerun step4_compute_embeddings.py.
A result of INVALID means the CSV has changed and embeddings must be rerun.

Usage:
  python step4_check_embeddings.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
SAMPLE_SIZE = 50    # number of rows to spot-check text hashing
SIMILARITY_THRESHOLD = 0.999  # re-encoded vectors should be ~identical

DATASETS = [
    {
        "name":     "eligible",
        "csv_path": "data/interim/patents_eligible_controls.csv",
        "emb_path": "data/interim/embeddings_eligible.npy",
        "ids_path": "data/interim/embeddings_eligible_ids.csv",
    },
    {
        "name":     "validation",
        "csv_path": "data/interim/patents_validation.csv",
        "emb_path": "data/interim/embeddings_validation.npy",
        "ids_path": "data/interim/embeddings_validation_ids.csv",
    },
    {
        "name":     "prior_art",
        "csv_path": "data/interim/patents_prior_art.csv",
        "emb_path": "data/interim/embeddings_prior_art.npy",
        "ids_path": "data/interim/embeddings_prior_art_ids.csv",
    },
    {
        "name":     "acquiror",
        "csv_path": "data/raw/patents_acquiror.csv",
        "emb_path": "data/interim/embeddings_acquiror.npy",
        "ids_path": "data/interim/embeddings_acquiror_ids.csv",
    },
    {
        "name":     "control",
        "csv_path": "data/interim/patents_control.csv",
        "emb_path": "data/interim/embeddings_control.npy",
        "ids_path": "data/interim/embeddings_control_ids.csv",
    },
]

MAX_CHARS = 1800  # must match step4_compute_embeddings.py

def build_smart_full_text(row):
    title    = str(row["title_text"]).strip()    if pd.notna(row.get("title_text"))    else ""
    abstract = str(row["abstract_text"]).strip() if pd.notna(row.get("abstract_text")) else ""
    claims   = str(row["all_claims_text"]).strip() if pd.notna(row.get("all_claims_text")) else ""
    text = title
    remaining = MAX_CHARS - len(text)
    if remaining > 0 and abstract:
        text += " " + abstract[:remaining]
    remaining = MAX_CHARS - len(text)
    if remaining > 0 and claims:
        text += " " + claims[:remaining]
    return text.strip()

# =============================================================================
# LOAD SBERT MODEL (only if text check is needed)
# =============================================================================
model = None

def get_model():
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("  Loading SBERT model for text spot-check...", flush=True)
            model = SentenceTransformer("all-mpnet-base-v2")
            print("  Model loaded", flush=True)
        except ImportError:
            print("  sentence-transformers not installed — skipping text check",
                  flush=True)
    return model

# =============================================================================
# CHECK EACH DATASET
# =============================================================================
print("=" * 65, flush=True)
print("EMBEDDING VALIDITY CHECK", flush=True)
print("=" * 65, flush=True)

results = {}

for dataset in DATASETS:
    name     = dataset["name"]
    csv_path = dataset["csv_path"]
    emb_path = dataset["emb_path"]
    ids_path = dataset["ids_path"]

    print(f"\n{'=' * 65}", flush=True)
    print(f"DATASET: {name.upper()}", flush=True)
    print(f"{'=' * 65}", flush=True)

    issues  = []
    passing = []

    # -------------------------------------------------------------------------
    # Check 0 — files exist
    # -------------------------------------------------------------------------
    if not os.path.exists(emb_path):
        print(f"  MISSING  {emb_path}", flush=True)
        results[name] = "MISSING"
        continue

    if not os.path.exists(csv_path):
        print(f"  MISSING  {csv_path}", flush=True)
        results[name] = "MISSING"
        continue

    emb_mtime = os.path.getmtime(emb_path)
    csv_mtime = os.path.getmtime(csv_path)
    import time as _time
    print(f"  Embedding created: {_time.ctime(emb_mtime)}", flush=True)
    print(f"  CSV modified:      {_time.ctime(csv_mtime)}", flush=True)
    if csv_mtime > emb_mtime:
        issues.append("CSV modified AFTER embeddings were created")
        print(f"  WARNING: CSV is newer than embeddings — content may have changed",
              flush=True)

    # -------------------------------------------------------------------------
    # Check 1 — row count match
    # -------------------------------------------------------------------------
    embeddings = np.load(emb_path)
    df         = pd.read_csv(csv_path, low_memory=False)

    n_emb = embeddings.shape[0]
    n_csv = len(df)

    if n_emb == n_csv:
        passing.append(f"Row count match: {n_emb:,} rows")
        print(f"  OK  Row count: {n_emb:,} embeddings = {n_csv:,} CSV rows",
              flush=True)
    else:
        issues.append(f"Row count mismatch: {n_emb:,} embeddings vs {n_csv:,} CSV rows")
        print(f"  FAIL  Row count: {n_emb:,} embeddings != {n_csv:,} CSV rows",
              flush=True)

    # -------------------------------------------------------------------------
    # Check 2 — publication number alignment via ids file
    # -------------------------------------------------------------------------
    if os.path.exists(ids_path):
        ids_df  = pd.read_csv(ids_path, index_col=0)
        csv_ids = df["publication_number"].tolist()
        emb_ids = ids_df["publication_number"].tolist()

        if len(emb_ids) == len(csv_ids):
            mismatches = sum(1 for a, b in zip(emb_ids, csv_ids) if a != b)
            if mismatches == 0:
                passing.append("Publication number order matches IDs file")
                print(f"  OK  Publication numbers: all {len(csv_ids):,} match IDs file",
                      flush=True)
            else:
                issues.append(f"Publication number order: {mismatches:,} mismatches")
                print(f"  FAIL  Publication numbers: {mismatches:,} rows out of order",
                      flush=True)
        else:
            issues.append(f"IDs file row count mismatch: "
                          f"{len(emb_ids):,} vs CSV {len(csv_ids):,}")
            print(f"  FAIL  IDs file has {len(emb_ids):,} rows, CSV has {len(csv_ids):,}",
                  flush=True)
    else:
        print(f"  SKIP  IDs file not found: {ids_path}", flush=True)

    # -------------------------------------------------------------------------
    # Check 3 — embedding norm check (should all be ~1.0)
    # -------------------------------------------------------------------------
    norms    = np.linalg.norm(embeddings, axis=1)
    norm_min = norms.min()
    norm_max = norms.max()
    norm_ok  = (norm_min > 0.98) and (norm_max < 1.02)

    if norm_ok:
        passing.append(f"Norms: min={norm_min:.4f} max={norm_max:.4f}")
        print(f"  OK  Norms: min={norm_min:.4f} max={norm_max:.4f} (expected ~1.0)",
              flush=True)
    else:
        issues.append(f"Unexpected norms: min={norm_min:.4f} max={norm_max:.4f}")
        print(f"  FAIL  Norms out of range: min={norm_min:.4f} max={norm_max:.4f}",
              flush=True)

    # -------------------------------------------------------------------------
    # Check 4 — text spot-check (re-encode sample, compare cosine similarity)
    # Only runs if SBERT model is available
    # -------------------------------------------------------------------------
    required_cols = ["title_text", "abstract_text", "all_claims_text"]
    has_text_cols = all(c in df.columns for c in required_cols)

    if has_text_cols and n_emb == n_csv:
        sbert = get_model()
        if sbert is not None:
            np.random.seed(42)
            sample_idx = np.random.choice(n_csv, min(SAMPLE_SIZE, n_csv),
                                          replace=False)
            sample_df  = df.iloc[sample_idx].copy()

            print(f"  Re-encoding {len(sample_idx)} random rows for text check...",
                  flush=True)

            sample_texts = [
                build_smart_full_text(row)
                for _, row in sample_df.iterrows()
            ]
            sample_new_embs  = sbert.encode(
                sample_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            sample_orig_embs = embeddings[sample_idx]

            # Cosine similarity between original and re-encoded
            # (both are normalised so dot product = cosine similarity)
            similarities = np.sum(sample_orig_embs * sample_new_embs, axis=1)
            mean_sim = float(similarities.mean())
            min_sim  = float(similarities.min())

            if min_sim >= SIMILARITY_THRESHOLD:
                passing.append(
                    f"Text spot-check: mean sim={mean_sim:.6f} "
                    f"min sim={min_sim:.6f}"
                )
                print(
                    f"  OK  Text spot-check: mean similarity={mean_sim:.6f}  "
                    f"min={min_sim:.6f}  (threshold={SIMILARITY_THRESHOLD})",
                    flush=True
                )
            else:
                low_sim = (similarities < SIMILARITY_THRESHOLD).sum()
                issues.append(
                    f"Text mismatch: {low_sim} of {len(sample_idx)} "
                    f"sampled rows have similarity < {SIMILARITY_THRESHOLD} "
                    f"(mean={mean_sim:.4f}, min={min_sim:.4f})"
                )
                print(
                    f"  FAIL  Text spot-check: {low_sim} rows have "
                    f"similarity < {SIMILARITY_THRESHOLD}  "
                    f"(mean={mean_sim:.4f} min={min_sim:.4f})",
                    flush=True
                )
        else:
            print(f"  SKIP  Text spot-check (SBERT not available)", flush=True)
    else:
        if not has_text_cols:
            print(f"  SKIP  Text spot-check (text columns not in CSV)", flush=True)

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    if not issues:
        results[name] = "VALID"
        print(f"\n  VERDICT: VALID — embeddings match CSV, no rerun needed",
              flush=True)
    else:
        results[name] = "INVALID"
        print(f"\n  VERDICT: INVALID — embeddings must be rerun", flush=True)
        for issue in issues:
            print(f"    - {issue}", flush=True)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'=' * 65}", flush=True)
print(f"  {'Dataset':<15} {'Status'}", flush=True)
print(f"  {'-' * 30}", flush=True)

needs_rerun = []
for name, status in results.items():
    flag = "  <-- RERUN NEEDED" if status in ("INVALID", "MISSING") else ""
    print(f"  {name:<15} {status}{flag}", flush=True)
    if status in ("INVALID", "MISSING"):
        needs_rerun.append(name)

if not needs_rerun:
    print(f"\n  All embeddings are valid — step5_compute_novelty.py is safe to run",
          flush=True)
else:
    print(f"\n  Rerun step4_compute_embeddings.py for: {', '.join(needs_rerun)}",
          flush=True)
    print(f"  Then re-run step5_compute_novelty.py", flush=True)