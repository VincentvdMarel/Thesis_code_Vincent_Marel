"""
Step 4 — Compute SBERT Embeddings
====================================
Encodes every patent's full_text into a 768-dimensional vector using
the sentence-transformers library. Produces embedding arrays for both
the eligible patent dataset (DiD analysis) and the validation dataset
(NLP validation regression).

Model: all-mpnet-base-v2
  - Best general-purpose SBERT model for semantic similarity tasks
  - 768 dimensions, trained on 1B+ sentence pairs
  - normalize_embeddings=True so cosine similarity = dot product
    which is required for FAISS in Step 5

Install dependencies first:
  pip install sentence-transformers torch

Inputs:
  data/interim/patents_eligible_controls.csv  — output of step3g
  data/interim/patents_validation.csv         — output of step3b
  data/interim/patents_prior_art.csv          — output of step3b
  data/raw/patents_acquiror.csv               — BigQuery acquiror export
  data/interim/patents_control.csv            — output of step3b

Outputs:
  data/interim/embeddings_eligible.npy        — shape (n_eligible_patents, 768)
  data/interim/embeddings_validation.npy      — shape (n_validation_patents, 768)
  data/interim/embeddings_prior_art.npy       — shape (n_prior_art_patents, 768)
  data/interim/embeddings_acquiror.npy        — shape (n_acquiror_patents, 768)
  data/interim/embeddings_control.npy         — shape (n_control_patents, 768)
  data/interim/embeddings_*_ids.csv           — publication numbers in row order

CRITICAL: Do not reorder or reshuffle any input CSV after running this
script. The embedding arrays are postion-matched to the CSV rows.
If you need to re-filter or reorder, re-run this script from scratch.

Expected runtime:
  CPU only:  2-4 hours for 65k patents, 6-10 hours for 239k patents
  GPU:       30-60 minutes total
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os
import time

# =============================================================================
# FULL TEXT CONSTRUCTION
# Builds patent full_text within the approximate 512-token budget by
# prioritising the most semantically rich content in order:
#   1. Title     — always included in full
#   2. Abstract  — included up to remaining budget
#   3. Claims    — included up to remaining budget
# =============================================================================
MAX_CHARS = 1800  # ~450 tokens — leaves room for SBERT special tokens

def build_smart_full_text(row):
    """
    Constructs full_text within MAX_CHARS budget.
    Title is always encoded fully. Abstract and claims fill
    the remaining space in order, ensuring the most distinctive
    semantic content from each patent is always represented.
    """
    title    = str(row["title_text"]).strip()      if pd.notna(row["title_text"])      else ""
    abstract = str(row["abstract_text"]).strip()   if pd.notna(row["abstract_text"])   else ""
    claims   = str(row["all_claims_text"]).strip() if pd.notna(row["all_claims_text"]) else ""

    # Step 1 — title always in full
    text = title

    # Step 2 — abstract up to remaining budget
    remaining = MAX_CHARS - len(text)
    if remaining > 0 and abstract:
        text += " " + abstract[:remaining]

    # Step 3 — claims up to remaining budget
    remaining = MAX_CHARS - len(text)
    if remaining > 0 and claims:
        text += " " + claims[:remaining]

    return text.strip()

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_NAME  = "all-mpnet-base-v2"
BATCH_SIZE  = 32     # reduce to 16 if you run out of memory
MAX_LENGTH  = 512    # SBERT truncates at this token length — standard setting

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

# =============================================================================
# CHECK DEPENDENCIES
# =============================================================================
print("="*60)
print("STEP 4 — SBERT EMBEDDINGS")
print("="*60)

try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers: OK")
except ImportError:
    print("ERROR: sentence-transformers not installed")
    print("Run: pip install sentence-transformers torch")
    sys.exit(1)

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch device:        {device.upper()}")
    if device == "cuda":
        print(f"GPU:                   {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    device = "cpu"
    print("PyTorch device:        CPU")

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"\nLoading model: {MODEL_NAME}")
print("(First run downloads ~420MB — subsequent runs load from cache)")

model_load_start = time.time()
model = SentenceTransformer(MODEL_NAME)
model_load_time = time.time() - model_load_start
print(f"Model loaded in {model_load_time:.1f}s")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

# =============================================================================
# ENCODE EACH DATASET
# =============================================================================
for dataset in DATASETS:
    name     = dataset["name"]
    csv_path = dataset["csv_path"]
    emb_path = dataset["emb_path"]
    ids_path = dataset["ids_path"]

    print(f"\n{'='*60}")
    print(f"ENCODING: {name.upper()} DATASET")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # CHECK IF ALREADY DONE — ask before recomputing
    # -------------------------------------------------------------------------
    if os.path.exists(emb_path):
        existing   = np.load(emb_path)
        size_mb    = os.path.getsize(emb_path) / (1024 * 1024)
        print(f"Found existing {emb_path}")
        print(f"  Shape:    {existing.shape}")
        print(f"  Size:     {size_mb:.0f} MB")
        print(f"  Created:  {time.ctime(os.path.getmtime(emb_path))}")
        answer = input(f"  Re-compute embeddings for '{name}'? (yes/no): ").strip().lower()
        if answer != "yes":
            print(f"  Skipping '{name}' — using existing embeddings")
            continue
        print(f"  Re-computing '{name}' embeddings from scratch...")

    # -------------------------------------------------------------------------
    # LOAD CSV
    # -------------------------------------------------------------------------
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found — skipping")
        continue

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.columns.tolist()}")

    # -------------------------------------------------------------------------
    # BUILD full_text USING SMART TRUNCATION
    # Always rebuilds using build_smart_full_text to ensure the token
    # budget is respected consistently across both datasets.
    # This prioritises title → abstract → claims within MAX_CHARS budget
    # -------------------------------------------------------------------------
    required_cols = ["title_text", "abstract_text", "all_claims_text"]
    missing_text_cols = [c for c in required_cols if c not in df.columns]

    if missing_text_cols:
        print(f"  WARNING: Missing text columns {missing_text_cols}")
        print("  Falling back to existing full_text column if available")
        if "full_text" not in df.columns:
            print("  ERROR: No text columns available — cannot encode")
            continue
        df["full_text"] = df["full_text"].fillna("")
    else:
        print(f"  Building smart full_text (title + abstract + claims, max {MAX_CHARS} chars)...")
        df["full_text"] = df.apply(build_smart_full_text, axis=1)

    # Quality check
    empty      = (df["full_text"].str.strip() == "").sum()
    very_short = (df["full_text"].str.len() < 50).sum()
    print(f"  Empty full_text:        {empty:,}")
    print(f"  Very short (<50 chars): {very_short:,}")
    print(f"  Mean text length:       {df['full_text'].str.len().mean():.0f} chars")
    print(f"  Median text length:     {df['full_text'].str.len().median():.0f} chars")
    print(f"  Max text length:        {df['full_text'].str.len().max():.0f} chars")
    print(f"  Texts over MAX_CHARS:   {(df['full_text'].str.len() > MAX_CHARS).sum():,} (expected ~0)")

    if empty > 0:
        print(f"  WARNING: {empty} patents have empty text — will get zero-like embeddings")

    # -------------------------------------------------------------------------
    # SAVE PUBLICATION NUMBER ORDER BEFORE ENCODING
    # This is critical — the row order of the CSV must match the embedding rows
    # -------------------------------------------------------------------------
    ids_df = df[["publication_number"]].copy()
    if "bvd_id" in df.columns:
        ids_df["bvd_id"] = df["bvd_id"]
    ids_df.to_csv(ids_path, index=True)  # keep index as row number
    print(f"  Row order saved to {ids_path}")

    # -------------------------------------------------------------------------
    # ENCODE
    # -------------------------------------------------------------------------
    texts = df["full_text"].tolist()
    total = len(texts)

    print(f"\nEncoding {total:,} patents with batch_size={BATCH_SIZE}...")
    print(f"Estimated time on {device.upper()}: "
          f"{'~15-30 min' if device == 'cuda' else '~2-4 hours'}")
    print("Progress bar below:")

    encode_start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for cosine similarity via dot product
        convert_to_numpy=True,
        device=device,
    )

    encode_time = time.time() - encode_start

    # -------------------------------------------------------------------------
    # VERIFY OUTPUT
    # -------------------------------------------------------------------------
    print(f"\nEncoding complete in {encode_time/60:.1f} minutes")
    print(f"  Output shape:        {embeddings.shape}")
    print(f"  Expected shape:      ({total}, 768)")
    print(f"  dtype:               {embeddings.dtype}")

    # Check normalisation — all row norms should be ~1.0
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Row norm min/max:    {norms.min():.4f} / {norms.max():.4f}  (should be ~1.0)")

    # Check for NaN or Inf
    n_nan = np.isnan(embeddings).sum()
    n_inf = np.isinf(embeddings).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {n_nan} NaN, {n_inf} Inf values found in embeddings")
    else:
        print(f"  No NaN or Inf values — embeddings are clean")

    # Verify row count matches CSV
    if embeddings.shape[0] != total:
        print(f"  ERROR: row count mismatch — embeddings {embeddings.shape[0]} vs CSV {total}")
    else:
        print(f"  Row count matches CSV — position alignment verified")

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    np.save(emb_path, embeddings)
    file_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
    print(f"  Saved to {emb_path} — {file_size_mb:.1f} MB")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print(f"STEP 4 COMPLETE — SUMMARY")
print(f"{'='*60}")

for dataset in DATASETS:
    emb_path = dataset["emb_path"]
    ids_path = dataset["ids_path"]
    if os.path.exists(emb_path):
        arr = np.load(emb_path)
        size_mb = os.path.getsize(emb_path) / (1024 * 1024)
        print(f"\n  {dataset['name']}:")
        print(f"    {emb_path:<45} {arr.shape}  {size_mb:.0f} MB")
        print(f"    {ids_path:<45} row order saved")
    else:
        print(f"\n  {dataset['name']}: NOT YET ENCODED")

print(f"""
Next steps:
  1. Verify all .npy files exist and have the expected shapes:
     data/interim/embeddings_eligible.npy   — should be (~26000, 768)
     data/interim/embeddings_validation.npy — should be (~239000, 768)
     data/interim/embeddings_prior_art.npy  — should be (large, 768)
     data/interim/embeddings_acquiror.npy   — should be (medium, 768)
     data/interim/embeddings_control.npy    — should be (large, 768)

  2. Do NOT reorder or reshuffle any input CSV after running this script.
     The embedding rows are position-matched to the CSV rows.

  3. Run step5_compute_novelty.py to compute patent novelty scores.
     This uses the embeddings to find backward similarity in the
     same CPC subclass over the prior 5 years.
""")