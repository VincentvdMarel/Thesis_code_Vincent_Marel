"""
Step 3b — Combine BigQuery Export Shards
==========================================
Combines all CSV shards from BigQuery exports into three single files:
  - patents_validation.csv  (2005-2015, with forward citations)
  - patents_prior_art.csv   (2010-2024, full patent universe)
  - patents_control.csv     (2010-2024, 10% sample of non-acquired firms)

Inputs:
  data/raw/Validation files/patents_validation*  — BigQuery validation shards
  data/raw/Prior_art files/patents_prior_art*    — BigQuery prior art shards
  data/raw/Control files/control_psm*            — BigQuery control group shards

Outputs:
  data/interim/patents_validation.csv
  data/interim/patents_prior_art.csv
  data/interim/patents_control.csv

Usage:
  python step3b_combine_shards.py

Notes:
  Set OVERWRITE_VALIDATION, OVERWRITE_PRIOR_ART, or OVERWRITE_CONTROL to
  True to re-combine the relevant shards even if the output already exists.
  Default is False for all — existing files are reused to save time.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import glob
import os
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
VALIDATION_FOLDER   = "data/raw/Validation files/"
PRIOR_ART_FOLDER    = "data/raw/Prior_art files/"
CONTROL_FOLDER      = "data/raw/Control files/"

VALIDATION_PATTERN  = "patents_validation*"
PRIOR_ART_PATTERN   = "patents_prior_art*"
CONTROL_PATTERN     = "control_psm*"

VALIDATION_OUTPUT   = "data/interim/patents_validation.csv"
PRIOR_ART_OUTPUT    = "data/interim/patents_prior_art.csv"
CONTROL_OUTPUT      = "data/interim/patents_control.csv"

CHUNKSIZE           = 50_000

# Set to True to re-combine shards even if the output file already exists
OVERWRITE_VALIDATION = False
OVERWRITE_PRIOR_ART  = False
OVERWRITE_CONTROL    = False

# =============================================================================
# SETUP
# =============================================================================
print("=" * 55, flush=True)
print("STEP 3b — COMBINE BIGQUERY EXPORT SHARDS", flush=True)
print("=" * 55, flush=True)

os.makedirs("data/interim", exist_ok=True)

# =============================================================================
# HELPER FUNCTION
# =============================================================================
def combine_shards(
    shard_folder,
    file_pattern,
    output_path,
    check_cols,
    overwrite=False,
    label="dataset",
):
    """
    Combines all CSV shards in shard_folder matching file_pattern
    into a single CSV file, with sanity checks and deduplication.

    Args:
        shard_folder  — folder containing the shard files
        file_pattern  — glob pattern, e.g. 'patents_validation*'
        output_path   — where to save the combined CSV
        check_cols    — columns to report missing-value counts for
        overwrite     — if False, skip combining if output already exists
        label         — display name used in print output

    Returns:
        combined DataFrame, or None if no shard files were found
    """
    print(f"\n{'=' * 55}", flush=True)
    print(f"COMBINING SHARDS: {label.upper()}", flush=True)
    print(f"{'=' * 55}", flush=True)

    # -------------------------------------------------------------------------
    # Check if output already exists
    # -------------------------------------------------------------------------
    if os.path.exists(output_path) and not overwrite:
        size_gb  = os.path.getsize(output_path) / (1024 ** 3)
        modified = time.ctime(os.path.getmtime(output_path))
        print(f"  Output already exists: {output_path}", flush=True)
        print(f"  Size:     {size_gb:.2f} GB", flush=True)
        print(f"  Modified: {modified}", flush=True)
        print(f"  Skipping — set overwrite=True to re-combine", flush=True)
        return pd.read_csv(output_path, low_memory=False)

    if os.path.exists(output_path) and overwrite:
        size_gb  = os.path.getsize(output_path) / (1024 ** 3)
        modified = time.ctime(os.path.getmtime(output_path))
        print(f"  Output already exists: {output_path}", flush=True)
        print(f"  Size:     {size_gb:.2f} GB  |  Modified: {modified}", flush=True)
        print(f"  Overwrite=True — re-combining from shards...", flush=True)

    # -------------------------------------------------------------------------
    # Find shard files
    # -------------------------------------------------------------------------
    if not os.path.exists(shard_folder):
        print(f"  ERROR: shard folder not found: {shard_folder}", flush=True)
        print(f"  Run step3a_generate_target_sql.py and export from BigQuery first",
              flush=True)
        return None

    all_files = sorted(glob.glob(os.path.join(shard_folder, file_pattern)))

    if not all_files:
        print(f"  ERROR: no files found matching '{file_pattern}' "
              f"in '{shard_folder}'", flush=True)
        return None

    print(f"\n  Found {len(all_files)} shard files:", flush=True)
    total_size = 0
    for f in all_files:
        size_mb   = os.path.getsize(f) / (1024 ** 2)
        total_size += size_mb
        print(f"    {os.path.basename(f):<50} {size_mb:>7.1f} MB", flush=True)
    print(f"\n  Total shard size: {total_size / 1024:.2f} GB", flush=True)

    # -------------------------------------------------------------------------
    # Load all shards
    # -------------------------------------------------------------------------
    chunks     = []
    start_time = time.time()

    for i, filepath in enumerate(all_files):
        print(f"\n  Loading {i + 1}/{len(all_files)}: "
              f"{os.path.basename(filepath)}", flush=True)
        file_chunks = []
        for chunk in pd.read_csv(filepath, chunksize=CHUNKSIZE, low_memory=False):
            file_chunks.append(chunk)
        file_df = pd.concat(file_chunks, ignore_index=True)
        print(f"    Rows: {len(file_df):,}", flush=True)
        chunks.append(file_df)

    print(f"\n  Combining {len(all_files)} shards...", flush=True)
    combined = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows before deduplication: {len(combined):,}", flush=True)

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    print(f"\n  {'=' * 45}", flush=True)
    print(f"  COMBINED DATASET SUMMARY — {label.upper()}", flush=True)
    print(f"  {'=' * 45}", flush=True)
    print(f"  Total rows:             {len(combined):,}", flush=True)
    print(f"  Unique publication nos: "
          f"{combined['publication_number'].nunique():,}", flush=True)
    print(f"  Filing date range:      {combined['filing_date'].min()} "
          f"– {combined['filing_date'].max()}", flush=True)

    if "assignee_name" in combined.columns:
        print(f"  Unique assignees:       "
              f"{combined['assignee_name'].nunique():,}", flush=True)

    # Missing values
    available_check = [c for c in check_cols if c in combined.columns]
    missing_check   = [c for c in check_cols if c not in combined.columns]

    if missing_check:
        print(f"\n  Note: expected columns not present: {missing_check}",
              flush=True)
    if available_check:
        print(f"\n  Missing values:", flush=True)
        print(combined[available_check].isnull().sum().to_string(), flush=True)

    # Citation distribution (validation dataset only)
    if "forward_citations_5yr" in combined.columns:
        print(f"\n  Forward citation distribution:", flush=True)
        print(combined["forward_citations_5yr"].describe().round(2).to_string(),
              flush=True)
        zero_pct = (combined["forward_citations_5yr"] == 0).mean() * 100
        print(f"  Zero-citation patents: {zero_pct:.1f}%", flush=True)

    # CPC distribution
    if "cpc_code" in combined.columns:
        combined["cpc_4digit"] = combined["cpc_code"].str[:4].str.upper()
        print(f"\n  Patents by CPC class:", flush=True)
        print(combined["cpc_4digit"].value_counts().to_string(), flush=True)
        combined = combined.drop(columns=["cpc_4digit"])

    # -------------------------------------------------------------------------
    # Deduplicate
    # -------------------------------------------------------------------------
    dupes = combined.duplicated(subset="publication_number", keep=False)
    print(f"\n  Duplicate publication numbers: {dupes.sum():,}", flush=True)
    if dupes.sum() > 0:
        combined = combined.drop_duplicates(
            subset="publication_number", keep="first"
        )
        print(f"  Rows after deduplication: {len(combined):,}", flush=True)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    combined.to_csv(output_path, index=False)
    elapsed = time.time() - start_time
    size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print(f"\n  Saved to {output_path}", flush=True)
    print(f"  Final rows: {len(combined):,}  |  "
          f"Size: {size_gb:.2f} GB  |  "
          f"Time: {elapsed / 60:.1f} min", flush=True)
    print(f"  Columns: {list(combined.columns)}", flush=True)

    return combined


# =============================================================================
# COMBINE VALIDATION SHARDS (2005-2015, includes forward citations)
# =============================================================================
patents_validation = combine_shards(
    shard_folder = VALIDATION_FOLDER,
    file_pattern = VALIDATION_PATTERN,
    output_path  = VALIDATION_OUTPUT,
    check_cols   = [
        "abstract_text", "title_text", "all_claims_text",
        "cpc_code", "forward_citations_5yr",
    ],
    overwrite    = OVERWRITE_VALIDATION,
    label        = "validation (2005-2015)",
)

# =============================================================================
# COMBINE PRIOR ART SHARDS (2010-2024, full patent universe)
# =============================================================================
patents_prior_art = combine_shards(
    shard_folder = PRIOR_ART_FOLDER,
    file_pattern = PRIOR_ART_PATTERN,
    output_path  = PRIOR_ART_OUTPUT,
    check_cols   = [
        "abstract_text", "title_text", "all_claims_text", "cpc_code",
    ],
    overwrite    = OVERWRITE_PRIOR_ART,
    label        = "prior art (2010-2024)",
)

# =============================================================================
# COMBINE CONTROL GROUP SHARDS (2010-2024, 10% sample of non-acquired firms)
# Used as PSM control candidates in Step 7
# =============================================================================
patents_control = combine_shards(
    shard_folder = CONTROL_FOLDER,
    file_pattern = CONTROL_PATTERN,
    output_path  = CONTROL_OUTPUT,
    check_cols   = [
        "abstract_text", "title_text", "all_claims_text", "cpc_code",
    ],
    overwrite    = OVERWRITE_CONTROL,
    label        = "control (2010-2024)",
)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"STEP 3b COMPLETE", flush=True)
print(f"{'=' * 55}", flush=True)

all_ok = True
for label, df, path in [
    ("patents_validation.csv", patents_validation, VALIDATION_OUTPUT),
    ("patents_prior_art.csv",  patents_prior_art,  PRIOR_ART_OUTPUT),
    ("patents_control.csv",    patents_control,    CONTROL_OUTPUT),
]:
    if df is not None:
        size_gb = os.path.getsize(path) / (1024 ** 3)
        print(f"\n  {label}", flush=True)
        print(f"    Rows:    {len(df):,}", flush=True)
        print(f"    Size:    {size_gb:.2f} GB", flush=True)
    else:
        print(f"\n  {label}  -- NOT CREATED (no shard files found)", flush=True)
        all_ok = False

if all_ok:
    print(f"""
Next step: step3c_check_eligibility.py
  Reads  data/raw/patents_main.csv
  Reads  data/interim/patents_validation.csv
  Reads  data/interim/crosswalk_final_clean.csv
  Writes data/interim/patent_count_eligibility.csv
""", flush=True)
else:
    print(f"""
WARNING: one or more output files were not created.
  Check that BigQuery exports are in the correct folders:
    {VALIDATION_FOLDER}  (pattern: {VALIDATION_PATTERN})
    {PRIOR_ART_FOLDER}   (pattern: {PRIOR_ART_PATTERN})
    {CONTROL_FOLDER}     (pattern: {CONTROL_PATTERN})
  Then re-run this script.
""", flush=True)