"""
Step 3f — Generate BigQuery SQL for Acquiror Patents
======================================================
Reads patents_eligible.csv and acquiror_name_map.csv, extracts all
BigQuery acquiror name variants (including one-to-many mappings),
and generates a BigQuery SQL query that pulls full patent text for those
acquirors. The resulting patents_acquiror.csv is used in:
  - step3g_build_controls.py  — log_acquiror_pre_patents control variable
  - step4_compute_embeddings.py       — SBERT embeddings for knowledge overlap
  - step6b_compute_overlap.py          — knowledge overlap computation (H2 moderator)

Inputs:
  data/interim/patents_eligible.csv           — output of step3d
  data/interim/acquiror_name_map.csv          — output of step3e

Outputs:
  queries/bigquery_acquiror_full_query.sql    — paste into BigQuery console

Usage:
  python step3f_generate_acquiror_sql.py

After running:
  1. Open queries/bigquery_acquiror_full_query.sql in BigQuery console
  2. Run and export result as data/raw/patents_acquiror.csv
  Then run: step3g_build_controls.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PATENTS_ELIGIBLE_PATH = "data/interim/patents_eligible.csv"
NAME_MAP_PATH         = "data/interim/acquiror_name_map.csv"
QUERIES_DIR           = "queries/"
OUTPUT_QUERY_PATH     = os.path.join(QUERIES_DIR, "bigquery_acquiror_full_query.sql")

# =============================================================================
# CHECKS
# =============================================================================
print("=" * 55, flush=True)
print("STEP 3f — GENERATE BIGQUERY SQL FOR ACQUIROR PATENTS", flush=True)
print("=" * 55, flush=True)

if not os.path.exists(PATENTS_ELIGIBLE_PATH):
    print(f"ERROR: {PATENTS_ELIGIBLE_PATH} not found", flush=True)
    print("  Run step3d_build_eligible_patents.py first", flush=True)
    sys.exit(1)

os.makedirs(QUERIES_DIR, exist_ok=True)

# =============================================================================
# LOAD AND EXTRACT ACQUIROR NAMES
# =============================================================================
# Check name map exists
if not os.path.exists(NAME_MAP_PATH):
    print(f"ERROR: {NAME_MAP_PATH} not found", flush=True)
    print("  Run step3e_map_acquirors.py first", flush=True)
    sys.exit(1)

print(f"\nLoading {PATENTS_ELIGIBLE_PATH}...", flush=True)
eligible = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)
print(f"  Rows: {len(eligible):,}  |  "
      f"Unique BvD IDs: {eligible['bvd_id'].nunique():,}", flush=True)

print(f"\nLoading {NAME_MAP_PATH}...", flush=True)
name_map = pd.read_csv(NAME_MAP_PATH)
print(f"  Rows: {len(name_map):,}  |  "
      f"Unique Orbis names: {name_map['orbis_acquiror_name'].nunique():,}", flush=True)

# Use BigQuery name variants from name_map — NOT Orbis names directly.
# This ensures the SQL IN clause uses the correct USPTO filing names,
# including all one-to-many variants (e.g. Cisco -> 2 names, Dell -> 3).
acquiror_names = (
    name_map["bq_acquiror_name"]
    .dropna()
    .str.strip()
    .unique()
    .tolist()
)

print(f"\n  Unique BigQuery name variants: {len(acquiror_names):,}", flush=True)
for name in sorted(acquiror_names):
    print(f"    {name}", flush=True)

if len(acquiror_names) == 0:
    print("ERROR: no acquiror names found in patents_eligible.csv",
          flush=True)
    print("  Check that 'Acquiror name' column is present", flush=True)
    sys.exit(1)

# =============================================================================
# FORMAT AS SQL IN CLAUSE
# =============================================================================
formatted = ", ".join([
    f"'{name.replace(chr(39), chr(39)*2)}'"
    for name in acquiror_names
])

# =============================================================================
# ACQUIROR PATENTS QUERY (2000-2024)
# Pulls full patent text for SBERT embedding and knowledge overlap
# computation. Date range starts at 2000 to capture the full pre-merger
# patent history needed for the 5-year acquiror patent count control.
# =============================================================================
ACQUIROR_QUERY = f"""
SELECT
  base.publication_number,
  base.filing_date,
  assignee_harmonized.name              AS assignee_name,
  base.cpc_code,
  base.abstract_text,
  base.title_text,
  STRING_AGG(claim.text, ' ')           AS all_claims_text

FROM (
  SELECT
    publication_number,
    filing_date,
    assignee_harmonized,
    claims_localized,
    (SELECT cpc.code FROM UNNEST(cpc) AS cpc
     WHERE cpc.inventive = TRUE
     AND (  cpc.code LIKE 'G06F%'
         OR cpc.code LIKE 'H01L%'
         OR cpc.code LIKE 'G06N%'
         OR cpc.code LIKE 'H04L%')
     LIMIT 1)                           AS cpc_code,
    (SELECT text FROM UNNEST(abstract_localized)
     WHERE language = 'en' LIMIT 1)     AS abstract_text,
    (SELECT text FROM UNNEST(title_localized)
     WHERE language = 'en' LIMIT 1)     AS title_text
  FROM
    `patents-public-data.patents.publications`
  WHERE
    filing_date BETWEEN 20000101 AND 20241231
    AND country_code = 'US'
    AND EXISTS (
      SELECT 1 FROM UNNEST(cpc) AS cpc
      WHERE cpc.inventive = TRUE
      AND (  cpc.code LIKE 'G06F%'
          OR cpc.code LIKE 'H01L%'
          OR cpc.code LIKE 'G06N%'
          OR cpc.code LIKE 'H04L%')
    )
) AS base
CROSS JOIN UNNEST(base.assignee_harmonized) AS assignee_harmonized
CROSS JOIN UNNEST(base.claims_localized)    AS claim

WHERE
  assignee_harmonized.country_code = 'US'
  AND (claim.language = 'en' OR claim.language IS NULL)
  AND assignee_harmonized.name IN ({formatted})

GROUP BY
  base.publication_number,
  base.filing_date,
  assignee_harmonized.name,
  base.cpc_code,
  base.abstract_text,
  base.title_text
"""

# =============================================================================
# SAVE
# =============================================================================
with open(OUTPUT_QUERY_PATH, "w") as f:
    f.write(ACQUIROR_QUERY)

print(f"\n{'=' * 55}", flush=True)
print(f"SAVED", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  Output:       {OUTPUT_QUERY_PATH}", flush=True)
print(f"  BigQuery name variants in query: {len(acquiror_names):,}", flush=True)

print(f"""
>>> MANUAL STEP REQUIRED:

  1. Open {OUTPUT_QUERY_PATH} in BigQuery console
  2. Run the query
  3. Export result as a single file: data/raw/patents_acquiror.csv

Next step: step3g_build_controls.py
  Reads  data/interim/patents_eligible.csv
  Reads  data/raw/patents_acquiror.csv
  Reads  data/interim/acquiror_name_map.csv
  Writes data/interim/patents_eligible_controls.csv
""", flush=True)