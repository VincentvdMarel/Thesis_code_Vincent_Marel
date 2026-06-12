"""
Step 3a — Generate BigQuery SQL Queries
=========================================
Loads crosswalk_final_clean.csv, extracts all verified USPTO assignee
names, and generates four SQL query files ready to run in BigQuery.

Inputs:
  data/interim/crosswalk_final_clean.csv    — output of step2g

Outputs:
  queries/bigquery_main_query.sql           → export as data/raw/patents_main.csv
  queries/bigquery_validation_query.sql     → export shards to data/raw/Validation files/
  queries/bigquery_prior_art_query.sql      → export shards to data/raw/Prior_art files/
  queries/bigquery_control_query.sql        → export shards to data/raw/Control files/

Usage:
  python step3a_generate_target_sql.py

After running:
  1. Open queries/bigquery_main_query.sql in BigQuery console
     Run and export result as data/raw/patents_main.csv

  2. Open queries/bigquery_validation_query.sql in BigQuery console
     Run and export shards to data/raw/Validation files/

  3. Open queries/bigquery_prior_art_query.sql in BigQuery console
     Run and export shards to data/raw/Prior_art files/
     Note: prior art scans the full patent universe — export will be large.
     Use a wildcard filename (e.g. prior_art_*) so BigQuery shards automatically.

  4. Open queries/bigquery_control_query.sql in BigQuery console
     Run and export shards to data/raw/Control files/
     Use filename prefix: patents_control_*
     Note: excludes all treated target firm assignee names via NOT IN.
     Does NOT apply a secondary Orbis M&A non-acquisition screen — document
     this limitation in thesis §4.3.4.

  Then run step3b_combine_shards.py to combine shard folders into single files.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
CROSSWALK_PATH      = "data/interim/crosswalk_final_clean.csv"
QUERIES_DIR         = "queries/"

MAIN_QUERY_PATH     = os.path.join(QUERIES_DIR, "bigquery_main_query.sql")
VAL_QUERY_PATH      = os.path.join(QUERIES_DIR, "bigquery_validation_query.sql")
PRIOR_ART_PATH      = os.path.join(QUERIES_DIR, "bigquery_prior_art_query.sql")
CONTROL_QUERY_PATH  = os.path.join(QUERIES_DIR, "bigquery_control_query.sql")

# =============================================================================
# CHECKS
# =============================================================================
print("=" * 55, flush=True)
print("STEP 3a — GENERATE BIGQUERY SQL QUERIES", flush=True)
print("=" * 55, flush=True)

if not os.path.exists(CROSSWALK_PATH):
    print(f"ERROR: {CROSSWALK_PATH} not found", flush=True)
    print("Run step2d_clean_crosswalk.py first", flush=True)
    sys.exit(1)

os.makedirs(QUERIES_DIR, exist_ok=True)

# =============================================================================
# LOAD CROSSWALK AND PREPARE ASSIGNEE LIST
# =============================================================================
print(f"\nLoading {CROSSWALK_PATH}...", flush=True)
crosswalk = pd.read_csv(CROSSWALK_PATH)
crosswalk["uspto_assignee_name"] = crosswalk["uspto_assignee_name"].str.upper()
crosswalk = crosswalk.drop_duplicates()

# Add FARADAY spaced variant if not present
if "FARADAY & FUTURE INC" not in crosswalk["uspto_assignee_name"].values:
    faraday_row = crosswalk[
        crosswalk["uspto_assignee_name"] == "FARADAY&FUTURE INC"
    ].copy()
    if len(faraday_row) > 0:
        faraday_row["uspto_assignee_name"] = "FARADAY & FUTURE INC"
        crosswalk = pd.concat([crosswalk, faraday_row], ignore_index=True)
        print("  Added FARADAY & FUTURE INC variant", flush=True)

assignee_names = crosswalk["uspto_assignee_name"].dropna().unique().tolist()
print(f"  Unique USPTO assignee names: {len(assignee_names):,}", flush=True)

# Format as SQL IN / NOT IN list — escape any internal single quotes
# Reused as IN  for queries 1 (main — treated firms only)
# Reused as NOT IN for query 4  (control — all other firms)
formatted = ", ".join([
    f"'{name.replace(chr(39), chr(39)*2)}'"
    for name in assignee_names
])

# =============================================================================
# QUERY 1 — MAIN ANALYSIS QUERY (2000–2024)
# Target firms only — used to build patents_main.csv → patents_eligible.csv
# Includes full text (abstract, title, claims) for SBERT encoding
# =============================================================================
MAIN_QUERY = """
SELECT
  base.publication_number,
  base.application_number,
  base.filing_date,
  assignee_harmonized.name              AS assignee_name,
  base.cpc_code,
  base.abstract_text,
  base.title_text,
  STRING_AGG(claim.text, ' ')           AS all_claims_text

FROM (
  SELECT
    publication_number,
    application_number,
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
  AND assignee_harmonized.name IN ({assignee_list})

GROUP BY
  base.publication_number,
  base.application_number,
  base.filing_date,
  assignee_harmonized.name,
  base.cpc_code,
  base.abstract_text,
  base.title_text
"""

query_main = MAIN_QUERY.format(assignee_list=formatted)
with open(MAIN_QUERY_PATH, "w") as f:
    f.write(query_main)
print(f"\n  Saved {MAIN_QUERY_PATH}", flush=True)
print(f"  Assignee names in query: {len(assignee_names):,}", flush=True)

# =============================================================================
# QUERY 2 — VALIDATION QUERY (2005–2015, includes forward citations)
# Full patent universe — no assignee filter
# Used for: (1) NLP validation regression, (2) prior art for 2010-2015 focal patents
# Sampling: MOD(..., 3) = 0 keeps ~33% for manageable file size
# =============================================================================
VALIDATION_QUERY = """
SELECT
  base.publication_number,
  base.application_number,
  base.filing_date,
  assignee_harmonized.name              AS assignee_name,
  base.cpc_code,
  base.abstract_text,
  base.title_text,
  STRING_AGG(claim.text, ' ')           AS all_claims_text,
  COUNT(DISTINCT citations.citing_pub)  AS forward_citations_5yr

FROM (
  SELECT
    pub.publication_number,
    pub.application_number,
    pub.filing_date,
    pub.assignee_harmonized,
    pub.claims_localized,
    (SELECT cpc.code FROM UNNEST(pub.cpc) AS cpc
     WHERE cpc.inventive = TRUE
     AND (  cpc.code LIKE 'G06F%'
         OR cpc.code LIKE 'H01L%'
         OR cpc.code LIKE 'G06N%'
         OR cpc.code LIKE 'H04L%')
     LIMIT 1)                           AS cpc_code,
    (SELECT text FROM UNNEST(pub.abstract_localized)
     WHERE language = 'en' LIMIT 1)     AS abstract_text,
    (SELECT text FROM UNNEST(pub.title_localized)
     WHERE language = 'en' LIMIT 1)     AS title_text
  FROM
    `patents-public-data.patents.publications` AS pub
  WHERE
    pub.filing_date BETWEEN 20050101 AND 20151231
    AND pub.country_code = 'US'
    AND MOD(ABS(FARM_FINGERPRINT(pub.publication_number)), 3) = 0
    AND EXISTS (
      SELECT 1 FROM UNNEST(pub.cpc) AS cpc
      WHERE cpc.inventive = TRUE
      AND (  cpc.code LIKE 'G06F%'
          OR cpc.code LIKE 'H01L%'
          OR cpc.code LIKE 'G06N%'
          OR cpc.code LIKE 'H04L%')
    )
) AS base

CROSS JOIN UNNEST(base.assignee_harmonized) AS assignee_harmonized
CROSS JOIN UNNEST(base.claims_localized)    AS claim

LEFT JOIN (
  SELECT
    citation.publication_number                         AS cited_pub,
    citing.publication_number                           AS citing_pub,
    citing.filing_date                                  AS citing_date,
    PARSE_DATE('%Y%m%d', CAST(citing.filing_date AS STRING)) AS citing_date_parsed
  FROM
    `patents-public-data.patents.publications` AS citing
  CROSS JOIN UNNEST(citing.citation) AS citation
  WHERE
    citing.filing_date BETWEEN 20050101 AND 20201231
    AND citing.country_code = 'US'
) AS citations

  ON  citations.cited_pub = base.publication_number
  AND citations.citing_date_parsed
      BETWEEN
        PARSE_DATE('%Y%m%d', CAST(base.filing_date AS STRING))
      AND
        DATE_ADD(
          PARSE_DATE('%Y%m%d', CAST(base.filing_date AS STRING)),
          INTERVAL 1826 DAY
        )

WHERE
  assignee_harmonized.country_code = 'US'
  AND (claim.language = 'en' OR claim.language IS NULL)

GROUP BY
  base.publication_number,
  base.application_number,
  base.filing_date,
  assignee_harmonized.name,
  base.cpc_code,
  base.abstract_text,
  base.title_text
"""

with open(VAL_QUERY_PATH, "w") as f:
    f.write(VALIDATION_QUERY)
print(f"  Saved {VAL_QUERY_PATH}", flush=True)

# =============================================================================
# QUERY 3 — PRIOR ART QUERY (2010–2024)
# Full patent universe — no assignee filter
# Used for: computing backward similarity novelty scores for patents_eligible.csv
#
# Why a separate query from the main analysis query:
#   patents_main.csv contains only your 208 target firms.
#   Novelty must be computed against ALL patents in each CPC subclass,
#   not just the patents of your specific sample firms.
#
# Includes title, abstract and claims for SBERT embedding in Step 4.
# No sampling filter — full universe needed for reliable similarity scores.
# =============================================================================
PRIOR_ART_QUERY = """
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
    filing_date BETWEEN 20100101 AND 20241231
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

GROUP BY
  base.publication_number,
  base.filing_date,
  assignee_harmonized.name,
  base.cpc_code,
  base.abstract_text,
  base.title_text
"""

with open(PRIOR_ART_PATH, "w") as f:
    f.write(PRIOR_ART_QUERY)
print(f"  Saved {PRIOR_ART_PATH}", flush=True)

# =============================================================================
# QUERY 4 — CONTROL GROUP QUERY (2010–2024, 10% sample, non-acquired firms)
# Full patent universe EXCLUDING all treated target assignee names.
# Used for: PSM control candidates in Step 7.
#
# Design decisions:
#   NOT IN:             Excludes the {n} treated target assignee names from
#                       crosswalk_final_clean.csv. This is the only
#                       non-acquisition filter applied — no secondary Orbis
#                       M&A screen is performed. Firms acquired by parties
#                       outside the 2015-2021 US-to-US sample may appear in
#                       the control pool. Document this in thesis §4.3.4.
#   FARM_FINGERPRINT:   Deterministic 10% sample — same publication_number
#                       always lands in the same bucket, so re-running the
#                       query produces identical results.
#   Date range:         2010-2024 covers the full ±3yr event window around
#                       all deals (earliest deal 2015 → needs back to 2012;
#                       latest deal 2021 → needs forward to 2024).
#   No application_number: not needed for PSM or novelty scoring.
#   Text columns:       abstract, title, claims required for SBERT encoding
#                       in step4_compute_embeddings.py.
# =============================================================================
CONTROL_QUERY = """
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
    filing_date BETWEEN 20100101 AND 20241231
    AND country_code = 'US'
    -- Deterministic 10% fingerprint sample.
    -- FARM_FINGERPRINT is stable: same publication_number always maps to
    -- the same bucket, so re-running produces the same dataset.
    AND MOD(ABS(FARM_FINGERPRINT(publication_number)), 10) = 0
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
  AND assignee_harmonized.name NOT IN ({assignee_list})

GROUP BY
  base.publication_number,
  base.filing_date,
  assignee_harmonized.name,
  base.cpc_code,
  base.abstract_text,
  base.title_text
"""

query_control = CONTROL_QUERY.format(assignee_list=formatted)
with open(CONTROL_QUERY_PATH, "w") as f:
    f.write(query_control)
print(f"  Saved {CONTROL_QUERY_PATH}", flush=True)
print(f"  Assignee names excluded (NOT IN): {len(assignee_names):,}", flush=True)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 55}", flush=True)
print(f"QUERIES GENERATED", flush=True)
print(f"{'=' * 55}", flush=True)
print(f"  {MAIN_QUERY_PATH:<45} → data/raw/patents_main.csv", flush=True)
print(f"  {VAL_QUERY_PATH:<45} → data/raw/Validation files/ (shards)", flush=True)
print(f"  {PRIOR_ART_PATH:<45} → data/raw/Prior_art files/ (shards)", flush=True)
print(f"  {CONTROL_QUERY_PATH:<45} → data/raw/Control files/ (shards)", flush=True)

print(f"""
>>> MANUAL STEP REQUIRED — run in this order in BigQuery console:

  1. Run bigquery_main_query.sql
     Export result as a single file: data/raw/patents_main.csv

  2. Run bigquery_validation_query.sql
     Export shards to:               data/raw/Validation files/
     Use filename prefix:            patents_validation_*

  3. Run bigquery_prior_art_query.sql
     Export shards to:               data/raw/Prior_art files/
     Use filename prefix:            patents_prior_art_*
     Note: scans full patent universe — sharding is required.

  4. Run bigquery_control_query.sql
     Export shards to:               data/raw/Control files/
     Use filename prefix:            patents_control_*
     Note: excludes {len(assignee_names):,} treated target firm names via NOT IN.
           Does NOT apply a secondary Orbis M&A non-acquisition screen.
           Document this limitation in thesis §4.3.4.

Then run: step3b_combine_shards.py
""", flush=True)