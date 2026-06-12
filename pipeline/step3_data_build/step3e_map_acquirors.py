"""
Step 3e — Build Acquiror Name Mapping (Orbis -> USPTO)
=======================================================
Builds the mapping between Orbis acquiror names and their correct
BigQuery/USPTO patent filing names. This mapping is required because
Orbis uses legal entity names while USPTO uses assignee names as filed,
which rarely match exactly.

Approach:
  1. Load acquiror names from patents_eligible.csv (one per deal)
  2. Resolve known major acquirors using KNOWN_VARIANTS dictionary
  3. Mark all remaining acquirors as NEEDS REVIEW
  4. If a manually verified file exists (from a previous run),
     merge confirmed matches back in automatically

No BigQuery pre-query or fuzzy matching required. Known variants are
authoritative. Unknown acquirors are flagged for manual lookup.

Inputs:
  data/interim/patents_eligible.csv                    — output of step3d
  data/manual/acquiror_needs_review_verified.csv       — optional,
      fill in bq_acquiror_name for NEEDS REVIEW rows and re-run

Outputs:
  data/interim/acquiror_name_map.csv                   — MAPPED rows only,
      used by step3f and step3g
  data/manual/acquiror_needs_review.csv                — NEEDS REVIEW rows,
      open and fill in bq_acquiror_name for each, then re-run

Usage:
  python step3e_map_acquirors.py

  First run:  generates acquiror_needs_review.csv for manual lookup
  Second run: after filling in acquiror_needs_review_verified.csv,
              merges confirmed names into acquiror_name_map.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PATENTS_ELIGIBLE_PATH = "data/interim/patents_eligible.csv"
OUTPUT_MAP_PATH       = "data/interim/acquiror_name_map.csv"
NEEDS_REVIEW_PATH     = "data/manual/acquiror_needs_review.csv"
MANUAL_VERIFIED_PATH  = "data/manual/acquiror_needs_review_verified.csv"

# =============================================================================
# KNOWN USPTO NAME VARIANTS — authoritative mapping for major acquirors
# Orbis name (exactly as it appears in patents_eligible.csv)
#   -> list of BigQuery USPTO filing names
#
# One acquiror can map to multiple BigQuery names because large companies
# file patents under subsidiaries and holding entities. All variants are
# included so step3f pulls the full patent portfolio for each acquiror.
# =============================================================================
KNOWN_VARIANTS = {
    "ALPHABET INC.":                        ["GOOGLE LLC", "GOOGLE INC."],
    "AMAZON.COM INC.":                      ["AMAZON TECHNOLOGIES INC.",
                                             "AMAZON.COM INC."],
    "ANALOG DEVICES INC.":                  ["ANALOG DEVICES INC.",
                                             "ANALOG DEVICES INTERNATIONAL UNLIMITED COMPANY"],
    "APPLE INC.":                           ["APPLE INC."],
    "BROADCOM INC.":                        ["BROADCOM INC.", "BROADCOM CORPORATION",
                                             "AVAGO TECHNOLOGIES INTERNATIONAL SALES PTE. LIMITED"],
    "CA INC.":                              ["CA INC.", "CA TECHNOLOGIES"],
    "CENTURYLINK INC.":                     ["CENTURYLINK IP LLC", "CENTURYLINK INC.",
                                             "QWEST COMMUNICATIONS COMPANY LLC"],
    "CISCO SYSTEMS INC.":                   ["CISCO TECHNOLOGY INC.",
                                             "CISCO SYSTEMS INC."],
    "DELL TECHNOLOGIES INC.":              ["DELL PRODUCTS L.P.", "DELL INC.",
                                             "DELL TECHNOLOGIES INC."],
    "FIDELITY NATIONAL INFORMATION SERVICES INC.": [
                                             "FIDELITY NATIONAL INFORMATION SERVICES INC.",
                                             "FIS"],
    "HEWLETT PACKARD ENTERPRISE COMPANY":  ["HEWLETT PACKARD ENTERPRISE DEVELOPMENT LP",
                                             "HEWLETT PACKARD ENTERPRISE COMPANY"],
    "HEWLETT-PACKARD COMPANY":             ["HEWLETT-PACKARD DEVELOPMENT COMPANY L.P.",
                                             "HEWLETT-PACKARD COMPANY"],
    "II-VI INC.":                           ["II-VI INCORPORATED", "II-VI INC."],
    "INTEL CORPORATION":                    ["INTEL CORPORATION"],
    "INTERNATIONAL BUSINESS MACHINES CORPORATION": [
                                             "INTERNATIONAL BUSINESS MACHINES CORPORATION"],
    "MASTERCARD INC.":                      ["MASTERCARD INTERNATIONAL INC.",
                                             "MASTERCARD INC."],
    "MARVELL TECHNOLOGY INC.":             ["MARVELL TECHNOLOGY GROUP LTD.",
                                             "MARVELL SEMICONDUCTOR INC."],
    "MICROSOFT CORPORATION":               ["MICROSOFT CORPORATION",
                                             "MICROSOFT TECHNOLOGY LICENSING LLC"],
    "MICROCHIP TECHNOLOGY INC.":           ["MICROCHIP TECHNOLOGY INC.",
                                             "MICROCHIP TECHNOLOGY INCORPORATED"],
    "ON SEMICONDUCTOR CORPORATION":        ["SEMICONDUCTOR COMPONENTS INDUSTRIES LLC",
                                             "ON SEMICONDUCTOR CORPORATION"],
    "ORACLE CORPORATION":                  ["ORACLE INTERNATIONAL CORPORATION",
                                             "ORACLE CORPORATION"],
    "QUALCOMM TECHNOLOGIES INC.":          ["QUALCOMM INCORPORATED",
                                             "QUALCOMM TECHNOLOGIES INC."],
    "SALESFORCE.COM INC.":                 ["SALESFORCE INC.", "SALESFORCE.COM INC.",
                                             "SALESFORCE.COM"],
    "SAP AMERICA INC.":                    ["SAP SE", "SAP AMERICA INC."],
    "T-MOBILE US INC.":                    ["T-MOBILE USA INC."],
    "TESSERA TECHNOLOGIES INC.":           ["XPERI INC.", "TESSERA TECHNOLOGIES INC.",
                                             "INVENSAS CORPORATION"],
    "UNITED TECHNOLOGIES CORPORATION":     ["RAYTHEON TECHNOLOGIES CORPORATION",
                                             "UNITED TECHNOLOGIES CORPORATION"],
    "VERIZON COMMUNICATIONS INC.":         ["VERIZON PATENT AND LICENSING INC.",
                                             "CELLCO PARTNERSHIP"],
    "VMWARE INC.":                         ["VMWARE INC.", "VMWARE LLC"],
    "XEROX HOLDINGS CORPORATION":          ["XEROX CORPORATION",
                                             "PALO ALTO RESEARCH CENTER INC."],
}

# =============================================================================
# SETUP
# =============================================================================
print("=" * 60, flush=True)
print("STEP 3e — BUILD ACQUIROR NAME MAPPING", flush=True)
print("=" * 60, flush=True)

os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/manual", exist_ok=True)

if not os.path.exists(PATENTS_ELIGIBLE_PATH):
    print(f"ERROR: {PATENTS_ELIGIBLE_PATH} not found", flush=True)
    print("  Run step3d_build_eligible_patents.py first", flush=True)
    sys.exit(1)

# =============================================================================
# STEP 1 — LOAD ACQUIROR NAMES FROM patents_eligible.csv
# One row per deal (one acquiror name per target firm BvD ID)
# =============================================================================
print(f"\n--- Step 1: Load acquiror names ---", flush=True)

eligible = pd.read_csv(PATENTS_ELIGIBLE_PATH, low_memory=False)

deal_info = (
    eligible[["bvd_id", "orbis_target_name", "Acquiror name", "deal_date"]]
    .drop_duplicates(subset="bvd_id")
    .copy()
)
deal_info["acq_name"] = deal_info["Acquiror name"].str.strip()
acquiror_list = deal_info.dropna(subset=["acq_name"])

print(f"  Eligible firms:   {len(deal_info):,}", flush=True)
print(f"  Unique acquirors: {acquiror_list['acq_name'].nunique():,}", flush=True)

# =============================================================================
# STEP 2 — APPLY KNOWN_VARIANTS
# Each acquiror in KNOWN_VARIANTS gets one row per BigQuery name variant.
# Everything else is marked NEEDS REVIEW with the Orbis name as placeholder.
# =============================================================================
print(f"\n--- Step 2: Apply KNOWN_VARIANTS ---", flush=True)

mapped_rows = []
review_rows = []

for _, deal in acquiror_list.iterrows():
    orbis_name  = deal["acq_name"]
    bvd_id      = deal["bvd_id"]
    target_name = deal["orbis_target_name"]

    variants = KNOWN_VARIANTS.get(orbis_name)

    if variants:
        for bq_name in variants:
            mapped_rows.append({
                "orbis_acquiror_name": orbis_name,
                "bq_acquiror_name":    bq_name,
                "status":              "MAPPED",
                "bvd_id":              bvd_id,
                "target_name":         target_name,
            })
    else:
        review_rows.append({
            "orbis_acquiror_name": orbis_name,
            "bq_acquiror_name":    orbis_name,   # placeholder — fill in manually
            "status":              "NEEDS REVIEW",
            "bvd_id":              bvd_id,
            "target_name":         target_name,
        })

n_mapped_acquirors = len(set(r["orbis_acquiror_name"] for r in mapped_rows))
n_review_acquirors = len(set(r["orbis_acquiror_name"] for r in review_rows))

print(f"  MAPPED (known variants):  {n_mapped_acquirors:,} acquirors  "
      f"({len(mapped_rows):,} BQ name variants)", flush=True)
print(f"  NEEDS REVIEW:             {n_review_acquirors:,} acquirors", flush=True)

# =============================================================================
# STEP 3 — MERGE MANUALLY VERIFIED NAMES
# After filling in bq_acquiror_name in acquiror_needs_review_verified.csv
# and changing status to MAPPED, re-run this script.
# Confirmed rows are automatically promoted and removed from NEEDS REVIEW.
# =============================================================================
print(f"\n--- Step 3: Merge manually verified names ---", flush=True)

manually_mapped_rows = []

if os.path.exists(MANUAL_VERIFIED_PATH):
    manual = pd.read_csv(MANUAL_VERIFIED_PATH)

    required = ["orbis_acquiror_name", "bq_acquiror_name", "status"]
    missing  = [c for c in required if c not in manual.columns]

    if missing:
        print(f"  ERROR: {MANUAL_VERIFIED_PATH} missing columns: {missing}",
              flush=True)
        print(f"  Expected columns: {required}", flush=True)
    else:
        confirmed = manual[
            manual["status"].str.strip().str.upper() == "MAPPED"
        ].copy()

        # Remove confirmed names from review_rows
        confirmed_names = set(confirmed["orbis_acquiror_name"].str.strip())
        review_rows = [r for r in review_rows
                       if r["orbis_acquiror_name"] not in confirmed_names]

        for _, row in confirmed.iterrows():
            manually_mapped_rows.append({
                "orbis_acquiror_name": str(row["orbis_acquiror_name"]).strip(),
                "bq_acquiror_name":    str(row["bq_acquiror_name"]).strip(),
                "status":              "MAPPED",
                "bvd_id":              row.get("bvd_id", ""),
                "target_name":         row.get("target_name", ""),
            })

        print(f"  Found {MANUAL_VERIFIED_PATH}", flush=True)
        print(f"  Manually resolved:  "
              f"{len(confirmed_names):,} acquirors  "
              f"({len(manually_mapped_rows):,} BQ name variants)", flush=True)
        print(f"  Still NEEDS REVIEW: "
              f"{len(set(r['orbis_acquiror_name'] for r in review_rows)):,}",
              flush=True)
else:
    print(f"  {MANUAL_VERIFIED_PATH} not found — skipping", flush=True)

# =============================================================================
# STEP 4 — BUILD FINAL MAPPING TABLE
# =============================================================================
print(f"\n--- Step 4: Build mapping table ---", flush=True)

all_rows   = mapped_rows + manually_mapped_rows + review_rows
mapping_df = pd.DataFrame(all_rows)

# Deduplicate — keep first per (orbis_name, bq_name) pair
mapping_df = mapping_df.drop_duplicates(
    subset=["orbis_acquiror_name", "bq_acquiror_name"]
)

n_total_mapped  = int((mapping_df["status"] == "MAPPED").sum())
n_total_review  = int((mapping_df["status"] == "NEEDS REVIEW").sum())
n_unique_mapped = mapping_df[
    mapping_df["status"] == "MAPPED"
]["orbis_acquiror_name"].nunique()
n_unique_review = mapping_df[
    mapping_df["status"] == "NEEDS REVIEW"
]["orbis_acquiror_name"].nunique()

print(f"  MAPPED:       {n_unique_mapped:,} acquirors  "
      f"({n_total_mapped:,} BQ name variants)", flush=True)
print(f"  NEEDS REVIEW: {n_unique_review:,} acquirors", flush=True)

# =============================================================================
# STEP 5 — SAVE OUTPUTS
# =============================================================================
print(f"\n--- Step 5: Save outputs ---", flush=True)

# acquiror_name_map.csv — MAPPED rows only, used by step3f and step3g
mapped_df = mapping_df[mapping_df["status"] == "MAPPED"].copy()
mapped_df.to_csv(OUTPUT_MAP_PATH, index=False)
print(f"  Saved {OUTPUT_MAP_PATH}  ({len(mapped_df):,} rows)", flush=True)

# acquiror_needs_review.csv — open and fill in correct BigQuery names
review_df = mapping_df[mapping_df["status"] == "NEEDS REVIEW"].copy()
if len(review_df) > 0:
    review_df.to_csv(NEEDS_REVIEW_PATH, index=False)
    print(f"  Saved {NEEDS_REVIEW_PATH}  ({len(review_df):,} rows)",
          flush=True)
else:
    print(f"  No NEEDS REVIEW rows — all acquirors resolved", flush=True)

# =============================================================================
# STEP 6 — SUMMARY REPORT
# =============================================================================
print(f"\n{'=' * 60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'=' * 60}", flush=True)

print(f"\n  MAPPED acquirors ({n_unique_mapped:,}):", flush=True)
prev_name = None
for _, row in mapped_df.sort_values(
    ["orbis_acquiror_name", "bq_acquiror_name"]
).iterrows():
    if row["orbis_acquiror_name"] != prev_name:
        print(f"\n    {row['orbis_acquiror_name']}", flush=True)
        prev_name = row["orbis_acquiror_name"]
    print(f"      -> {row['bq_acquiror_name']}", flush=True)

if len(review_df) > 0:
    print(f"\n  NEEDS REVIEW ({n_unique_review:,} acquirors):", flush=True)
    print(f"  {'Orbis name':<50} {'Target'}", flush=True)
    print(f"  {'-' * 80}", flush=True)
    seen = set()
    for _, row in review_df.sort_values("orbis_acquiror_name").iterrows():
        name = row["orbis_acquiror_name"]
        if name not in seen:
            print(f"  {name:<50} {str(row['target_name'])[:28]}",
                  flush=True)
            seen.add(name)

print(f"""
{'=' * 60}
NEXT STEPS
{'=' * 60}
1. Open {NEEDS_REVIEW_PATH}
   For each row, find the correct USPTO filing name:
     Search https://patents.google.com for the company name
     Or query BigQuery:
       SELECT DISTINCT assignee_harmonized.name, COUNT(*) AS n
       FROM `patents-public-data.patents.publications`
       CROSS JOIN UNNEST(assignee_harmonized) AS assignee_harmonized
       WHERE assignee_harmonized.name LIKE 'COMPANY_NAME%'
       GROUP BY 1 ORDER BY 2 DESC LIMIT 20

   Fill in bq_acquiror_name with the correct USPTO name.
   Change status from NEEDS REVIEW to MAPPED.
   For acquirors with multiple USPTO filing names, add one row per name.
   Save as: {MANUAL_VERIFIED_PATH}

2. Re-run this script — resolved names merge automatically.

3. Add frequently recurring acquirors to KNOWN_VARIANTS in this script
   so future runs resolve them without manual lookup.

4. Once satisfied with {OUTPUT_MAP_PATH}:
   Run step3f_generate_acquiror_sql.py
   -> generates queries/bigquery_acquiror_full_query.sql
   -> run in BigQuery, export as data/raw/patents_acquiror.csv
   -> then run step3g_build_controls.py
""", flush=True)
