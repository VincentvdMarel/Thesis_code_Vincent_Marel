"""
Step 2a — Linking Orbis Targets to USPTO Assignee Records
==========================================================
Inputs:
  - Dataset_thesis.xlsx    : exported from Orbis, must contain columns
                             'bvd_id' and 'target_name'
  - Patents_matching.csv    : exported from BigQuery query (assignee_name,
                             assignee_country, patent_count)

Output:
  - crosswalk_table.csv    : verified mapping of bvd_id → uspto_assignee_name
  - manual_review.csv      : matches in the 85-94 score band for manual check
  - no_match.csv           : targets with no match found, need investigation
"""

import re
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import openpyxl


# =============================================================================
# CONFIGURATION
# =============================================================================

ORBIS_PATH = "data/raw/Matching_Thesis.xlsx"
USPTO_PATH = "data/raw/Patents_matching.csv"

CROSSWALK_OUTPUT   = "data/interim/crosswalk_table.csv"
MANUAL_REVIEW_OUTPUT = "data/manual/manual_review.csv"
NO_MATCH_OUTPUT    = "data/interim/no_match.csv"

# Score thresholds
AUTO_ACCEPT_THRESHOLD   = 90   # auto-accept, spot check 10%
MANUAL_REVIEW_THRESHOLD = 85   # manual review required
# Below 85 → no match / investigate individually

# How many top candidates to return per Orbis name (keeps best N for review)
TOP_N_CANDIDATES = 3


# =============================================================================
# STAGE 2 — LOAD DATA
# =============================================================================

def load_data(orbis_path, uspto_path):
    print("Loading data...")

    orbis = pd.read_excel(orbis_path, sheet_name="Results")
    uspto = pd.read_csv(uspto_path)

    # Validate required columns
    assert "Target BvD ID number" in orbis.columns, "Orbis file must contain 'Target BvD ID number' column"
    orbis = orbis.rename(columns={"Target BvD ID number": "bvd_id"})
    assert "Target name" in orbis.columns, "Orbis file must contain 'Target name' column"
    orbis = orbis.rename(columns={"Target name": "target_name"})
    assert "assignee_name" in uspto.columns, "USPTO file must contain 'assignee_name' column"

    print(f"  Orbis targets loaded:    {len(orbis):,} rows")
    print(f"  USPTO assignees loaded:  {len(uspto):,} rows")

    return orbis, uspto


# =============================================================================
# STAGE 3 — PREPROCESSING
# =============================================================================

LEGAL_SUFFIXES = r'\b(inc|llc|ltd|corp|co|limited|corporation|incorporated|lp|plc|gmbh|bv|nv|sa)\b'

def clean_name(name: str) -> str:
    """
    Normalises a company name for fuzzy matching:
      1. Lowercase and strip whitespace
      2. Remove legal suffixes (Inc, LLC, Ltd, Corp, etc.)
      3. Remove punctuation
      4. Collapse multiple spaces
    """
    if not isinstance(name, str) or name.strip() == "":
        return ""
    name = name.lower().strip()
    name = re.sub(LEGAL_SUFFIXES, "", name)
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def preprocess(orbis: pd.DataFrame, uspto: pd.DataFrame):
    print("\nPreprocessing names...")

    orbis = orbis.copy()
    uspto = uspto.copy()

    orbis["target_name_clean"]      = orbis["target_name"].apply(clean_name)
    uspto["assignee_name_clean"]    = uspto["assignee_name"].apply(clean_name)

    # Drop rows where cleaning produced an empty string
    orbis_empty = orbis[orbis["target_name_clean"] == ""]
    if len(orbis_empty) > 0:
        print(f"  Warning: {len(orbis_empty)} Orbis targets had empty names after cleaning — dropped")
    orbis = orbis[orbis["target_name_clean"] != ""].reset_index(drop=True)

    # Remove duplicates in USPTO reference list on the clean name
    before = len(uspto)
    uspto = uspto.drop_duplicates(subset="assignee_name_clean").reset_index(drop=True)
    print(f"  USPTO duplicates removed: {before - len(uspto):,}")
    print(f"  Clean Orbis targets:  {len(orbis):,}")
    print(f"  Clean USPTO names:    {len(uspto):,}")

    return orbis, uspto


# =============================================================================
# STAGE 4 — FUZZY MATCHING
# =============================================================================

def run_fuzzy_match(orbis: pd.DataFrame, uspto: pd.DataFrame, top_n: int = TOP_N_CANDIDATES):
    """
    For each Orbis target, find the top N matching USPTO assignee names
    using token_sort_ratio scorer. Returns a flat dataframe of all candidates.
    """
    print(f"\nRunning fuzzy matching (top {top_n} candidates per target)...")

    print("  Step A: Checking for instant exact matches...")
    exact_matches = pd.merge(
        orbis, uspto,
        left_on="target_name_clean",
        right_on="assignee_name_clean",
        how="inner"
    )

    # Format the exact matches to match our output schema
    exact_results = []
    for _, row in exact_matches.iterrows():
        exact_results.append({
            "bvd_id":               row["bvd_id"],
            "orbis_target_name":    row["target_name"],
            "target_name_clean":    row["target_name_clean"],
            "uspto_assignee_name":  row["assignee_name"],
            "match_score":          100.0,
            "patent_count":         row.get("patent_count", None),
            "match_status":         "auto_accept" 
        })

    exact_df = pd.DataFrame(exact_results)
    matched_bvd_ids = exact_df["bvd_id"].tolist() if not exact_df.empty else []
    print(f"  --> Found {len(matched_bvd_ids):,} exact matches. Skipping fuzzy logic for these.")

    # Filter out the exact matches from Orbis so we don't calculate them
    orbis_leftovers = orbis[~orbis["bvd_id"].isin(matched_bvd_ids)].reset_index(drop=True)


    #MATRIX MATH (cdist) FOR THE REST
    print(f"  Step B: Running matrix fuzzy matching on remaining {len(orbis_leftovers):,} targets...")
    
    uspto_clean_names = uspto["assignee_name_clean"].tolist()
    uspto_raw_names   = uspto["assignee_name"].tolist()
    patent_counts     = uspto["patent_count"].tolist() if "patent_count" in uspto.columns else [None] * len(uspto)
    orbis_clean_names = orbis_leftovers["target_name_clean"].tolist()

    fuzzy_results = []

    BATCH_SIZE = 1000

    if len(orbis_leftovers) > 0:

       # Loop through the data in chunks
        for start_idx in range(0, len(orbis_clean_names), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(orbis_clean_names))
            print(f"    Processing batch {start_idx:,} to {end_idx:,}...")
            
            batch_queries = orbis_clean_names[start_idx:end_idx]

            # Compute matrix just for this chunk
            score_matrix = process.cdist(
                batch_queries,
                uspto_clean_names,
                scorer=fuzz.token_sort_ratio,
                workers=-1  
            )

            # Process the results of this chunk
            for i, scores in enumerate(score_matrix):
                actual_row_idx = start_idx + i # Track the actual row in the dataframe
                
                valid_indices = np.where(scores >= MANUAL_REVIEW_THRESHOLD)[0]

                if len(valid_indices) == 0:
                    fuzzy_results.append({
                        "bvd_id":               orbis_leftovers.iloc[actual_row_idx]["bvd_id"],
                        "orbis_target_name":    orbis_leftovers.iloc[actual_row_idx]["target_name"],
                        "target_name_clean":    orbis_leftovers.iloc[actual_row_idx]["target_name_clean"],
                        "uspto_assignee_name":  None,
                        "match_score":          None,
                        "patent_count":         None,
                        "match_status":         "no_match"
                    })
                    continue

                valid_scores = scores[valid_indices]
                top_indices = valid_indices[np.argsort(-valid_scores)][:top_n]

                for idx in top_indices:
                    fuzzy_results.append({
                        "bvd_id":               orbis_leftovers.iloc[actual_row_idx]["bvd_id"],
                        "orbis_target_name":    orbis_leftovers.iloc[actual_row_idx]["target_name"],
                        "target_name_clean":    orbis_leftovers.iloc[actual_row_idx]["target_name_clean"],
                        "uspto_assignee_name":  uspto_raw_names[idx],
                        "match_score":          scores[idx],
                        "patent_count":         patent_counts[idx],
                        "match_status":         None 
                    })

    fuzzy_df = pd.DataFrame(fuzzy_results)

    # Combine Exact Matches and Fuzzy Matches back together
    if exact_df.empty:
        results_df = fuzzy_df
    elif fuzzy_df.empty:
        results_df = exact_df
    else:
        results_df = pd.concat([exact_df, fuzzy_df], ignore_index=True)

    print(f"  Total candidate rows generated: {len(results_df):,}")
    return results_df

# =============================================================================
# STAGE 5 — BUCKET ASSIGNMENT
# =============================================================================

def assign_buckets(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns each match to one of three buckets:
      - auto_accept   : score >= 90 (Exact matches are 100)
      - manual_review : score 85–89
      - no_match      : no candidate found
    """
    def bucket(row):
        # Keep "auto_accept" if it was already flagged as an exact match
        if row["match_status"] in ["no_match", "auto_accept"]:
            return row["match_status"]
        
        score = row["match_score"]
        if score >= AUTO_ACCEPT_THRESHOLD:
            return "auto_accept"
        else:
            return "manual_review"

    results_df["match_status"] = results_df.apply(bucket, axis=1)
    return results_df

# =============================================================================
# STAGE 6 — SPLIT AND EXPORT
# =============================================================================

def split_and_export(results_df: pd.DataFrame):
    """
    Splits results into three output files:
      1. crosswalk_table.csv   — auto-accepted matches (the usable crosswalk)
      2. manual_review.csv     — matches needing human verification
      3. no_match.csv          — targets with no match, need investigation
    """
    print("\nSplitting results into output files...")

    # --- Auto-accepted matches ---
    auto = results_df[results_df["match_status"] == "auto_accept"].copy()
    auto["verified"] = "auto"
    auto = auto[[
        "bvd_id", "orbis_target_name", "uspto_assignee_name",
        "match_score", "patent_count", "verified"
    ]]
    auto.to_csv(CROSSWALK_OUTPUT, index=False)
    print(f"  Auto-accepted matches:  {len(auto):,} rows → {CROSSWALK_OUTPUT}")

    # --- Manual review ---
    manual = results_df[results_df["match_status"] == "manual_review"].copy()
    # Add empty column for the researcher to fill in
    manual["verified"] = ""
    manual["notes"] = ""
    manual = manual[[
        "bvd_id", "orbis_target_name", "uspto_assignee_name",
        "match_score", "patent_count", "verified", "notes"
    ]]
    manual.to_csv(MANUAL_REVIEW_OUTPUT, index=False)
    print(f"  Manual review matches:  {len(manual):,} rows → {MANUAL_REVIEW_OUTPUT}")

    # --- No match ---
    no_match = results_df[results_df["match_status"] == "no_match"].copy()
    no_match = no_match[["bvd_id", "orbis_target_name"]].drop_duplicates()
    no_match["notes"] = ""
    no_match.to_csv(NO_MATCH_OUTPUT, index=False)
    print(f"  No-match targets:       {len(no_match):,} rows → {NO_MATCH_OUTPUT}")

    return auto, manual, no_match


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary(orbis: pd.DataFrame, auto: pd.DataFrame,
                  manual: pd.DataFrame, no_match: pd.DataFrame):

    total_targets = orbis["bvd_id"].nunique()  # unique companies, not deal-rows
    auto_targets    = auto["bvd_id"].nunique()
    manual_targets  = manual["bvd_id"].nunique()
    no_match_targets = len(no_match)

    # Targets with at least one match (auto or manual candidate)
    matched         = auto_targets + manual_targets
    match_rate      = matched / total_targets * 100

    print("\n" + "="*55)
    print("MATCHING SUMMARY")
    print("="*55)
    print(f"  Total Unique Orbis targets:          {total_targets:>6,}")
    print(f"  Auto-accepted (score ≥ 90):   {auto_targets:>6,}")
    print(f"  Manual review (score 85-89):  {manual_targets:>6,}")
    print(f"  No match found (score < 85):  {no_match_targets:>6,}")
    print(f"  Overall match rate:           {match_rate:>6.1f}%")
    print("="*55)
    print("\nNext steps:")
    print(f"  1. Open '{MANUAL_REVIEW_OUTPUT}' and fill in the 'verified' column")
    print(f"     with 'yes' or 'no' for each candidate match")
    print(f"  2. Open '{NO_MATCH_OUTPUT}' and manually search Google Patents")
    print(f"     for each unmatched target")
    print(f"  3. Merge verified manual matches back into '{CROSSWALK_OUTPUT}'")


# =============================================================================
# MERGE MANUAL REVIEWS BACK INTO CROSSWALK
# =============================================================================

def merge_verified_manual(crosswalk_path: str, manual_review_path: str,
                           output_path: str = "crosswalk_final.csv"):
    """
    Run this function AFTER you have manually filled in the 'verified' column
    in manual_review.csv. It appends confirmed matches to the crosswalk.

    Call separately once manual review is complete:
        merge_verified_manual("crosswalk_table.csv", "manual_review.csv")
    """
    crosswalk = pd.read_csv(crosswalk_path)
    manual    = pd.read_csv(manual_review_path)

    # Keep only rows the researcher confirmed as correct
    confirmed = manual[manual["verified"].str.lower() == "yes"].copy()
    confirmed = confirmed.drop(columns=["notes"], errors="ignore")

    combined = pd.concat([crosswalk, confirmed], ignore_index=True)
    combined = combined.sort_values(["bvd_id", "match_score"], ascending=[True, False])
    combined.to_csv(output_path, index=False)

    print(f"Final crosswalk saved to '{output_path}'")
    print(f"  Auto-accepted rows:  {len(crosswalk):,}")
    print(f"  Manually verified:   {len(confirmed):,}")
    print(f"  Total rows:          {len(combined):,}")
    print(f"  Unique BvD IDs:      {combined['bvd_id'].nunique():,}")

    return combined


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Stage 2 — Load
    orbis, uspto = load_data(ORBIS_PATH, USPTO_PATH)

    # Stage 3 — Preprocess
    orbis, uspto = preprocess(orbis, uspto)

    # Stage 4 — Fuzzy match
    results_df = run_fuzzy_match(orbis, uspto, top_n=TOP_N_CANDIDATES)

    # Stage 5 — Assign buckets
    results_df = assign_buckets(results_df)

    # Stage 6 — Split and export
    auto, manual, no_match = split_and_export(results_df)

    # Summary
    print_summary(orbis, auto, manual, no_match)

    # -------------------------------------------------------------------------
    # Once you have manually reviewed manual_review.csv, run this separately:
    # -------------------------------------------------------------------------
    merge_verified_manual("data/interim/crosswalk_table.csv", "data/manual/manual_review_fixed.csv")
