import pandas as pd
import os

# =============================================================================
# SET THIS TO True AFTER YOU HAVE FILLED IN manual_review_96_98.csv
# =============================================================================
MANUAL_REVIEW_COMPLETE = True

df = pd.read_csv("data/interim/crosswalk_final.csv")

# Tier 1 — Auto accept (scores 98-100), already validated
tier1 = df[df["match_score"] >= 98].copy()

# Tier 2 — Requires manual review (scores 96-98)
tier2 = df[(df["match_score"] >= 96) & (df["match_score"] < 98)].copy()

# Tier 3 — Pre-verified manual matches (scores < 90)
# These are your 45 manually approved rows from previous steps!
pre_verified = df[df["match_score"] < 90].copy()

# Tier 4 — Discarded (scores 90 to 96)
# These are the bands that failed your spot check
discarded = df[(df["match_score"] >= 90) & (df["match_score"] < 96)].copy()

if not os.path.exists("data/manual/manual_review_96_98.csv"):
    tier2["correct"] = ""
    tier2.to_csv("data/manual/manual_review_96_98.csv", index=False)
    print("Generated manual_review_96_98.csv — please fill in the correct column")
else:
    print("manual_review_96_98.csv already exists — skipping regeneration")

print(f"Auto-accepted (≥98):             {len(tier1):,} rows")
print(f"Manual review (96-98):           {len(tier2):,} rows")
print(f"Pre-verified from earlier (<90): {len(pre_verified):,} rows")
print(f"Discarded (90 to <96):           {len(discarded):,} rows")

if not MANUAL_REVIEW_COMPLETE:
    print("\nNext step:")
    print("  Open manual_review_96_98.csv and fill in the")
    print("  'correct' column with yes or no for each row.")
    print("  Then set MANUAL_REVIEW_COMPLETE = True and re-run.")
else:
    reviewed  = pd.read_csv("data/manual/manual_review_96_98.csv")
    confirmed = reviewed[reviewed["correct"].str.strip().str.lower() == "yes"]
    rejected  = reviewed[reviewed["correct"].str.strip().str.lower() == "no"]

    # --- Manual review acceptance summary ---
    print(f"\nManual review results (96-98 band):")
    print(f"  Total reviewed:  {len(reviewed):,} rows")
    print(f"  Accepted (yes):  {len(confirmed):,} rows")
    print(f"  Rejected (no):   {len(rejected):,} rows")

    # Combine Auto-accepted, Newly confirmed, AND the Pre-verified 45 rows
    final = pd.concat([tier1, confirmed, pre_verified], ignore_index=True)
    
    # Save paths fixed to route correctly to data/interim/
    verified_path = "data/interim/crosswalk_verified.csv"
    duplicates_path = "data/interim/crosswalk_duplicates.csv"
    
    if not os.path.exists(verified_path):
        final.to_csv(verified_path, index=False)
        print(f"\nFinal crosswalk rows:    {len(final):,}")
        print(f"Unique BvD IDs:          {final['bvd_id'].nunique():,}")
        print(f"Saved to {verified_path}")
    else:
        print(f"\n{verified_path} already exists — skipping save to prevent overwrite.")

    # --- Duplicate BvD ID report ---
    duplicate_bvd_ids = final[final.duplicated(subset="bvd_id", keep=False)]
    duplicate_bvd_ids = duplicate_bvd_ids.sort_values(["bvd_id", "match_score"], ascending=[True, False])

    if len(duplicate_bvd_ids) > 0:
        if not os.path.exists(duplicates_path):
            duplicate_bvd_ids.to_csv(duplicates_path, index=False)
            print(f"\nDuplicate BvD IDs found:")
            print(f"  Rows with duplicated BvD IDs: {len(duplicate_bvd_ids):,}")
            print(f"  Unique BvD IDs duplicated:    {duplicate_bvd_ids['bvd_id'].nunique():,}")
            print(f"Saved to {duplicates_path}")
        else:
            print(f"{duplicates_path} already exists — skipping save.")
    else:
        print(f"\nNo duplicate BvD IDs found in {verified_path}")