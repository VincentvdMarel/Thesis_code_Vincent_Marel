"""
Spot Check — Crosswalk Final Validation
=========================================
Draws a stratified 10% random sample from crosswalk_final.csv across
match score bands, saves it for manual inspection, and after you have
filled in the results, computes and reports precision statistics.

Inputs:
  - crosswalk_final.csv     : your completed crosswalk from Step 2

Outputs:
  - spot_check_sample.csv   : stratified sample to manually verify
  - spot_check_results.csv  : your completed verdicts (you fill this in)
  - spot_check_report.txt   : precision statistics by score band
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

CROSSWALK_PATH      = "data\interim\crosswalk_final.csv"
SAMPLE_OUTPUT       = "data\manual\spot_check_sample.csv"
RESULTS_PATH        = "data\interim\spot_check_results.csv"   # you fill this in after
REPORT_OUTPUT       = "outputs\spot_check_report.txt"

SAMPLE_FRACTION     = 0.10   # 10% spot check
RANDOM_SEED         = 42     # for reproducibility — keep this fixed

# Score bands for stratified sampling
# Ensures riskier low-score matches are checked as thoroughly as high-score ones
SCORE_BINS   = [90, 92, 94, 96, 98, 100]
SCORE_LABELS = ["90-92", "92-94", "94-96", "96-98", "98-100"]


# =============================================================================
# STEP 1 — DRAW STRATIFIED SAMPLE
# Run this first to generate spot_check_sample.csv
# =============================================================================

def draw_sample(crosswalk_path: str, output_path: str):
    print("Loading crosswalk...")
    df = pd.read_csv(crosswalk_path)
    print(f"  Total rows in crosswalk: {len(df):,}")
    print(f"  Unique BvD IDs:          {df['bvd_id'].nunique():,}")

    # Assign score band to each row
    df["score_band"] = pd.cut(
        df["match_score"],
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        include_lowest=True,
        right=True
    )

    print(f"\nScore band distribution:")
    band_counts = df["score_band"].value_counts().sort_index()
    for band, count in band_counts.items():
        sample_n = max(1, int(np.ceil(count * SAMPLE_FRACTION)))
        print(f"  {band}: {count:>5,} rows  →  {sample_n:>4} sampled")

    # Stratified sample: 10% per band, minimum 1 per band
    sample = (
        df.groupby("score_band", observed=True)
        .apply(lambda x: x.sample(
            n=max(1, int(np.ceil(len(x) * SAMPLE_FRACTION))),
            random_state=RANDOM_SEED
        ))
        .reset_index(drop=True)
    )

    # Add empty columns for manual verification
    sample["correct"]   = ""   # fill with: yes / no
    sample["notes"]     = ""   # optional: reason if incorrect

    # Drop the score band helper column from the crosswalk before saving
    sample_out = sample.drop(columns=["score_band"], errors="ignore")

    sample_out.to_csv(output_path, index=False)

    print(f"\nSpot check sample saved to '{output_path}'")
    print(f"  Total rows sampled: {len(sample_out):,}")
    print(f"\nNext step:")
    print(f"  Open '{output_path}' and fill in the 'correct' column")
    print(f"  with 'yes' or 'no' for each row, then run compute_precision()")

    return sample_out


# =============================================================================
# STEP 2 — COMPUTE PRECISION
# Run this after you have filled in spot_check_sample.csv and saved it
# as spot_check_results.csv
# =============================================================================

def compute_precision(results_path: str, report_output: str):
    print("Loading completed spot check results...")
    df = pd.read_csv(results_path)

    # Validate that the correct column has been filled in
    unfilled = df["correct"].isna() | (df["correct"].str.strip() == "")
    if unfilled.sum() > 0:
        print(f"  Warning: {unfilled.sum()} rows still have empty 'correct' values")
        print(f"  These rows are excluded from precision calculations")
        df = df[~unfilled].copy()

    # Normalise values
    df["correct"] = df["correct"].str.strip().str.lower()

    # Validate values
    valid_values = {"yes", "no"}
    invalid = ~df["correct"].isin(valid_values)
    if invalid.sum() > 0:
        print(f"  Warning: {invalid.sum()} rows have invalid values (expected 'yes' or 'no')")
        print(f"  Invalid values found: {df.loc[invalid, 'correct'].unique()}")
        df = df[~invalid].copy()

    # Re-assign score bands for reporting
    df["score_band"] = pd.cut(
        df["match_score"],
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        include_lowest=True,
        right=True
    )

    # Overall precision
    total       = len(df)
    correct     = (df["correct"] == "yes").sum()
    incorrect   = (df["correct"] == "no").sum()
    precision   = correct / total * 100

    # Precision by score band
    band_stats = (
        df.groupby("score_band", observed=True)["correct"]
        .apply(lambda x: (x == "yes").sum() / len(x) * 100)
        .reset_index()
        .rename(columns={"correct": "precision_pct"})
    )

    band_counts = (
        df.groupby("score_band", observed=True)["correct"]
        .agg(
            total="count",
            correct=lambda x: (x == "yes").sum(),
            incorrect=lambda x: (x == "no").sum()
        )
        .reset_index()
    )
    band_counts["precision_pct"] = band_counts["correct"] / band_counts["total"] * 100

    # Build report
    report_lines = [
        "=" * 55,
        "SPOT CHECK PRECISION REPORT",
        "=" * 55,
        f"  Total rows verified:    {total:>6,}",
        f"  Correct matches (yes):  {correct:>6,}",
        f"  Incorrect matches (no): {incorrect:>6,}",
        f"  Overall precision:      {precision:>6.1f}%",
        "",
        "Precision by score band:",
        "-" * 55,
        f"  {'Band':<12} {'Total':>6}  {'Correct':>8}  {'Incorrect':>10}  {'Precision':>10}",
        "-" * 55,
    ]

    for _, row in band_counts.iterrows():
        report_lines.append(
            f"  {str(row['score_band']):<12} "
            f"{int(row['total']):>6}  "
            f"{int(row['correct']):>8}  "
            f"{int(row['incorrect']):>10}  "
            f"{row['precision_pct']:>9.1f}%"
        )

    report_lines += [
        "-" * 55,
        "",
        "Interpretation:",
    ]

    # Automated interpretation
    if precision >= 95:
        report_lines.append(
            f"  PASS: Overall precision of {precision:.1f}% confirms the"
        )
        report_lines.append(
            f"  matching threshold is reliable. This can be reported"
        )
        report_lines.append(
            f"  in the thesis methods section as validation evidence."
        )
    elif precision >= 85:
        report_lines.append(
            f"  ACCEPTABLE: Precision of {precision:.1f}% is acceptable but"
        )
        report_lines.append(
            f"  consider raising the auto-accept threshold or flagging"
        )
        report_lines.append(
            f"  the lower-scoring bands as a limitation."
        )
    else:
        report_lines.append(
            f"  WARNING: Precision of {precision:.1f}% is below acceptable"
        )
        report_lines.append(
            f"  levels. Consider raising the auto-accept threshold."
        )

    # Check if any specific band is problematic
    for _, row in band_counts.iterrows():
        if row["precision_pct"] < 80:
            report_lines.append(
                f"\n  NOTE: Band {row['score_band']} has precision below 80%"
                f" ({row['precision_pct']:.1f}%). Consider moving this band"
                f" back to manual review."
            )

    report_lines += [
        "",
        "Suggested thesis methods text:",
        "-" * 55,
        f'  "Of the {len(pd.read_csv(CROSSWALK_PATH)):,} auto-accepted matches,',
        f'  a stratified random 10% sample (n={total:,}) was manually',
        f'  verified across five score bands, confirming an overall',
        f'  precision rate of {precision:.1f}%, validating the matching',
        f'  threshold as reliable (see Appendix Table X)."',
        "=" * 55,
    ]

    report_text = "\n".join(report_lines)

    # Print to console
    print("\n" + report_text)

    # Save to file
    with open(report_output, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to '{report_output}'")

    return precision, band_counts


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    import os

    # -------------------------------------------------------------------------
    # STEP 1: Generate the sample
    # Always run this first
    # -------------------------------------------------------------------------
    sample = draw_sample(CROSSWALK_PATH, SAMPLE_OUTPUT)

    # -------------------------------------------------------------------------
    # STEP 2: Compute precision
    # Only run this after you have manually filled in spot_check_sample.csv
    # and saved it as spot_check_results.csv
    # -------------------------------------------------------------------------
    if os.path.exists(RESULTS_PATH):
        print("\n" + "=" * 55)
        print("Found completed results file — computing precision...")
        print("=" * 55)
        compute_precision(RESULTS_PATH, REPORT_OUTPUT)
    else:
        print(f"\nOnce you have filled in '{SAMPLE_OUTPUT}',")
        print(f"save it as '{RESULTS_PATH}' and re-run this script")
        print(f"to generate your precision report.")