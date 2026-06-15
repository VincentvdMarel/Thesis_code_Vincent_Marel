# M&A and Patent Data Pipeline: Orbis to USPTO 📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Data](https://img.shields.io/badge/Data-Orbis%20%7C%20USPTO-green)
![NLP](https://img.shields.io/badge/NLP-SBERT%20%7C%20FAISS-purple)
![Econometrics](https://img.shields.io/badge/Econometrics-DiD%20%7C%20PSM-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

This repository contains a comprehensive data engineering and econometric analysis pipeline designed to merge Mergers & Acquisitions (M&A) data from **Orbis** with patent data from the **USPTO** (via Google BigQuery). 

The pipeline facilitates a **Difference-in-Differences (DiD)** causal analysis to evaluate the impact of corporate acquisitions on the technological novelty of target firms. It handles complex entity resolution (fuzzy matching), BigQuery SQL generation, dataset sharding, deep learning-based NLP embeddings, and Propensity Score Matching (PSM).

> **⚠️ Data Not Included:** Please note that this repository contains **only the Python pipeline scripts**. Due to licensing agreements (Orbis data privacy) and massive file size constraints (BigQuery CSV exports, `.npy` SBERT embeddings), all raw, interim, and manual review data files have been excluded (e.g., via `.gitignore`). You must supply your own Orbis and BigQuery data to execute this pipeline.

---

## 📂 Pipeline Architecture

### Phase 2: Entity Resolution (Orbis Target to USPTO Assignee)
* **`step2a_match_names.py`** — Links Orbis target company names to USPTO records using RapidFuzz matrix matching.
* **`step2b_spot_check.py`** — Extracts a 10% stratified random sample to validate matching precision.
* **`step2c_review_tiers.py`** — Manages integration of manually verified edge cases across confidence tiers.
* **`step2d_clean_crosswalk.py`** — Compiles the final verified crosswalk, aligning deal dates and patent count thresholds.

### Phase 3: BigQuery Interfacing & Cohort Construction
* **`step3a_generate_target_sql.py`** — Dynamically generates BigQuery SQL for targets, validation, prior art, and control pools.
* **`step3b_combine_shards.py`** — Merges and deduplicates massive CSV shards exported from BigQuery.
* **`step3c_check_eligibility.py`** — Enforces DiD structural requirements (e.g., 3+ patents pre/post-merger).
* **`step3d_build_eligible_patents.py`** — Constructs the core firm-patent dataset with relative event-time indices.
* **`step3e_map_acquirors.py`** — Maps Orbis acquiror names to their USPTO filing variants.
* **`step3f_generate_acquiror_sql.py`** — Generates SQL to extract full historical patent portfolios of acquiring firms.
* **`step3g_build_controls.py`** — Computes and merges firm-level control variables (e.g., patent age, log deal value, acquiror assets, pre-merger patent volumes) from Orbis and BigQuery into the main dataset.

### Phase 4: SBERT NLP Embeddings
* **`step4_compute_embeddings.py`** — Uses the `sentence-transformers` library (`all-mpnet-base-v2`) to encode the full text of every patent (title + abstract + claims) into highly semantic 768-dimensional normalized vectors.
* **`step4_check_embeddings.py`** — A diagnostic utility that spot-checks `.npy` embedding arrays against their source CSVs to ensure row alignment, correct normal distributions, and text-hash validity.

### Phase 5: Novelty Scoring
* **`step5_compute_novelty.py`** — Computes the primary dependent variable: **Patent Novelty**. Builds localized `FAISS` indices per 4-digit CPC subclass and calculates backward similarity against a 5-year prior art window. *(Formula: Novelty = 1 - Backward Similarity)*.

### Phase 6: Knowledge Overlap (Moderator Variable)
* **`step6a_check_acquiror_coverage.py`** — A pre-flight check ensuring all acquiring entities have sufficient pre-merger patents mapped for centroid computation.
* **`step6b_compute_overlap.py`** — Calculates the pre-merger knowledge overlap between target and acquiror patent portfolios using centroid cosine similarity (following Han, Jo, and Kang 2018).

### Phase 7: Propensity Score Matching (PSM) & Panel Construction
* **`step7a_psm_diagnostics.py`** — Evaluates treated firms to ensure sufficient covariate coverage, checks for collinearity, and determines optimal winsorization thresholds for the matching variables.
* **`step7b_build_control_psm_vars.py`** — Generates PSM candidate rows for control group firms across multiple pseudo-deal years.
* **`step7c_psm_matching.py`** — Executes 1-to-1 nearest neighbor PSM (without replacement) within 4-digit CPC strata, matching on pre-merger patent volume and pre-merger mean novelty. Outputs standard balance tables (SMD).
* **`step7d_build_panel.py`** — Assembles the matched pairs into the final firm-event-time panel datasets. Aggregates data at quarterly (baseline), semi-annual, and annual frequencies for robust econometric testing.
* **`step7e_check_post_coverage.py`** — Validates the integrity of the constructed panel, dropping sparse control/treated matched pairs to ensure balanced parallel trends and post-merger observations.

### Phase 8: Metric Validation
* **`step8_validate_nlp_metric.py`** — Validates the SBERT/FAISS novelty metric via OLS regression. Regresses 5-year forward citations against the novelty score (using the 2010-2015 focal dataset).

### Phase 9: Descriptive Statistics & Econometrics
* **`step9a_descriptive_stats.py`** — Generates summary statistics (Table 1), correlation matrices, and density plots for the final thesis presentation.
* **`step9b_did_regression.py`** — Executes the core Two-Way Fixed Effects (TWFE) Difference-in-Differences regressions. Evaluates the base causal impact of M&A on target novelty, and incorporates interaction terms for the Knowledge Overlap moderator.
* **`step9c_robustness_poor_matches.py`** — Executes econometric robustness checks by re-running the DiD models after strictly filtering out matched pairs with poor PSM caliper distances (>0.25).

---

## ⚙️ Setup & Requirements

**Prerequisites:**
* Python 3.8+
* Data manipulation: `pandas`, `numpy`
* NLP & Deep Learning: `sentence-transformers`, `torch`, `faiss-cpu`
* Econometrics & ML: `statsmodels`, `scikit-learn`
* Visualization: `matplotlib`, `seaborn`
* Google Cloud Platform (BigQuery) access for executing the generated `.sql` files.

**Directory Setup:**
To run the scripts locally, you must recreate the following folder structure in your root directory and place your raw data inside the `data/raw/` folders accordingly:
```text
data/
├── raw/                 # Place your Orbis exports and BigQuery raw CSVs here
│   ├── Validation files/
│   ├── Prior_art files/
│   └── Control files/
├── interim/             # Scripts will generate crosswalks, flags, and .npy files here
└── manual/              # Scripts will generate CSVs requiring human-in-the-loop review here
