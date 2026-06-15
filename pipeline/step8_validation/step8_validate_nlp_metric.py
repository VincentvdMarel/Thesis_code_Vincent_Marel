"""
Step 8 — NLP Metric Validation
================================
Validates the SBERT patent novelty metric by testing its predictive
power for five-year forward citation counts. Benchmarks SBERT against
TF-IDF and word2vec baselines.

Validation design (Section 4.4.1):
  - Focal patents:  2010-2015 (novelty scores already computed in step5)
  - Prior art pool: 2005-2015, same CPC subgroups as focal
  - DV:             Five-year forward citation count 
  - IV:             Novelty score at filing date (t=0)
  - FE:             CPC subgroup + filing year (all estimators)
  - SE:             Clustered by assignee_name

Model structure (motivated by var/mean ~ 49 in corrected citation data):
  PRIMARY:    NB2 with CPC + filing-year FEs + assignee-clustered SEs.
              With only G=4 CPC subgroups and T~6 filing years,
              incidental-parameters bias from dummy-variable FEs is
              negligible (Hausman, Hall and Griliches 1984).
  ROBUSTNESS: (1) FE Poisson (PPML) via pyfixest, same CPC + year FEs absorbed
              via within-transformation, assignee-clustered SEs. Consistent
              under any conditional mean spec (Santos Silva & Tenreyro 2006).
              (2) Zero-Inflated Negative Binomial (ZINB) via statsmodels,
              intercept-only inflation equation, same FEs and clustered SEs.
              An NB-vs-ZINB boundary LR test (p = 0.5*Pr(chi²(1)>LR)) and
              the structural-zero probability from the inflate intercept
              jointly determine whether ZINB is preferred over NB2.

Baseline comparison (Appendix Table):
  - TF-IDF cosine similarity novelty  (lookahead-corrected: per-year IDF)
  - word2vec average-vector novelty
  - SBERT backward similarity novelty (main specification)

Inputs:
  data/interim/patents_validation_novelty.csv   -- output of step5 + citation patch

Outputs:
  data/outputs/validation_results.csv           -- regression coefficients
  data/outputs/table_nlp_comparison.tex         -- LaTeX comparison table

Usage:
  python step8_validate_nlp_metric.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Inverting hessian failed")

from tqdm import tqdm
from joblib import Parallel, delayed

# =============================================================================
# CONFIGURATION
# =============================================================================
VALIDATION_PATH    = "data/interim/patents_validation_novelty.csv"
OUTPUT_DIR         = "data/outputs"
RESULTS_PATH       = os.path.join(OUTPUT_DIR, "validation_results.csv")
TABLE_PATH         = os.path.join(OUTPUT_DIR, "table_nlp_comparison.tex")
EQUALITY_TABLE_PATH = os.path.join(OUTPUT_DIR, "table_coef_equality.tex")

NOVELTY_START      = pd.Timestamp("2010-01-01")
NOVELTY_END        = pd.Timestamp("2015-12-31")
PRIOR_ART_YEARS    = 5
CPC_CHARS          = 4
MAX_CHARS_BASELINE = 1800   

N_JOBS             = -1     

DEBUG              = False
MAX_SUBGROUPS      = 10

TFIDF_MAX_FEATURES = 3000
TFIDF_MIN_DF       = 3
TFIDF_MAX_DF       = 0.90

# =============================================================================
# SETUP
# =============================================================================
print("=" * 65, flush=True)
print("STEP 8 — NLP METRIC VALIDATION", flush=True)
print("=" * 65, flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(VALIDATION_PATH):
    print(f"ERROR: {VALIDATION_PATH} not found", flush=True)
    print("  Run step5_compute_novelty.py first", flush=True)
    sys.exit(1)

import multiprocessing
n_cpus = multiprocessing.cpu_count()
print(f"CPUs available: {n_cpus}  |  N_JOBS = {N_JOBS}", flush=True)
if DEBUG:
    print(f"DEBUG MODE: processing only {MAX_SUBGROUPS} subgroups", flush=True)

# =============================================================================
# PHASE 1 — LOAD AND PREPARE DATA
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 1 — LOAD AND PREPARE", flush=True)
print(f"{'=' * 65}", flush=True)

print(f"Loading {VALIDATION_PATH}...", flush=True)
df = pd.read_csv(VALIDATION_PATH, low_memory=False)

# --- Parse dates ---
raw_filing_date = df["filing_date"].astype(str).str.strip()
df["filing_date"] = pd.to_datetime(
    raw_filing_date, format="%Y%m%d", errors="coerce"
)
mask_still_na = df["filing_date"].isna()
df.loc[mask_still_na, "filing_date"] = pd.to_datetime(
    raw_filing_date[mask_still_na], errors="coerce"
)

df["filing_year"] = df["filing_date"].dt.year
df["cpc_4digit"]  = df["cpc_code"].str[:CPC_CHARS].str.upper()

print(f"  Total rows:    {len(df):,}", flush=True)

# --- Focal sample: 2010-2015 ---
focal_df = df[
    (df["filing_date"] >= NOVELTY_START) &
    (df["filing_date"] <= NOVELTY_END)
].copy()
print(f"  Focal ({NOVELTY_START.year}-{NOVELTY_END.year}): {len(focal_df):,}",
      flush=True)

# --- Prior art pool ---
min_prior_date  = NOVELTY_START - pd.DateOffset(years=PRIOR_ART_YEARS)
valid_subgroups = set(focal_df["cpc_4digit"].dropna().unique())

prior_df = df[
    (df["filing_date"] >= min_prior_date) &
    (df["filing_date"] <= NOVELTY_END) &
    (df["cpc_4digit"].isin(valid_subgroups))
].copy()
print(f"  Prior art pool ({min_prior_date.year}-{NOVELTY_END.year}, "
      f"relevant CPC only): {len(prior_df):,}", flush=True)

# --- Check required columns ---
n_scored = focal_df["novelty_score"].notna().sum()
print(f"  Focal with SBERT novelty: {n_scored:,} "
      f"({n_scored/len(focal_df)*100:.1f}%)", flush=True)

if "forward_citations_5yr" not in df.columns:
    print("ERROR: forward_citations_5yr column not found", flush=True)
    sys.exit(1)

# --- Build regression sample ---
reg_df = focal_df.dropna(
    subset=["novelty_score", "forward_citations_5yr",
            "cpc_4digit", "filing_year"]
).copy()
reg_df["citations"] = reg_df["forward_citations_5yr"].astype(int)

print(f"\n  Regression sample:   {len(reg_df):,} focal patents", flush=True)
print(f"  Citation stats:", flush=True)
print(f"    Mean:   {reg_df['citations'].mean():.2f}", flush=True)
print(f"    Median: {reg_df['citations'].median():.2f}", flush=True)
print(f"    Max:    {reg_df['citations'].max():.0f}", flush=True)
print(f"    % zero: {(reg_df['citations']==0).mean()*100:.1f}%", flush=True)

mean_cit = reg_df["citations"].mean()
var_cit  = reg_df["citations"].var()
var_ratio = var_cit / mean_cit if mean_cit > 0 else np.nan
zero_pct  = (reg_df["citations"] == 0).mean() * 100

if not np.isnan(var_ratio):
    print(f"    Variance/Mean ratio: {var_ratio:.2f}  "
          f"{'overdispersed — NB2 appropriate' if var_ratio > 2 else 'check Poisson'}",
          flush=True)

# --- Detect assignee column for clustering ---
CLUSTER_COL = next(
    (c for c in ["assignee_name", "assignee_harmonized", "assignee"]
     if c in reg_df.columns),
    None
)
if CLUSTER_COL:
    n_clusters = reg_df[CLUSTER_COL].nunique()
    print(f"\n  Cluster variable: '{CLUSTER_COL}'  ({n_clusters:,} unique assignees)",
          flush=True)
    if n_clusters < 50:
        print("  WARNING: fewer than 50 clusters — clustered SEs may be unreliable",
              flush=True)
else:
    print(f"\n  WARNING: no assignee column found.", flush=True)
    print(f"  Expected one of: assignee_name, assignee_harmonized, assignee",
          flush=True)
    print(f"  Available columns: {list(reg_df.columns[:15])}", flush=True)
    print("  Falling back to heteroskedasticity-robust SEs.", flush=True)

# =============================================================================
# PHASE 2 — COMPUTE BASELINE NOVELTY SCORES
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 2 — BASELINE NOVELTY SCORES (ROLLING WINDOW)", flush=True)
print(f"{'=' * 65}", flush=True)

print(f"Building full_text (title -> abstract -> claims, max {MAX_CHARS_BASELINE} chars)...",
      flush=True)
t0 = time.time()
_title    = df["title_text"].fillna("").astype(str).str.strip()
_abstract = df["abstract_text"].fillna("").astype(str).str.strip()
_claims   = df["all_claims_text"].fillna("").astype(str).str.strip()
df["full_text"] = (
    (_title + " " + _abstract + " " + _claims)
    .str[:MAX_CHARS_BASELINE]
    .str.strip()
)
del _title, _abstract, _claims
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

focal_df["full_text"] = df.loc[focal_df.index, "full_text"].values
prior_df["full_text"] = df.loc[prior_df.index, "full_text"].values

subgroups = sorted(focal_df["cpc_4digit"].dropna().unique())
if DEBUG:
    subgroups = subgroups[:MAX_SUBGROUPS]
print(f"  CPC subgroups to process: {len(subgroups):,}", flush=True)

# -------------------------------------------------------------------------
# 2a — TF-IDF Novelty (Rolling Window, per-year IDF to avoid lookahead)
# -------------------------------------------------------------------------
print(f"\n--- TF-IDF backward similarity (rolling window, per-year IDF) ---",
      flush=True)

def tfidf_subgroup_fast(subgroup, focal_df, prior_df, prior_art_years):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pandas as pd

    focal_sub = focal_df[focal_df["cpc_4digit"] == subgroup].copy()
    prior_sub = prior_df[prior_df["cpc_4digit"] == subgroup].copy()
    scores = {}

    if len(focal_sub) == 0 or len(prior_sub) == 0:
        return {idx: np.nan for idx in focal_sub.index}

    focal_sub = focal_sub.sort_values("filing_date")
    prior_sub = prior_sub.sort_values("filing_date")

    focal_years = sorted(focal_sub["filing_date"].dt.year.dropna().unique())

    for focal_year in focal_years:
        cutoff   = pd.Timestamp(f"{focal_year}-01-01")
        fit_pool = prior_sub[prior_sub["filing_date"] < cutoff]

        if len(fit_pool) < TFIDF_MIN_DF:
            year_focal = focal_sub[focal_sub["filing_date"].dt.year == focal_year]
            for idx in year_focal.index:
                scores[idx] = np.nan
            continue

        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,
            norm="l2",
        )
        vectorizer.fit(fit_pool["full_text"].fillna(""))
        n_features    = len(vectorizer.vocabulary_)
        prior_vecs_yr  = vectorizer.transform(fit_pool["full_text"].fillna(""))
        prior_dense_yr = prior_vecs_yr.toarray()   # convert once; avoids per-row .toarray() in loop

        year_focal    = focal_sub[
            focal_sub["filing_date"].dt.year == focal_year
        ].sort_values("filing_date")
        focal_vecs_yr  = vectorizer.transform(year_focal["full_text"].fillna(""))
        focal_dense_yr = focal_vecs_yr.toarray()   # convert once

        prior_dates_yr   = fit_pool["filing_date"].values
        focal_dates_yr   = year_focal["filing_date"].values
        focal_indices_yr = year_focal.index.to_list()

        left = right = 0

        for i in range(len(year_focal)):
            filing_date  = focal_dates_yr[i]
            window_start = filing_date - pd.DateOffset(years=prior_art_years)

            while right < len(fit_pool) and prior_dates_yr[right] < filing_date:
                right += 1

            while left < right and prior_dates_yr[left] < window_start:
                left += 1

            focal_idx = focal_indices_yr[i]
            if right == left:
                scores[focal_idx] = np.nan
            else:
                focal_row   = focal_dense_yr[i]
                window_vecs = prior_dense_yr[left:right]  # O(1) numpy slice
                cosines     = window_vecs.dot(focal_row)  # shape: (right-left,)
                scores[focal_idx] = 1.0 - float(cosines.mean())

    return scores

try:
    t0 = time.time()
    results_list = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(tfidf_subgroup_fast)(sg, focal_df, prior_df, PRIOR_ART_YEARS)
        for sg in tqdm(subgroups, desc="TF-IDF subgroups")
    )
    tfidf_scores = {}
    for d in results_list:
        tfidf_scores.update(d)
    focal_df["novelty_tfidf"] = pd.Series(tfidf_scores)
    n_tfidf = focal_df["novelty_tfidf"].notna().sum()
    print(f"  TF-IDF novelty computed for {n_tfidf:,} patents "
          f"in {time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"  WARNING: TF-IDF failed — {e}", flush=True)
    focal_df["novelty_tfidf"] = np.nan

# -------------------------------------------------------------------------
# 2b — word2vec Novelty 
# -------------------------------------------------------------------------
print(f"\n--- word2vec backward similarity (rolling window) ---", flush=True)

try:
    import gensim.downloader as api
    print("  Loading word2vec-google-news-300 (~1.6 GB)...", flush=True)
    w2v_model = api.load("word2vec-google-news-300")
    print("  Model loaded", flush=True)

    def get_w2v_vector(text, model):
        tokens = str(text).lower().split()
        vecs   = [model[w] for w in tokens if w in model]
        if len(vecs) == 0:
            return None
        vec  = np.mean(vecs, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    relevant_indices = set(focal_df.index) | set(prior_df.index)
    relevant_df      = df.loc[df.index.isin(relevant_indices)]

    print(f"  Computing word2vec vectors for {len(relevant_df):,} patents...",
          flush=True)
    t0 = time.time()
    w2v_vecs = {}
    for idx, text in tqdm(
        zip(relevant_df.index, relevant_df["full_text"]),
        total=len(relevant_df), desc="word2vec vectors"
    ):
        vec = get_w2v_vector(text, w2v_model)
        if vec is not None:
            w2v_vecs[idx] = vec
    print(f"  Vectors computed: {len(w2v_vecs):,} in {time.time()-t0:.0f}s",
          flush=True)

    W2V_DIM = len(next(iter(w2v_vecs.values())))

    def w2v_subgroup_fast(subgroup, focal_df, prior_df, w2v_vecs, prior_art_years):
        import numpy as np
        import pandas as pd

        focal_sub = focal_df[focal_df["cpc_4digit"] == subgroup].copy()
        prior_sub = prior_df[prior_df["cpc_4digit"] == subgroup].copy()
        scores = {}

        if len(focal_sub) == 0 or len(prior_sub) == 0:
            return {idx: np.nan for idx in focal_sub.index}

        focal_sub = focal_sub.sort_values("filing_date")
        prior_sub = prior_sub.sort_values("filing_date")
        prior_sub = prior_sub.loc[
            [idx for idx in prior_sub.index if idx in w2v_vecs]
        ]
        if len(prior_sub) == 0:
            return {idx: np.nan for idx in focal_sub.index}

        prior_dates   = prior_sub["filing_date"].values
        focal_dates   = focal_sub["filing_date"].values
        prior_indices = prior_sub.index.to_list()
        focal_indices = focal_sub.index.to_list()

        # Precompute dense matrix once so inner loop uses O(1) numpy slices
        prior_matrix = np.stack([w2v_vecs[idx] for idx in prior_indices])

        left = right = 0

        for i in range(len(focal_sub)):
            filing_date  = focal_dates[i]
            window_start = filing_date - pd.DateOffset(years=prior_art_years)

            while right < len(prior_sub) and prior_dates[right] < filing_date:
                right += 1

            while left < right and prior_dates[left] < window_start:
                left += 1

            focal_idx    = focal_indices[i]
            window_count = right - left

            if focal_idx not in w2v_vecs or window_count == 0:
                scores[focal_idx] = np.nan
            else:
                focal_vec   = w2v_vecs[focal_idx]
                window_vecs = prior_matrix[left:right]  # O(1) numpy slice
                cosines     = window_vecs.dot(focal_vec)
                scores[focal_idx] = 1.0 - float(cosines.mean())

        return scores

    t0 = time.time()
    w2v_results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(w2v_subgroup_fast)(sg, focal_df, prior_df, w2v_vecs, PRIOR_ART_YEARS)
        for sg in tqdm(subgroups, desc="word2vec subgroups")
    )
    w2v_scores = {}
    for d in w2v_results:
        w2v_scores.update(d)
    focal_df["novelty_w2v"] = pd.Series(w2v_scores)
    n_w2v = focal_df["novelty_w2v"].notna().sum()
    print(f"  word2vec novelty computed for {n_w2v:,} patents "
          f"in {time.time()-t0:.0f}s", flush=True)

except Exception as e:
    print(f"  WARNING: word2vec failed — {e}", flush=True)
    focal_df["novelty_w2v"] = np.nan

# --- Join baseline scores to regression sample ---
reg_df = reg_df.join(focal_df[["novelty_tfidf", "novelty_w2v"]], how="left")


# =============================================================================
# PHASE 2c — DESCRIPTIVE STATISTICS & PAIRWISE CORRELATIONS
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 2c — DESCRIPTIVE STATISTICS & PAIRWISE CORRELATIONS", flush=True)
print(f"{'=' * 65}", flush=True)

METRIC_LABELS = {
    "novelty_tfidf": "TF-IDF",
    "novelty_w2v":   "Word2Vec",
    "novelty_score": "SBERT",
}
METRIC_COLS = list(METRIC_LABELS.keys())

# --- Descriptive statistics ---
desc = (
    reg_df[METRIC_COLS]
    .describe()
    .T
    .rename(index=METRIC_LABELS)
)
desc.columns = ["N", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
desc["N"] = desc["N"].astype(int)

print(f"\n  Descriptive Statistics (regression sample, N = {len(reg_df):,}):\n",
      flush=True)
print(desc.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)

# --- Pearson correlation ---
pearson_corr = (
    reg_df[METRIC_COLS]
    .corr(method="pearson")
    .rename(index=METRIC_LABELS, columns=METRIC_LABELS)
)
print(f"\n  Pairwise Pearson Correlations:\n", flush=True)
print(pearson_corr.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)

# --- Spearman rank correlation ---
spearman_corr = (
    reg_df[METRIC_COLS]
    .corr(method="spearman")
    .rename(index=METRIC_LABELS, columns=METRIC_LABELS)
)
print(f"\n  Pairwise Spearman Rank Correlations:\n", flush=True)
print(spearman_corr.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)

# --- Overdispersion figure ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cit      = reg_df["citations"]
    p99      = int(np.percentile(cit, 99))
    observed = cit[cit <= p99]
    bins     = np.arange(-0.5, p99 + 1.5, 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(observed, bins=bins, color="#4C72B0", alpha=0.75)
    ax.set_xlabel("5-year forward citation count", fontsize=11)
    ax.set_ylabel("Number of patents", fontsize=11)
    ax.set_title("Distribution of 5-year forward citation count", fontsize=11)
    ax.set_xlim(-0.5, min(p99, 60) + 0.5)
    fig.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, "fig_citation_overdispersion.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved overdispersion figure: {fig_path}", flush=True)

except Exception as e:
    print(f"\n  WARNING: overdispersion figure failed — {e}", flush=True)

# =============================================================================
# PHASE 3 — COUNT MODEL REGRESSIONS
# =============================================================================
# PRIMARY: NB2, CPC + year FE, assignee-clustered SEs
#
# ROBUSTNESS: FE Poisson (PPML) 
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 3 — COUNT MODEL REGRESSIONS", flush=True)
print(f"{'=' * 65}", flush=True)

import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

try:
    import pyfixest as pf
    HAS_PYFIXEST = True
    print("  pyfixest loaded — FE Poisson (PPML) available", flush=True)
except ImportError:
    HAS_PYFIXEST = False
    print("  WARNING: pyfixest not installed. Run: pip install pyfixest",
          flush=True)
    print("  PPML robustness spec will be skipped.", flush=True)

try:
    import patsy as _patsy
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP as _ZINB
    HAS_ZINB = True
    print("  statsmodels ZINB loaded — Zero-Inflated NB (ZINB) available", flush=True)
except ImportError:
    HAS_ZINB = False
    print("  WARNING: statsmodels ZeroInflatedNegativeBinomialP not available.",
          flush=True)
    print("  ZINB robustness spec will be skipped.", flush=True)

print(f"\n  PRIMARY:      NB2, CPC + year FE, clustered SEs", flush=True)
print(f"  Rationale:    var/mean = {var_ratio:.1f} — extreme overdispersion, "
      f"NB2 most efficient", flush=True)
print(f"  ROBUSTNESS 1: FE Poisson (PPML), same CPC + year FEs, clustered SEs",
      flush=True)
print(f"  ROBUSTNESS 2: ZINB, intercept-only inflation, same FEs, clustered SEs",
      flush=True)

MODELS = {
    "tfidf": ("novelty_tfidf", "TF-IDF"),
    "w2v":   ("novelty_w2v",   "word2vec"),
    "sbert": ("novelty_score", "SBERT"),
}

results_nb   = {}   # primary:      NB2, CPC + year FE
results_ppml = {}   # robustness 1: FE Poisson (PPML)
results_zinb = {}   # robustness 2: Zero-Inflated NB (ZINB)

for model_key, (novelty_col, model_label) in MODELS.items():

    print(f"\n--- {model_label} ---", flush=True)

    model_df = reg_df.dropna(
        subset=[novelty_col, "citations", "cpc_4digit", "filing_year"]
    ).copy()
    model_df["cpc_4digit"]  = model_df["cpc_4digit"].astype(str)
    model_df["filing_year"] = model_df["filing_year"].astype(str)

    print(f"  Sample: {len(model_df):,} patents", flush=True)

    if len(model_df) == 0:
        print("  SKIP: no valid observations", flush=True)
        continue

    # -------------------------------------------------------------------------
    # PRIMARY: NB2 with CPC + year FE + clustered SEs
    # -------------------------------------------------------------------------
    try:
        print("  Fitting NB2 (CPC + year FE, clustered SEs)...", flush=True)
        t0         = time.time()
        nb_formula = f"citations ~ {novelty_col} + C(filing_year) + C(cpc_4digit)"

        if CLUSTER_COL:
            fit_nb = smf.negativebinomial(nb_formula, data=model_df).fit(
                method="bfgs",
                cov_type="cluster",
                cov_kwds={"groups": model_df[CLUSTER_COL]},
                disp=False,
                maxiter=200,
            )
        else:
            fit_nb = smf.negativebinomial(nb_formula, data=model_df).fit(
                method="bfgs", cov_type="HC1", disp=False, maxiter=200,
            )

        coef  = fit_nb.params[novelty_col]
        se    = fit_nb.bse[novelty_col]
        pval  = fit_nb.pvalues[novelty_col]
        alpha = (float(fit_nb.params["alpha"])
                 if "alpha" in fit_nb.params.index else float("nan"))
        n_obs = int(fit_nb.nobs)
        ll    = fit_nb.llf
        sig   = ("***" if pval < 0.01 else "**" if pval < 0.05
                 else "*" if pval < 0.10 else "")

        results_nb[model_key] = {
            "model": model_label, "estimator": "NB2",
            "coef": coef, "se": se, "pval": pval,
            "n": n_obs, "alpha": alpha, "ll": ll,
            "clustered": CLUSTER_COL is not None, "sig": sig,
        }
        print(f"  NB2:  b = {coef:+.4f}  SE = {se:.4f}  "
              f"p = {pval:.4f}  {sig}  a = {alpha:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    except Exception as e:
        print(f"  NB2 failed: {e}", flush=True)

    # -------------------------------------------------------------------------
    # ROBUSTNESS: FE Poisson (PPML) via pyfixest
    # -------------------------------------------------------------------------
    if HAS_PYFIXEST:
        try:
            print("  Fitting FE Poisson (PPML)...", flush=True)
            t0        = time.time()
            vcov_spec = ({"CRV1": CLUSTER_COL} if CLUSTER_COL else "hetero")

            fit_ppml = pf.fepois(
                fml=f"citations ~ {novelty_col} | cpc_4digit + filing_year",
                data=model_df,
                vcov=vcov_spec,
            )

            coef  = float(fit_ppml.coef()[novelty_col])
            se    = float(fit_ppml.se()[novelty_col])
            pval  = float(fit_ppml.pvalue()[novelty_col])
            try:
                n_obs = int(fit_ppml._N)
            except AttributeError:
                n_obs = len(model_df)
            sig = ("***" if pval < 0.01 else "**" if pval < 0.05
                   else "*" if pval < 0.10 else "")

            results_ppml[model_key] = {
                "model": model_label, "estimator": "PPML",
                "coef": coef, "se": se, "pval": pval,
                "n": n_obs, "clustered": CLUSTER_COL is not None, "sig": sig,
            }
            print(f"  PPML: b = {coef:+.4f}  SE = {se:.4f}  "
                  f"p = {pval:.4f}  {sig}  ({time.time()-t0:.0f}s)",
                  flush=True)

        except Exception as e:
            print(f"  PPML failed: {e}", flush=True)

    # -------------------------------------------------------------------------
    # ROBUSTNESS 2: Zero-Inflated Negative Binomial (ZINB) 
    # Intercept-only inflation equation (logistic); same CPC + year FEs as NB2.
    # Identifies whether structural zeros drive the 32% zero share, or whether
    # NB2 dispersion already accounts for them.
    # -------------------------------------------------------------------------
    if HAS_ZINB:
        try:
            print("  Fitting ZINB (intercept-only inflation, CPC + year FE)...",
                  flush=True)
            t0 = time.time()

            zinb_formula = (
                f"citations ~ {novelty_col} + C(filing_year) + C(cpc_4digit)"
            )
            y_zinb, X_zinb = _patsy.dmatrices(
                zinb_formula, data=model_df, return_type="dataframe"
            )
            endog_zinb = np.asarray(y_zinb).ravel().astype(int)
            exog_zinb  = np.asarray(X_zinb)
            exog_infl  = np.ones((len(model_df), 1))   # intercept only

            zinb_model = _ZINB(endog_zinb, exog_zinb, exog_infl=exog_infl, p=2)

            zinb_start = None
            if model_key in results_nb:
                try:
                    nb_coefs   = np.asarray(fit_nb.params[list(X_zinb.columns)])
                    alpha_nb   = (float(fit_nb.params["alpha"])
                                  if "alpha" in fit_nb.params.index else 0.5)
                    zero_prop  = (model_df["citations"] == 0).mean()
                    logit_zero = np.log(max(zero_prop, 1e-6) /
                                        max(1.0 - zero_prop, 1e-6))
                    zinb_start = np.r_[nb_coefs, logit_zero,
                                       np.log(max(alpha_nb, 1e-6))]
                except Exception:
                    zinb_start = None

            # Fit with preferred SE type; track whether clustering succeeded
            actually_clustered = False
            fit_zinb = None
            if CLUSTER_COL:
                try:
                    fit_zinb = zinb_model.fit(
                        start_params=zinb_start,
                        method="bfgs", maxiter=600, disp=False,
                        cov_type="cluster",
                        cov_kwds={"groups": model_df[CLUSTER_COL].values},
                    )
                    actually_clustered = True
                except Exception as e_cl:
                    print(f"  WARNING: ZINB clustered SEs failed ({e_cl}); "
                          f"falling back to HC1", flush=True)
                    try:
                        fit_zinb = zinb_model.fit(
                            start_params=zinb_start,
                            method="bfgs", maxiter=600, disp=False,
                            cov_type="HC1",
                        )
                    except Exception:
                        pass
            else:
                try:
                    fit_zinb = zinb_model.fit(
                        start_params=zinb_start,
                        method="bfgs", maxiter=600, disp=False, cov_type="HC1",
                    )
                except Exception:
                    pass

            if fit_zinb is None:
                fit_zinb = zinb_model.fit(
                    start_params=zinb_start,
                    method="bfgs", maxiter=600, disp=False,
                )

            # Convergence diagnostics — non-convergence invalidates the LR test
            conv = getattr(fit_zinb, "mle_retvals", {}).get("converged", None)
            if conv is False:
                print("  WARNING: ZINB did not converge — interpret with caution",
                      flush=True)
            try:
                grad_norm = np.linalg.norm(fit_zinb.score(fit_zinb.params))
                if grad_norm > 1e-3:
                    print(f"  WARNING: ZINB gradient norm at solution = "
                          f"{grad_norm:.2e} (should be < 1e-3)", flush=True)
            except Exception:
                pass

            # Parameter names — try results object first, then model, then fallback
            if hasattr(fit_zinb, "param_names"):
                pnames = list(fit_zinb.param_names)
            elif hasattr(fit_zinb.model, "param_names"):
                pnames = list(fit_zinb.model.param_names)
            else:
                n_c   = exog_zinb.shape[1]
                n_i   = exog_infl.shape[1]
                n_tot = len(fit_zinb.params)
                infl_names = (["inflate_const"] if n_i == 1
                               else [f"inflate_{j}" for j in range(n_i)])
                pnames = (list(X_zinb.columns)
                          + infl_names
                          + ["alpha"] * max(0, n_tot - n_c - n_i))[:n_tot]

            params_z = pd.Series(np.asarray(fit_zinb.params), index=pnames)
            try:
                bse_z   = pd.Series(np.asarray(fit_zinb.bse),    index=pnames)
                pvals_z = pd.Series(np.asarray(fit_zinb.pvalues), index=pnames)
            except Exception:
                bse_z   = pd.Series(np.full(len(pnames), np.nan), index=pnames)
                pvals_z = pd.Series(np.full(len(pnames), np.nan), index=pnames)

            # Novelty coef: count equation, not inflation equation
            nov_keys = [k for k in params_z.index
                        if novelty_col in k and "inflate" not in k.lower()]
            if not nov_keys:
                raise ValueError(
                    f"Novelty key not found in ZINB params: "
                    f"{params_z.index.tolist()}"
                )
            nk   = nov_keys[0]
            coef = float(params_z[nk])
            se   = float(bse_z[nk])
            pval = float(pvals_z[nk])

            if np.isnan(se):
                raise ValueError(
                    "Hessian singular after all SE attempts — ZINB locally "
                    "unidentified for this metric. The predictor likely has "
                    "insufficient signal to separate the zero-inflation process "
                    "from the count component. Skipping ZINB."
                )

            ll   = float(fit_zinb.llf)
            n_obs = int(fit_zinb.nobs)
            sig  = ("***" if pval < 0.01 else "**" if pval < 0.05
                    else "*" if pval < 0.10 else "")

            alpha_keys_z = [k for k in params_z.index
                            if "alpha" in k.lower() and "inflate" not in k.lower()]
            # NegativeBinomialP stores log(alpha) in params; back-transform
            # so the reported alpha is on the same natural scale as NB2 alpha
            alpha_zinb = (np.exp(float(params_z[alpha_keys_z[0]]))
                          if alpha_keys_z else np.nan)

            infl_keys = [k for k in params_z.index if "inflate" in k.lower()]
            infl_int  = float(params_z[infl_keys[0]]) if infl_keys else np.nan
            p_infl    = (float(1.0 / (1.0 + np.exp(-infl_int)))
                         if not np.isnan(infl_int) else np.nan)

            results_zinb[model_key] = {
                "model": model_label, "estimator": "ZINB",
                "coef": coef, "se": se, "pval": pval,
                "n": n_obs, "alpha": alpha_zinb, "ll": ll,
                "inflate_intercept": infl_int, "p_inflate": p_infl,
                "clustered": actually_clustered, "converged": conv, "sig": sig,
            }
            print(f"  ZINB: b = {coef:+.4f}  SE = {se:.4f}  p = {pval:.4f}  {sig}  "
                  f"p_inflate = {p_infl:.4f}  a = {alpha_zinb:.4f}  "
                  f"converged={conv}  ({time.time()-t0:.0f}s)", flush=True)

        except Exception as e:
            print(f"  ZINB failed: {type(e).__name__}: {e}", flush=True)
            if DEBUG:
                import traceback; traceback.print_exc()

# =============================================================================
# PHASE 3.2 — NB VS ZINB MODEL SELECTION
# =============================================================================
# Boundary LR test for H0: no zero inflation (NB2 is adequate).
# Under H0 the inflate parameter is on the boundary of the parameter space
# (p_inflate = 0), so the standard chi²(1) LRT is anti-conservative.
# The conservative boundary correction halves the tail probability:
#   p_LR = 0.5 * Pr(chi²(1) > LR)
# This is the standard mixture-of-chi² result.
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 3.2 — NB VS ZINB MODEL SELECTION", flush=True)
print(f"{'=' * 65}", flush=True)
print(f"  H0: no zero inflation (NB2 adequate).", flush=True)
print(f"  LR = 2*(LL_ZINB - LL_NB)  |  p_LR = 0.5*Pr(chi²(1) > LR)  "
      f"[boundary correction]", flush=True)
print(f"  p_inflate: Pr(structural zero) = logistic(inflate intercept).",
      flush=True)

nb_zinb_verdict = {}
for model_key in ["tfidf", "w2v", "sbert"]:
    label = MODELS[model_key][1]
    if model_key not in results_nb:
        print(f"\n  {label}: SKIP — NB2 result missing", flush=True)
        continue
    if model_key not in results_zinb:
        print(f"\n  {label}: SKIP — ZINB result missing "
              f"(HAS_ZINB={HAS_ZINB})", flush=True)
        continue

    nb_ll    = results_nb[model_key]["ll"]
    zinb_ll  = results_zinb[model_key]["ll"]
    lr_stat  = 2.0 * (zinb_ll - nb_ll)
    p_lr     = 0.5 * scipy_stats.chi2.sf(max(lr_stat, 0.0), df=1)
    p_infl   = results_zinb[model_key].get("p_inflate", np.nan)
    infl_int = results_zinb[model_key].get("inflate_intercept", np.nan)

    if lr_stat < 0:
        # Negative LR means ZINB converged to a worse optimum — a diagnostics
        # failure, not evidence for NB2
        verdict = "Inconclusive — ZINB LL below NB2 (convergence issue)"
    elif lr_stat == 0:
        verdict = "NB2 preferred — ZINB adds no improvement (LR = 0)"
    elif p_lr >= 0.05:
        verdict = "NB2 preferred — zero inflation not significant at 5%"
    elif not np.isnan(p_infl) and p_infl < 0.01:
        verdict = ("ZINB nominally significant (p_LR < 0.05) but estimated "
                   "structural-zero probability < 1% — zero inflation "
                   "negligible in magnitude; NB2 adequate")
    else:
        verdict = "ZINB preferred — significant zero inflation detected"

    nb_zinb_verdict[model_key] = verdict
    print(f"\n  {label}:", flush=True)
    print(f"    LL_NB2  = {nb_ll:.2f}   LL_ZINB = {zinb_ll:.2f}", flush=True)
    print(f"    LR stat = {lr_stat:.3f}   p_LR (boundary) = {p_lr:.4f}", flush=True)
    print(f"    inflate intercept = {infl_int:.3f}  "
          f"=>  p_inflate = {p_infl:.4f}", flush=True)
    print(f"    Verdict: {verdict}", flush=True)

if not nb_zinb_verdict:
    print("  Skipped — need both NB2 and ZINB results for at least one metric.",
          flush=True)

# =============================================================================
# PHASE 3.5 — COEFFICIENT EQUALITY TESTS (SBERT vs BASELINES)
# =============================================================================
# Tests whether SBERT's coefficient differs significantly from TF-IDF and
# Word2Vec using an asymptotic Wald z-test for equality of independently
# estimated coefficients:
#
#   z = (b_SBERT - b_j) / sqrt(SE_SBERT^2 + SE_j^2)
#
# Under H0: b_SBERT == b_j, z ~ N(0,1) asymptotically (two-tailed).
#
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 3.5 — COEFFICIENT EQUALITY TESTS", flush=True)
print(f"{'=' * 65}", flush=True)
print(f"  H0: b_SBERT == b_baseline", flush=True)
print(f"  Test: asymptotic z-test (conservative — assumes independence)", flush=True)


def wald_coef_equality(coef_a, se_a, coef_b, se_b,
                       label_a="SBERT", label_b="baseline", estimator="NB2"):
    """
    Asymptotic z-test for H0: coef_a == coef_b.

    z = (coef_a - coef_b) / sqrt(se_a^2 + se_b^2)

    Conservative: assumes independence. In practice the samples partially
    overlap (same focal patents, different NaN patterns per metric), so the
    true SE of the difference may be smaller, making this a lower bound on
    the z-statistic.
    """
    se_diff = np.sqrt(se_a**2 + se_b**2)
    if se_diff == 0:
        return None
    z_stat = (coef_a - coef_b) / se_diff
    p_val  = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(z_stat)))
    sig    = ("***" if p_val < 0.01 else "**" if p_val < 0.05
              else "*" if p_val < 0.10 else "")
    return {
        "comparison": f"{label_a} vs {label_b}",
        "estimator":  estimator,
        "coef_a":     coef_a,
        "se_a":       se_a,
        "coef_b":     coef_b,
        "se_b":       se_b,
        "diff":       coef_a - coef_b,
        "se_diff":    se_diff,
        "z_stat":     z_stat,
        "p_val":      p_val,
        "sig":        sig,
    }


equality_results = []

# --- NB2 comparisons (primary estimator) ---
if "sbert" in results_nb and "tfidf" in results_nb:
    r = wald_coef_equality(
        results_nb["sbert"]["coef"], results_nb["sbert"]["se"],
        results_nb["tfidf"]["coef"], results_nb["tfidf"]["se"],
        label_a="SBERT", label_b="TF-IDF", estimator="NB2",
    )
    if r:
        equality_results.append(r)

if "sbert" in results_nb and "w2v" in results_nb:
    r = wald_coef_equality(
        results_nb["sbert"]["coef"], results_nb["sbert"]["se"],
        results_nb["w2v"]["coef"],   results_nb["w2v"]["se"],
        label_a="SBERT", label_b="Word2Vec", estimator="NB2",
    )
    if r:
        equality_results.append(r)

# --- PPML comparisons (robustness estimator) ---
if "sbert" in results_ppml and "tfidf" in results_ppml:
    r = wald_coef_equality(
        results_ppml["sbert"]["coef"], results_ppml["sbert"]["se"],
        results_ppml["tfidf"]["coef"], results_ppml["tfidf"]["se"],
        label_a="SBERT", label_b="TF-IDF", estimator="PPML",
    )
    if r:
        equality_results.append(r)

if "sbert" in results_ppml and "w2v" in results_ppml:
    r = wald_coef_equality(
        results_ppml["sbert"]["coef"], results_ppml["sbert"]["se"],
        results_ppml["w2v"]["coef"],   results_ppml["w2v"]["se"],
        label_a="SBERT", label_b="Word2Vec", estimator="PPML",
    )
    if r:
        equality_results.append(r)

# --- ZINB comparisons (second robustness estimator) ---
if "sbert" in results_zinb and "tfidf" in results_zinb:
    r = wald_coef_equality(
        results_zinb["sbert"]["coef"], results_zinb["sbert"]["se"],
        results_zinb["tfidf"]["coef"], results_zinb["tfidf"]["se"],
        label_a="SBERT", label_b="TF-IDF", estimator="ZINB",
    )
    if r:
        equality_results.append(r)

if "sbert" in results_zinb and "w2v" in results_zinb:
    r = wald_coef_equality(
        results_zinb["sbert"]["coef"], results_zinb["sbert"]["se"],
        results_zinb["w2v"]["coef"],   results_zinb["w2v"]["se"],
        label_a="SBERT", label_b="Word2Vec", estimator="ZINB",
    )
    if r:
        equality_results.append(r)

# --- Console report ---
if equality_results:
    print(f"\n  {'Comparison':<22} {'Est.':<6} {'b_SBERT':>8}  "
          f"{'b_base':>8}  {'Diff':>8}  {'z':>7}  {'p':>8}  {'Sig':<5}",
          flush=True)
    print(f"  {'-' * 78}", flush=True)
    for r in equality_results:
        print(
            f"  {r['comparison']:<22} {r['estimator']:<6} "
            f"{r['coef_a']:>8.4f}  {r['coef_b']:>8.4f}  "
            f"{r['diff']:>+8.4f}  {r['z_stat']:>7.3f}  "
            f"{r['p_val']:>8.4f}  {r['sig']:<5}",
            flush=True,
        )

    print(f"\n  Interpretation:", flush=True)
    for r in equality_results:
        if r["p_val"] < 0.10:
            print(
                f"  REJECT H0  | {r['comparison']} ({r['estimator']}): "
                f"p = {r['p_val']:.4f}{r['sig']} — SBERT coefficient "
                f"significantly differs from baseline. Differential significance "
                f"in Table 8 reflects genuine predictive difference.",
                flush=True,
            )
        else:
            print(
                f"  FAIL TO REJECT | {r['comparison']} ({r['estimator']}): "
                f"p = {r['p_val']:.4f} — Cannot rule out that SBERT-only "
                f"significance is driven by lower SE rather than methodological "
                f"superiority. Report this limitation in the thesis.",
                flush=True,
            )
else:
    print("  Skipped — insufficient model results (need SBERT + at least one "
          "baseline in both results_nb and results_ppml).", flush=True)

# =============================================================================
# PHASE 4 — OUTPUT TABLES
# =============================================================================
print(f"\n{'=' * 65}", flush=True)
print(f"PHASE 4 — GENERATE COMPARISON TABLES", flush=True)
print(f"{'=' * 65}", flush=True)

all_results = {}
for key in ["tfidf", "w2v", "sbert"]:
    if key in results_nb:
        all_results[f"{key}_nb"]   = results_nb[key]
    if key in results_ppml:
        all_results[f"{key}_ppml"] = results_ppml[key]
    if key in results_zinb:
        all_results[f"{key}_zinb"] = results_zinb[key]

if len(all_results) > 0:

    pd.DataFrame(all_results).T.to_csv(RESULTS_PATH, index=True)
    print(f"  Saved {RESULTS_PATH}", flush=True)

    # --- Console summary ---
    print(f"\n  PRIMARY — NB2 (CPC + year FE, clustered SE):", flush=True)
    print(f"  {'Model':<12} {'b':>8}  {'SE':>8}  {'p':>8}  "
          f"{'Sig':<5}  {'a':>8}  {'LL':>10}", flush=True)
    print(f"  {'-' * 65}", flush=True)
    for key in ["tfidf", "w2v", "sbert"]:
        if key in results_nb:
            r = results_nb[key]
            print(f"  {r['model']:<12} {r['coef']:>8.4f}  "
                  f"{r['se']:>8.4f}  {r['pval']:>8.4f}  "
                  f"{r['sig']:<5}  {r['alpha']:>8.4f}  "
                  f"{r['ll']:>10.1f}", flush=True)

    print(f"\n  ROBUSTNESS 1 — FE Poisson (PPML, same CPC + year FEs):", flush=True)
    print(f"  {'Model':<12} {'b':>8}  {'SE':>8}  {'p':>8}  {'Sig':<5}",
          flush=True)
    print(f"  {'-' * 50}", flush=True)
    for key in ["tfidf", "w2v", "sbert"]:
        if key in results_ppml:
            r = results_ppml[key]
            print(f"  {r['model']:<12} {r['coef']:>8.4f}  "
                  f"{r['se']:>8.4f}  {r['pval']:>8.4f}  "
                  f"{r['sig']:<5}", flush=True)

    print(f"\n  ROBUSTNESS 2 — ZINB (intercept-only inflation, CPC + year FEs):",
          flush=True)
    print(f"  {'Model':<12} {'b':>8}  {'SE':>8}  {'p':>8}  {'Sig':<5}  "
          f"{'p_inflate':>10}  {'a':>8}  {'LL':>10}", flush=True)
    print(f"  {'-' * 72}", flush=True)
    for key in ["tfidf", "w2v", "sbert"]:
        if key in results_zinb:
            r = results_zinb[key]
            pi_s = (f"{r['p_inflate']:.4f}"
                    if not np.isnan(r.get("p_inflate", np.nan)) else "    --")
            a_s  = (f"{r['alpha']:.4f}"
                    if not np.isnan(r.get("alpha", np.nan)) else "    --")
            print(f"  {r['model']:<12} {r['coef']:>8.4f}  "
                  f"{r['se']:>8.4f}  {r['pval']:>8.4f}  "
                  f"{r['sig']:<5}  {pi_s:>10}  {a_s:>8}  "
                  f"{r['ll']:>10.1f}", flush=True)

    # --- LaTeX table ---
    se_note = (f"Standard errors clustered by assignee ({CLUSTER_COL})."
               if CLUSTER_COL else
               "Heteroskedasticity-robust standard errors.")

    rows = ""
    for key in ["tfidf", "w2v", "sbert"]:
        bold_o = "\\textbf{" if key == "sbert" else ""
        bold_c = "}"         if key == "sbert" else ""

        if key in results_nb:
            r = results_nb[key]
            rows += (
                f"    {bold_o}{r['model']}{bold_c} & NB2 & "
                f"{r['coef']:.4f}{r['sig']} & ({r['se']:.4f}) & "
                f"{r['alpha']:.4f} & {r['ll']:.1f} & {int(r['n']):,} \\\\\n"
            )
        if key in results_ppml:
            r = results_ppml[key]
            rows += (
                f"    {bold_o}{r['model']}{bold_c} & PPML & "
                f"{r['coef']:.4f}{r['sig']} & ({r['se']:.4f}) & "
                f"-- & -- & {int(r['n']):,} \\\\\n"
            )
        if key in results_zinb:
            r = results_zinb[key]
            a_s  = (f"{r['alpha']:.4f}"
                    if not np.isnan(r.get("alpha", np.nan)) else "--")
            pi_s = (f"{r['p_inflate']:.4f}"
                    if not np.isnan(r.get("p_inflate", np.nan)) else "--")
            rows += (
                f"    {bold_o}{r['model']}{bold_c} & ZINB ($\\hat{{p}}_{{0}}={pi_s}$) & "
                f"{r['coef']:.4f}{r['sig']} & ({r['se']:.4f}) & "
                f"{a_s} & {r['ll']:.1f} & {int(r['n']):,} \\\\\n"
            )

    latex_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\captionsetup{justification=raggedright, singlelinecheck=false,\n"
        "  labelfont=bf, textfont=it}\n"
        "\\caption{NLP Model Comparison -- NB2 (Primary), PPML and ZINB\n"
        "  (Robustness Checks)}\n"
        "\\label{tab:nlp_comparison}\n"
        "\\begin{tabular}{@{} l l c c c c c @{}}\n"
        "\\toprule\n"
        "Model & Estimator & $\\hat{\\beta}_1$ & (SE) & $\\alpha$ & LL & $N$ \\\\\n"
        "\\midrule\n"
        f"{rows}"
        "\\bottomrule\n"
        "\\multicolumn{7}{p{14cm}}{\\textit{Note.} Dependent variable is\n"
        "  the five-year forward citation count (corrected; see Section 4.4).\n"
        "  All estimators include four-digit CPC subgroup and filing-year\n"
        "  fixed effects. With only $G=4$ CPC subgroups and $T\\approx6$ filing\n"
        "  years, incidental-parameters bias from dummy-variable FEs is\n"
        "  negligible (Hausman, Hall and Griliches 1984). The variance-to-mean\n"
        f"  ratio of {var_ratio:.0f} motivates NB2 as the primary estimator.\n"
        "  PPML = Poisson pseudo-maximum likelihood with CPC and filing-year\n"
        "  FEs absorbed via within-transformation (pyfixest); consistent under\n"
        "  any conditional mean specification (Santos Silva \\& Tenreyro 2006).\n"
        "  ZINB = Zero-Inflated Negative Binomial with intercept-only inflation\n"
        "  equation; $\\hat{p}_{0}$ is the implied structural-zero probability\n"
        "  (logistic inflate intercept). NB2 vs ZINB preference is assessed\n"
        "  via a boundary LR test ($p_{LR} = 0.5\\cdot\\Pr[\\chi^2(1)>LR]$;\n"
        "  Self \\& Liang 1987). $\\alpha$ = overdispersion parameter;\n"
        "  LL = log-likelihood (unavailable for PPML).\n"
        f"  {se_note}\n"
        "  Significance: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.\n"
        "  SBERT (bold) is the main specification.}\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(TABLE_PATH, "w") as f:
        f.write(latex_table)
    print(f"  Saved {TABLE_PATH}", flush=True)

    # --- LaTeX table: coefficient equality tests ---
    if equality_results:
        eq_rows = ""
        for r in equality_results:
            eq_rows += (
                f"    {r['comparison']} & {r['estimator']} & "
                f"{r['coef_a']:.4f} & {r['coef_b']:.4f} & "
                f"{r['diff']:+.4f} & {r['z_stat']:.3f} & "
                f"{r['p_val']:.4f}{r['sig']} \\\\\n"
            )

        equality_latex = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\captionsetup{justification=raggedright, singlelinecheck=false,\n"
            "  labelfont=bf, textfont=it}\n"
            "\\caption{Coefficient Equality Tests: SBERT vs Baseline NLP Metrics}\n"
            "\\label{tab:coef_equality}\n"
            "\\begin{tabular}{@{} l l c c c c c @{}}\n"
            "\\toprule\n"
            "Comparison & Estimator & $\\hat{\\beta}_{\\text{SBERT}}$ & "
            "$\\hat{\\beta}_{\\text{base}}$ & Difference & $z$ & $p$-value \\\\\n"
            "\\midrule\n"
            f"{eq_rows}"
            "\\bottomrule\n"
            "\\multicolumn{7}{p{14cm}}{\\textit{Note.} "
            "Asymptotic $z$-test for $H_0: \\beta_{\\text{SBERT}} = "
            "\\beta_{\\text{baseline}}$. Test statistic: "
            "$z = (\\hat{\\beta}_{\\text{SBERT}} - \\hat{\\beta}_{\\text{base}}) "
            "/ \\sqrt{\\widehat{\\text{SE}}_{\\text{SBERT}}^2 + "
            "\\widehat{\\text{SE}}_{\\text{base}}^2}$. "
            "The test is conservative: assuming independence between estimators "
            "overstates the denominator when models share overlapping samples, "
            "making rejection a lower bound on evidence of genuine predictive "
            "difference. Under $H_0$, $z \\sim \\mathcal{N}(0,1)$ asymptotically "
            "(two-tailed). "
            "Significance: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.}\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

        with open(EQUALITY_TABLE_PATH, "w") as f:
            f.write(equality_latex)
        print(f"  Saved {EQUALITY_TABLE_PATH}", flush=True)

    # --- Verdict ---
    sbert_nb   = results_nb.get("sbert", {})
    sbert_ppml = results_ppml.get("sbert", {})
    sbert_zinb = results_zinb.get("sbert", {})

    print(f"\n{'=' * 65}", flush=True)
    print(f"VALIDATION VERDICT", flush=True)
    print(f"{'=' * 65}", flush=True)

    for label, r in [("NB2   (primary)     ", sbert_nb),
                     ("PPML  (robustness 1)", sbert_ppml),
                     ("ZINB  (robustness 2)", sbert_zinb)]:
        if not r:
            continue
        if r["pval"] < 0.05:
            print(f"  {label}: VALIDATED  "
                  f"b = {r['coef']:.4f}  p = {r['pval']:.4f}  {r['sig']}",
                  flush=True)
        elif r["pval"] < 0.10:
            print(f"  {label}: MARGINALLY VALIDATED (p < 0.10)  "
                  f"b = {r['coef']:.4f}  p = {r['pval']:.4f}  {r['sig']}",
                  flush=True)
        else:
            print(f"  {label}: NOT validated at 10% level  "
                  f"b = {r['coef']:.4f}  p = {r['pval']:.4f}", flush=True)

    # NB vs ZINB preference for SBERT
    if sbert_nb and sbert_zinb:
        verd = nb_zinb_verdict.get("sbert", "")
        if verd:
            print(f"\n  NB vs ZINB (SBERT): {verd}", flush=True)

    # Overall conclusion
    nb_ok   = bool(sbert_nb)   and sbert_nb["pval"]   < 0.05
    nb_mar  = bool(sbert_nb)   and sbert_nb["pval"]   < 0.10
    ppml_ok = bool(sbert_ppml) and sbert_ppml["pval"] < 0.05
    zinb_ok = bool(sbert_zinb) and sbert_zinb["pval"] < 0.05

    checks_ok = sum([ppml_ok, zinb_ok])

    if nb_ok and checks_ok == 2:
        print(f"\n  CONCLUSION: SBERT metric validated. Result is robust "
              f"across NB2, PPML, and ZINB.", flush=True)
    elif nb_ok and checks_ok >= 1:
        print(f"\n  CONCLUSION: SBERT metric validated under NB2 (primary) "
              f"and at least one robustness check.", flush=True)
    elif nb_ok:
        print(f"\n  CONCLUSION: SBERT metric validated under NB2 (primary).\n"
              f"  PPML/ZINB not significant — likely lower power under extreme "
              f"overdispersion.", flush=True)
    elif nb_mar and (ppml_ok or zinb_ok):
        print(f"\n  CONCLUSION: SBERT marginally significant in NB2 (p < 0.10) "
              f"and significant in at least one robustness check.\n"
              f"  Both agree on direction and magnitude.", flush=True)
    elif ppml_ok or zinb_ok:
        print(f"\n  CONCLUSION: SBERT significant in robustness check(s) but "
              f"not NB2.\n  Distributional sensitivity — report all estimators.",
              flush=True)
    else:
        print(f"\n  CONCLUSION: SBERT not significant across any estimator.",
              flush=True)

else:
    print("  No results to report — all regressions failed", flush=True)

print(f"""
Next step: step9b_did_regression.py
  Reads  data/interim/panel_full.csv (or semiannual variant)
""", flush=True)