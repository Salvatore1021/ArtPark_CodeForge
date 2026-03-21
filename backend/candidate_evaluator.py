"""
candidate_evaluator.py
-----------------------
Scores candidates using O*NET-weighted benchmarks with category-aware
hard/soft skill split ratios.

Scoring model
─────────────
1. Split-Weighted Fit
   Skills in the benchmark are classified as HARD or SOFT.
   Each group gets a sub-fit score (confidence-weighted average).
   The two sub-scores are blended by the category ratio:

     TECH:       95% hard × hard_fit + 05% soft × soft_fit
     NON_TECH:   70% hard × hard_fit + 30% soft × soft_fit
     MANAGEMENT: 50% hard × hard_fit + 50% soft × soft_fit
     HEALTHCARE: 65% hard × hard_fit + 35% soft × soft_fit
     EDUCATION:  55% hard × hard_fit + 45% soft × soft_fit
     CREATIVE:   60% hard × hard_fit + 40% soft × soft_fit

2. Weighted Cosine Similarity
   Benchmark text repeated ∝ O*NET weight → TF-IDF cosine vs resume.
   Retained even when LLM JSON is used — it captures semantic context
   beyond exact skill-name matches.

3. Composite = (Split-Weighted Fit + Weighted Cosine) / 2

Confidence source priority
──────────────────────────
  1. LLM JSON  (llm_json param)  — richest signal: per-skill confidence,
     proficiency level, and direct evidence from the resume text.
  2. TF-IDF skill vectors        — fallback when LLM is not available
     (CSV / batch mode).
  3. Binary extracted_set match  — last resort when neither is available.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CATEGORY_TO_JOB_TITLE,
    GRADE_THRESHOLDS,
    FUNDAMENTAL_GRADES,
    WEEK_COUNT_FUNDAMENTAL,
    WEEK_COUNT_ADVANCED,
    BENCHMARK_TOP_N,
    MIN_SKILL_WEIGHT,
)
from taxonomy_adapter import build_weighted_benchmark_text, find_job_title
from skill_extractor import get_weighted_benchmark, build_benchmark_text
from skill_confidence import score_all_benchmark_skills, STRONG_THRESHOLD
from skill_classifier import (
    compute_split_weighted_fit,
    get_category_type,
    get_ratio,
)
from enhanced_taxonomy import ENHANCED_TAXONOMY_DATA


# ── Default skill weights (built once at import) ──────────────────────────────

DEFAULT_SKILL_WEIGHTS: dict[str, float] = {}
for _domain, _roles in ENHANCED_TAXONOMY_DATA.items():
    if isinstance(_roles, dict):
        for _role, _data in _roles.items():
            if isinstance(_data, dict) and "WEIGHTS" in _data:
                for _skill, _weight in _data["WEIGHTS"].items():
                    _k = _skill.lower()
                    DEFAULT_SKILL_WEIGHTS[_k] = max(
                        DEFAULT_SKILL_WEIGHTS.get(_k, 0.0), _weight
                    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def determine_grade(score: float) -> str:
    for grade, threshold in sorted(
        GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
    ):
        if score > threshold:
            return grade
    return "F"


def resolve_job_title(category: str) -> str:
    upper = category.upper().strip()
    if upper in CATEGORY_TO_JOB_TITLE:
        return CATEGORY_TO_JOB_TITLE[upper]
    found = find_job_title(category)
    return found if found else category


def _build_confidence_from_llm(
    llm_json: dict,
    benchmark: dict,
) -> dict:
    """
    Build the confidence_scores dict from LLM extraction output.

    The LLM gives per-skill confidence for skills the candidate demonstrably
    has. Benchmark skills not found in the LLM output receive confidence=0.0,
    meaning the evaluator treats them as gaps.

    Parameters
    ----------
    llm_json  : output of extract_all_skills() —
                must contain a 'skills' key with list of skill dicts
    benchmark : {skill_name: onet_weight} for the target job

    Returns
    -------
    {
      skill_lower: {
        "confidence":  float,   # from LLM (0.0 if not found)
        "onet_weight": float,   # from O*NET benchmark
        "proficiency": str,     # from LLM ("unknown" if not found)
      }
    }
    """
    # Lookup: skill_name_lower → full LLM skill dict
    llm_lookup: dict[str, dict] = {
        s["skill"].lower(): s
        for s in llm_json.get("skills", [])
    }

    result = {}
    for skill, onet_weight in benchmark.items():
        skill_lower = skill.lower()
        llm_skill   = llm_lookup.get(skill_lower)
        result[skill_lower] = {
            "confidence":  float(llm_skill["confidence"]) if llm_skill else 0.0,
            "onet_weight": onet_weight,
            "proficiency": llm_skill["proficiency"]       if llm_skill else "unknown",
        }
    return result


# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate_candidate(
    candidate_row: pd.Series,
    vectorizer: TfidfVectorizer,
    skill_vectors: dict | None = None,
    category: str | None = None,
    resume_col: str = "Resume_str",
    jd_skills: set[str] | list[str] | None = None,
    llm_json: dict | None = None,
) -> dict:
    """
    Evaluate a single candidate against a job benchmark.

    Parameters
    ----------
    candidate_row : pd.Series with Resume_str, Extracted_Skills, Category, ID
    vectorizer    : fitted TF-IDF vectorizer
    skill_vectors : pre-computed skill TF-IDF vectors (used as fallback
                    confidence source when llm_json is not provided)
    category      : override for the Category field in candidate_row
    resume_col    : column name for resume text (default 'Resume_str')
    jd_skills     : skills extracted from the job description PDF;
                    when provided, benchmark weights are boosted for these skills
    llm_json      : output of extract_all_skills() — when provided, LLM
                    confidence scores are used directly instead of TF-IDF
                    vector similarity. This is the primary confidence source
                    for the PDF pipeline.
    """
    cand_id      = candidate_row.get("ID", 0)
    raw_category = category or candidate_row.get("Category", "UNKNOWN")
    job_title    = resolve_job_title(str(raw_category))
    resume_text  = str(candidate_row.get(resume_col, ""))

    # ── Benchmark & JD dynamic weighting ─────────────────────────────────────
    if jd_skills:
        benchmark: dict[str, float] = {}
        for skill in jd_skills:
            s_lower        = skill.lower()
            initial_weight = DEFAULT_SKILL_WEIGHTS.get(s_lower, 1.0)
            if s_lower in DEFAULT_SKILL_WEIGHTS:
                benchmark[skill] = max(4.0, initial_weight * 1.5)
            else:
                benchmark[skill] = initial_weight
    else:
        benchmark = get_weighted_benchmark(
            job_title, top_n=BENCHMARK_TOP_N, min_weight=MIN_SKILL_WEIGHT
        )
        if not benchmark:
            benchmark = get_weighted_benchmark(
                raw_category, top_n=BENCHMARK_TOP_N, min_weight=MIN_SKILL_WEIGHT
            )

    extracted_skills: list[str] = list(candidate_row.get("Extracted_Skills") or [])
    extracted_set = {s.lower() for s in extracted_skills}
    cand_vec      = vectorizer.transform([resume_text])

    # ── Confidence scores — LLM JSON first, TF-IDF fallback ──────────────────
    if llm_json:
        # Primary path: PDF pipeline with LLM extraction
        confidence_scores = _build_confidence_from_llm(llm_json, benchmark)
    elif skill_vectors:
        # Fallback: CSV / batch mode without LLM
        confidence_scores = score_all_benchmark_skills(
            benchmark, resume_text, extracted_skills, cand_vec, skill_vectors
        )
    else:
        confidence_scores = {}

    # ── Split-weighted fit (hard/soft ratio) ─────────────────────────────────
    if confidence_scores:
        split_result = compute_split_weighted_fit(
            benchmark, confidence_scores, raw_category, job_title
        )
        weighted_fit = split_result["split_fit"]
        hard_fit     = split_result["hard_fit"]
        soft_fit     = split_result["soft_fit"]
        ratio        = get_ratio(raw_category, job_title)
        cat_type     = split_result["category_type"]
        ratio_label  = split_result["ratio_label"]
    else:
        # Last resort: plain O*NET-weighted binary fit
        total_w      = sum(benchmark.values())
        matched_w    = sum(w for s, w in benchmark.items() if s.lower() in extracted_set)
        weighted_fit = matched_w / total_w if total_w > 0 else 0.0
        hard_fit     = weighted_fit
        soft_fit     = weighted_fit
        split_result = {}
        cat_type     = get_category_type(raw_category, job_title)
        ratio        = get_ratio(raw_category, job_title)
        ratio_label  = ratio.label

    # ── Weighted cosine similarity ────────────────────────────────────────────
    if jd_skills:
        bench_words: list[str] = []
        for s, w in benchmark.items():
            bench_words.extend([s] * max(1, int(round(w))))
        bench_text = " ".join(bench_words)
    else:
        bench_text = build_benchmark_text(job_title, top_n=BENCHMARK_TOP_N)

    # ── Composite, grade, pathway ─────────────────────────────────────────────
    composite     = weighted_fit
    grade         = determine_grade(composite)
    pathway_depth = (
        "Fundamental & Comprehensive"
        if grade in FUNDAMENTAL_GRADES
        else "Advanced & Fast-Track"
    )
    week_count = (
        WEEK_COUNT_FUNDAMENTAL if grade in FUNDAMENTAL_GRADES else WEEK_COUNT_ADVANCED
    )

    # ── Gap detection ─────────────────────────────────────────────────────────
    if confidence_scores:
        gap_weights = {
            skill: weight
            for skill, weight in benchmark.items()
            if confidence_scores.get(skill.lower(), {}).get("confidence", 0.0)
               < STRONG_THRESHOLD
        }
    else:
        gap_weights = {
            skill: weight
            for skill, weight in benchmark.items()
            if skill.lower() not in extracted_set
        }

    return {
        "ID":                  cand_id,
        "Category":            raw_category,
        "Job_Title":           job_title,
        "Category_Type":       cat_type,
        "Ratio_Label":         ratio_label,
        "Hard_Fit":            round(hard_fit, 4),
        "Soft_Fit":            round(soft_fit, 4),
        "Split_Weighted_Fit":  round(weighted_fit, 4),
        "Weighted_Fit":        round(weighted_fit, 4),
        "Weighted_Cosine":     round(wt_cosine, 4),
        "Composite_Score":     round(composite, 4),
        "Confidence_Scores":   confidence_scores,
        "Split_Details":       split_result,
        "Grade":               grade,
        "Pathway_Depth":       pathway_depth,
        "Duration":            week_count,
        "Gaps":                sorted(gap_weights.keys()),
        "Gap_Weights":         gap_weights,
        "Extracted_Skills":    sorted(extracted_set),
    }


def batch_evaluate(
    resume_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    skill_vectors: dict | None = None,
    resume_col: str = "Resume_str",
    jd_skills: set[str] | list[str] | None = None,
) -> list[dict]:
    """
    Evaluate all rows in a DataFrame. CSV/batch mode — no LLM JSON.
    Uses TF-IDF skill vectors for confidence scoring.
    """
    results = []
    for _, row in resume_df.iterrows():
        try:
            row_jd_skills = row.get("JD_Skills") or jd_skills   # fix: avoid NaN ambiguity
            m = evaluate_candidate(
                row, vectorizer, skill_vectors,
                resume_col=resume_col, jd_skills=row_jd_skills,
                llm_json=None,   # explicit: batch mode never uses LLM JSON
            )
            results.append(m)
        except Exception as exc:
            results.append({"ID": row.get("ID"), "error": str(exc)})
    return results