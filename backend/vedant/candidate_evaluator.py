"""
candidate_evaluator.py
----------------------
Computes the 70/30 Weighted Fit Score, Vector Similarity, and Composite
Score for a candidate row, then assigns a grade and pathway depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CATEGORY_BENCHMARKS,
    DEFAULT_BENCHMARK,
    GRADE_THRESHOLDS,
    HARD_SKILL_WEIGHT,
    SOFT_SKILL_WEIGHT,
    FUNDAMENTAL_GRADES,
    WEEK_COUNT_FUNDAMENTAL,
    WEEK_COUNT_ADVANCED,
)
from skills_taxonomy import SKILLS_TAXONOMY


# ── Soft-skill set (from taxonomy) ───────────────────────────────────────────
_SOFT_SKILLS_SET = {s.lower() for s in SKILLS_TAXONOMY.get("soft_skills", [])}


# ── Grade assignment ─────────────────────────────────────────────────────────

def determine_grade(score: float) -> str:
    """Map a composite score [0-1] to a letter grade."""
    thresholds = sorted(GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True)
    for grade, threshold in thresholds:
        if score > threshold:
            return grade
    return "F"


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_candidate(
    candidate_row: pd.Series,
    vectorizer: TfidfVectorizer,
    category: str | None = None,
    resume_col: str = "Resume_str",
) -> dict:
    """
    Evaluate a single candidate and return a metrics dict.

    Parameters
    ----------
    candidate_row : a row from resume_df (must contain at least resume_col)
    vectorizer    : fitted TF-IDF vectorizer (shared across the engine)
    category      : override the row's Category field if supplied
    resume_col    : column name holding the raw resume text

    Returns
    -------
    dict with keys:
        ID, Category, Weighted_Fit, Vector_Similarity, Composite_Score,
        Grade, Pathway_Depth, Duration, Gaps, Extracted_Skills
    """
    cand_id       = candidate_row.get("ID", 0)
    cand_category = (category or candidate_row.get("Category", "UNKNOWN")).upper()
    resume_text   = candidate_row.get(resume_col, "")

    # ── Benchmark skills for this category ──────────────────────────────────
    benchmark_skills = CATEGORY_BENCHMARKS.get(cand_category, DEFAULT_BENCHMARK)

    required_skills  = {s.lower() for s in benchmark_skills}
    extracted_skills = {s.lower() for s in (candidate_row.get("Extracted_Skills") or [])}

    # ── 70/30 Weighted Fit Score ─────────────────────────────────────────────
    req_soft = required_skills & _SOFT_SKILLS_SET
    req_hard = required_skills - _SOFT_SKILLS_SET

    soft_match = (
        len(extracted_skills & req_soft) / len(req_soft)
        if req_soft else 1.0
    )
    hard_match = (
        len(extracted_skills & req_hard) / len(req_hard)
        if req_hard else 1.0
    )
    weighted_fit_score = (hard_match * HARD_SKILL_WEIGHT) + (soft_match * SOFT_SKILL_WEIGHT)

    # ── Vector Similarity Score ──────────────────────────────────────────────
    bench_text     = " ".join(benchmark_skills)
    cand_vec       = vectorizer.transform([resume_text])
    bench_vec      = vectorizer.transform([bench_text])
    vector_sim     = float(cosine_similarity(cand_vec, bench_vec)[0][0])

    # ── Composite Score, Grade, Pathway ─────────────────────────────────────
    composite_score = (weighted_fit_score + vector_sim) / 2
    grade           = determine_grade(composite_score)
    pathway_depth   = (
        "Fundamental & Comprehensive"
        if grade in FUNDAMENTAL_GRADES
        else "Advanced & Fast-Track"
    )
    week_count = (
        WEEK_COUNT_FUNDAMENTAL
        if grade in FUNDAMENTAL_GRADES
        else WEEK_COUNT_ADVANCED
    )

    # ── Skill Gaps ───────────────────────────────────────────────────────────
    skill_gaps = sorted(required_skills - extracted_skills)

    return {
        "ID":               cand_id,
        "Category":         cand_category,
        "Weighted_Fit":     round(weighted_fit_score, 4),
        "Vector_Similarity":round(vector_sim, 4),
        "Composite_Score":  round(composite_score, 4),
        "Grade":            grade,
        "Pathway_Depth":    pathway_depth,
        "Duration":         week_count,
        "Gaps":             skill_gaps,
        "Extracted_Skills": list(extracted_skills),
        "_cand_vec":        cand_vec,  # kept for gap prioritisation
    }


def batch_evaluate(
    resume_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    resume_col: str = "Resume_str",
) -> list[dict]:
    """Evaluate every row in resume_df and return a list of metric dicts."""
    results = []
    for _, row in resume_df.iterrows():
        try:
            metrics = evaluate_candidate(row, vectorizer, resume_col=resume_col)
            results.append(metrics)
        except Exception as exc:
            results.append({"ID": row.get("ID"), "error": str(exc)})
    return results
