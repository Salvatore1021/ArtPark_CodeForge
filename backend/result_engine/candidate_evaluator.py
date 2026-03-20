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

# ── NEW: Import and Flatten default_JD weights ───────────────────────────────
from default_JD import ENHANCED_TAXONOMY_DATA

DEFAULT_SKILL_WEIGHTS = {}
for domain, roles in ENHANCED_TAXONOMY_DATA.items():
    if isinstance(roles, dict):
        for role, data in roles.items():
            if isinstance(data, dict) and "WEIGHTS" in data:
                for skill, weight in data["WEIGHTS"].items():
                    skill_lower = skill.lower()
                    # Keep the highest weight if a skill appears in multiple roles
                    if skill_lower not in DEFAULT_SKILL_WEIGHTS:
                        DEFAULT_SKILL_WEIGHTS[skill_lower] = weight
                    else:
                        DEFAULT_SKILL_WEIGHTS[skill_lower] = max(DEFAULT_SKILL_WEIGHTS[skill_lower], weight)

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
    jd_skills: set[str] | list[str] | None = None,
) -> dict:
    """
    Evaluate a single candidate and return a metrics dict.
    Dynamically adjusts skill weights based on JD presence and default_JD values.
    """
    cand_id       = candidate_row.get("ID", 0)
    cand_category = (category or candidate_row.get("Category", "UNKNOWN")).upper()
    resume_text   = candidate_row.get(resume_col, "")

    # ── 1. Determine Required Skills ──
    # If JD is uploaded, use its skills. Otherwise, fall back to Category benchmarks.
    is_jd_uploaded = False
    if jd_skills:
        required_skills = {s.lower() for s in jd_skills}
        is_jd_uploaded = True
    else:
        benchmark_skills = CATEGORY_BENCHMARKS.get(cand_category, DEFAULT_BENCHMARK)
        required_skills = {s.lower() for s in benchmark_skills}

    extracted_skills = {s.lower() for s in (candidate_row.get("Extracted_Skills") or [])}

    # ── 2. Assign Dynamic Weights (Resets per CV) ──
    skill_weights = {}
    for skill in required_skills:
        initial_weight = DEFAULT_SKILL_WEIGHTS.get(skill, 1.0) # Default to 1.0 if not in taxonomy
        
        # If JD is uploaded and skill is found in both JD and default_JD, boost it
        if is_jd_uploaded and skill in DEFAULT_SKILL_WEIGHTS:
            skill_weights[skill] = max(4.0, initial_weight * 1.5)
        else:
            skill_weights[skill] = initial_weight

    # ── 3. 70/30 Weighted Fit Score (Weight-based) ───────────────────────────
    req_soft = required_skills & _SOFT_SKILLS_SET
    req_hard = required_skills - _SOFT_SKILLS_SET

    soft_matched_weight = sum(skill_weights[s] for s in (extracted_skills & req_soft))
    soft_total_weight   = sum(skill_weights[s] for s in req_soft)

    hard_matched_weight = sum(skill_weights[s] for s in (extracted_skills & req_hard))
    hard_total_weight   = sum(skill_weights[s] for s in req_hard)

    soft_match = soft_matched_weight / soft_total_weight if soft_total_weight > 0 else 1.0
    hard_match = hard_matched_weight / hard_total_weight if hard_total_weight > 0 else 1.0

    weighted_fit_score = (hard_match * HARD_SKILL_WEIGHT) + (soft_match * SOFT_SKILL_WEIGHT)

    # ── 4. Vector Similarity Score ───────────────────────────────────────────
    bench_text     = " ".join(required_skills)
    cand_vec       = vectorizer.transform([resume_text])
    bench_vec      = vectorizer.transform([bench_text])
    vector_sim     = float(cosine_similarity(cand_vec, bench_vec))

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
        "_cand_vec":        cand_vec, 
    }

def batch_evaluate(
    resume_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    resume_col: str = "Resume_str",
    jd_skills: set[str] | list[str] | None = None,
) -> list[dict]:
    """Evaluate every row in resume_df and return a list of metric dicts."""
    results = []
    for _, row in resume_df.iterrows():
        try:
            # Allows individual rows to have their own JD skills, or uses the batch parameter
            row_jd_skills = row.get("JD_Skills", jd_skills)
            metrics = evaluate_candidate(row, vectorizer, resume_col=resume_col, jd_skills=row_jd_skills)
            results.append(metrics)
        except Exception as exc:
            results.append({"ID": row.get("ID"), "error": str(exc)})
    return results
