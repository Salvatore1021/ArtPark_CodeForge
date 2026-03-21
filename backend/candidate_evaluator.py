"""
candidate_evaluator.py
-----------------------
Scores candidates using O*NET-weighted benchmarks with category-aware
hard/soft skill split ratios.

Scoring model
-------------
1. Split-Weighted Fit
   Skills in the benchmark are classified as HARD or SOFT.
   Each group gets a sub-fit score (confidence-weighted average).
   The two sub-scores are blended by the category ratio.

2. Weighted Cosine Similarity
   Benchmark text repeated in proportion to O*NET weight is compared
   against the resume with TF-IDF cosine similarity.

3. Composite = 0.75 x Split-Weighted Fit + 0.25 x Weighted Cosine
   This keeps the final percentage much closer to actual skill coverage,
   while still using semantic document similarity as a secondary signal.

Confidence source priority
--------------------------
1. LLM JSON  (llm_json param)
   Uses directly extracted skill evidence and proficiency.
2. TF-IDF skill vectors
   Provides fallback confidence for adjacent or semantically similar skills.
3. Binary extracted_set match
   Last resort when neither of the above is available.
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


PROFICIENCY_MULTIPLIER = {
    "beginner": 0.75,
    "intermediate": 0.9,
    "advanced": 1.0,
    "expert": 1.0,
    "unknown": 0.85,
}


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
    benchmark: dict[str, float],
    resume_text: str = "",
    extracted_skills: list[str] | None = None,
    resume_vector=None,
    skill_vectors: dict | None = None,
) -> dict[str, dict]:
    """
    Build benchmark-aligned confidence scores from LLM output.

    Exact benchmark matches use LLM evidence directly. When available,
    TF-IDF confidence is also computed and used as a fallback so that
    semantically adjacent skills can still receive partial credit.
    """
    extracted_skills = extracted_skills or []

    llm_lookup: dict[str, dict] = {}
    for skill_info in llm_json.get("skills", []):
        aliases = {
            str(skill_info.get("skill", "")).lower().strip(),
            str(skill_info.get("original_skill", "")).lower().strip(),
        } - {""}
        for alias in aliases:
            llm_lookup[alias] = skill_info

    fallback_scores: dict[str, dict] = {}
    if skill_vectors and resume_vector is not None:
        fallback_scores = score_all_benchmark_skills(
            benchmark, resume_text, extracted_skills, resume_vector, skill_vectors
        )

    result: dict[str, dict] = {}
    for skill, onet_weight in benchmark.items():
        skill_lower = skill.lower()
        llm_skill = llm_lookup.get(skill_lower)
        fallback = fallback_scores.get(skill, {})

        proficiency = "unknown"
        llm_confidence = 0.0
        if llm_skill:
            proficiency = str(llm_skill.get("proficiency", "unknown")).lower().strip()
            multiplier = PROFICIENCY_MULTIPLIER.get(proficiency, 0.85)
            llm_confidence = min(
                1.0, float(llm_skill.get("confidence", 0.0)) * multiplier
            )

        final_confidence = round(
            max(llm_confidence, float(fallback.get("confidence", 0.0))), 4
        )
        result[skill_lower] = {
            "confidence": final_confidence,
            "onet_weight": onet_weight,
            "proficiency": proficiency,
            "tier": (
                "strong"
                if final_confidence >= STRONG_THRESHOLD
                else "partial"
                if final_confidence > 0.0
                else fallback.get("tier", "weak")
            ),
            "weighted_confidence": round(final_confidence * onet_weight, 4),
        }
    return result


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
    skill_vectors : pre-computed skill TF-IDF vectors
    category      : override for the Category field in candidate_row
    resume_col    : column name for resume text
    jd_skills     : skills extracted from the job description
    llm_json      : output of extract_all_skills()
    """
    cand_id = candidate_row.get("ID", 0)
    raw_category = category or candidate_row.get("Category", "UNKNOWN")
    job_title = resolve_job_title(str(raw_category))
    resume_text = str(candidate_row.get(resume_col, ""))

    if jd_skills:
        benchmark: dict[str, float] = {}
        for skill in jd_skills:
            s_lower = skill.lower()
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
    cand_vec = vectorizer.transform([resume_text])

    if llm_json:
        confidence_scores = _build_confidence_from_llm(
            llm_json,
            benchmark,
            resume_text=resume_text,
            extracted_skills=extracted_skills,
            resume_vector=cand_vec,
            skill_vectors=skill_vectors,
        )
    elif skill_vectors:
        confidence_scores = score_all_benchmark_skills(
            benchmark, resume_text, extracted_skills, cand_vec, skill_vectors
        )
    else:
        confidence_scores = {}

    if confidence_scores:
        split_result = compute_split_weighted_fit(
            benchmark, confidence_scores, raw_category, job_title
        )
        weighted_fit = split_result["split_fit"]
        hard_fit = split_result["hard_fit"]
        soft_fit = split_result["soft_fit"]
        ratio = get_ratio(raw_category, job_title)
        cat_type = split_result["category_type"]
        ratio_label = split_result["ratio_label"]
    else:
        total_w = sum(benchmark.values())
        matched_w = sum(w for s, w in benchmark.items() if s.lower() in extracted_set)
        weighted_fit = matched_w / total_w if total_w > 0 else 0.0
        hard_fit = weighted_fit
        soft_fit = weighted_fit
        split_result = {}
        cat_type = get_category_type(raw_category, job_title)
        ratio = get_ratio(raw_category, job_title)
        ratio_label = ratio.label

    if jd_skills:
        bench_words: list[str] = []
        for skill, weight in benchmark.items():
            bench_words.extend([skill] * max(1, int(round(weight))))
        bench_text = " ".join(bench_words)
    else:
        bench_text = build_benchmark_text(job_title, top_n=BENCHMARK_TOP_N)

    try:
        bench_vec = vectorizer.transform([bench_text])
        wt_cosine = float(cosine_similarity(cand_vec, bench_vec)[0][0])
    except Exception:
        wt_cosine = 0.0

    composite = 0.75 * weighted_fit + 0.25 * wt_cosine
    grade = determine_grade(composite)
    pathway_depth = (
        "Fundamental & Comprehensive"
        if grade in FUNDAMENTAL_GRADES
        else "Advanced & Fast-Track"
    )
    week_count = (
        WEEK_COUNT_FUNDAMENTAL if grade in FUNDAMENTAL_GRADES else WEEK_COUNT_ADVANCED
    )

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
        "ID": cand_id,
        "Category": raw_category,
        "Job_Title": job_title,
        "Category_Type": cat_type,
        "Ratio_Label": ratio_label,
        "Hard_Fit": round(hard_fit, 4),
        "Soft_Fit": round(soft_fit, 4),
        "Weighted_Fit": round(weighted_fit, 4),
        "Split_Weighted_Fit": round(weighted_fit, 4),
        "Weighted_Cosine": round(wt_cosine, 4),
        "Composite_Score": round(composite, 4),
        "Confidence_Scores": confidence_scores,
        "Split_Details": split_result,
        "Grade": grade,
        "Pathway_Depth": pathway_depth,
        "Duration": week_count,
        "Gaps": sorted(gap_weights.keys()),
        "Gap_Weights": gap_weights,
        "Extracted_Skills": sorted(extracted_set),
        "_cand_vec": cand_vec,
    }


def batch_evaluate(
    resume_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    skill_vectors: dict | None = None,
    resume_col: str = "Resume_str",
    jd_skills: set[str] | list[str] | None = None,
) -> list[dict]:
    """
    Evaluate all rows in a DataFrame. CSV/batch mode uses TF-IDF confidence.
    """
    results = []
    for _, row in resume_df.iterrows():
        try:
            row_jd_skills = row.get("JD_Skills") or jd_skills
            m = evaluate_candidate(
                row,
                vectorizer,
                skill_vectors,
                resume_col=resume_col,
                jd_skills=row_jd_skills,
                llm_json=None,
            )
            results.append(m)
        except Exception as exc:
            results.append({"ID": row.get("ID"), "error": str(exc)})
    return results
