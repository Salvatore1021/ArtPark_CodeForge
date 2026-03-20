"""
role_recommender.py
--------------------
Automatically finds the best-fitting job titles from the taxonomy for a
given resume, without requiring the user to specify a role upfront.

Algorithm
---------
For each candidate taxonomy job title:
  1. Extract skills from resume that appear in that job's benchmark.
  2. Compute a Weighted Fit Score  (O*NET-weighted skill match ratio).
  3. Compute a Weighted Cosine Similarity between resume vector and benchmark vector.
  4. Composite = 0.6 × Weighted Fit + 0.4 × Weighted Cosine
     (Fit weighted higher here since we want hard-skill alignment for recommendations)

Returns the top-N roles ranked by composite, with explanations of why each role fits.

Usage
-----
    from role_recommender import recommend_roles
    results = recommend_roles(resume_text, vectorizer, top_n=5)
    for r in results:
        print(r['job_title'], r['composite'], r['matched_skills'])
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy_adapter import (
    list_jobs,
    get_job_benchmark,
    build_weighted_benchmark_text,
    get_sector,
    TECH_SKILL_AUGMENTATION,
    PROXY_MAP,
)
from skill_extractor import extract_skills, FULL_SKILL_LIBRARY
from config import BENCHMARK_TOP_N


# Limit how many roles to score (most roles are irrelevant for tech candidates)
# Score all roles in these priority sectors first, then sample from others
_PRIORITY_SECTORS = {
    "Data & AI", "Backend", "Frontend/Web", "Mobile", "IT Infrastructure",
    "Engineering & Tech", "Management & Business", "Healthcare",
    "Other Professional Services", "Education",
}

# Roles to skip (very specialised manual trades unlikely to match tech resumes)
_SKIP_SECTORS = set()


def _score_resume_against_role(
    resume_text: str,
    extracted_skills: list[str],
    job_title: str,
    vectorizer: TfidfVectorizer,
) -> dict:
    """Score a single resume against a single job title. Returns a score dict."""
    benchmark = get_job_benchmark(job_title, top_n=BENCHMARK_TOP_N)
    if not benchmark:
        return {}

    extracted_set = set(s.lower() for s in extracted_skills)
    total_weight   = sum(benchmark.values())
    matched_weight = sum(w for s, w in benchmark.items() if s.lower() in extracted_set)

    weighted_fit = matched_weight / total_weight if total_weight > 0 else 0.0

    bench_text = build_weighted_benchmark_text(job_title, top_n=BENCHMARK_TOP_N)
    try:
        cand_vec  = vectorizer.transform([resume_text])
        bench_vec = vectorizer.transform([bench_text])
        wt_cosine = float(cosine_similarity(cand_vec, bench_vec)[0][0])
    except Exception:
        wt_cosine = 0.0

    composite = 0.6 * weighted_fit + 0.4 * wt_cosine

    matched_skills = sorted(s for s in benchmark if s.lower() in extracted_set)
    gap_skills     = sorted(s for s in benchmark if s.lower() not in extracted_set)

    return {
        "job_title":      job_title,
        "sector":         get_sector(job_title) or "Unknown",
        "weighted_fit":   round(weighted_fit, 4),
        "weighted_cosine":round(wt_cosine, 4),
        "composite":      round(composite, 4),
        "matched_skills": matched_skills,
        "gap_skills":     gap_skills[:8],   # top gaps only
        "match_count":    len(matched_skills),
        "benchmark_size": len(benchmark),
    }


def recommend_roles(
    resume_text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 5,
    min_match_count: int = 2,
    candidate_id: int | str | None = None,
) -> list[dict]:
    """
    Find the top-N best-fitting job titles for a resume.

    Parameters
    ----------
    resume_text     : raw resume text (full body)
    vectorizer      : fitted TF-IDF vectorizer from the engine
    top_n           : number of top roles to return
    min_match_count : minimum matched skills for a role to be considered
    candidate_id    : optional ID for logging

    Returns
    -------
    list of dicts sorted by composite score (descending), each with:
        job_title, sector, weighted_fit, weighted_cosine, composite,
        matched_skills, gap_skills, match_count, benchmark_size
    """
    extracted = extract_skills(resume_text, FULL_SKILL_LIBRARY)
    extracted_set = set(extracted)

    # Collect job titles to score
    # Score tech roles (have augmentation) + O*NET roles
    tech_titles = list(TECH_SKILL_AUGMENTATION.keys()) + list(PROXY_MAP.keys())
    all_titles  = sorted(set(list_jobs()) )

    results: list[dict] = []
    for title in all_titles:
        score = _score_resume_against_role(resume_text, extracted, title, vectorizer)
        if not score:
            continue
        if score["match_count"] < min_match_count:
            continue
        results.append(score)

    # Sort by composite descending
    results.sort(key=lambda x: x["composite"], reverse=True)
    return results[:top_n]


def format_recommendations(results: list[dict], candidate_id: int | str = "") -> str:
    """Pretty-print the role recommendation results."""
    lines = [
        f"\n{'═'*65}",
        f"  Role Recommendations  {'for candidate ' + str(candidate_id) if candidate_id else ''}",
        f"{'═'*65}",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"\n  #{i}  {r['job_title']}  [{r['sector']}]",
            f"      Composite : {r['composite']:.4f}  "
            f"(Fit {r['weighted_fit']:.3f} + Cosine {r['weighted_cosine']:.3f})",
            f"      Matched   : {', '.join(r['matched_skills']) or 'none'}",
            f"      Top gaps  : {', '.join(r['gap_skills'][:5]) or 'none'}",
        ]
    lines.append(f"\n{'═'*65}")
    return "\n".join(lines)
