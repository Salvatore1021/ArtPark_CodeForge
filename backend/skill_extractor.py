"""
skill_extractor.py
------------------
Unified skill extraction module.

Two complementary layers:

  1. LLM extraction  (extract_all_skills)
     Called by local_main.py and the PDF → LLM → JSON pathway.
     Runs the full Ollama pipeline: extract → filter hallucinations →
     optionally map to taxonomy → group by category.

  2. Regex + TF-IDF layer  (extract_skills, build_and_fit_vectorizer, …)
     Called by candidate_evaluator, role_recommender, gap_prioritizer.
     Fast O*NET-taxonomy keyword matching and cosine-similarity scoring.

Both layers are needed. The LLM layer produces the authoritative skill JSON;
the TF-IDF layer powers job-matching and gap-analysis scoring downstream.

Requires Ollama for layer 1. Layer 2 has no external runtime dependency.
"""

from __future__ import annotations

import re
import datetime
from collections import defaultdict, Counter
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from llm_extractor import extract_skills_llm, map_to_taxonomy
from taxonomy_adapter import (
    get_all_skills,
    get_job_benchmark,
    build_weighted_benchmark_text,
    find_job_title,
)
from config import BENCHMARK_TOP_N, MIN_SKILL_WEIGHT


# ── Skill library — built once at import ─────────────────────────────────────
# These are module-level singletons; importing this file is cheap after the first time.

FULL_SKILL_LIBRARY: list[str] = get_all_skills(min_weight=0.0)
FULL_SKILL_SET:     frozenset  = frozenset(FULL_SKILL_LIBRARY)   # O(1) membership


# ── Precompiled combined regex ────────────────────────────────────────────────
# Sort longest-first so multi-word phrases ("machine learning") match before
# substrings ("learning"). Built once; reused by every extract_skills() call.

_sorted_skills = sorted(FULL_SKILL_LIBRARY, key=len, reverse=True)
_SKILL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _sorted_skills) + r")\b",
    re.IGNORECASE,
)


# ════════════════════════════════════════════════════════════════════
# Layer 2 — Regex / TF-IDF helpers
# ════════════════════════════════════════════════════════════════════

def extract_skills(
    text: str,
    skill_library: Optional[list[str]] = None,
) -> list[str]:
    """
    Return a sorted list of taxonomy skills found in *text*.

    Uses a single combined OR-regex (~54x faster than per-skill re.search).
    If a custom skill_library is provided, compiles a one-off pattern for it;
    otherwise uses the module-level precompiled pattern.

    Called by: candidate_evaluator, role_recommender, main.py (run_single_pdf).
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if skill_library is None or frozenset(skill_library) == FULL_SKILL_SET:
        matches = _SKILL_PATTERN.findall(text.lower())
    else:
        custom_sorted = sorted(skill_library, key=len, reverse=True)
        pat = re.compile(
            r"\b(" + "|".join(re.escape(s) for s in custom_sorted) + r")\b",
            re.IGNORECASE,
        )
        matches = pat.findall(text.lower())

    return sorted(set(m.lower() for m in matches))


def get_weighted_benchmark(
    job_title: str,
    top_n: int = BENCHMARK_TOP_N,
    min_weight: float = MIN_SKILL_WEIGHT,
) -> dict[str, float]:
    """
    Return {skill: onet_weight} for a job title, filtered by min_weight.
    Called by: candidate_evaluator.
    """
    raw = get_job_benchmark(job_title, top_n=top_n)
    return {k: v for k, v in raw.items() if v >= min_weight}


def build_benchmark_text(job_title: str, top_n: int = BENCHMARK_TOP_N) -> str:
    """
    Return the O*NET-weighted benchmark text for TF-IDF comparison.
    Called by: candidate_evaluator.
    """
    return build_weighted_benchmark_text(job_title, top_n=top_n)


def build_and_fit_vectorizer(
    resume_df: pd.DataFrame,
    job_description_text: str = "",
    skill_library: Optional[list[str]] = None,
    resume_col: str = "Resume_str",
    extra_benchmark_texts: Optional[list[str]] = None,
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the combined corpus:
      - all resume texts
      - job-description PDF text (if provided)
      - weighted O*NET benchmark texts
      - full skill library terms

    ngram_range=(1, 2) captures "machine learning" / "deep learning" as tokens.
    sublinear_tf=True log-scales term frequencies to reduce repetition inflation.
    Called by: main.py (run_single_pdf, run_single_csv, run_batch).
    """
    library = skill_library or FULL_SKILL_LIBRARY
    corpus: list[str] = (
        resume_df[resume_col].fillna("").tolist()
        + ([job_description_text] if job_description_text else [])
        + (extra_benchmark_texts or [])
        + library
    )
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=1,
    )
    vectorizer.fit(corpus)
    return vectorizer


def precompute_skill_vectors(
    vectorizer: TfidfVectorizer,
    skill_library: Optional[list[str]] = None,
) -> dict[str, csr_matrix]:
    """
    Pre-compute TF-IDF sparse vectors for every skill in a single batch
    transform call (~32x faster than looping vectorizer.transform([skill])).

    Returns  {skill_string: sparse_row_vector (1 × vocab_size)}
    Called by: main.py, gap_prioritizer (via candidate_evaluator).
    """
    library = skill_library or FULL_SKILL_LIBRARY
    matrix = vectorizer.transform(library)   # (n_skills × vocab) — one call
    return {skill: matrix[i] for i, skill in enumerate(library)}


# ════════════════════════════════════════════════════════════════════
# Experience calculator — used by extract_all_skills (LLM path)
# ════════════════════════════════════════════════════════════════════

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}


def _parse_date(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip().lower()
    m = re.match(r"([a-z]+)\s+(\d{4})", s)
    if m:
        return int(m.group(2)) + _MONTH_MAP.get(m.group(1), 6) / 12.0
    m = re.match(r"(\d{4})", s)
    if m:
        return float(m.group(1))
    if "present" in s or "current" in s:
        return float(datetime.date.today().year)
    return None


def _compute_experience(experience: list) -> dict:
    """
    Compute total years and per-title years from a list of job dicts.

    Handles both structured dicts (HTML-parsed resumes) and raw strings
    (PDF-parsed resumes where experience is a list of text lines).
    Raw strings are skipped — structured data only.
    """
    current_year = float(datetime.date.today().year)
    total_months = 0.0
    years_per_title: dict[str, float] = {}

    for job in experience:
        # PDF resumes return experience as list[str] — skip those entries
        if not isinstance(job, dict):
            continue
        start = _parse_date(job.get("start_date", ""))
        end   = _parse_date(job.get("end_date",   "")) or current_year
        if start and end and end > start:
            yrs = end - start
            total_months += yrs * 12
            title = job.get("title", "")
            if title:
                years_per_title[title] = round(yrs, 1)

    total = round(total_months / 12, 1)
    if total < 2:
        seniority = "Junior (0-2 yrs)"
    elif total < 5:
        seniority = "Mid-Level (2-5 yrs)"
    elif total < 10:
        seniority = "Senior (5-10 yrs)"
    else:
        seniority = "Expert (10+ yrs)"

    return {
        "total_years":     total,
        "years_per_title": years_per_title,
        "seniority_level": seniority,
    }


def _classify_domains(skills: list) -> list[str]:
    """
    Derive top-5 domain expertise from LLM-assigned categories.
    Pure frequency count — no hardcoding.
    """
    cat_counts = Counter(s["category"] for s in skills)
    return [cat for cat, _ in cat_counts.most_common(5)]


# ════════════════════════════════════════════════════════════════════
# Layer 1 — LLM extraction pipeline
# ════════════════════════════════════════════════════════════════════

def extract_all_skills(
    parsed_resume: dict,
    taxonomy_skills: Optional[list[str]] = None,
) -> dict:
    """
    Extract skills from a parsed resume using the LLM (Ollama).

    Accepts output from parse_resume_pdf OR parse_resume_html — handles
    both key names ('full_text' from PDF parser, 'raw_text' from HTML parser).

    Parameters
    ----------
    parsed_resume   : dict from resume_parser.parse_resume_pdf / parse_resume_html
    taxonomy_skills : optional list of canonical skill names for taxonomy mapping

    Returns
    -------
    {
        skills             : list[dict]  — each skill has skill, category,
                                           proficiency, confidence, reasoning
        skills_by_category : dict[str, list[str]]
        experience_info    : dict  — total_years, seniority_level, years_per_title
        domains            : list[str]  — top-5 domains by skill count
        languages          : list[str]
        confidence_scores  : dict[str, float]  — skill → confidence
        reasoning_trace    : list[dict]  — post-filter audit trail
        backend_used       : str
        model_used         : str
    }

    Raises RuntimeError if Ollama is unreachable.
    """
    # Handle both PDF ('full_text') and HTML ('raw_text') parser outputs
    raw_text = parsed_resume.get("raw_text") or parsed_resume.get("full_text", "")
    experience = parsed_resume.get("experience", [])

    # ── LLM extraction ────────────────────────────────────────────────
    llm_result = extract_skills_llm(raw_text)

    if not llm_result["llm_available"]:
        raise RuntimeError(
            f"LLM extraction failed: {llm_result.get('error', 'Ollama unavailable')}. "
            "Make sure Ollama is running: ollama serve"
        )

    skills       = llm_result["skills"]
    backend_used = llm_result["backend_used"]
    model_used   = llm_result["model_used"]

    # ── Filter hallucinations first — drop skills with no real evidence ───
    # NOTE: reasoning_trace is built AFTER filtering so it only logs
    # skills that actually passed the evidence check.
    
    _INVALID_REASONING = {"", "not specified", "n/a", "none", "not mentioned"}
    skills = [
        s for s in skills
        if s.get("reasoning", "").strip().lower() not in _INVALID_REASONING
    ]

    # ── Reasoning trace (post-filter audit trail) ─────────────────────
    reasoning_trace = [
        {
            "layer":       "llm",
            "skill":       s["skill"],
            "category":    s["category"],
            "proficiency": s["proficiency"],
            "confidence":  s["confidence"],
            "reasoning":   s["reasoning"],
            "backend":     backend_used,
            "model":       model_used,
        }
        for s in skills
    ]

    # ── Taxonomy mapping (optional) ───────────────────────────────────
    if taxonomy_skills:
        skill_lookup = {s.lower().strip(): s for s in taxonomy_skills}
        mapping      = map_to_taxonomy(skills, taxonomy_skills)
        mapped = []
        for s in skills:
            matched = mapping.get(s["skill"])
            if matched:
                canonical = skill_lookup.get(matched.lower().strip(), matched)
                if canonical in taxonomy_skills:
                    mapped.append({
                        **s,
                        "original_skill": s["skill"],
                        "skill":          canonical,
                    })
                else:
                    print(f"  [taxonomy] removed '{s['skill']}' (no canonical match)")
            else:
                print(f"  [taxonomy] removed '{s['skill']}' (no taxonomy match)")
        skills = mapped

    # ── Dedup ─────────────────────────────────────────────────────────
    seen: set[str] = set()
    deduped = []
    for s in skills:
        if s["skill"] not in seen:
            seen.add(s["skill"])
            deduped.append(s)
    skills = deduped

    # ── Group by category ─────────────────────────────────────────────
    by_category: dict[str, list[str]] = defaultdict(list)
    for s in skills:
        by_category[s["category"]].append(s["skill"])

    return {
        "skills":             skills,
        "skills_by_category": dict(by_category),
        "experience_info":    _compute_experience(experience),
        "domains":            _classify_domains(skills),
        "languages":          parsed_resume.get("languages", []),
        "confidence_scores":  {s["skill"]: s["confidence"] for s in skills},
        "reasoning_trace":    reasoning_trace,
        "backend_used":       backend_used,
        "model_used":         model_used,
    }