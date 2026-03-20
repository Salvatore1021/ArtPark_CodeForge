"""
skill_extractor.py
------------------
Builds the unified skill library from SKILLS_TAXONOMY, extracts skills
from free text, and fits / manages the TF-IDF vectorizer.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from skills_taxonomy import SKILLS_TAXONOMY, SKILL_TO_CATEGORY


# ── Build the flat skill library from taxonomy ───────────────────────────────

def build_skill_library(
    extra_skills: Optional[list[str]] = None,
    include_categories: Optional[list[str]] = None,
) -> list[str]:
    """
    Flatten SKILLS_TAXONOMY into a single deduplicated list of skill strings.

    Parameters
    ----------
    extra_skills       : additional skills not in the taxonomy
    include_categories : if given, only these taxonomy categories are used;
                         otherwise ALL categories are included

    Returns
    -------
    Sorted list of unique lowercase skill strings.
    """
    skills: set[str] = set()

    for category, skill_list in SKILLS_TAXONOMY.items():
        if include_categories and category not in include_categories:
            continue
        for s in skill_list:
            skills.add(s.lower())

    if extra_skills:
        for s in extra_skills:
            skills.add(s.lower())

    return sorted(skills)


# Pre-built full library (used as default throughout the engine)
FULL_SKILL_LIBRARY: list[str] = build_skill_library()


# ── Skill extraction ─────────────────────────────────────────────────────────

def extract_skills(
    text: str,
    skill_library: Optional[list[str]] = None,
) -> list[str]:
    """
    Return a sorted list of skills found in *text* using whole-word matching.

    Parameters
    ----------
    text          : free-form text (resume body, job description, etc.)
    skill_library : list of skill strings to search for;
                    defaults to FULL_SKILL_LIBRARY
    """
    if not isinstance(text, str) or not text.strip():
        return []

    library = skill_library or FULL_SKILL_LIBRARY
    text_lower = text.lower()

    found = [
        skill for skill in library
        if re.search(rf"\b{re.escape(skill)}\b", text_lower)
    ]
    return sorted(set(found))


def get_skill_category(skill: str) -> Optional[str]:
    """Return the taxonomy category for a skill, or None if unknown."""
    return SKILL_TO_CATEGORY.get(skill.lower())


# ── TF-IDF vectorizer management ─────────────────────────────────────────────

def build_and_fit_vectorizer(
    resume_df: pd.DataFrame,
    job_description_text: str = "",
    skill_library: Optional[list[str]] = None,
    resume_col: str = "Resume_str",
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the combined corpus of:
      - all resume texts
      - the job-description text
      - the skill library terms

    Returns the fitted vectorizer.
    """
    library = skill_library or FULL_SKILL_LIBRARY

    corpus = (
        resume_df[resume_col].fillna("").tolist()
        + ([job_description_text] if job_description_text else [])
        + library
    )

    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectorizer.fit(corpus)
    return vectorizer


def precompute_skill_vectors(
    vectorizer: TfidfVectorizer,
    skill_library: Optional[list[str]] = None,
) -> dict:
    """
    Pre-compute TF-IDF sparse vectors for every skill in the library.

    Returns dict  skill_string → sparse matrix (1 × vocab_size)
    """
    library = skill_library or FULL_SKILL_LIBRARY
    return {skill: vectorizer.transform([skill]) for skill in library}
