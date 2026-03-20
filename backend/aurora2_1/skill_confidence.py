"""
skill_confidence.py
--------------------
Replaces binary skill presence (0/1) with a continuous confidence score per skill.

Confidence model
----------------
For each skill S in the benchmark:

  text_match    = 1.0 if skill is detected in resume text by regex, else 0.0
  semantic_sim  = cosine_similarity(resume_tfidf, skill_tfidf)   [0–1]
  frequency     = min(count_of_skill_in_resume / 3, 1.0)         [0–1, caps at 3 mentions]

  confidence = 0.5 × text_match + 0.3 × semantic_sim + 0.2 × frequency

Interpretation
--------------
  > 0.70  → Strong — skill clearly demonstrated
  0.40–0.70 → Partial — implied / adjacent knowledge
  < 0.40  → Weak / missing — include in learning plan

The confidence vector replaces the binary match in the Weighted Fit Score,
giving a smoother signal that rewards candidates who have adjacent knowledge.
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Confidence tier labels and thresholds
STRONG_THRESHOLD  = 0.70
PARTIAL_THRESHOLD = 0.40

CONFIDENCE_TIERS = {
    "strong":  STRONG_THRESHOLD,
    "partial": PARTIAL_THRESHOLD,
    "weak":    0.0,
}


def _count_occurrences(text: str, skill: str) -> int:
    """Count whole-word occurrences of skill in text (case-insensitive)."""
    return len(re.findall(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE))


def compute_skill_confidence(
    skill: str,
    resume_text: str,
    extracted_skills: list[str],
    resume_vector,
    skill_vectors: dict,
    epsilon: float = 1e-6,
) -> float:
    """
    Return a [0, 1] confidence score for a single skill against a resume.

    Parameters
    ----------
    skill            : skill string to score
    resume_text      : raw resume text
    extracted_skills : list of skills already detected by regex
    resume_vector    : TF-IDF sparse vector of the resume
    skill_vectors    : pre-computed skill TF-IDF vectors dict
    """
    s = skill.lower().strip()

    # Component 1: regex text match
    text_match = 1.0 if s in {x.lower() for x in extracted_skills} else 0.0

    # Component 2: semantic similarity via TF-IDF cosine
    skill_vec = skill_vectors.get(s)
    if skill_vec is not None and resume_vector.nnz > 0:
        sim = float(cosine_similarity(resume_vector, skill_vec)[0][0])
        semantic_sim = sim if not np.isnan(sim) else 0.0
    else:
        semantic_sim = 0.0

    # Component 3: frequency (how often skill words appear, capped at 3 mentions)
    count = _count_occurrences(resume_text, s)
    frequency = min(count / 3.0, 1.0)

    confidence = 0.5 * text_match + 0.3 * semantic_sim + 0.2 * frequency
    return round(min(confidence, 1.0), 4)


def score_all_benchmark_skills(
    benchmark: dict[str, float],
    resume_text: str,
    extracted_skills: list[str],
    resume_vector,
    skill_vectors: dict,
) -> dict[str, dict]:
    """
    Score every benchmark skill and return a detailed confidence breakdown.

    Returns
    -------
    dict  skill → {
        confidence: float,
        tier: str ('strong'|'partial'|'weak'),
        onet_weight: float,
        weighted_confidence: float  (confidence × onet_weight)
    }
    """
    results: dict[str, dict] = {}
    for skill, onet_weight in benchmark.items():
        conf = compute_skill_confidence(
            skill, resume_text, extracted_skills, resume_vector, skill_vectors
        )
        tier = (
            "strong"  if conf >= STRONG_THRESHOLD  else
            "partial" if conf >= PARTIAL_THRESHOLD else
            "weak"
        )
        results[skill] = {
            "confidence":          conf,
            "tier":                tier,
            "onet_weight":         onet_weight,
            "weighted_confidence": round(conf * onet_weight, 4),
        }
    return results


def confidence_weighted_fit(
    confidence_scores: dict[str, dict],
) -> float:
    """
    Compute a fit score using confidence (not binary) weighted by O*NET importance.

    Score = Σ(onet_weight_i × confidence_i) / Σ(onet_weight_i)
    """
    total_weight    = sum(v["onet_weight"] for v in confidence_scores.values())
    weighted_conf   = sum(v["weighted_confidence"] for v in confidence_scores.values())
    return round(weighted_conf / total_weight, 4) if total_weight > 0 else 0.0


def get_partial_skills(
    confidence_scores: dict[str, dict],
    min_confidence: float = PARTIAL_THRESHOLD,
    max_confidence: float = STRONG_THRESHOLD,
) -> list[str]:
    """Return skills in the partial knowledge zone — candidate has adjacent knowledge."""
    return [
        skill for skill, data in confidence_scores.items()
        if min_confidence <= data["confidence"] < max_confidence
    ]


def format_confidence_report(
    confidence_scores: dict[str, dict],
    top_n: int = 15,
) -> str:
    """Pretty-print a confidence breakdown table."""
    rows = sorted(
        confidence_scores.items(),
        key=lambda x: x[1]["weighted_confidence"],
        reverse=True,
    )[:top_n]

    lines = [
        f"\n{'─'*70}",
        f"  {'Skill':<38} {'Conf':>6}  {'Tier':<8}  {'O*NET':>5}  {'Wtd Conf':>8}",
        f"{'─'*70}",
    ]
    for skill, data in rows:
        tier_sym = {"strong": "●", "partial": "◑", "weak": "○"}.get(data["tier"], " ")
        lines.append(
            f"  {skill:<38} {data['confidence']:>6.3f}  "
            f"{tier_sym} {data['tier']:<6}  {data['onet_weight']:>5.2f}  "
            f"{data['weighted_confidence']:>8.4f}"
        )
    lines.append(f"{'─'*70}")
    return "\n".join(lines)
