"""
gap_analyzer.py — Compute skill gaps between resume skills and JD requirements.

Gap severity scores drive pathway prioritization.
"""

import re
import json
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

_TAXONOMY_CACHE: dict | None = None


def _load_taxonomy() -> dict:
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "skill_taxonomy.json")
        with open(path, "r") as f:
            _TAXONOMY_CACHE = json.load(f)
    except Exception as e:
        logger.error(f"Could not load taxonomy: {e}")
        _TAXONOMY_CACHE = {"categories": {}}
    return _TAXONOMY_CACHE


def _normalize_skill_name(skill_name: str) -> str:
    """Lowercase, strip, collapse whitespace for matching."""
    return re.sub(r"\s+", " ", skill_name.strip().lower())


def _build_alias_map(taxonomy: dict) -> dict[str, str]:
    """Returns alias→canonical mapping from taxonomy."""
    alias_map: dict[str, str] = {}
    for category in taxonomy["categories"].values():
        for canonical, data in category["skills"].items():
            alias_map[_normalize_skill_name(canonical)] = canonical
            for alias in data.get("aliases", []):
                alias_map[_normalize_skill_name(alias)] = canonical
    return alias_map


def _resolve_canonical(skill_name: str, alias_map: dict[str, str]) -> str:
    """Resolve a skill name to its canonical taxonomy form, or return as-is."""
    key = _normalize_skill_name(skill_name)
    return alias_map.get(key, skill_name)


def _match_skill(required_canonical: str, resume_skills: list[dict], alias_map: dict) -> dict | None:
    """
    Find the best matching skill in resume for a required canonical skill.
    Returns the matching skill dict or None.
    """
    req_norm = _normalize_skill_name(required_canonical)
    for res_skill in resume_skills:
        res_canonical = _resolve_canonical(res_skill["skill_name"], alias_map)
        if _normalize_skill_name(res_canonical) == req_norm:
            return res_skill
    return None


def compute_skill_gap(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare resume skills against JD requirements and compute gap metadata.

    Returns:
    {
        "gaps": [
            {
                "skill_name": str,
                "canonical_name": str,
                "category": str,
                "required_level": int (1-5),
                "current_level": int (0 = absent),
                "gap_magnitude": float (0.0 – 1.0),
                "is_mandatory": bool,
                "priority_score": float,   # drives ordering in pathway
                "status": "missing" | "below_required" | "met" | "exceeds"
            }
        ],
        "matched_skills": [...],   # skills that are already met/exceeded
        "gap_summary": {
            "total_required": int,
            "total_gaps": int,
            "critical_gaps": int,   # mandatory + missing
            "coverage_pct": float
        },
        "reasoning": str           # human-readable summary
    }
    """
    taxonomy = _load_taxonomy()
    alias_map = _build_alias_map(taxonomy)

    resume_skills: list[dict] = resume_data.get("skills", [])
    jd_skills: list[dict] = jd_data.get("skills", [])

    gaps = []
    matched = []

    for jd_skill in jd_skills:
        skill_name = jd_skill["skill_name"]
        canonical = _resolve_canonical(skill_name, alias_map)
        required_level = int(jd_skill.get("required_level", 3))
        is_mandatory = bool(jd_skill.get("is_mandatory", True))
        category = jd_skill.get("category", "other")

        # Try to find this skill in resume
        resume_match = _match_skill(canonical, resume_skills, alias_map)
        current_level = int(resume_match.get("proficiency_level", 0)) if resume_match else 0
        evidence = resume_match.get("evidence", "") if resume_match else ""

        if current_level == 0:
            status = "missing"
            gap_magnitude = required_level / 5.0
        elif current_level < required_level:
            status = "below_required"
            gap_magnitude = (required_level - current_level) / 5.0
        elif current_level == required_level:
            status = "met"
            gap_magnitude = 0.0
        else:
            status = "exceeds"
            gap_magnitude = 0.0

        # Priority score:
        # mandatory × 2 + gap_magnitude × required_level
        # Higher = more urgent to address in the pathway
        mandatory_weight = 2.0 if is_mandatory else 1.0
        priority_score = round(mandatory_weight * gap_magnitude * required_level, 4)

        entry = {
            "skill_name": skill_name,
            "canonical_name": canonical,
            "category": category,
            "required_level": required_level,
            "current_level": current_level,
            "gap_magnitude": round(gap_magnitude, 4),
            "is_mandatory": is_mandatory,
            "priority_score": priority_score,
            "status": status,
            "evidence": evidence,
            "context": jd_skill.get("context", ""),
        }

        if status in ("missing", "below_required"):
            gaps.append(entry)
        else:
            matched.append(entry)

    # Sort gaps: highest priority first
    gaps.sort(key=lambda x: (-x["priority_score"], x["skill_name"]))

    total_required = len(jd_skills)
    total_gaps = len(gaps)
    critical_gaps = sum(1 for g in gaps if g["is_mandatory"] and g["status"] == "missing")
    coverage_pct = round((total_required - total_gaps) / max(total_required, 1) * 100, 1)

    # Human-readable reasoning trace
    gap_names = [g["skill_name"] for g in gaps[:5]]
    reasoning = (
        f"Out of {total_required} required skills, the candidate already meets {len(matched)} "
        f"({coverage_pct}% coverage). {total_gaps} gap(s) were identified, of which "
        f"{critical_gaps} are critical mandatory gaps. "
        f"Top priority gaps: {', '.join(gap_names) if gap_names else 'None'}. "
        f"The learning pathway is ordered by priority score (mandatory × gap magnitude × required level)."
    )

    return {
        "gaps": gaps,
        "matched_skills": matched,
        "gap_summary": {
            "total_required": total_required,
            "total_gaps": total_gaps,
            "critical_gaps": critical_gaps,
            "coverage_pct": coverage_pct,
        },
        "reasoning": reasoning,
    }
