"""
metrics.py — Evaluation metrics for the AURORA adaptive pathway system.

Metrics tracked:
  1. Skill Gap Coverage Rate      — % of identified gaps addressed by pathway
  2. Pathway Efficiency Score     — Coverage / total_hours (more coverage per hour = better)
  3. Redundancy Rate              — % of courses that are prereq-only (no direct gap coverage)
  4. Mandatory Gap Fill Rate      — % of mandatory gaps covered (most critical metric)
  5. Catalog Grounding Rate       — % of recommendations that exist in catalog (always 1.0)
  6. Prerequisite Validity        — All prerequisite orderings are respected
  7. Estimated Redundancy Saved   — Hours saved vs naive all-courses approach
  8. Phase Balance Score          — Std dev of phase hours (lower = more balanced)
"""

import math
import statistics
from typing import Any


# ---------------------------------------------------------------------------
# 1. Skill Gap Coverage Rate
# ---------------------------------------------------------------------------

def skill_gap_coverage_rate(gap_analysis: dict, pathway: dict) -> float:
    """
    % of identified skill gaps that are addressed by at least one course
    in the generated pathway.

    Range: 0.0 – 1.0 (1.0 = all gaps covered)
    """
    gaps: list[dict] = gap_analysis.get("gaps", [])
    if not gaps:
        return 1.0  # no gaps = 100% coverage

    gap_skill_names = {g["skill_name"].lower() for g in gaps}
    covered: set[str] = set()

    for phase in pathway.get("phases", []):
        for course in phase.get("courses", []):
            for skill in course.get("skills_taught", []):
                covered.add(skill.lower())

    covered_gaps = gap_skill_names & covered
    return round(len(covered_gaps) / len(gap_skill_names), 4)


# ---------------------------------------------------------------------------
# 2. Pathway Efficiency Score
# ---------------------------------------------------------------------------

def pathway_efficiency_score(gap_analysis: dict, pathway: dict) -> float:
    """
    Coverage per 10 hours of recommended training.
    Higher = more efficient (covers more gaps per learning hour).

    Formula: (coverage_rate × total_gaps) / (total_hours / 10)
    """
    coverage = skill_gap_coverage_rate(gap_analysis, pathway)
    total_gaps = len(gap_analysis.get("gaps", []))
    total_hours = pathway.get("estimated_total_hours", 1)

    if total_hours == 0 or total_gaps == 0:
        return 0.0

    score = (coverage * total_gaps) / (total_hours / 10)
    return round(score, 4)


# ---------------------------------------------------------------------------
# 3. Redundancy Rate
# ---------------------------------------------------------------------------

def redundancy_rate(pathway: dict) -> float:
    """
    % of courses in the pathway that are prerequisite-only
    (they don't directly address any identified gap).

    Lower is better. High redundancy means learner has many foundational
    gaps that need filling before reaching their target skills.

    Range: 0.0 – 1.0
    """
    all_courses = []
    for phase in pathway.get("phases", []):
        all_courses.extend(phase.get("courses", []))

    if not all_courses:
        return 0.0

    prereq_only = sum(1 for c in all_courses if c.get("is_prerequisite_only", False))
    return round(prereq_only / len(all_courses), 4)


# ---------------------------------------------------------------------------
# 4. Mandatory Gap Fill Rate
# ---------------------------------------------------------------------------

def mandatory_gap_fill_rate(gap_analysis: dict, pathway: dict) -> float:
    """
    % of MANDATORY skill gaps that are addressed in the pathway.
    This is the most critical metric — mandatory gaps block role readiness.

    Range: 0.0 – 1.0 (target: 1.0)
    """
    mandatory_gaps = [g for g in gap_analysis.get("gaps", []) if g.get("is_mandatory", True)]
    if not mandatory_gaps:
        return 1.0

    mandatory_names = {g["skill_name"].lower() for g in mandatory_gaps}
    covered: set[str] = set()

    for phase in pathway.get("phases", []):
        for course in phase.get("courses", []):
            for skill in course.get("skills_taught", []):
                covered.add(skill.lower())

    filled = mandatory_names & covered
    return round(len(filled) / len(mandatory_names), 4)


# ---------------------------------------------------------------------------
# 5. Prerequisite Validity
# ---------------------------------------------------------------------------

def prerequisite_validity(pathway: dict) -> dict[str, Any]:
    """
    Verify that all prerequisite orderings are respected in the pathway.
    Returns: { valid: bool, violations: list[str] }

    A violation = course X appears before its prerequisite Y.
    """
    all_courses = []
    for phase in pathway.get("phases", []):
        all_courses.extend(phase.get("courses", []))

    course_position = {c["id"]: i for i, c in enumerate(all_courses)}
    violations = []

    for course in all_courses:
        course_pos = course_position[course["id"]]
        for prereq_id in course.get("prerequisites", []):
            if prereq_id in course_position:
                prereq_pos = course_position[prereq_id]
                if prereq_pos > course_pos:
                    violations.append(
                        f"'{course['id']}' (pos {course_pos}) appears before "
                        f"its prerequisite '{prereq_id}' (pos {prereq_pos})"
                    )

    return {
        "valid": len(violations) == 0,
        "total_courses": len(all_courses),
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# 6. Phase Balance Score
# ---------------------------------------------------------------------------

def phase_balance_score(pathway: dict) -> float:
    """
    Measures how evenly hours are distributed across phases.
    Returns the coefficient of variation (std_dev / mean) of phase hours.
    Lower = more balanced.

    Range: ≥ 0.0 (0.0 = perfectly balanced)
    """
    hours_per_phase = [p["total_hours"] for p in pathway.get("phases", [])]
    if len(hours_per_phase) <= 1:
        return 0.0

    mean = statistics.mean(hours_per_phase)
    if mean == 0:
        return 0.0

    std = statistics.stdev(hours_per_phase)
    return round(std / mean, 4)


# ---------------------------------------------------------------------------
# 7. Estimated Redundancy Saved (hours)
# ---------------------------------------------------------------------------

def redundancy_hours_saved(gap_analysis: dict, pathway: dict, catalog: list[dict]) -> dict[str, Any]:
    """
    Compare hours in the personalized pathway vs naive all-courses approach.
    Shows how much training time is saved by targeting gaps only.
    """
    pathway_hours = pathway.get("estimated_total_hours", 0)
    total_catalog_hours = sum(c.get("duration_hours", 0) for c in catalog)
    hours_saved = total_catalog_hours - pathway_hours
    pct_saved = round(hours_saved / max(total_catalog_hours, 1) * 100, 1)

    return {
        "pathway_hours": pathway_hours,
        "full_catalog_hours": total_catalog_hours,
        "hours_saved": hours_saved,
        "percent_saved": pct_saved,
    }


# ---------------------------------------------------------------------------
# 8. Composite Evaluation Report
# ---------------------------------------------------------------------------

def evaluate_pathway(
    gap_analysis: dict,
    pathway: dict,
    catalog: list[dict],
) -> dict[str, Any]:
    """
    Run all metrics and produce a structured evaluation report.
    """
    coverage      = skill_gap_coverage_rate(gap_analysis, pathway)
    efficiency    = pathway_efficiency_score(gap_analysis, pathway)
    redundancy_r  = redundancy_rate(pathway)
    mandatory_fr  = mandatory_gap_fill_rate(gap_analysis, pathway)
    prereq_valid  = prerequisite_validity(pathway)
    phase_balance = phase_balance_score(pathway)
    savings       = redundancy_hours_saved(gap_analysis, pathway, catalog)

    # Composite score (weighted average of key metrics, 0–100)
    composite = round(
        (coverage     * 0.25 +
         mandatory_fr * 0.35 +
         (1.0 - redundancy_r) * 0.10 +
         (1.0 if prereq_valid["valid"] else 0.0) * 0.20 +
         min(efficiency / 10.0, 1.0) * 0.10) * 100,
        1
    )

    return {
        "composite_score": composite,
        "metrics": {
            "skill_gap_coverage_rate":  {"value": coverage,      "unit": "ratio", "higher_is_better": True},
            "mandatory_gap_fill_rate":  {"value": mandatory_fr,   "unit": "ratio", "higher_is_better": True},
            "pathway_efficiency_score": {"value": efficiency,     "unit": "gaps/10h", "higher_is_better": True},
            "redundancy_rate":          {"value": redundancy_r,   "unit": "ratio", "higher_is_better": False},
            "phase_balance_cv":         {"value": phase_balance,  "unit": "ratio", "higher_is_better": False},
            "prerequisite_validity":    {"value": prereq_valid,   "unit": "struct", "higher_is_better": True},
        },
        "training_savings": savings,
        "gap_summary": gap_analysis.get("gap_summary", {}),
        "interpretation": _interpret_scores(coverage, mandatory_fr, prereq_valid),
    }


def _interpret_scores(coverage: float, mandatory_fr: float, prereq_valid: dict) -> str:
    parts = []
    if coverage >= 0.90:
        parts.append("Excellent gap coverage (≥90% of gaps addressed).")
    elif coverage >= 0.70:
        parts.append("Good gap coverage, but some skills may not have matching courses.")
    else:
        parts.append("Low coverage — consider expanding the course catalog.")

    if mandatory_fr == 1.0:
        parts.append("All mandatory gaps are covered — the learner will reach role readiness.")
    elif mandatory_fr >= 0.8:
        parts.append("Most mandatory gaps covered, but a few critical skills remain unaddressed.")
    else:
        parts.append("WARNING: Several mandatory gaps lack course coverage. Role readiness is at risk.")

    if not prereq_valid["valid"]:
        parts.append(f"ALERT: {len(prereq_valid['violations'])} prerequisite ordering violation(s) detected.")

    return " ".join(parts)
