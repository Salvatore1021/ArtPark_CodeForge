"""
tests/test_metrics.py — Unit tests for the metrics/evaluation module.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from metrics import (
    skill_gap_coverage_rate,
    mandatory_gap_fill_rate,
    redundancy_rate,
    pathway_efficiency_score,
    prerequisite_validity,
    phase_balance_score,
    redundancy_hours_saved,
    evaluate_pathway,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GAP_ANALYSIS = {
    "gaps": [
        {"skill_name": "Python",      "is_mandatory": True},
        {"skill_name": "PyTorch",     "is_mandatory": True},
        {"skill_name": "SQL",         "is_mandatory": True},
        {"skill_name": "Kubernetes",  "is_mandatory": False},
    ],
    "matched_skills": [],
    "gap_summary": {"total_required": 4, "total_gaps": 4, "critical_gaps": 3, "coverage_pct": 0.0},
    "reasoning": "4 gaps.",
}

PATHWAY_FULL_COVERAGE = {
    "estimated_total_hours": 60,
    "estimated_weeks": 3,
    "phases": [
        {
            "phase": 1,
            "title": "Foundations",
            "total_hours": 30,
            "courses": [
                {
                    "id": "PROG-001",
                    "name": "Python Fundamentals",
                    "skills_taught": ["Python", "Programming Basics"],
                    "prerequisites": [],
                    "duration_hours": 20,
                    "is_prerequisite_only": False,
                },
                {
                    "id": "PROG-003",
                    "name": "SQL",
                    "skills_taught": ["SQL", "Database Design"],
                    "prerequisites": [],
                    "duration_hours": 10,
                    "is_prerequisite_only": False,
                },
            ],
        },
        {
            "phase": 2,
            "title": "Core",
            "total_hours": 30,
            "courses": [
                {
                    "id": "ML-002",
                    "name": "Deep Learning",
                    "skills_taught": ["PyTorch", "Deep Learning"],
                    "prerequisites": ["PROG-001"],
                    "duration_hours": 20,
                    "is_prerequisite_only": False,
                },
                {
                    "id": "CLOUD-004",
                    "name": "Kubernetes",
                    "skills_taught": ["Kubernetes"],
                    "prerequisites": [],
                    "duration_hours": 10,
                    "is_prerequisite_only": True,  # prereq-only flag
                },
            ],
        },
    ],
}

PATHWAY_PARTIAL = {
    "estimated_total_hours": 20,
    "estimated_weeks": 1,
    "phases": [
        {
            "phase": 1,
            "title": "Foundations",
            "total_hours": 20,
            "courses": [
                {
                    "id": "PROG-001",
                    "name": "Python Fundamentals",
                    "skills_taught": ["Python"],
                    "prerequisites": [],
                    "duration_hours": 20,
                    "is_prerequisite_only": False,
                }
            ],
        }
    ],
}

SAMPLE_CATALOG = [
    {"id": "A", "duration_hours": 20},
    {"id": "B", "duration_hours": 30},
    {"id": "C", "duration_hours": 50},
]


# ---------------------------------------------------------------------------
# Tests: skill_gap_coverage_rate
# ---------------------------------------------------------------------------

class TestSkillGapCoverageRate:

    def test_full_coverage(self):
        rate = skill_gap_coverage_rate(GAP_ANALYSIS, PATHWAY_FULL_COVERAGE)
        assert rate == 1.0

    def test_partial_coverage(self):
        rate = skill_gap_coverage_rate(GAP_ANALYSIS, PATHWAY_PARTIAL)
        assert 0.0 < rate < 1.0

    def test_no_gaps_is_100pct(self):
        no_gaps = {"gaps": [], "matched_skills": [], "gap_summary": {}}
        rate = skill_gap_coverage_rate(no_gaps, PATHWAY_FULL_COVERAGE)
        assert rate == 1.0

    def test_empty_pathway(self):
        rate = skill_gap_coverage_rate(GAP_ANALYSIS, {"phases": []})
        assert rate == 0.0

    def test_range(self):
        for p in [PATHWAY_FULL_COVERAGE, PATHWAY_PARTIAL, {"phases": []}]:
            rate = skill_gap_coverage_rate(GAP_ANALYSIS, p)
            assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# Tests: mandatory_gap_fill_rate
# ---------------------------------------------------------------------------

class TestMandatoryGapFillRate:

    def test_full_mandatory_fill(self):
        rate = mandatory_gap_fill_rate(GAP_ANALYSIS, PATHWAY_FULL_COVERAGE)
        assert rate == 1.0

    def test_partial_mandatory_fill(self):
        rate = mandatory_gap_fill_rate(GAP_ANALYSIS, PATHWAY_PARTIAL)
        # Only Python is covered out of 3 mandatory (Python, PyTorch, SQL)
        assert rate == pytest.approx(1/3, abs=0.02)

    def test_no_mandatory_gaps_returns_1(self):
        no_mandatory = {
            "gaps": [{"skill_name": "X", "is_mandatory": False}],
            "gap_summary": {},
        }
        rate = mandatory_gap_fill_rate(no_mandatory, PATHWAY_PARTIAL)
        assert rate == 1.0


# ---------------------------------------------------------------------------
# Tests: redundancy_rate
# ---------------------------------------------------------------------------

class TestRedundancyRate:

    def test_one_prereq_only_course(self):
        rate = redundancy_rate(PATHWAY_FULL_COVERAGE)
        # 1 out of 4 courses is prereq_only
        assert rate == pytest.approx(0.25, abs=0.01)

    def test_no_prereq_courses(self):
        rate = redundancy_rate(PATHWAY_PARTIAL)
        assert rate == 0.0

    def test_empty_pathway(self):
        rate = redundancy_rate({"phases": []})
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Tests: prerequisite_validity
# ---------------------------------------------------------------------------

class TestPrerequisiteValidity:

    def test_valid_pathway(self):
        result = prerequisite_validity(PATHWAY_FULL_COVERAGE)
        assert result["valid"] is True
        assert len(result["violations"]) == 0

    def test_violation_detected(self):
        bad_pathway = {
            "phases": [
                {
                    "phase": 1,
                    "total_hours": 20,
                    "courses": [
                        {"id": "ML-001", "prerequisites": ["PROG-001"], "duration_hours": 20},
                        {"id": "PROG-001", "prerequisites": [], "duration_hours": 20},
                    ],
                }
            ]
        }
        # ML-001 appears at position 0 but its prereq PROG-001 is at position 1 — violation!
        result = prerequisite_validity(bad_pathway)
        assert result["valid"] is False
        assert len(result["violations"]) > 0


# ---------------------------------------------------------------------------
# Tests: phase_balance_score
# ---------------------------------------------------------------------------

class TestPhaseBalanceScore:

    def test_perfectly_balanced(self):
        equal_pathway = {
            "phases": [
                {"total_hours": 20, "courses": []},
                {"total_hours": 20, "courses": []},
                {"total_hours": 20, "courses": []},
            ]
        }
        score = phase_balance_score(equal_pathway)
        assert score == 0.0

    def test_imbalanced(self):
        unequal = {
            "phases": [
                {"total_hours": 5, "courses": []},
                {"total_hours": 60, "courses": []},
            ]
        }
        score = phase_balance_score(unequal)
        assert score > 0.3  # high imbalance

    def test_single_phase_zero(self):
        score = phase_balance_score({"phases": [{"total_hours": 20, "courses": []}]})
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: redundancy_hours_saved
# ---------------------------------------------------------------------------

class TestRedundancyHoursSaved:

    def test_savings_positive(self):
        result = redundancy_hours_saved(GAP_ANALYSIS, PATHWAY_PARTIAL, SAMPLE_CATALOG)
        assert result["hours_saved"] > 0
        assert result["pathway_hours"] < result["full_catalog_hours"]

    def test_pct_saved_range(self):
        result = redundancy_hours_saved(GAP_ANALYSIS, PATHWAY_PARTIAL, SAMPLE_CATALOG)
        assert 0.0 <= result["percent_saved"] <= 100.0


# ---------------------------------------------------------------------------
# Tests: evaluate_pathway (composite)
# ---------------------------------------------------------------------------

class TestEvaluatePathway:

    def test_returns_composite_score(self):
        result = evaluate_pathway(GAP_ANALYSIS, PATHWAY_FULL_COVERAGE, SAMPLE_CATALOG)
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 100.0

    def test_returns_all_metric_keys(self):
        result = evaluate_pathway(GAP_ANALYSIS, PATHWAY_FULL_COVERAGE, SAMPLE_CATALOG)
        expected = {
            "skill_gap_coverage_rate",
            "mandatory_gap_fill_rate",
            "pathway_efficiency_score",
            "redundancy_rate",
            "phase_balance_cv",
            "prerequisite_validity",
        }
        assert expected == set(result["metrics"].keys())

    def test_interpretation_is_string(self):
        result = evaluate_pathway(GAP_ANALYSIS, PATHWAY_FULL_COVERAGE, SAMPLE_CATALOG)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 10

    def test_full_coverage_higher_composite_than_partial(self):
        full = evaluate_pathway(GAP_ANALYSIS, PATHWAY_FULL_COVERAGE, SAMPLE_CATALOG)
        partial = evaluate_pathway(GAP_ANALYSIS, PATHWAY_PARTIAL, SAMPLE_CATALOG)
        assert full["composite_score"] >= partial["composite_score"]
