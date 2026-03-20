"""
tests/test_gap_analyzer.py — Unit tests for the gap analyzer module.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from gap_analyzer import compute_skill_gap, _normalize_skill_name, _resolve_canonical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RESUME_DATA_STRONG = {
    "candidate_name": "Alice Engineer",
    "years_total_experience": 5,
    "current_role": "Senior Data Scientist",
    "skills": [
        {"skill_name": "Python",         "proficiency_level": 4, "category": "programming", "evidence": "5 years Python"},
        {"skill_name": "Machine Learning","proficiency_level": 4, "category": "data-science", "evidence": "led ML projects"},
        {"skill_name": "SQL",            "proficiency_level": 3, "category": "data-engineering", "evidence": "daily SQL use"},
        {"skill_name": "Scikit-learn",   "proficiency_level": 3, "category": "data-science", "evidence": "sklearn experience"},
        {"skill_name": "Docker",         "proficiency_level": 2, "category": "cloud-devops", "evidence": "basic Docker"},
    ]
}

RESUME_DATA_JUNIOR = {
    "candidate_name": "Bob Newgrad",
    "years_total_experience": 0,
    "current_role": None,
    "skills": [
        {"skill_name": "Python", "proficiency_level": 2, "category": "programming", "evidence": "coursework"},
    ]
}

JD_DATA_ML_ENGINEER = {
    "job_title": "Machine Learning Engineer",
    "department": "AI Platform",
    "seniority": "mid",
    "skills": [
        {"skill_name": "Python",     "required_level": 4, "is_mandatory": True,  "category": "programming", "context": "core language"},
        {"skill_name": "PyTorch",    "required_level": 4, "is_mandatory": True,  "category": "data-science", "context": "DL framework"},
        {"skill_name": "SQL",        "required_level": 3, "is_mandatory": True,  "category": "data-engineering", "context": "data querying"},
        {"skill_name": "Docker",     "required_level": 3, "is_mandatory": True,  "category": "cloud-devops", "context": "containerization"},
        {"skill_name": "MLOps",      "required_level": 3, "is_mandatory": False, "category": "data-science", "context": "nice to have"},
        {"skill_name": "Kubernetes", "required_level": 2, "is_mandatory": False, "category": "cloud-devops", "context": "preferred"},
    ]
}


# ---------------------------------------------------------------------------
# Tests: _normalize_skill_name
# ---------------------------------------------------------------------------

class TestNormalizeSkillName:
    def test_basic_lowercase(self):
        assert _normalize_skill_name("Python") == "python"

    def test_strip_whitespace(self):
        assert _normalize_skill_name("  SQL  ") == "sql"

    def test_collapse_inner_whitespace(self):
        assert _normalize_skill_name("Machine  Learning") == "machine learning"

    def test_already_normalized(self):
        assert _normalize_skill_name("pytorch") == "pytorch"


# ---------------------------------------------------------------------------
# Tests: compute_skill_gap
# ---------------------------------------------------------------------------

class TestComputeSkillGap:

    def test_returns_required_keys(self):
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        assert "gaps" in result
        assert "matched_skills" in result
        assert "gap_summary" in result
        assert "reasoning" in result

    def test_python_met(self):
        """Alice has Python level 4, JD requires 4 → should be matched, not a gap."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        gap_names = [g["skill_name"] for g in result["gaps"]]
        matched_names = [m["skill_name"] for m in result["matched_skills"]]
        assert "Python" not in gap_names
        assert "Python" in matched_names

    def test_pytorch_missing(self):
        """Alice has no PyTorch → should be a gap with status=missing."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        pytorch_gap = next((g for g in result["gaps"] if g["skill_name"] == "PyTorch"), None)
        assert pytorch_gap is not None
        assert pytorch_gap["status"] == "missing"
        assert pytorch_gap["current_level"] == 0

    def test_docker_below_required(self):
        """Alice has Docker level 2, JD requires 3 → below_required."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        docker_gap = next((g for g in result["gaps"] if g["skill_name"] == "Docker"), None)
        assert docker_gap is not None
        assert docker_gap["status"] == "below_required"
        assert docker_gap["current_level"] == 2
        assert docker_gap["required_level"] == 3

    def test_mandatory_gaps_have_higher_priority(self):
        """Mandatory gaps should have priority_score > optional gaps of same magnitude."""
        result = compute_skill_gap(RESUME_DATA_JUNIOR, JD_DATA_ML_ENGINEER)
        mandatory_scores   = [g["priority_score"] for g in result["gaps"] if g["is_mandatory"]]
        optional_scores    = [g["priority_score"] for g in result["gaps"] if not g["is_mandatory"]]
        if mandatory_scores and optional_scores:
            assert max(optional_scores) <= max(mandatory_scores)

    def test_gaps_sorted_by_priority_descending(self):
        """Gaps list must be sorted highest priority first."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        scores = [g["priority_score"] for g in result["gaps"]]
        assert scores == sorted(scores, reverse=True)

    def test_gap_magnitude_range(self):
        """gap_magnitude must be in [0.0, 1.0]."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        for g in result["gaps"]:
            assert 0.0 <= g["gap_magnitude"] <= 1.0, f"Out of range: {g['skill_name']} = {g['gap_magnitude']}"

    def test_coverage_pct_range(self):
        """coverage_pct must be between 0 and 100."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        pct = result["gap_summary"]["coverage_pct"]
        assert 0.0 <= pct <= 100.0

    def test_empty_resume(self):
        """Empty resume → all JD skills should be missing gaps."""
        empty_resume = {"skills": []}
        result = compute_skill_gap(empty_resume, JD_DATA_ML_ENGINEER)
        assert result["gap_summary"]["total_gaps"] == len(JD_DATA_ML_ENGINEER["skills"])
        for g in result["gaps"]:
            assert g["status"] == "missing"
            assert g["current_level"] == 0

    def test_empty_jd(self):
        """Empty JD → no gaps, no matched skills."""
        empty_jd = {"skills": []}
        result = compute_skill_gap(RESUME_DATA_STRONG, empty_jd)
        assert len(result["gaps"]) == 0
        assert len(result["matched_skills"]) == 0
        assert result["gap_summary"]["total_required"] == 0

    def test_total_counts_consistent(self):
        """gaps + matched == total_required."""
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        s = result["gap_summary"]
        assert s["total_gaps"] + len(result["matched_skills"]) == s["total_required"]

    def test_reasoning_is_nonempty_string(self):
        result = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 10

    def test_junior_candidate_many_gaps(self):
        """Junior candidate should have more gaps than senior."""
        result_senior = compute_skill_gap(RESUME_DATA_STRONG, JD_DATA_ML_ENGINEER)
        result_junior = compute_skill_gap(RESUME_DATA_JUNIOR, JD_DATA_ML_ENGINEER)
        assert result_junior["gap_summary"]["total_gaps"] >= result_senior["gap_summary"]["total_gaps"]

    def test_exceeds_status(self):
        """If candidate has level > required, status should be 'exceeds'."""
        jd_easy = {
            "skills": [
                {"skill_name": "Python", "required_level": 2, "is_mandatory": True, "category": "programming", "context": ""}
            ]
        }
        result = compute_skill_gap(RESUME_DATA_STRONG, jd_easy)
        # Python level 4 vs required 2 → exceeds
        exceeds = next((m for m in result["matched_skills"] if m["skill_name"] == "Python"), None)
        assert exceeds is not None
        assert exceeds["status"] == "exceeds"
