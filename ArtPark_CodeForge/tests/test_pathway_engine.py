"""
tests/test_pathway_engine.py — Unit tests for the adaptive pathway engine.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from pathway_engine import (
    generate_learning_pathway,
    load_course_catalog,
    _find_courses_for_gap,
    _topological_sort_with_priority,
    _build_graph,
    _group_into_phases,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GAPS = {
    "gaps": [
        {
            "skill_name": "Python",
            "canonical_name": "Python",
            "category": "programming",
            "required_level": 4,
            "current_level": 0,
            "gap_magnitude": 0.8,
            "is_mandatory": True,
            "priority_score": 6.4,
            "status": "missing",
            "evidence": "",
            "context": "core language",
        },
        {
            "skill_name": "Machine Learning",
            "canonical_name": "Machine Learning",
            "category": "data-science",
            "required_level": 3,
            "current_level": 0,
            "gap_magnitude": 0.6,
            "is_mandatory": True,
            "priority_score": 3.6,
            "status": "missing",
            "evidence": "",
            "context": "core skill",
        },
    ],
    "matched_skills": [],
    "gap_summary": {"total_required": 2, "total_gaps": 2, "critical_gaps": 2, "coverage_pct": 0.0},
    "reasoning": "2 gaps identified.",
}

SAMPLE_JD = {
    "job_title": "Data Scientist",
    "seniority": "mid",
    "skills": [
        {"skill_name": "Python", "required_level": 4, "is_mandatory": True, "category": "programming", "context": ""},
        {"skill_name": "Machine Learning", "required_level": 3, "is_mandatory": True, "category": "data-science", "context": ""},
    ]
}

NO_GAPS = {
    "gaps": [],
    "matched_skills": [{"skill_name": "Python", "status": "met"}],
    "gap_summary": {"total_required": 1, "total_gaps": 0, "critical_gaps": 0, "coverage_pct": 100.0},
    "reasoning": "No gaps.",
}


# ---------------------------------------------------------------------------
# Tests: load_course_catalog
# ---------------------------------------------------------------------------

class TestLoadCourseCatalog:

    def test_returns_list(self):
        catalog = load_course_catalog()
        assert isinstance(catalog, list)

    def test_has_courses(self):
        catalog = load_course_catalog()
        assert len(catalog) > 0

    def test_each_course_has_required_fields(self):
        catalog = load_course_catalog()
        required_fields = {"id", "name", "skills_taught", "prerequisites", "duration_hours", "level"}
        for course in catalog:
            for field in required_fields:
                assert field in course, f"Course {course.get('id', '?')} missing field '{field}'"

    def test_prerequisites_are_valid_ids(self):
        """Every prerequisite ID must refer to a real course in the catalog."""
        catalog = load_course_catalog()
        all_ids = {c["id"] for c in catalog}
        for course in catalog:
            for prereq in course.get("prerequisites", []):
                assert prereq in all_ids, f"Course {course['id']} has unknown prereq '{prereq}'"


# ---------------------------------------------------------------------------
# Tests: _find_courses_for_gap
# ---------------------------------------------------------------------------

class TestFindCoursesForGap:

    def test_finds_python_course(self):
        catalog = load_course_catalog()
        gap = {"skill_name": "Python", "canonical_name": "Python"}
        matched = _find_courses_for_gap(gap, catalog)
        assert len(matched) > 0
        ids = [c["id"] for c in matched]
        assert "PROG-001" in ids  # Python Fundamentals

    def test_no_match_for_unknown_skill(self):
        catalog = load_course_catalog()
        gap = {"skill_name": "QuantumFortranXYZ", "canonical_name": "QuantumFortranXYZ"}
        matched = _find_courses_for_gap(gap, catalog)
        assert len(matched) == 0

    def test_ml_course_found(self):
        catalog = load_course_catalog()
        gap = {"skill_name": "Machine Learning", "canonical_name": "Machine Learning"}
        matched = _find_courses_for_gap(gap, catalog)
        assert any(c["id"] == "ML-001" for c in matched)


# ---------------------------------------------------------------------------
# Tests: generate_learning_pathway
# ---------------------------------------------------------------------------

class TestGenerateLearningPathway:

    def test_no_gaps_returns_empty_pathway(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(NO_GAPS, {"job_title": "Dev", "seniority": "mid"}, catalog)
        assert pathway["estimated_total_hours"] == 0
        assert pathway["phases"] == []
        assert pathway["gap_coverage"] == 100.0

    def test_returns_required_keys(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        required = {"job_title", "seniority", "estimated_total_hours", "estimated_weeks", "phases", "gap_coverage", "algorithm", "reasoning_summary"}
        for key in required:
            assert key in pathway, f"Missing key: {key}"

    def test_phases_is_list(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        assert isinstance(pathway["phases"], list)

    def test_each_phase_has_courses(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        for phase in pathway["phases"]:
            assert "courses" in phase
            assert len(phase["courses"]) > 0

    def test_no_duplicate_courses(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        all_course_ids = []
        for phase in pathway["phases"]:
            for course in phase["courses"]:
                all_course_ids.append(course["id"])
        assert len(all_course_ids) == len(set(all_course_ids)), "Duplicate courses in pathway"

    def test_prerequisite_ordering_respected(self):
        """For any two courses A and B where A is a prereq of B, A must appear before B."""
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        all_courses = []
        for phase in pathway["phases"]:
            all_courses.extend(phase["courses"])

        position = {c["id"]: i for i, c in enumerate(all_courses)}
        for course in all_courses:
            for prereq_id in course.get("prerequisites", []):
                if prereq_id in position:
                    assert position[prereq_id] < position[course["id"]], (
                        f"Ordering violation: {prereq_id} (pos {position[prereq_id]}) "
                        f"should come before {course['id']} (pos {position[course['id']]})"
                    )

    def test_courses_have_reasoning(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        for phase in pathway["phases"]:
            for course in phase["courses"]:
                assert "reasoning" in course
                assert isinstance(course["reasoning"], str)
                assert len(course["reasoning"]) > 5

    def test_estimated_hours_positive(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        assert pathway["estimated_total_hours"] > 0

    def test_estimated_weeks_positive(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        assert pathway["estimated_weeks"] >= 1

    def test_gap_coverage_in_range(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        assert 0.0 <= pathway["gap_coverage"] <= 100.0

    def test_all_courses_from_catalog(self):
        """GROUNDING CHECK: Every recommended course must exist in the catalog."""
        catalog = load_course_catalog()
        catalog_ids = {c["id"] for c in catalog}
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        for phase in pathway["phases"]:
            for course in phase["courses"]:
                assert course["id"] in catalog_ids, (
                    f"Hallucinated course '{course['id']}' not in catalog!"
                )

    def test_job_title_preserved(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        assert pathway["job_title"] == SAMPLE_JD["job_title"]

    def test_phase_hours_sums_to_total(self):
        catalog = load_course_catalog()
        pathway = generate_learning_pathway(SAMPLE_GAPS, SAMPLE_JD, catalog)
        phase_total = sum(p["total_hours"] for p in pathway["phases"])
        assert phase_total == pathway["estimated_total_hours"]
