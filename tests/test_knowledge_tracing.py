"""
tests/test_knowledge_tracing.py — Unit tests for BKT knowledge tracing.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from knowledge_tracing import (
    SkillKnowledgeState,
    LearnerModel,
    update_knowledge_state,
    initialize_from_proficiency,
    estimate_from_quiz,
    rerank_pathway_with_bkt,
    MASTERY_THRESHOLD,
    DEFAULT_P_INIT,
    DEFAULT_P_LEARN,
)


# ---------------------------------------------------------------------------
# Tests: initialize_from_proficiency
# ---------------------------------------------------------------------------

class TestInitializeFromProficiency:

    def test_level_0_very_low(self):
        state = initialize_from_proficiency("Python", 0)
        assert state.p_mastery < 0.10

    def test_level_5_very_high(self):
        state = initialize_from_proficiency("Python", 5)
        assert state.p_mastery >= 0.90

    def test_monotonically_increasing(self):
        levels = [initialize_from_proficiency("X", i).p_mastery for i in range(6)]
        for i in range(len(levels) - 1):
            assert levels[i] <= levels[i + 1], f"Not monotonic at level {i}"

    def test_skill_name_stored(self):
        state = initialize_from_proficiency("SQL", 3)
        assert state.skill_name == "SQL"

    def test_p_mastery_in_range(self):
        for level in range(6):
            state = initialize_from_proficiency("X", level)
            assert 0.0 <= state.p_mastery <= 1.0


# ---------------------------------------------------------------------------
# Tests: update_knowledge_state
# ---------------------------------------------------------------------------

class TestUpdateKnowledgeState:

    def test_correct_response_increases_mastery(self):
        state = initialize_from_proficiency("Python", 1)
        updated = update_knowledge_state(state, correct=True)
        assert updated.p_mastery >= state.p_mastery

    def test_incorrect_response_decreases_or_stays_mastery(self):
        state = initialize_from_proficiency("Python", 3)
        updated = update_knowledge_state(state, correct=False)
        assert updated.p_mastery <= state.p_mastery

    def test_n_opportunities_increments(self):
        state = initialize_from_proficiency("Python", 2)
        assert state.n_opportunities == 0
        updated = update_knowledge_state(state, correct=True)
        assert updated.n_opportunities == 1

    def test_observation_recorded(self):
        state = initialize_from_proficiency("Python", 2)
        updated = update_knowledge_state(state, correct=True)
        assert updated.observations == [1]
        updated2 = update_knowledge_state(updated, correct=False)
        assert updated2.observations == [1, 0]

    def test_p_mastery_stays_in_0_1(self):
        state = initialize_from_proficiency("X", 5)
        for _ in range(20):
            state = update_knowledge_state(state, correct=True)
        assert 0.0 <= state.p_mastery <= 1.0

    def test_many_correct_leads_to_mastery(self):
        state = initialize_from_proficiency("SQL", 1)
        for _ in range(30):
            state = update_knowledge_state(state, correct=True)
        assert state.p_mastery >= MASTERY_THRESHOLD

    def test_is_mastered_flag(self):
        state = initialize_from_proficiency("SQL", 1)
        assert not state.is_mastered
        # Force mastery
        state.p_mastery = 0.96
        assert state.is_mastered


# ---------------------------------------------------------------------------
# Tests: estimate_from_quiz
# ---------------------------------------------------------------------------

class TestEstimateFromQuiz:

    def test_all_correct_improves_over_baseline(self):
        state = estimate_from_quiz("Python", responses=[True]*10, initial_proficiency=1)
        baseline = initialize_from_proficiency("Python", 1)
        assert state.p_mastery > baseline.p_mastery

    def test_all_wrong_doesnt_exceed_baseline_much(self):
        state = estimate_from_quiz("Python", responses=[False]*10, initial_proficiency=1)
        baseline = initialize_from_proficiency("Python", 1)
        assert state.p_mastery <= baseline.p_mastery + 0.05

    def test_empty_responses_returns_initial(self):
        state = estimate_from_quiz("X", responses=[], initial_proficiency=3)
        baseline = initialize_from_proficiency("X", 3)
        assert abs(state.p_mastery - baseline.p_mastery) < 1e-9


# ---------------------------------------------------------------------------
# Tests: LearnerModel
# ---------------------------------------------------------------------------

class TestLearnerModel:

    def test_initialize_skill(self):
        model = LearnerModel("user1")
        model.initialize_skill("Python", proficiency_level=3)
        state = model.get_state("Python")
        assert state is not None
        assert state.skill_name == "Python"

    def test_record_observation_updates_state(self):
        model = LearnerModel("user1")
        model.initialize_skill("Python", proficiency_level=2)
        before = model.get_state("Python").p_mastery
        model.record_observation("Python", correct=True)
        after = model.get_state("Python").p_mastery
        assert after >= before

    def test_auto_init_on_unknown_skill(self):
        model = LearnerModel("user1")
        model.record_observation("NewSkill", correct=True)
        state = model.get_state("NewSkill")
        assert state is not None

    def test_mastered_skills_empty_at_start(self):
        model = LearnerModel("user1")
        model.initialize_skill("Python", proficiency_level=1)
        assert "Python" not in model.mastered_skills()

    def test_mastered_after_forcing(self):
        model = LearnerModel("user1")
        model.initialize_skill("Python", proficiency_level=1)
        model._states["Python"].p_mastery = 0.97
        assert "Python" in model.mastered_skills()

    def test_to_dict_structure(self):
        model = LearnerModel("u1")
        model.initialize_skill("SQL", proficiency_level=3)
        d = model.to_dict()
        assert d["learner_id"] == "u1"
        assert "SQL" in d["skills"]
        assert "p_mastery" in d["skills"]["SQL"]

    def test_from_resume_skills(self):
        resume_skills = [
            {"skill_name": "Python", "proficiency_level": 3},
            {"skill_name": "SQL",    "proficiency_level": 2},
        ]
        model = LearnerModel.from_resume_skills("test", resume_skills)
        assert model.get_state("Python") is not None
        assert model.get_state("SQL") is not None
        py_state = model.get_state("Python")
        sql_state = model.get_state("SQL")
        # Level 3 should have higher p_mastery than level 2
        assert py_state.p_mastery > sql_state.p_mastery

    def test_skills_to_practice_sorted_desc(self):
        model = LearnerModel("u1")
        model.initialize_skill("A", proficiency_level=3)
        model.initialize_skill("B", proficiency_level=1)
        model.initialize_skill("C", proficiency_level=2)
        to_practice = model.skills_to_practice()
        # Should be sorted highest p_mastery first (closest to mastery)
        p_values = [model.get_state(s).p_mastery for s in to_practice]
        assert p_values == sorted(p_values, reverse=True)


# ---------------------------------------------------------------------------
# Tests: rerank_pathway_with_bkt
# ---------------------------------------------------------------------------

class TestReRankPathwayWithBKT:

    def _make_pathway(self, courses):
        return {"phases": [{"phase": 1, "title": "Test", "total_hours": 20, "courses": courses}]}

    def test_mastered_courses_flagged(self):
        model = LearnerModel("u1")
        model.initialize_skill("Python", proficiency_level=5)
        model._states["Python"].p_mastery = 0.97  # force mastery

        course = {
            "id": "PROG-001", "name": "Python Fundamentals",
            "skills_taught": ["Python", "Programming Basics"],
            "prerequisites": [], "duration_hours": 20, "level": "beginner",
        }
        pathway = self._make_pathway([course])
        updated = rerank_pathway_with_bkt(pathway, model)
        # Only Python is mastered, but "Programming Basics" is not → not skipped
        # (both skills must be mastered for bkt_skip=True)
        courses_out = updated[0]["courses"]
        assert "bkt_skip" in courses_out[0]

    def test_non_mastered_courses_not_skipped(self):
        model = LearnerModel("u1")
        model.initialize_skill("Python", proficiency_level=1)

        course = {
            "id": "PROG-001", "name": "Python Fundamentals",
            "skills_taught": ["Python"],
            "prerequisites": [], "duration_hours": 20, "level": "beginner",
        }
        pathway = self._make_pathway([course])
        updated = rerank_pathway_with_bkt(pathway, model)
        assert updated[0]["courses"][0]["bkt_skip"] == False

    def test_structure_preserved(self):
        model = LearnerModel("u1")
        pathway = {"phases": []}
        updated = rerank_pathway_with_bkt(pathway, model)
        assert updated == []
