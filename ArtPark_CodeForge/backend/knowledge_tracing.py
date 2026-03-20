"""
knowledge_tracing.py — Bayesian Knowledge Tracing (BKT) Implementation

BKT is a Hidden Markov Model that estimates the probability a learner
has mastered a skill, given their observed performance history.

State: binary (mastered / not-mastered)
Observation: binary (correct / incorrect response)

Parameters per skill:
  p_init   — P(mastered at start)
  p_learn  — P(transition: not-mastered → mastered) per opportunity
  p_slip   — P(wrong answer | mastered)
  p_guess  — P(correct answer | not-mastered)

This module uses default BKT parameters derived from literature
(Corbett & Anderson 1994) as a baseline. Parameters can be fitted
per-skill if performance log data is available.

In AURORA, BKT is used to:
  1. Estimate current mastery per skill from resume evidence + activity
  2. Re-order the learning pathway as the learner progresses
  3. Flag when a skill is "mastered" to skip remaining modules
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Default BKT parameters (literature baseline)
# ---------------------------------------------------------------------------
DEFAULT_P_INIT  = 0.10   # 10% chance of already knowing a skill cold
DEFAULT_P_LEARN = 0.30   # 30% chance of learning per practice opportunity
DEFAULT_P_SLIP  = 0.10   # 10% chance of getting it wrong even if mastered
DEFAULT_P_GUESS = 0.20   # 20% chance of getting it right by guessing

MASTERY_THRESHOLD = 0.95  # P(mastered) ≥ 0.95 → skill considered mastered


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SkillKnowledgeState:
    """Current BKT state for one learner × one skill."""
    skill_name: str
    p_mastery: float = DEFAULT_P_INIT       # P(L_n) — current mastery belief
    p_init: float    = DEFAULT_P_INIT
    p_learn: float   = DEFAULT_P_LEARN
    p_slip: float    = DEFAULT_P_SLIP
    p_guess: float   = DEFAULT_P_GUESS
    n_opportunities: int = 0                 # number of practice events seen
    observations: list[int] = field(default_factory=list)  # 1=correct, 0=incorrect

    @property
    def is_mastered(self) -> bool:
        return self.p_mastery >= MASTERY_THRESHOLD

    @property
    def mastery_pct(self) -> float:
        return round(self.p_mastery * 100, 1)


# ---------------------------------------------------------------------------
# Core BKT update
# ---------------------------------------------------------------------------

def update_knowledge_state(
    state: SkillKnowledgeState,
    correct: bool,
) -> SkillKnowledgeState:
    """
    Perform one BKT update step given a new observation.

    BKT forward algorithm:
      1. P(L_n | obs) = P(obs | L_n) * P(L_n) / P(obs)
      2. P(L_{n+1}) = P(L_n | obs) + (1 - P(L_n | obs)) * P_learn
    """
    p_l = state.p_mastery
    p_slip  = state.p_slip
    p_guess = state.p_guess
    p_learn = state.p_learn

    obs = 1 if correct else 0

    # Step 1: P(correct | L), P(correct | ¬L)
    if obs == 1:
        p_obs_given_mastered     = 1.0 - p_slip   # mastered and didn't slip
        p_obs_given_not_mastered = p_guess          # not mastered but guessed
    else:
        p_obs_given_mastered     = p_slip           # mastered but slipped
        p_obs_given_not_mastered = 1.0 - p_guess    # not mastered, wrong (expected)

    # Step 2: Bayesian update — P(L | obs)
    numerator   = p_obs_given_mastered * p_l
    denominator = numerator + p_obs_given_not_mastered * (1.0 - p_l)

    if denominator < 1e-10:
        p_l_given_obs = p_l  # numerical safety
    else:
        p_l_given_obs = numerator / denominator

    # Step 3: Learning transition
    p_l_next = p_l_given_obs + (1.0 - p_l_given_obs) * p_learn
    p_l_next = max(0.0, min(1.0, p_l_next))  # clamp to [0, 1]

    # Create updated state
    new_state = SkillKnowledgeState(
        skill_name      = state.skill_name,
        p_mastery       = p_l_next,
        p_init          = state.p_init,
        p_learn         = state.p_learn,
        p_slip          = state.p_slip,
        p_guess         = state.p_guess,
        n_opportunities = state.n_opportunities + 1,
        observations    = state.observations + [obs],
    )
    return new_state


# ---------------------------------------------------------------------------
# Resume-based initialization
# ---------------------------------------------------------------------------

def initialize_from_proficiency(
    skill_name: str,
    proficiency_level: int,  # 1–5 scale
) -> SkillKnowledgeState:
    """
    Bootstrap a BKT state from a resume-inferred proficiency level.

    Maps the 5-point proficiency scale to an initial P(mastery):
      1 (Aware)        → 0.10
      2 (Beginner)     → 0.30
      3 (Intermediate) → 0.60
      4 (Advanced)     → 0.85
      5 (Expert)       → 0.97
    """
    proficiency_to_p_mastery = {
        0: 0.02,
        1: 0.10,
        2: 0.30,
        3: 0.60,
        4: 0.85,
        5: 0.97,
    }
    p_init = proficiency_to_p_mastery.get(max(0, min(5, proficiency_level)), 0.10)
    return SkillKnowledgeState(
        skill_name = skill_name,
        p_mastery  = p_init,
        p_init     = p_init,
    )


# ---------------------------------------------------------------------------
# Learner model: tracks all skills for one learner
# ---------------------------------------------------------------------------

class LearnerModel:
    """
    Maintains BKT states for all skills of one learner.
    Can be persisted to/from a plain dict for API serialization.
    """

    def __init__(self, learner_id: str):
        self.learner_id = learner_id
        self._states: dict[str, SkillKnowledgeState] = {}

    def initialize_skill(
        self,
        skill_name: str,
        proficiency_level: int = 0,
        p_learn: Optional[float] = None,
    ) -> None:
        state = initialize_from_proficiency(skill_name, proficiency_level)
        if p_learn is not None:
            state.p_learn = p_learn
        self._states[skill_name] = state

    def record_observation(self, skill_name: str, correct: bool) -> SkillKnowledgeState:
        """Update knowledge state for a skill after one practice event."""
        if skill_name not in self._states:
            self.initialize_skill(skill_name)
        self._states[skill_name] = update_knowledge_state(self._states[skill_name], correct)
        return self._states[skill_name]

    def get_state(self, skill_name: str) -> Optional[SkillKnowledgeState]:
        return self._states.get(skill_name)

    def mastered_skills(self) -> list[str]:
        return [s for s, st in self._states.items() if st.is_mastered]

    def skills_to_practice(self, threshold: float = MASTERY_THRESHOLD) -> list[str]:
        """Skills not yet mastered, sorted by how close to mastery (desc)."""
        not_mastered = [
            (s, st.p_mastery) for s, st in self._states.items()
            if st.p_mastery < threshold
        ]
        return [s for s, _ in sorted(not_mastered, key=lambda x: -x[1])]

    def to_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "skills": {
                name: {
                    "p_mastery":      round(st.p_mastery, 4),
                    "n_opportunities": st.n_opportunities,
                    "is_mastered":    st.is_mastered,
                    "mastery_pct":    st.mastery_pct,
                }
                for name, st in self._states.items()
            },
        }

    @classmethod
    def from_resume_skills(cls, learner_id: str, resume_skills: list[dict]) -> "LearnerModel":
        """
        Build a LearnerModel directly from extracted resume skills.
        resume_skills: list of {skill_name, proficiency_level, ...}
        """
        model = cls(learner_id)
        for skill in resume_skills:
            model.initialize_skill(
                skill_name       = skill["skill_name"],
                proficiency_level = int(skill.get("proficiency_level", 0)),
            )
        return model


# ---------------------------------------------------------------------------
# Pathway adaptation: re-score courses based on current knowledge state
# ---------------------------------------------------------------------------

def rerank_pathway_with_bkt(
    pathway_phases: list[dict],
    learner_model: LearnerModel,
) -> list[dict]:
    """
    Adjust pathway: skip or deprioritize courses whose skills are already mastered.
    Returns a new list of phases with `bkt_skip=True` on mastered courses.
    """
    mastered = set(s.lower() for s in learner_model.mastered_skills())
    updated_phases = []

    for phase in pathway_phases:
        updated_courses = []
        for course in phase.get("courses", []):
            taught = {s.lower() for s in course.get("skills_taught", [])}
            # Skip if ALL skills taught by this course are mastered
            all_mastered = bool(taught) and taught.issubset(mastered)
            course_copy = dict(course)
            course_copy["bkt_skip"] = all_mastered
            if all_mastered:
                course_copy["skip_reason"] = (
                    "All skills taught by this course are already mastered "
                    "according to your current knowledge state."
                )
            updated_courses.append(course_copy)
        phase_copy = dict(phase)
        phase_copy["courses"] = updated_courses
        updated_phases.append(phase_copy)

    return updated_phases


# ---------------------------------------------------------------------------
# Diagnostic: estimate P(mastery) from a short quiz sequence
# ---------------------------------------------------------------------------

def estimate_from_quiz(
    skill_name: str,
    responses: list[bool],
    initial_proficiency: int = 0,
) -> SkillKnowledgeState:
    """
    Given a sequence of quiz responses (True=correct), return final BKT state.
    Useful for diagnostic assessments at onboarding.
    """
    state = initialize_from_proficiency(skill_name, initial_proficiency)
    for correct in responses:
        state = update_knowledge_state(state, correct)
    return state


# ---------------------------------------------------------------------------
# Quick sanity check (run as __main__)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== BKT Sanity Check ===")

    # Simulate a learner practicing Python
    state = initialize_from_proficiency("Python", proficiency_level=2)
    print(f"Initial P(mastery): {state.mastery_pct}%")

    responses = [True, False, True, True, True, True, True, True]
    for i, correct in enumerate(responses):
        state = update_knowledge_state(state, correct)
        print(f"  Obs {i+1} ({'✓' if correct else '✗'}) → P(mastery): {state.mastery_pct}%  {'[MASTERED]' if state.is_mastered else ''}")

    print()

    # Test LearnerModel
    model = LearnerModel("test_user")
    model.initialize_skill("Python", proficiency_level=2)
    model.initialize_skill("SQL",    proficiency_level=3)
    model.initialize_skill("PyTorch", proficiency_level=0)

    print(f"Mastered skills: {model.mastered_skills()}")
    print(f"To practice:     {model.skills_to_practice()}")
    print(f"Learner dict:    {model.to_dict()}")
