"""
roadmap_builder.py
-------------------
Generates the narrative week-by-week learning roadmap from prioritised gaps.
Weeks are allocated proportional to O*NET weight — the most critical skills
get dedicated weeks; lower-weight skills may be grouped.
"""

from __future__ import annotations

import pandas as pd

from config import LEARNING_META, LEARNING_META_DEFAULT
from gap_prioritizer import split_foundational_advanced


def build_roadmap(cand_eval_metrics: dict) -> pd.DataFrame:
    """
    Create a week-by-week roadmap DataFrame.

    Parameters
    ----------
    cand_eval_metrics : dict from evaluate_candidate() with 'Prioritized_Gaps' added

    Returns
    -------
    pd.DataFrame  columns: Week, Skill, ONET_Weight, Level, Objective, Success

    Notes
    -----
    The roadmap only generates a week for each unique skill gap — it never
    repeats a skill to pad the week count, and never inserts hardcoded
    fallback skills when a pool is empty. If a candidate has fewer gaps
    than the target week_count, the roadmap is shorter. That is honest.
    """
    gaps       = cand_eval_metrics.get("Prioritized_Gaps", [])
    week_count = cand_eval_metrics.get("Duration", 8)

    if not gaps:
        return pd.DataFrame(
            columns=["Week", "Skill", "ONET_Weight", "Level", "Objective", "Success"]
        )

    foundational, advanced = split_foundational_advanced(gaps)

    # Allocate weeks proportionally but never exceed what's actually available.
    # If one pool is empty, give all weeks to the other — no padding.
    first_half  = max(1, week_count // 2)
    second_half = week_count - first_half

    found_slots = min(first_half, len(foundational))
    adv_slots   = min(second_half, len(advanced))

    # If foundational pool ran short, give leftover slots to advanced (and v/v)
    found_leftover = first_half - found_slots
    adv_leftover   = second_half - adv_slots

    adv_slots   = min(adv_slots   + found_leftover, len(advanced))
    found_slots = min(found_slots + adv_leftover,   len(foundational))

    schedule = (
        foundational[:found_slots]
        + advanced[:adv_slots]
    )

    rows = []
    for i, skill_data in enumerate(schedule, start=1):
        skill = skill_data["Skill"]
        meta  = LEARNING_META.get(skill.lower(), LEARNING_META_DEFAULT)
        rows.append({
            "Week":        f"Week {i}",
            "Skill":       skill,
            "ONET_Weight": skill_data.get("ONET_Weight", 0.0),
            "Level":       skill_data.get("Level", 0),
            "Objective":   meta["Objective"],
            "Success":     meta["Success"],
        })

    return pd.DataFrame(rows)


def roadmap_to_text(roadmap_df: pd.DataFrame, candidate_id) -> str:
    lines = [
        f"{'='*65}",
        f"  Learning Roadmap  —  Candidate {candidate_id}",
        f"{'='*65}",
    ]
    for _, row in roadmap_df.iterrows():
        lines.append(
            f"\n{row['Week']}  ▸  {row['Skill']}"
            f"  (Level {row['Level']}, O*NET weight {row['ONET_Weight']:.2f})"
        )
        lines.append(f"  Objective : {row['Objective']}")
        lines.append(f"  Success   : {row['Success']}")
    lines.append(f"\n{'='*65}")
    return "\n".join(lines)