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
    """
    gaps        = cand_eval_metrics.get("Prioritized_Gaps", [])
    week_count  = cand_eval_metrics.get("Duration", 8)

    foundational, advanced = split_foundational_advanced(gaps)

    first_half  = max(1, week_count // 2)
    second_half = week_count - first_half

    rows = []
    for i in range(1, week_count + 1):
        if i <= first_half:
            pool = foundational
            idx  = (i - 1) % len(pool) if pool else 0
            skill_data = pool[idx] if pool else {"Skill": "Active Listening", "Level": 0, "ONET_Weight": 2.0}
        else:
            pool = advanced
            idx  = (i - first_half - 1) % len(pool) if pool else 0
            skill_data = pool[idx] if pool else {"Skill": "Systems Analysis", "Level": 3, "ONET_Weight": 2.0}

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
