"""
roadmap_builder.py
------------------
Generates the narrative week-by-week learning roadmap from prioritised gaps,
taking the pathway depth (Fundamental vs. Advanced) into account.
"""

from __future__ import annotations

import pandas as pd

from config import LEARNING_META, LEARNING_META_DEFAULT
from gap_prioritizer import split_foundational_advanced


def build_roadmap(
    cand_eval_metrics: dict,
) -> pd.DataFrame:
    """
    Create a week-by-week roadmap DataFrame from the candidate's
    evaluation metrics (output of candidate_evaluator.evaluate_candidate).

    Parameters
    ----------
    cand_eval_metrics : dict returned by evaluate_candidate(), must contain
                        'Prioritized_Gaps', 'Duration', and 'Pathway_Depth'

    Returns
    -------
    pd.DataFrame with columns: Week, Skill, Objective, Success, Level
    """
    gaps       = cand_eval_metrics.get("Prioritized_Gaps", [])
    week_count = cand_eval_metrics.get("Duration", 8)

    foundational, advanced = split_foundational_advanced(gaps)

    # ── Half-and-half split ──────────────────────────────────────────────────
    first_half  = max(1, week_count // 2)
    second_half = week_count - first_half

    roadmap_rows = []

    for i in range(1, week_count + 1):
        if i <= first_half:
            pool = foundational
            idx  = (i - 1) % len(pool) if pool else 0
            skill_data = pool[idx] if pool else {"Skill": "Business Ethics", "Level": 0}
        else:
            pool = advanced
            idx  = (i - first_half - 1) % len(pool) if pool else 0
            skill_data = pool[idx] if pool else {"Skill": "Strategic Management", "Level": 3}

        skill_name = skill_data["Skill"]
        meta       = LEARNING_META.get(skill_name, LEARNING_META_DEFAULT)

        roadmap_rows.append({
            "Week":      f"Week {i}",
            "Skill":     skill_name,
            "Objective": meta["Objective"],
            "Success":   meta["Success"],
            "Level":     skill_data.get("Level", 0),
        })

    return pd.DataFrame(roadmap_rows)


def roadmap_to_text(roadmap_df: pd.DataFrame, candidate_id: int | str) -> str:
    """
    Produce a human-readable text summary of the roadmap, suitable for
    printing to console or saving as a plain-text report.
    """
    lines = [f"{'='*60}", f"  8-Week Learning Roadmap for Candidate {candidate_id}", f"{'='*60}"]
    for _, row in roadmap_df.iterrows():
        lines.append(f"\n{row['Week']}  ▸  {row['Skill']}  (Level {row['Level']})")
        lines.append(f"  Objective : {row['Objective']}")
        lines.append(f"  Success   : {row['Success']}")
    lines.append(f"\n{'='*60}")
    return "\n".join(lines)
