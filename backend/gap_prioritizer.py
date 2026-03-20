"""
gap_prioritizer.py
-------------------
Orders skill gaps using four keys:
  1. Graph level (foundational first)
  2. O*NET importance weight (desc — most critical first)
  3. Confidence score (asc — least confident first)
  4. Cosine similarity tie-break (asc — biggest knowledge gap first)

Confidence integration: if skill_confidence data is available, the priority
score uses `1 - confidence` instead of raw cosine similarity for a richer signal.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from dependency_graph import get_skill_level


def calculate_robust_similarity(
    resume_vector,
    skill_name: str,
    skill_vectors: dict,
    epsilon: float = 1e-6,
) -> float:
    try:
        skill_vec = skill_vectors.get(skill_name.lower())
        if skill_vec is None:
            return 0.0
        if resume_vector.nnz == 0:
            return epsilon
        sim = cosine_similarity(resume_vector, skill_vec)[0][0]
        return float(sim) if not np.isnan(sim) else epsilon
    except Exception:
        return 0.0


def prioritize_gaps(
    gaps: list[str],
    gap_weights: dict[str, float],
    resume_vector,
    skill_vectors: dict,
    graph: nx.DiGraph,
    confidence_scores: dict | None = None,
) -> list[dict]:
    """
    Return a sorted list of gap dicts.

    Sort key (all ascending → lower = higher priority):
      (graph_level, -onet_weight, gap_score)
      where gap_score = (1 - confidence) if confidence_scores available,
                        else cosine_similarity.

    Parameters
    ----------
    gaps              : list of skill strings the candidate is missing / weak
    gap_weights       : {skill: onet_weight}
    resume_vector     : candidate TF-IDF sparse vector
    skill_vectors     : pre-computed skill TF-IDF vectors
    graph             : skill dependency DAG
    confidence_scores : per-skill confidence from skill_confidence module (optional)
    """
    records = []
    for gap in gaps:
        level  = get_skill_level(graph, gap.lower())
        onet_w = gap_weights.get(gap, 1.0)

        # Gap score: lower = candidate knows less about this skill
        if confidence_scores and gap.lower() in confidence_scores:
            conf       = confidence_scores[gap.lower()]["confidence"]
            gap_score  = round(1.0 - conf, 4)   # 1 = no knowledge, 0 = near-strong
        else:
            sim       = calculate_robust_similarity(resume_vector, gap, skill_vectors)
            gap_score = round(1.0 - sim, 4)

        sim = calculate_robust_similarity(resume_vector, gap, skill_vectors)
        records.append({
            "Skill":       gap,
            "Level":       level,
            "ONET_Weight": round(onet_w, 2),
            "Gap_Score":   gap_score,
            "Similarity":  round(sim, 4),
        })

    # Sort: low level → high onet → high gap_score (most unknown)
    sorted_gaps = sorted(
        records,
        key=lambda x: (x["Level"], -x["ONET_Weight"], -x["Gap_Score"]),
    )
    for i, rec in enumerate(sorted_gaps, start=1):
        rec["Priority"] = i

    return sorted_gaps


def split_foundational_advanced(
    prioritized_gaps: list[dict],
    threshold_level: int = 2,
) -> tuple[list[dict], list[dict]]:
    foundational = [g for g in prioritized_gaps if g["Level"] <= threshold_level]
    advanced     = [g for g in prioritized_gaps if g["Level"] >  threshold_level]
    return foundational, advanced
