"""
gap_prioritizer.py
------------------
Orders a candidate's skill gaps using:
  1. Graph depth (foundational gaps first via dependency graph)
  2. Cosine similarity tie-breaking (larger gap = lower similarity = higher priority)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from dependency_graph import get_skill_level


# ── Robust similarity helper ─────────────────────────────────────────────────

def calculate_robust_similarity(
    resume_vector,
    skill_name: str,
    skill_vectors: dict,
    epsilon: float = 1e-6,
) -> float:
    """
    Safe cosine similarity between a resume TF-IDF vector and a skill vector.
    Returns epsilon (near-zero) on any error or empty vector.
    """
    try:
        skill_vec = skill_vectors.get(skill_name)
        if skill_vec is None:
            return 0.0
        if resume_vector.nnz == 0:
            return epsilon
        sim = cosine_similarity(resume_vector, skill_vec)[0][0]
        return float(sim) if not np.isnan(sim) else epsilon
    except Exception:
        return 0.0


# ── Gap prioritisation ────────────────────────────────────────────────────────

def prioritize_gaps(
    gaps: list[str],
    resume_vector,
    skill_vectors: dict,
    graph: nx.DiGraph,
) -> list[dict]:
    """
    Return a sorted list of gap dicts, ordered so:
      - Foundational skills (low graph depth / level) come first
      - Among skills at the same level, the largest gap (lowest similarity) is first

    Parameters
    ----------
    gaps          : list of skill strings the candidate is missing
    resume_vector : candidate's TF-IDF sparse vector
    skill_vectors : pre-computed skill TF-IDF vectors (from skill_extractor)
    graph         : the skill dependency DAG

    Returns
    -------
    list of dicts: [{ 'Skill', 'Level', 'Similarity', 'Priority' }, …]
    """
    gap_records = []

    for gap in gaps:
        level = get_skill_level(graph, gap)
        sim   = calculate_robust_similarity(resume_vector, gap, skill_vectors)
        gap_records.append({
            "Skill":      gap,
            "Level":      level,
            "Similarity": round(sim, 4),
        })

    # Sort: ascending Level (foundational first), then ascending Similarity (bigger gap first)
    sorted_gaps = sorted(gap_records, key=lambda x: (x["Level"], x["Similarity"]))

    # Tag with Priority rank
    for i, record in enumerate(sorted_gaps, start=1):
        record["Priority"] = i

    return sorted_gaps


def split_foundational_advanced(
    prioritized_gaps: list[dict],
    threshold_level: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    Partition prioritised gaps into foundational (level ≤ threshold) and advanced.

    Returns (foundational, advanced)
    """
    foundational = [g for g in prioritized_gaps if g["Level"] <= threshold_level]
    advanced     = [g for g in prioritized_gaps if g["Level"] >  threshold_level]
    return foundational, advanced
