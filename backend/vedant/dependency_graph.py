"""
dependency_graph.py
-------------------
Builds and queries the Skill Dependency Graph (directed acyclic graph)
used to sequence the learning pathway logically.
"""

from __future__ import annotations

import networkx as nx

from config import SKILL_DEPENDENCY_EDGES


def build_skill_dependency_graph(
    extra_edges: list[tuple[str, str]] | None = None,
) -> nx.DiGraph:
    """
    Construct a Directed Acyclic Graph (DAG) of skill prerequisites.

    Parameters
    ----------
    extra_edges : optional list of additional (prerequisite, advanced) tuples

    Returns
    -------
    nx.DiGraph  –  nodes are skill strings, edges point prerequisite → advanced
    """
    G = nx.DiGraph()
    edges = SKILL_DEPENDENCY_EDGES + (extra_edges or [])
    G.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(
            f"Skill dependency graph contains cycles: {cycles}. "
            "Fix SKILL_DEPENDENCY_EDGES in config.py."
        )
    return G


def get_skill_level(graph: nx.DiGraph, skill: str) -> int:
    """
    Return the 'depth' of a skill node = number of its ancestors in the graph.
    Skills not present in the graph are treated as level 0 (foundational).
    """
    if skill not in graph:
        return 0
    return len(nx.ancestors(graph, skill))


def get_prerequisites(graph: nx.DiGraph, skill: str) -> list[str]:
    """Return all direct predecessors (immediate prerequisites) of a skill."""
    if skill not in graph:
        return []
    return list(graph.predecessors(skill))


def get_all_ancestors(graph: nx.DiGraph, skill: str) -> set[str]:
    """Return the full transitive set of prerequisites for a skill."""
    if skill not in graph:
        return set()
    return nx.ancestors(graph, skill)


def get_subgraph_for_skills(graph: nx.DiGraph, skills: list[str]) -> nx.DiGraph:
    """
    Return a subgraph containing the given skills plus all their ancestors.
    Useful for the 'Skill Bridge' visualisation.
    """
    nodes: set[str] = set()
    for skill in skills:
        nodes.add(skill)
        nodes.update(get_all_ancestors(graph, skill))
    return graph.subgraph(nodes).copy()


def graph_summary(graph: nx.DiGraph) -> dict:
    """Return a quick-stats dict for the graph."""
    return {
        "nodes":  graph.number_of_nodes(),
        "edges":  graph.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "roots":  [n for n in graph.nodes if graph.in_degree(n) == 0],
        "leaves": [n for n in graph.nodes if graph.out_degree(n) == 0],
    }
