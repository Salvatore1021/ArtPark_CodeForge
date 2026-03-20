"""
dependency_graph.py
--------------------
Builds and queries the Skill Dependency Graph (DAG).
Edges from config.py cover both O*NET competency progression and tech-stack prerequisites.
"""

from __future__ import annotations
import networkx as nx
from config import SKILL_DEPENDENCY_EDGES


def build_skill_dependency_graph(
    extra_edges: list[tuple[str, str]] | None = None,
) -> nx.DiGraph:
    """
    Build a directed acyclic graph of skill prerequisites.
    All skill strings are stored lowercase for consistent lookup.
    Raises ValueError if cycles are detected.
    """
    G = nx.DiGraph()
    edges = [(a.lower(), b.lower()) for a, b in SKILL_DEPENDENCY_EDGES]
    if extra_edges:
        edges += [(a.lower(), b.lower()) for a, b in extra_edges]
    G.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Dependency graph contains cycles: {cycles}")
    return G


def get_skill_level(graph: nx.DiGraph, skill: str) -> int:
    """
    Depth of a skill = number of ancestors.
    Skills absent from the graph are Level 0 (foundational).
    """
    s = skill.lower()
    if s not in graph:
        return 0
    return len(nx.ancestors(graph, s))


def get_prerequisites(graph: nx.DiGraph, skill: str) -> list[str]:
    """Direct predecessors (immediate prerequisites) of a skill."""
    s = skill.lower()
    return list(graph.predecessors(s)) if s in graph else []


def get_all_ancestors(graph: nx.DiGraph, skill: str) -> set[str]:
    """Full transitive set of prerequisites for a skill."""
    s = skill.lower()
    return nx.ancestors(graph, s) if s in graph else set()


def get_subgraph_for_skills(graph: nx.DiGraph, skills: list[str]) -> nx.DiGraph:
    """Subgraph containing skills + all their ancestors (for Skill Bridge visualisation)."""
    nodes: set[str] = set()
    for skill in skills:
        s = skill.lower()
        nodes.add(s)
        nodes.update(get_all_ancestors(graph, s))
    # keep only nodes that actually exist in the graph
    nodes = {n for n in nodes if n in graph}
    return graph.subgraph(nodes).copy()


def graph_summary(graph: nx.DiGraph) -> dict:
    return {
        "nodes":  graph.number_of_nodes(),
        "edges":  graph.number_of_edges(),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "roots":  [n for n in graph.nodes if graph.in_degree(n)  == 0],
        "leaves": [n for n in graph.nodes if graph.out_degree(n) == 0],
    }
