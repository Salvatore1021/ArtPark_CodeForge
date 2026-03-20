"""
pathway_engine.py — Graph-based adaptive learning pathway generator.

Algorithm:
1. Map each skill gap → candidate courses from catalog (by skills_taught overlap).
2. Build a prerequisite-aware directed graph (NetworkX DiGraph).
3. Include prerequisite courses transitively (even if the skill is already met).
4. Apply priority-weighted topological sort:
   - Priority = max gap priority_score of skills the course addresses
   - Mandatory gaps weighted 2×
5. Group the ordered courses into phases (weeks) by estimated hours.
6. Attach reasoning traces to every course recommendation.

Strict grounding: ONLY courses in the catalog are ever recommended.
Zero hallucinations — no invented courses.
"""

import json
import os
import logging
from typing import Any

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    logging.warning("networkx not installed — falling back to simple topological sort")

logger = logging.getLogger(__name__)

_CATALOG_CACHE: list[dict] | None = None
MAX_HOURS_PER_PHASE = 40  # ~1 week of focused learning


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_course_catalog() -> list[dict]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "course_catalog.json")
        with open(path, "r") as f:
            data = json.load(f)
        _CATALOG_CACHE = data["courses"]
    except Exception as e:
        logger.error(f"Could not load course catalog: {e}")
        _CATALOG_CACHE = []
    return _CATALOG_CACHE


# ---------------------------------------------------------------------------
# Course → skill gap matching
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.strip().lower()


def _skill_overlap_score(course: dict, gap_skills: set[str]) -> float:
    """
    How many gap skills does this course address?
    Returns a 0.0–1.0 relevance score.
    """
    taught = {_normalize(s) for s in course.get("skills_taught", [])}
    overlap = taught & {_normalize(s) for s in gap_skills}
    if not taught:
        return 0.0
    return len(overlap) / len(taught)


def _find_courses_for_gap(gap: dict, catalog: list[dict]) -> list[dict]:
    """Return catalog courses that teach the skill in a gap."""
    skill_name_norm = _normalize(gap["skill_name"])
    canonical_norm = _normalize(gap.get("canonical_name", gap["skill_name"]))

    matched = []
    for course in catalog:
        taught_norms = {_normalize(s) for s in course.get("skills_taught", [])}
        tags_norms = {_normalize(t) for t in course.get("tags", [])}
        if skill_name_norm in taught_norms or canonical_norm in taught_norms:
            matched.append(course)
        elif skill_name_norm in tags_norms or canonical_norm in tags_norms:
            matched.append(course)
    return matched


# ---------------------------------------------------------------------------
# Graph construction + topological sort
# ---------------------------------------------------------------------------

def _build_graph(course_ids: set[str], catalog: list[dict]) -> "nx.DiGraph | dict":
    """
    Build directed graph where edge A→B means 'A must be done before B'.
    Edges come from course prerequisites.
    """
    id_to_course = {c["id"]: c for c in catalog}

    if _HAS_NX:
        G = nx.DiGraph()
        for cid in course_ids:
            G.add_node(cid)
            course = id_to_course.get(cid, {})
            for prereq in course.get("prerequisites", []):
                if prereq in id_to_course:
                    G.add_node(prereq)
                    G.add_edge(prereq, cid)  # prereq → course
                    # Ensure prereq's prereqs are also in the graph
                    for pp in id_to_course[prereq].get("prerequisites", []):
                        if pp in id_to_course:
                            G.add_node(pp)
                            G.add_edge(pp, prereq)
        return G
    else:
        # Simple adjacency dict fallback
        graph: dict[str, list[str]] = {cid: [] for cid in course_ids}
        for cid in course_ids:
            course = id_to_course.get(cid, {})
            for prereq in course.get("prerequisites", []):
                if prereq in id_to_course:
                    graph.setdefault(prereq, [])
                    graph[prereq].append(cid)
        return graph


def _topological_sort_with_priority(
    graph: Any,
    course_ids: set[str],
    priority_map: dict[str, float],
    catalog: list[dict],
) -> list[str]:
    """
    Topological sort respecting prerequisites, with priority tie-breaking.
    Higher priority_map[course_id] → schedule earlier among peers at same depth.
    """
    id_to_course = {c["id"]: c for c in catalog}
    all_nodes = set(course_ids)

    if _HAS_NX:
        # Add all nodes from the graph (includes transitively pulled prereqs)
        all_nodes = set(graph.nodes())
        # kahn's algorithm with priority queue
        import heapq
        in_degree = {n: graph.in_degree(n) for n in all_nodes}
        # Max-heap (negate priority for min-heap)
        heap = [(-priority_map.get(n, 0.0), n) for n in all_nodes if in_degree[n] == 0]
        import heapq
        heapq.heapify(heap)
        result = []
        while heap:
            _, node = heapq.heappop(heap)
            result.append(node)
            for successor in graph.successors(node):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    heapq.heappush(heap, (-priority_map.get(successor, 0.0), successor))
        return result
    else:
        # Simple DFS-based topological sort
        visited = set()
        result = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for successor in graph.get(node, []):
                dfs(successor)
            result.append(node)

        for node in sorted(all_nodes, key=lambda n: -priority_map.get(n, 0.0)):
            dfs(node)
        result.reverse()
        return result


# ---------------------------------------------------------------------------
# Phase grouping
# ---------------------------------------------------------------------------

def _group_into_phases(ordered_ids: list[str], catalog: list[dict]) -> list[dict]:
    """Group ordered courses into weekly phases by hours budget."""
    id_to_course = {c["id"]: c for c in catalog}
    phases = []
    current_phase: list[dict] = []
    current_hours = 0
    phase_num = 1

    for cid in ordered_ids:
        course = id_to_course.get(cid)
        if not course:
            continue
        hours = course.get("duration_hours", 10)
        if current_hours + hours > MAX_HOURS_PER_PHASE and current_phase:
            phases.append({
                "phase": phase_num,
                "title": f"Phase {phase_num}",
                "total_hours": current_hours,
                "courses": current_phase,
            })
            phase_num += 1
            current_phase = []
            current_hours = 0
        current_phase.append(course)
        current_hours += hours

    if current_phase:
        phases.append({
            "phase": phase_num,
            "title": f"Phase {phase_num}",
            "total_hours": current_hours,
            "courses": current_phase,
        })

    # Name phases semantically
    _name_phases(phases)
    return phases


PHASE_NAMES = [
    "Foundations",
    "Core Skills",
    "Applied Techniques",
    "Advanced Specialization",
    "Mastery & Integration",
]


def _name_phases(phases: list[dict]) -> None:
    for i, phase in enumerate(phases):
        if i < len(PHASE_NAMES):
            phase["title"] = PHASE_NAMES[i]
        else:
            phase["title"] = f"Advanced Track {i - len(PHASE_NAMES) + 2}"


# ---------------------------------------------------------------------------
# Reasoning trace builder
# ---------------------------------------------------------------------------

def _build_course_reasoning(
    course: dict,
    gaps: list[dict],
    is_prerequisite_only: bool,
) -> str:
    """Generate a human-readable reasoning trace for why this course is included."""
    skill_set = {s.lower() for s in course.get("skills_taught", [])}
    addressed_gaps = [
        g for g in gaps if g["skill_name"].lower() in skill_set
        or g.get("canonical_name", "").lower() in skill_set
    ]

    if is_prerequisite_only:
        return (
            f"Included as a prerequisite for other courses in your pathway. "
            f"Completing '{course['name']}' unlocks subsequent courses in the plan."
        )

    if not addressed_gaps:
        return f"Supports your overall learning journey for this role."

    gap_descs = []
    for g in addressed_gaps[:3]:
        lvl_diff = g["required_level"] - g["current_level"]
        if g["status"] == "missing":
            gap_descs.append(f"'{g['skill_name']}' (you have no current exposure; role requires level {g['required_level']}/5)")
        else:
            gap_descs.append(
                f"'{g['skill_name']}' (current level {g['current_level']}/5 → target {g['required_level']}/5, gap of {lvl_diff})"
            )

    mandatory_count = sum(1 for g in addressed_gaps if g["is_mandatory"])
    reason = f"Addresses {len(addressed_gaps)} gap(s): {'; '.join(gap_descs)}."
    if mandatory_count:
        reason += f" {mandatory_count} of these are mandatory for the role."
    return reason


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_learning_pathway(
    gap_analysis: dict[str, Any],
    jd_data: dict[str, Any],
    catalog: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Build a complete, phase-grouped learning pathway from gap analysis.

    Returns:
    {
        "job_title": str,
        "seniority": str,
        "estimated_total_hours": int,
        "estimated_weeks": int,
        "phases": [ { phase, title, total_hours, courses: [...] } ],
        "prerequisite_courses": [...],   # pulled in transitively
        "gap_coverage": float,           # % of gaps addressed by pathway
        "algorithm": str,
        "reasoning_summary": str
    }
    """
    if catalog is None:
        catalog = load_course_catalog()

    gaps: list[dict] = gap_analysis.get("gaps", [])
    job_title = jd_data.get("job_title", "Target Role")
    seniority = jd_data.get("seniority", "mid")

    if not gaps:
        return {
            "job_title": job_title,
            "seniority": seniority,
            "estimated_total_hours": 0,
            "estimated_weeks": 0,
            "phases": [],
            "prerequisite_courses": [],
            "gap_coverage": 100.0,
            "algorithm": "graph-topological-sort-priority-weighted",
            "reasoning_summary": "No skill gaps identified. The candidate meets all requirements for this role.",
        }

    id_to_course = {c["id"]: c for c in catalog}

    # Step 1: Find directly relevant courses for each gap
    gap_course_map: dict[str, list[str]] = {}  # course_id → gap_skill_names
    direct_course_ids: set[str] = set()

    for gap in gaps:
        matched = _find_courses_for_gap(gap, catalog)
        for course in matched:
            cid = course["id"]
            direct_course_ids.add(cid)
            gap_course_map.setdefault(cid, []).append(gap["skill_name"])

    if not direct_course_ids:
        return {
            "job_title": job_title,
            "seniority": seniority,
            "estimated_total_hours": 0,
            "estimated_weeks": 0,
            "phases": [],
            "prerequisite_courses": [],
            "gap_coverage": 0.0,
            "algorithm": "graph-topological-sort-priority-weighted",
            "reasoning_summary": "No matching courses found in the catalog for the identified gaps.",
        }

    # Step 2: Build priority map for courses
    # Course priority = max priority_score of all gaps it addresses
    priority_map: dict[str, float] = {}
    for gap in gaps:
        matched = _find_courses_for_gap(gap, catalog)
        for course in matched:
            cid = course["id"]
            existing = priority_map.get(cid, 0.0)
            priority_map[cid] = max(existing, gap["priority_score"])

    # Step 3: Build graph (includes transitive prereqs)
    graph = _build_graph(direct_course_ids, catalog)

    # Identify prerequisite-only courses (in graph but not directly addressing a gap)
    if _HAS_NX:
        all_course_ids_in_graph = set(graph.nodes())
    else:
        all_course_ids_in_graph = set(graph.keys())

    prereq_only_ids = all_course_ids_in_graph - direct_course_ids

    # Assign minimal priority to prerequisite-only courses
    for cid in prereq_only_ids:
        if cid not in priority_map:
            priority_map[cid] = 0.1

    # Step 4: Topological sort with priority
    ordered_ids = _topological_sort_with_priority(
        graph, all_course_ids_in_graph, priority_map, catalog
    )

    # Step 5: Annotate each course with reasoning
    annotated_courses = []
    for cid in ordered_ids:
        course = id_to_course.get(cid)
        if not course:
            continue
        is_prereq_only = cid in prereq_only_ids
        course_copy = dict(course)
        course_copy["reasoning"] = _build_course_reasoning(course, gaps, is_prereq_only)
        course_copy["is_prerequisite_only"] = is_prereq_only
        course_copy["addresses_gaps"] = gap_course_map.get(cid, [])
        course_copy["priority_score"] = round(priority_map.get(cid, 0.0), 4)
        annotated_courses.append(course_copy)

    # Step 6: Group into phases
    phases = _group_into_phases(ordered_ids, catalog)
    # Inject annotated course data into phases
    annotation_map = {c["id"]: c for c in annotated_courses}
    for phase in phases:
        phase["courses"] = [
            {**annotation_map[c["id"]], **c}  # merge annotation into course
            if c["id"] in annotation_map else c
            for c in phase["courses"]
        ]

    # Compute stats
    total_hours = sum(p["total_hours"] for p in phases)
    # ~20 hours of focused learning per week
    estimated_weeks = max(1, round(total_hours / 20))

    # Gap coverage: how many unique gap skills are addressed?
    all_gap_skills = {g["skill_name"].lower() for g in gaps}
    covered_skills: set[str] = set()
    for phase in phases:
        for course in phase["courses"]:
            for s in course.get("skills_taught", []):
                covered_skills.add(s.lower())
    gap_coverage = round(len(all_gap_skills & covered_skills) / max(len(all_gap_skills), 1) * 100, 1)

    reasoning_summary = (
        f"Generated a {len(phases)}-phase pathway for '{job_title}' ({seniority} level) "
        f"covering {len(all_course_ids_in_graph)} courses over ~{estimated_weeks} weeks "
        f"({total_hours} total hours). "
        f"{len(prereq_only_ids)} prerequisite course(s) were automatically included to ensure "
        f"foundational readiness. The pathway addresses {gap_coverage}% of identified skill gaps. "
        f"Ordering was determined by a priority-weighted topological sort on the prerequisite "
        f"dependency graph — mandatory gaps with larger magnitude are scheduled earliest."
    )

    return {
        "job_title": job_title,
        "seniority": seniority,
        "estimated_total_hours": total_hours,
        "estimated_weeks": estimated_weeks,
        "phases": phases,
        "prerequisite_courses": list(prereq_only_ids),
        "gap_coverage": gap_coverage,
        "algorithm": "graph-topological-sort-priority-weighted",
        "reasoning_summary": reasoning_summary,
    }
