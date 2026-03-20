"""
dashboard.py
------------
Generates the full matplotlib onboarding dashboard:
  - Profile summary panel
  - Readiness indicator (donut chart)
  - Skill Bridge subgraph visualization
  - 8-Week narrative roadmap table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

from dependency_graph import get_subgraph_for_skills
from roadmap_builder import build_roadmap


# ── Colour palette ────────────────────────────────────────────────────────────
_GAP_COLOR  = "#e74c3c"   # red   → skills the candidate is missing
_BASE_COLOR = "#3498db"   # blue  → prerequisite / foundation skills
_READY_COLOR= "#27ae60"   # green → readiness arc
_BG_COLOR   = "#ecf0f1"   # light grey → remainder arc

_GRADE_COLOR = {
    "A": "#27ae60", "B": "#2ecc71",
    "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c",
}


def generate_dashboard(
    cand_eval_metrics: dict,
    skill_dependency_graph: nx.DiGraph,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Build and display the candidate onboarding dashboard.

    Parameters
    ----------
    cand_eval_metrics      : dict from evaluate_candidate() + 'Prioritized_Gaps' added
    skill_dependency_graph : the DAG from build_skill_dependency_graph()
    save_path              : if given, saves the figure to this path (PNG)
    show                   : whether to call plt.show()

    Returns
    -------
    matplotlib Figure object
    """
    gaps       = cand_eval_metrics.get("Prioritized_Gaps", [])
    gap_names  = [g["Skill"] for g in gaps]
    cand_id    = cand_eval_metrics["ID"]
    grade      = cand_eval_metrics["Grade"]
    score      = cand_eval_metrics["Composite_Score"]
    category   = cand_eval_metrics["Category"]
    pathway    = cand_eval_metrics["Pathway_Depth"]

    roadmap_df = build_roadmap(cand_eval_metrics)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1.6, 1.4])
    fig.suptitle(
        f"AI Onboarding Dashboard  ▸  Candidate {cand_id}",
        fontsize=22, fontweight="bold", color="#2c3e50",
    )

    # ── Panel 1: Profile Summary ──────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis("off")

    info_lines = [
        f"Category        :  {category}",
        f"Final Grade     :  {grade}",
        f"Pathway Depth   :  {pathway}",
        f"Duration        :  {cand_eval_metrics['Duration']} weeks",
        f"Composite Score :  {score:.4f}",
        f"Weighted Fit    :  {cand_eval_metrics['Weighted_Fit']:.4f}",
        f"Vector Sim.     :  {cand_eval_metrics['Vector_Similarity']:.4f}",
        f"Skill Gaps      :  {len(gap_names)}",
    ]
    ax_info.text(
        0.05, 0.5, "\n".join(info_lines),
        fontsize=12, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="aliceblue", alpha=0.7),
        transform=ax_info.transAxes,
    )
    ax_info.set_title("Profile Summary", fontsize=14, fontweight="bold", pad=8)

    # ── Panel 2: Readiness Indicator (Donut) ─────────────────────────────────
    ax_gauge = fig.add_subplot(gs[0, 1])
    pct = score * 100
    ax_gauge.pie(
        [pct, max(0, 100 - pct)],
        colors=[_GRADE_COLOR.get(grade, _READY_COLOR), _BG_COLOR],
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="white"),
    )
    ax_gauge.text(0, 0, f"{pct:.1f}%", ha="center", va="center",
                  fontsize=26, fontweight="bold", color=_GRADE_COLOR.get(grade, "#2c3e50"))
    ax_gauge.text(0, -0.3, f"Grade: {grade}", ha="center", va="center",
                  fontsize=14, color=_GRADE_COLOR.get(grade, "#2c3e50"))
    ax_gauge.set_title("Readiness Indicator", fontsize=14, fontweight="bold", pad=8)

    # ── Panel 3: Skill Bridge Subgraph ────────────────────────────────────────
    ax_graph = fig.add_subplot(gs[1, :])
    sub_g = get_subgraph_for_skills(skill_dependency_graph, gap_names)

    if sub_g.number_of_nodes() > 0:
        pos = nx.spring_layout(sub_g, seed=42, k=1.5)
        node_colors = [
            _GAP_COLOR if n in gap_names else _BASE_COLOR
            for n in sub_g.nodes()
        ]
        nx.draw(
            sub_g, pos, ax=ax_graph,
            with_labels=True, node_color=node_colors,
            node_size=2800, font_size=9, font_weight="bold",
            edge_color="#95a5a6", arrows=True,
            arrowsize=18, connectionstyle="arc3,rad=0.1",
        )
        legend_handles = [
            mpatches.Patch(color=_GAP_COLOR,  label="Skill Gap (needs training)"),
            mpatches.Patch(color=_BASE_COLOR, label="Foundation Prerequisite"),
        ]
        ax_graph.legend(handles=legend_handles, loc="upper left", fontsize=10)
    else:
        ax_graph.text(0.5, 0.5, "No dependency subgraph available for these gaps.",
                      ha="center", va="center", fontsize=13, transform=ax_graph.transAxes)
        ax_graph.axis("off")

    ax_graph.set_title(
        "Skill Bridge: Gaps (Red) → Foundation (Blue)",
        fontsize=14, fontweight="bold", pad=8,
    )

    # ── Panel 4: 8-Week Roadmap Table ─────────────────────────────────────────
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    display_df = roadmap_df[["Week", "Skill", "Objective", "Success"]]
    tbl = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.1)

    # Style header row
    for col_idx in range(len(display_df.columns)):
        tbl[(0, col_idx)].set_facecolor("#2c3e50")
        tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")

    ax_table.set_title(
        f"Narrative Learning Roadmap  ({cand_eval_metrics['Duration']} Weeks)",
        fontsize=14, fontweight="bold", pad=8,
    )

    # ── Save / Show ───────────────────────────────────────────────────────────
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[Dashboard] Saved to {save_path}")

    if show:
        plt.show()

    return fig
