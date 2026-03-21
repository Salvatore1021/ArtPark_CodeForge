"""
dashboard.py
------------
Clean, layman-friendly 5-panel onboarding dashboard.

Panels
------
1. Candidate Profile Card         — name, contact, education, role info
2. Overall Readiness Gauge        — simple % donut with plain-English label
3. Skills You Have vs To Learn    — two visual columns, no jargon
4. Top 3 Role Recommendations     — match bars with matched/gap skill lists
5. Your Learning Pathway          — clean week-by-week table
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd

from roadmap_builder import build_roadmap
from skill_classifier import classify_skill


# ── Colour palette ────────────────────────────────────────────────────────────
_GRADE_COLORS = {
    "A": "#27ae60", "B": "#2ecc71", "C": "#f39c12", "D": "#e67e22", "F": "#e74c3c",
}
_TECH_COLOR   = "#3498db"
_SOFT_COLOR   = "#9b59b6"
_GAP_COLOR    = "#e74c3c"
_MATCH_COLOR  = "#27ae60"
_BG_LIGHT     = "#f8f9fa"
_BORDER       = "#dee2e6"
_TEXT_DARK    = "#2c3e50"
_TEXT_MED     = "#555e6b"
_TEXT_LIGHT   = "#8492a6"
_WEEK_COLORS  = ["#3498db", "#2980b9", "#1abc9c", "#16a085",
                 "#e67e22", "#d35400", "#9b59b6", "#8e44ad"]


def _importance_label(weight: float) -> str:
    if weight >= 3.75: return "Essential"
    if weight >= 3.25: return "Very Important"
    if weight >= 2.75: return "Important"
    return "Useful"


def _readiness_label(grade: str) -> str:
    return {
        "A": "Excellent", "B": "Good", "C": "Developing",
        "D": "Needs Work", "F": "Beginner",
    }.get(grade, "")


def _wrap_text(text: str, max_len: int) -> list[str]:
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > max_len:
            if line: lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line: lines.append(line)
    return lines or [""]


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _panel_profile(ax: plt.Axes, metrics: dict, profile: dict | None) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Header background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 7.2), 10, 2.8, boxstyle="round,pad=0.1",
        facecolor=_TEXT_DARK, edgecolor="none",
    ))

    ax.text(5, 9.2, "CANDIDATE PROFILE", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

    name = (profile or {}).get("name", f"Candidate {metrics['ID']}")
    ax.text(5, 8.3, name, ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")

    job_title = metrics.get("Job_Title", "")
    ax.text(5, 7.55, f"Target Role: {job_title}", ha="center", va="center",
            fontsize=9, color="#b2bec3")

    # Contact details
    y = 6.7
    if profile:
        details = [
            ("Email",     profile.get("email", "")),
            ("Phone",     profile.get("phone", "")),
            ("LinkedIn",  profile.get("linkedin", "")),
        ]
        if profile.get("education"):
            details.append(("Education", profile["education"][0][:50]))

        for label, val in details:
            if val:
                ax.text(0.4, y, f"{label}:", fontsize=8.5, va="center",
                        color=_TEXT_MED, fontweight="bold")
                ax.text(2.6, y, str(val)[:45], fontsize=8.5, va="center",
                        color=_TEXT_DARK)
                y -= 1.1

    # Category type badge
    from taxonomy_adapter import get_sector
    from skill_classifier import get_category_type
    raw_cat  = metrics.get("Category", "")
    cat_type = get_category_type(raw_cat, job_title)
    sector   = get_sector(job_title) or ""
    cat_nice = {
        "TECH": "Technical", "MANAGEMENT": "Management",
        "HEALTHCARE": "Healthcare", "EDUCATION": "Education",
        "CREATIVE": "Creative", "NON_TECH": "Professional",
    }.get(cat_type, cat_type)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.3, 0.3), 4.2, 1.0, boxstyle="round,pad=0.15",
        facecolor=_BG_LIGHT, edgecolor=_BORDER, linewidth=0.8,
    ))
    ax.text(2.4, 0.8, f"Category: {cat_nice}", ha="center", va="center",
            fontsize=8.5, color=_TEXT_DARK)
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.3, 0.3), 4.4, 1.0, boxstyle="round,pad=0.15",
        facecolor=_BG_LIGHT, edgecolor=_BORDER, linewidth=0.8,
    ))
    ax.text(7.5, 0.8, f"Sector: {sector}", ha="center", va="center",
            fontsize=8.5, color=_TEXT_DARK)


def _panel_readiness(ax: plt.Axes, metrics: dict) -> None:
    grade  = metrics["Grade"]
    score  = metrics["Composite_Score"]
    pct    = score * 100
    color  = _GRADE_COLORS.get(grade, "#3498db")
    label  = _readiness_label(grade)
    split  = metrics.get("Split_Details", {})
    h_fit  = round(split.get("hard_fit", 0) * 100)
    s_fit  = round(split.get("soft_fit", 0) * 100)
    h_wt   = round(split.get("hard_weight", 0) * 100)
    s_wt   = round(split.get("soft_weight", 0) * 100)
    weeks  = metrics.get("Duration", 0)
    pathway= "Comprehensive" if "Fundamental" in metrics.get("Pathway_Depth","") else "Fast-Track"

    ax.pie(
        [pct, max(0, 100 - pct)],
        colors=[color, "#ecf0f1"],
        startangle=90,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
        radius=0.85,
    )
    ax.text(0, 0.18, f"{pct:.0f}%", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color)
    ax.text(0, -0.18, label, ha="center", va="center",
            fontsize=11, color=_TEXT_MED)
    ax.text(0, -0.55, f"{weeks}-Week {pathway} Plan",
            ha="center", va="center", fontsize=9, color=_TEXT_LIGHT)

    # Hard / soft mini bars below
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.5, 1.2)

    for i, (name, val, wt, col) in enumerate([
        ("Technical Skills", h_fit, h_wt, _TECH_COLOR),
        ("Soft Skills",      s_fit, s_wt, _SOFT_COLOR),
    ]):
        y_base = -1.1 - i * 0.35
        bar_w  = (val / 100) * 2.4
        ax.add_patch(mpatches.FancyBboxPatch(
            (-1.2, y_base - 0.08), 2.4, 0.16,
            boxstyle="round,pad=0.02", facecolor="#ecf0f1", edgecolor="none",
        ))
        if bar_w > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (-1.2, y_base - 0.08), bar_w, 0.16,
                boxstyle="round,pad=0.02", facecolor=col, edgecolor="none", alpha=0.85,
            ))
        ax.text(-1.25, y_base, f"{name} ({wt}%)", ha="left", va="center",
                fontsize=7.5, color=_TEXT_MED)
        ax.text(1.25, y_base, f"{val}%", ha="right", va="center",
                fontsize=7.5, color=_TEXT_DARK, fontweight="bold")

    ax.axis("off")


def _panel_skills(ax: plt.Axes, metrics: dict) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    job_title    = metrics.get("Job_Title", "")
    extracted    = sorted(metrics.get("Extracted_Skills", []))
    gaps         = [g["Skill"] for g in metrics.get("Prioritized_Gaps", [])]

    tech_have = [s for s in extracted if classify_skill(s, job_title) == "hard"]
    soft_have = [s for s in extracted if classify_skill(s, job_title) == "soft"]
    tech_gaps = [g for g in gaps if classify_skill(g, job_title) == "hard"][:8]
    soft_gaps = [g for g in gaps if classify_skill(g, job_title) == "soft"][:5]

    # Left column: skills you have
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.1, 5.2), 4.6, 4.6, boxstyle="round,pad=0.15",
        facecolor="#eafaf1", edgecolor="#27ae60", linewidth=0.8,
    ))
    ax.text(2.4, 9.4, f"✓ You Have  ({len(extracted)})", ha="center",
            fontsize=9.5, fontweight="bold", color="#27ae60")

    y = 8.9
    if tech_have:
        ax.text(0.4, y, "Technical:", fontsize=8, color=_TECH_COLOR, fontweight="bold")
        y -= 0.5
        for s in tech_have[:7]:
            ax.text(0.6, y, f"• {s.title()}", fontsize=7.8, color=_TEXT_DARK)
            y -= 0.42
    y -= 0.1
    if soft_have:
        ax.text(0.4, y, "Soft Skills:", fontsize=8, color=_SOFT_COLOR, fontweight="bold")
        y -= 0.5
        for s in soft_have[:4]:
            ax.text(0.6, y, f"• {s.title()}", fontsize=7.8, color=_TEXT_DARK)
            y -= 0.42

    # Right column: gaps to develop
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.3, 5.2), 4.6, 4.6, boxstyle="round,pad=0.15",
        facecolor="#fdedec", edgecolor="#e74c3c", linewidth=0.8,
    ))
    ax.text(7.6, 9.4, f"✗ To Develop  ({len(gaps)})", ha="center",
            fontsize=9.5, fontweight="bold", color="#e74c3c")

    y = 8.9
    if tech_gaps:
        ax.text(5.5, y, "Technical:", fontsize=8, color=_TECH_COLOR, fontweight="bold")
        y -= 0.5
        for s in tech_gaps:
            ax.text(5.7, y, f"• {s.title()}", fontsize=7.8, color=_TEXT_DARK)
            y -= 0.42
    y -= 0.1
    if soft_gaps:
        ax.text(5.5, y, "Soft Skills:", fontsize=8, color=_SOFT_COLOR, fontweight="bold")
        y -= 0.5
        for s in soft_gaps:
            ax.text(5.7, y, f"• {s.title()}", fontsize=7.8, color=_TEXT_DARK)
            y -= 0.42

    # Bottom summary bar
    total = len(extracted) + len(gaps)
    frac  = len(extracted) / total if total > 0 else 0
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.1, 4.5), 9.8, 0.5, boxstyle="round,pad=0.05",
        facecolor="#ecf0f1", edgecolor="none",
    ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.1, 4.5), 9.8 * frac, 0.5, boxstyle="round,pad=0.05",
        facecolor=_MATCH_COLOR, edgecolor="none", alpha=0.8,
    ))
    ax.text(5, 4.75, f"{len(extracted)} of {total} required skills demonstrated",
            ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    # Gap priority table
    ax.text(5, 4.1, "Priority Gaps to Address", ha="center", fontsize=9,
            fontweight="bold", color=_TEXT_DARK)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.1, 0.3), 9.8, 3.5, boxstyle="round,pad=0.1",
        facecolor=_BG_LIGHT, edgecolor=_BORDER, linewidth=0.6,
    ))

    headers = ["#", "Skill to Learn", "Type", "Importance"]
    xs      = [0.3, 1.1, 5.8, 7.8]
    y = 3.5
    for h, x in zip(headers, xs):
        ax.text(x, y, h, fontsize=8, fontweight="bold", color=_TEXT_MED)
    ax.plot([0.2, 9.8], [3.2, 3.2], color=_BORDER, linewidth=0.6)

    y = 2.9
    gap_weights = metrics.get("Gap_Weights", {})
    for g in metrics.get("Prioritized_Gaps", [])[:6]:
        skill   = g["Skill"].title()
        weight  = g["ONET_Weight"]
        imp     = _importance_label(weight)
        stype   = "Technical" if classify_skill(g["Skill"], metrics.get("Job_Title","")) == "hard" else "Soft"
        imp_col = {"Essential": "#e74c3c", "Very Important": "#e67e22",
                   "Important": "#f39c12"}.get(imp, _TEXT_MED)
        ax.text(0.3, y, str(g["Priority"]), fontsize=8, color=_TEXT_LIGHT)
        ax.text(1.1, y, skill[:28], fontsize=8, color=_TEXT_DARK)
        ax.text(5.8, y, stype, fontsize=8, color=_TEXT_MED)
        ax.text(7.8, y, imp, fontsize=8, color=imp_col, fontweight="bold")
        y -= 0.42


def _panel_recommendations(ax: plt.Axes, recommendations: list[dict]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    top3   = recommendations[:3]
    colors = [_GRADE_COLORS["A"], _GRADE_COLORS["B"], _GRADE_COLORS["C"]]
    medals = ["#1st", "#2nd", "#3rd"]

    for i, (rec, col, medal) in enumerate(zip(top3, colors, medals)):
        y_top = 9.5 - i * 3.15
        pct   = round(rec["composite"] * 100)
        bar_w = (pct / 100) * 8.5
        title = rec["job_title"]
        sec   = rec["sector"]

        # Card background
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, y_top - 2.8), 9.8, 2.9, boxstyle="round,pad=0.12",
            facecolor=_BG_LIGHT, edgecolor=col, linewidth=1.2,
        ))
        # Medal + title
        ax.text(0.45, y_top - 0.4, medal, fontsize=9, fontweight="bold", color=col)
        ax.text(1.5, y_top - 0.4, f"{title}", fontsize=9.5,
                fontweight="bold", color=_TEXT_DARK)
        ax.text(1.5, y_top - 0.9, f"{sec}", fontsize=8, color=_TEXT_LIGHT)

        # Match bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.3, y_top - 1.5), 8.5, 0.28, boxstyle="round,pad=0.02",
            facecolor="#ecf0f1", edgecolor="none",
        ))
        if bar_w > 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.3, y_top - 1.5), bar_w, 0.28, boxstyle="round,pad=0.02",
                facecolor=col, edgecolor="none", alpha=0.8,
            ))
        ax.text(9.0, y_top - 1.38, f"{pct}% match", ha="right", fontsize=8,
                fontweight="bold", color=col)

        # Matched skills
        matched_str = ", ".join(s.title() for s in rec["matched_skills"][:4])
        gap_str     = ", ".join(s.title() for s in rec["gap_skills"][:3])
        ax.text(0.35, y_top - 1.9, f"You have:  {matched_str or 'foundations'}",
                fontsize=7.8, color="#27ae60")
        ax.text(0.35, y_top - 2.45, f"To learn:  {gap_str or 'minor gaps only'}",
                fontsize=7.8, color="#e74c3c")


def _panel_pathway(ax: plt.Axes, metrics: dict, roadmap_df: pd.DataFrame) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    rows    = roadmap_df.iterrows()
    n_weeks = len(roadmap_df)
    row_h   = min(9.2 / (n_weeks + 1), 1.08)

    # Column headers
    ax.text(0.3, 9.6, "Week", fontsize=8.5, fontweight="bold", color=_TEXT_MED)
    ax.text(1.4, 9.6, "What You Will Learn", fontsize=8.5, fontweight="bold", color=_TEXT_MED)
    ax.text(6.2, 9.6, "Goal to Achieve", fontsize=8.5, fontweight="bold", color=_TEXT_MED)
    ax.plot([0.1, 9.9], [9.35, 9.35], color=_BORDER, linewidth=0.8)

    y = 9.1
    for idx, (_, row) in enumerate(roadmap_df.iterrows()):
        col    = _WEEK_COLORS[idx % len(_WEEK_COLORS)]
        skill  = row["Skill"].title()
        obj    = row["Objective"]
        suc    = row["Success"]

        # Week pill
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, y - row_h * 0.45), 1.0, row_h * 0.9,
            boxstyle="round,pad=0.04", facecolor=col, edgecolor="none", alpha=0.9,
        ))
        ax.text(0.6, y + row_h * 0.0, row["Week"], ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")

        # Skill name
        ax.text(1.45, y + row_h * 0.15, skill,
                fontsize=8.5, fontweight="bold", color=_TEXT_DARK, va="center")

        # Objective (truncated)
        obj_short = obj if len(obj) <= 40 else obj[:38] + ".."
        ax.text(1.45, y - row_h * 0.28, obj_short,
                fontsize=7.5, color=_TEXT_MED, va="center")

        # Success criteria
        suc_short = suc if len(suc) <= 32 else suc[:30] + ".."
        ax.text(6.2, y - row_h * 0.05, f"✓ {suc_short}",
                fontsize=7.5, color="#27ae60", va="center")

        # Divider
        if idx < n_weeks - 1:
            ax.plot([0.1, 9.9], [y - row_h * 0.5, y - row_h * 0.5],
                    color=_BORDER, linewidth=0.4, alpha=0.6)
        y -= row_h


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dashboard(
    cand_eval_metrics: dict,
    skill_dependency_graph: nx.DiGraph,
    recommendations: list[dict] | None = None,
    profile: dict | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Build the clean layman-friendly 5-panel onboarding dashboard.

    Panels
    ------
    Top row    : Candidate Profile  |  Readiness Gauge
    Middle row : Skills Have / Develop  |  Top 3 Role Recommendations
    Bottom row : Personalised Learning Pathway  (full width)
    """
    roadmap_df = build_roadmap(cand_eval_metrics)
    recs       = recommendations or []

    cand_id   = cand_eval_metrics["ID"]
    job_title = cand_eval_metrics.get("Job_Title", "")
    name      = (profile or {}).get("name", f"Candidate {cand_id}")

    fig = plt.figure(figsize=(22, 17), constrained_layout=True)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.1, 1.6, 1.5],
        hspace=0.08, wspace=0.06,
    )

    # Title banner
    fig.suptitle(
        f"Candidate Onboarding Report  —  {name}  —  {job_title}",
        fontsize=17, fontweight="bold", color=_TEXT_DARK, y=0.99,
    )

    # Panel 1: Profile
    ax1 = fig.add_subplot(gs[0, 0])
    _panel_profile(ax1, cand_eval_metrics, profile)
    ax1.set_title("Candidate Profile", fontsize=12, fontweight="bold",
                  color=_TEXT_DARK, pad=8, loc="left")

    # Panel 2: Readiness
    ax2 = fig.add_subplot(gs[0, 1])
    _panel_readiness(ax2, cand_eval_metrics)
    ax2.set_title("Overall Readiness", fontsize=12, fontweight="bold",
                  color=_TEXT_DARK, pad=8, loc="left")

    # Panel 3: Skills
    ax3 = fig.add_subplot(gs[1, 0])
    _panel_skills(ax3, cand_eval_metrics)
    ax3.set_title("Skills — Identified & To Develop", fontsize=12, fontweight="bold",
                  color=_TEXT_DARK, pad=8, loc="left")

    # Panel 4: Recommendations
    ax4 = fig.add_subplot(gs[1, 1])
    _panel_recommendations(ax4, recs)
    ax4.set_title("Top 3 Matching Roles For You", fontsize=12, fontweight="bold",
                  color=_TEXT_DARK, pad=8, loc="left")

    # Panel 5: Learning pathway
    ax5 = fig.add_subplot(gs[2, :])
    _panel_pathway(ax5, cand_eval_metrics, roadmap_df)
    weeks = cand_eval_metrics.get("Duration", 0)
    ax5.set_title(f"Your Personalised {weeks}-Week Learning Pathway",
                  fontsize=12, fontweight="bold", color=_TEXT_DARK, pad=8, loc="left")

    # Light card borders
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
            spine.set_linewidth(0.8)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"\n  Visual dashboard saved -> {save_path}")

    if show:
        plt.show()

    return fig
