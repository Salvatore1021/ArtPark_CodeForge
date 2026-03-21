"""
report_exporter.py
-------------------
Export evaluation results to structured JSON and a plain-text PDF-ready report.

Outputs
-------
- output/<id>_report.json   : full machine-readable evaluation record
- output/<id>_report.txt    : formatted human-readable onboarding summary

Usage
-----
    from report_exporter import export_json, export_text_report, export_all
    export_all(metrics, roadmap_df, confidence_scores, output_dir="output")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _sanitise(obj):
    """Recursively convert non-JSON-serialisable types (numpy floats, sets)."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(i) for i in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    if hasattr(obj, "item"):        # numpy scalar
        return obj.item()
    return obj


def export_json(
    metrics: dict,
    roadmap_df: pd.DataFrame,
    confidence_scores: dict | None = None,
    recommendations: list[dict] | None = None,
    output_dir: str = "output",
) -> Path:
    """
    Export a complete JSON record for this evaluation run.

    Includes: metrics, roadmap, per-skill confidence breakdown,
    top role recommendations (if provided).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cid  = metrics.get("ID", "unknown")
    path = Path(output_dir) / f"{cid}_report.json"

    record = {
        "meta": {
            "candidate_id":   cid,
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "engine_version": "2.0",
        },
        "evaluation": {
            "job_title":         metrics.get("Job_Title", ""),
            "category":          metrics.get("Category", ""),
            "grade":             metrics.get("Grade", ""),
            "composite_score":   metrics.get("Composite_Score", 0.0),
            "weighted_fit":      metrics.get("Weighted_Fit", 0.0),
            "weighted_cosine":   metrics.get("Weighted_Cosine", 0.0),
            "pathway_depth":     metrics.get("Pathway_Depth", ""),
            "duration_weeks":    metrics.get("Duration", 0),
            "extracted_skills":  sorted(metrics.get("Extracted_Skills", [])),
            "gaps":              sorted(metrics.get("Gaps", [])),
        },
        "roadmap": roadmap_df.to_dict(orient="records"),
        "prioritised_gaps": _sanitise(metrics.get("Prioritized_Gaps", [])),
    }

    if confidence_scores:
        record["skill_confidence"] = _sanitise(confidence_scores)

    if recommendations:
        record["role_recommendations"] = _sanitise(
            [{k: v for k, v in r.items()} for r in recommendations]
        )

    path.write_text(json.dumps(_sanitise(record), indent=2), encoding="utf-8")
    return path


def export_text_report(
    metrics: dict,
    roadmap_df: pd.DataFrame,
    confidence_scores: dict | None = None,
    recommendations: list[dict] | None = None,
    output_dir: str = "output",
) -> Path:
    """
    Export a formatted plain-text onboarding report.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cid  = metrics.get("ID", "unknown")
    path = Path(output_dir) / f"{cid}_report.txt"

    grade       = metrics.get("Grade", "?")
    composite   = metrics.get("Composite_Score", 0.0)
    job_title   = metrics.get("Job_Title", "")
    pathway     = metrics.get("Pathway_Depth", "")
    duration    = metrics.get("Duration", 0)
    wfit        = metrics.get("Weighted_Fit", 0.0)
    wcosine     = metrics.get("Weighted_Cosine", 0.0)
    skills      = sorted(metrics.get("Extracted_Skills", []))
    gaps        = sorted(metrics.get("Gaps", []))
    prio_gaps   = metrics.get("Prioritized_Gaps", [])

    lines = [
        "=" * 70,
        "  ONBOARDING ENGINE v2 — CANDIDATE EVALUATION REPORT",
        "=" * 70,
        f"  Generated      : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Candidate ID   : {cid}",
        f"  Target Role    : {job_title}",
        "",
        "─" * 70,
        "  SCORES",
        "─" * 70,
        f"  Grade            : {grade}",
        f"  Composite Score  : {composite:.4f}",
        f"  Weighted Fit     : {wfit:.4f}  (O*NET importance-weighted skill match)",
        f"  Weighted Cosine  : {wcosine:.4f}  (TF-IDF semantic similarity)",
        f"  Pathway          : {pathway}",
        f"  Recommended Plan : {duration} weeks",
        "",
        "─" * 70,
        "  DETECTED SKILLS",
        "─" * 70,
    ]

    if confidence_scores:
        for skill in sorted(skills):
            data = confidence_scores.get(skill.lower(), {})
            tier = data.get("tier", "")
            conf = data.get("confidence", 0.0)
            sym  = {"strong": "●", "partial": "◑", "weak": "○"}.get(tier, " ")
            lines.append(f"  {sym} {skill:<40} confidence={conf:.3f}  [{tier}]")
    else:
        for skill in skills:
            lines.append(f"  ● {skill}")

    lines += [
        "",
        "─" * 70,
        f"  SKILL GAPS  ({len(gaps)} identified)",
        "─" * 70,
    ]

    if prio_gaps:
        lines.append(f"  {'#':<4} {'Skill':<40} {'Level':>5}  {'O*NET':>5}")
        lines.append("  " + "─" * 55)
        for g in prio_gaps:
            lines.append(
                f"  {g['Priority']:<4} {g['Skill']:<40} "
                f"{g['Level']:>5}  {g['ONET_Weight']:>5.2f}"
            )
    else:
        for gap in gaps:
            lines.append(f"  ○ {gap}")

    lines += [
        "",
        "─" * 70,
        "  LEARNING ROADMAP",
        "─" * 70,
        f"  {'Week':<8} {'Skill':<35} {'O*NET':>5}  {'Objective'}",
        "  " + "─" * 65,
    ]
    for _, row in roadmap_df.iterrows():
        lines.append(
            f"  {row['Week']:<8} {row['Skill']:<35} "
            f"{row['ONET_Weight']:>5.2f}  {row['Objective']}"
        )

    if recommendations:
        lines += [
            "",
            "─" * 70,
            "  ROLE RECOMMENDATIONS",
            "─" * 70,
        ]
        for i, r in enumerate(recommendations[:5], 1):
            lines.append(
                f"  #{i}  {r['job_title']:<40} composite={r['composite']:.4f}"
            )
            lines.append(
                f"       Matched: {', '.join(r['matched_skills'][:5])}"
            )

    lines += ["", "=" * 70, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_all(
    metrics: dict,
    roadmap_df: pd.DataFrame,
    confidence_scores: dict | None = None,
    recommendations: list[dict] | None = None,
    output_dir: str = "output",
) -> dict[str, Path]:
    """Export both JSON and text report, return dict of paths."""
    return {
        "json": export_json(metrics, roadmap_df, confidence_scores, recommendations, output_dir),
        "text": export_text_report(metrics, roadmap_df, confidence_scores, recommendations, output_dir),
    }
