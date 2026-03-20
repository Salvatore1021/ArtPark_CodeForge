"""
pdf_parser.py
-------------
PDF utilities:
  - extract_text_from_pdf()  → raw text from any PDF (job descriptions, etc.)
  - parse_resume_pdf()       → structured candidate profile from a resume PDF
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ── Low-level helper ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Return the full concatenated text from every page of a PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


# ── Resume-specific patterns ──────────────────────────────────────────────────

_EMAIL_RE    = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)

# Section header keywords (case-insensitive)
_SECTION_KEYWORDS = {
    "education":   ["education", "academic", "qualification", "degree", "university", "college"],
    "experience":  ["experience", "employment", "work history", "career", "professional background"],
    "skills":      ["skills", "competencies", "technical skills", "core skills", "expertise"],
    "certifications": ["certification", "certificate", "license", "accreditation"],
    "projects":    ["project", "portfolio", "achievements"],
    "summary":     ["summary", "objective", "profile", "about me", "overview"],
}


def _find_section(lines: list[str], header_keywords: list[str]) -> list[str]:
    """
    Return the lines belonging to a section whose header matches any
    of the provided keywords.  Stops at the next detected section header.
    """
    all_headers = [kw for kws in _SECTION_KEYWORDS.values() for kw in kws]
    inside = False
    section_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Detect current section header
        if any(kw in lower for kw in header_keywords) and len(stripped) < 60:
            inside = True
            continue

        # Stop when another section header is encountered
        if inside and any(kw in lower for kw in all_headers) and len(stripped) < 60:
            break

        if inside and stripped:
            section_lines.append(stripped)

    return section_lines


def _extract_name(lines: list[str]) -> str:
    """
    Heuristic: the candidate's name is usually the first non-blank line
    that contains no digits, no email, and no URL.
    """
    for line in lines[:10]:
        stripped = line.strip()
        if (
            stripped
            and not re.search(r"\d", stripped)
            and not _EMAIL_RE.search(stripped)
            and not re.search(r"http|www\.|linkedin|github", stripped, re.I)
            and len(stripped.split()) <= 5  # names are short
        ):
            return stripped
    return "Unknown"


# ── Public API ────────────────────────────────────────────────────────────────

def parse_resume_pdf(pdf_path: str | Path) -> dict:
    """
    Parse a candidate's resume PDF and return a structured profile dict.

    Returns
    -------
    dict with keys:
        name, email, phone, linkedin, github,
        summary_text, education, experience, skills_raw,
        certifications, projects, full_text
    """
    full_text = extract_text_from_pdf(pdf_path)
    lines = full_text.splitlines()

    # ── Contact info ──────────────────────────────────────────────────────
    emails    = _EMAIL_RE.findall(full_text)
    phones    = _PHONE_RE.findall(full_text)
    linkedins = _LINKEDIN_RE.findall(full_text)
    githubs   = _GITHUB_RE.findall(full_text)

    # ── Sections ──────────────────────────────────────────────────────────
    education      = _find_section(lines, _SECTION_KEYWORDS["education"])
    experience     = _find_section(lines, _SECTION_KEYWORDS["experience"])
    skills_section = _find_section(lines, _SECTION_KEYWORDS["skills"])
    certifications = _find_section(lines, _SECTION_KEYWORDS["certifications"])
    projects       = _find_section(lines, _SECTION_KEYWORDS["projects"])
    summary_lines  = _find_section(lines, _SECTION_KEYWORDS["summary"])

    # ── Skills raw text (join section + full text fallback) ───────────────
    skills_raw = " ".join(skills_section) if skills_section else ""

    profile = {
        "name":          _extract_name(lines),
        "email":         emails[0]    if emails    else None,
        "phone":         phones[0]    if phones    else None,
        "linkedin":      linkedins[0] if linkedins else None,
        "github":        githubs[0]   if githubs   else None,
        "summary_text":  " ".join(summary_lines),
        "education":     education,
        "experience":    experience,
        "skills_raw":    skills_raw,
        "certifications":certifications,
        "projects":      projects,
        "full_text":     full_text,
    }

    return profile


def resume_profile_to_dataframe_row(profile: dict) -> dict:
    """
    Convert a parsed resume profile into a flat dict compatible with
    the 'resume_df' row format used by the rest of the engine.

    The returned dict has:
        ID          → auto-generated hash
        Category    → 'UPLOADED_PDF' (caller should override if known)
        Resume_str  → full_text (used for TF-IDF)
    """
    import hashlib
    uid = int(hashlib.md5(profile["full_text"].encode()).hexdigest(), 16) % (10**8)
    return {
        "ID":          uid,
        "Category":    "UPLOADED_PDF",
        "Resume_str":  profile["full_text"],
        "_name":       profile["name"],
        "_email":      profile["email"],
        "_phone":      profile["phone"],
        "_linkedin":   profile["linkedin"],
        "_github":     profile["github"],
    }
