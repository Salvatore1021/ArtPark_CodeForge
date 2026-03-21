"""
resume_parser.py
----------------
Unified resume parsing utilities.

Merges pdf_parser.py (PyMuPDF-based PDF pipeline, contact extraction,
section detection, DataFrame adapter) with resume_parser.py (text
cleaning, wordninja artifact fix, HTML parser, plaintext fallback).

PDF parsing  → PyMuPDF (fitz)   [primary]
HTML parsing → BeautifulSoup    [for templated resumes]
Plaintext    → regex splitter   [fallback]

Public API (drop-in for both old files):
    extract_text_from_pdf(pdf_path)             → str
    parse_resume_pdf(pdf_path)                  → dict
    parse_resume_html(html_content)             → dict
    parse_resume_plaintext(text)                → dict
    resume_profile_to_dataframe_row(profile)    → dict
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF  — pip install pymupdf
from bs4 import BeautifulSoup  # pip install beautifulsoup4


# ── Text-cleaning helpers (ported from old resume_parser.py) ─────────────────

_PROTECTED_TECH = {
    "numpy": "numpy", "pytorch": "pytorch", "tensorflow": "tensorflow",
    "matplotlib": "matplotlib", "scikit": "scikit", "sklearn": "sklearn",
    "yolov8": "yolov8", "deepsort": "deepsort", "resnet": "resnet",
    "sqlite": "sqlite", "mysql": "mysql", "postgresql": "postgresql",
    "nodejs": "nodejs", "expressjs": "expressjs", "django": "django",
    "github": "github", "gitlab": "gitlab", "linux": "linux",
    "opencv": "opencv", "fastapi": "fastapi", "mongodb": "mongodb",
}


def _fix_concatenated_words(text: str) -> str:
    """
    Fix words glued together without spaces — common in LaTeX-generated PDFs.
    Uses wordninja when available; falls back to camelCase splitting.
    Known tech names are protected from incorrect splits.
    """
    try:
        import wordninja
        lines = []
        for line in text.splitlines():
            tokens = line.split(" ")
            fixed = []
            for token in tokens:
                token_lower = token.lower().rstrip(".,;:)")
                if token_lower in _PROTECTED_TECH:
                    fixed.append(_PROTECTED_TECH[token_lower])
                elif len(token) > 20 and token.isalpha():
                    fixed.append(" ".join(wordninja.split(token)))
                else:
                    fixed.append(token)
            lines.append(" ".join(fixed))
        return "\n".join(lines)
    except ImportError:
        # camelCase fallback
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)


def _clean_pdf_text(text: str) -> str:
    """Normalise unicode punctuation, fix whitespace, strip PDF artifacts."""
    replacements = {
        "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2022": "-", "\u00b7": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = _fix_concatenated_words(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Regex patterns for contact extraction (ported from old pdf_parser.py) ────

_EMAIL_RE    = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE    = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[\w\-]+", re.IGNORECASE)

# Section keyword registry — used by both the section finder and aliases
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "summary":        ["summary", "objective", "profile", "about me",
                       "overview", "professional summary", "career summary"],
    "experience":     ["experience", "employment", "work history", "career",
                       "professional experience", "professional background"],
    "education":      ["education", "academic", "qualification", "degree",
                       "university", "college", "academics"],
    "skills":         ["skills", "competencies", "technical skills",
                       "core skills", "expertise", "core qualifications",
                       "key skills"],
    "certifications": ["certification", "certificate", "license",
                       "accreditation"],
    "projects":       ["project", "portfolio", "achievements"],
    "languages":      ["languages", "language skills"],
    "accomplishments": ["accomplishments", "awards", "honors", "recognition"],
    "additional":     ["additional", "additional information", "other",
                       "interests", "hobbies"],
}

# Flat list of all keywords — used to detect where one section ends
_ALL_SECTION_KEYWORDS: list[str] = [
    kw for kws in _SECTION_KEYWORDS.values() for kw in kws
]


# ── Section splitter ──────────────────────────────────────────────────────────

def _find_section(lines: list[str], header_keywords: list[str]) -> list[str]:
    """
    Return lines belonging to the first section whose heading matches any
    of *header_keywords*.  Stops at the next detected section heading.
    """
    inside = False
    section_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if any(kw in lower for kw in header_keywords) and len(stripped) < 60:
            inside = True
            continue
        if inside and any(kw in lower for kw in _ALL_SECTION_KEYWORDS) and len(stripped) < 60:
            break
        if inside and stripped:
            section_lines.append(stripped)
    return section_lines


def _extract_name(lines: list[str]) -> str:
    """
    Heuristic: the candidate's name is usually the first short, non-blank line
    that contains no digits, e-mail addresses, or URLs.
    """
    for line in lines[:10]:
        stripped = line.strip()
        if (
            stripped
            and not re.search(r"\d", stripped)
            and not _EMAIL_RE.search(stripped)
            and not re.search(r"http|www\.|linkedin|github", stripped, re.I)
            and len(stripped.split()) <= 5
        ):
            return stripped
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Return the full, cleaned plain-text extracted from every page of a PDF.
    Useful for non-resume PDFs (job descriptions, etc.).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    raw = "".join(page.get_text() for page in doc)
    doc.close()
    return _clean_pdf_text(raw)


def parse_resume_pdf(pdf_path: str | Path) -> dict:
    """
    Parse a candidate's resume PDF and return a structured profile dict.

    Returns
    -------
    {
        name, email, phone, linkedin, github,
        summary_text   : str,
        education      : list[str],   # raw section lines
        experience     : list[str],   # raw section lines
        skills_raw     : str,
        certifications : list[str],
        projects       : list[str],
        full_text      : str,
    }

    Notes
    -----
    experience / education are returned as raw text lines (not structured
    dicts) because reliable job/degree extraction from freeform PDFs
    requires an LLM or a far more complex parser.  Downstream code should
    either use full_text for embedding/TF-IDF or pass the section lines
    to a dedicated extractor.
    """
    full_text = extract_text_from_pdf(pdf_path)  # already cleaned
    lines = full_text.splitlines()

    emails    = _EMAIL_RE.findall(full_text)
    phones    = _PHONE_RE.findall(full_text)
    linkedins = _LINKEDIN_RE.findall(full_text)
    githubs   = _GITHUB_RE.findall(full_text)

    summary_lines  = _find_section(lines, _SECTION_KEYWORDS["summary"])
    experience     = _find_section(lines, _SECTION_KEYWORDS["experience"])
    education      = _find_section(lines, _SECTION_KEYWORDS["education"])
    skills_section = _find_section(lines, _SECTION_KEYWORDS["skills"])
    certifications = _find_section(lines, _SECTION_KEYWORDS["certifications"])
    projects       = _find_section(lines, _SECTION_KEYWORDS["projects"])

    return {
        "name":           _extract_name(lines),
        "email":          emails[0]    if emails    else None,
        "phone":          phones[0]    if phones    else None,
        "linkedin":       linkedins[0] if linkedins else None,
        "github":         githubs[0]   if githubs   else None,
        "summary_text":   " ".join(summary_lines),
        "education":      education,
        "experience":     experience,
        "skills_raw":     " ".join(skills_section),
        "certifications": certifications,
        "projects":       projects,
        "full_text":      full_text,
    }


def parse_resume_html(html_content: str) -> dict:
    """
    Parse a templated HTML resume and return a structured dict.

    Returns
    -------
    {
        raw_text       : str,
        summary        : str,
        experience     : list[dict],   # {title, company, start_date, end_date, responsibilities}
        education      : list[dict],   # {degree, field, institution, year}
        skills         : list[str],
        languages      : list[str],
        accomplishments: list[str],
        additional     : str,
    }

    Expected HTML structure: section IDs of the form SECTION_SUMM,
    SECTION_EXPR, SECTION_EDUC, SECTION_SKLL / SECTION_HILT,
    SECTION_LANG, SECTION_ACCM, SECTION_ADDI (Resume.io / Zety style).
    Falls back gracefully when sections are absent.
    """
    html_content = html_content.strip()
    if html_content.startswith('"') and html_content.endswith('"'):
        html_content = html_content[1:-1]
    html_content = html_content.replace('""', '"')

    soup = BeautifulSoup(html_content, "html.parser")
    result: dict = {
        "raw_text":        soup.get_text(separator=" ", strip=True),
        "summary":         "",
        "experience":      [],
        "education":       [],
        "skills":          [],
        "languages":       [],
        "accomplishments": [],
        "additional":      "",
    }

    # Summary
    sec = soup.find(id=re.compile(r"SECTION_SUMM", re.I))
    if sec:
        result["summary"] = sec.get_text(separator=" ", strip=True)

    # Experience
    sec = soup.find(id=re.compile(r"SECTION_EXPR", re.I))
    if sec:
        for para in sec.find_all("div", class_="paragraph"):
            job = {
                "title": "", "company": "",
                "start_date": "", "end_date": "",
                "responsibilities": [],
            }
            el = para.find(class_=re.compile(r"jobtitle"))
            if el:
                job["title"] = el.get_text(strip=True)
            el = para.find(class_=re.compile(r"companyname"))
            if el:
                job["company"] = el.get_text(strip=True)
            dates = para.find_all(class_=re.compile(r"jobdates"))
            if len(dates) >= 2:
                job["start_date"] = dates[0].get_text(strip=True)
                job["end_date"]   = dates[1].get_text(strip=True)
            elif len(dates) == 1:
                job["start_date"] = dates[0].get_text(strip=True)
            job["responsibilities"] = [
                li.get_text(strip=True) for li in para.find_all("li")
                if li.get_text(strip=True)
            ]
            if job["title"] or job["responsibilities"]:
                result["experience"].append(job)

    # Education
    sec = soup.find(id=re.compile(r"SECTION_EDUC", re.I))
    if sec:
        for para in sec.find_all("div", class_="paragraph"):
            edu = {"degree": "", "field": "", "institution": "", "year": ""}
            el = para.find(class_=re.compile(r"degree"))
            if el:
                edu["degree"] = el.get_text(strip=True)
            el = para.find(class_=re.compile(r"programline"))
            if el:
                edu["field"] = el.get_text(strip=True)
            el = para.find(class_=re.compile(r"companyname_educ"))
            if el:
                edu["institution"] = el.get_text(strip=True)
            el = para.find(class_=re.compile(r"jobdates"))
            if el:
                edu["year"] = el.get_text(strip=True)
            if any(edu.values()):
                result["education"].append(edu)

    # Skills
    sec = soup.find(id=re.compile(r"SECTION_SKLL|SECTION_HILT", re.I))
    if sec:
        text = sec.get_text(separator=",", strip=True)
        raw = [s.strip() for s in re.split(r"[,\n•]", text) if s.strip()]
        result["skills"] = [
            s for s in raw
            if len(s) > 1 and not any(
                alias in s.lower() for alias in ["skill", "qualification", "competenc"]
            )
        ]

    # Languages
    sec = soup.find(id=re.compile(r"SECTION_LANG", re.I))
    if sec:
        text = sec.get_text(separator=",", strip=True)
        langs = [l.strip() for l in re.split(r"[,/\n]", text) if l.strip()]
        result["languages"] = [l for l in langs if len(l) > 1 and "language" not in l.lower()]

    # Accomplishments
    sec = soup.find(id=re.compile(r"SECTION_ACCM", re.I))
    if sec:
        result["accomplishments"] = [
            li.get_text(strip=True) for li in sec.find_all("li")
            if li.get_text(strip=True)
        ]

    # Additional
    sec = soup.find(id=re.compile(r"SECTION_ADDI", re.I))
    if sec:
        result["additional"] = sec.get_text(separator=" ", strip=True)

    return result


def parse_resume_plaintext(text: str) -> dict:
    """
    Fallback parser for unstructured plain-text resumes.
    Splits on common section headings and returns raw text per section.
    Useful when PDF/HTML extraction fails or as a pre-processing step.
    """
    HEADING_RE = re.compile(
        r"^(summary|objective|experience|employment|education|skills|"
        r"certifications|languages|accomplishments|projects|references)"
        r"\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    sections: dict[str, str] = {}
    current = "header"
    buffer: list[str] = []

    for line in text.splitlines():
        if HEADING_RE.match(line.strip()):
            sections[current] = "\n".join(buffer).strip()
            current = line.strip().lower()
            buffer = []
        else:
            buffer.append(line)

    sections[current] = "\n".join(buffer).strip()
    return sections


def resume_profile_to_dataframe_row(profile: dict) -> dict:
    """
    Convert a parsed resume profile (output of parse_resume_pdf or
    parse_resume_html) into a flat dict compatible with the resume_df
    row format used by the matching engine.

    Keys:  ID, Category, Resume_str, _name, _email, _phone,
           _linkedin, _github
    """
    # Works for both PDF profiles (full_text) and HTML profiles (raw_text)
    text = profile.get("full_text") or profile.get("raw_text", "")
    uid = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10 ** 8)
    return {
        "ID":        uid,
        "Category":  "UPLOADED_PDF",   # caller should override if known
        "Resume_str": text,
        "_name":     profile.get("name",    profile.get("_name")),
        "_email":    profile.get("email",   profile.get("_email")),
        "_phone":    profile.get("phone",   profile.get("_phone")),
        "_linkedin": profile.get("linkedin",profile.get("_linkedin")),
        "_github":   profile.get("github",  profile.get("_github")),
    }