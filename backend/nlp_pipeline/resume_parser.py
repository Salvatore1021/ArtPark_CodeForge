"""
resume_parser.py
----------------
Parses HTML resume files into structured sections (experience, education,
skills, summary, etc.) using BeautifulSoup.

Supports:
  - Standard HTML resume templates (class-based selectors)
  - Plain-text fallback for unstructured HTML
"""

import re
from bs4 import BeautifulSoup


# ── Section title aliases ──────────────────────────────────────────────────
SECTION_ALIASES = {
    "summary":       ["summary", "profile", "objective", "about", "overview",
                      "professional summary", "career summary"],
    "experience":    ["experience", "employment", "work history", "career",
                      "work experience", "professional experience"],
    "education":     ["education", "academic", "qualification", "degree",
                      "academics"],
    "skills":        ["skills", "core qualifications", "competencies",
                      "technical skills", "key skills", "expertise"],
    "accomplishments": ["accomplishments", "achievements", "awards",
                        "certifications", "honors", "recognition"],
    "languages":     ["languages", "language skills"],
    "additional":    ["additional", "additional information", "other",
                      "interests", "hobbies"],
}


def _normalize_section_title(title: str) -> str:
    """Map a raw section heading string to a canonical section key."""
    title_lower = title.strip().lower()
    for canonical, aliases in SECTION_ALIASES.items():
        if any(alias in title_lower for alias in aliases):
            return canonical
    return title_lower  # keep as-is if no match


def parse_resume_html(html_content: str) -> dict:
    """
    Parse an HTML resume and return a dict with canonical section keys.

    Returns
    -------
    {
        "raw_text":    str,          # full plain-text of the resume
        "summary":     str,
        "experience":  list[dict],   # [{title, dates, responsibilities:[]}]
        "education":   list[dict],   # [{degree, field, institution, year}]
        "skills":      list[str],
        "languages":   list[str],
        "accomplishments": list[str],
        "additional":  str,
    }
    """

    # Strip outer wrapping quotes that appear when HTML is saved as a quoted string
    html_content = html_content.strip()
    if html_content.startswith('"') and html_content.endswith('"'):
        html_content = html_content[1:-1]
    # Unescape doubled quotes used as an escape sequence: "" -> "
    html_content = html_content.replace('""', '"')

    soup = BeautifulSoup(html_content, "html.parser")

    result = {
        "raw_text":        "",
        "summary":         "",
        "experience":      [],
        "education":       [],
        "skills":          [],
        "languages":       [],
        "accomplishments": [],
        "additional":      "",
    }

    # ── 1. Full plain text (used by skill extractor) ─────────────────────
    result["raw_text"] = soup.get_text(separator=" ", strip=True)

    # ── 2. Summary ────────────────────────────────────────────────────────
    summ_section = soup.find(id=re.compile(r"SECTION_SUMM", re.I))
    if summ_section:
        result["summary"] = summ_section.get_text(separator=" ", strip=True)

    # ── 3. Experience ─────────────────────────────────────────────────────
    expr_section = soup.find(id=re.compile(r"SECTION_EXPR", re.I))
    if expr_section:
        for para in expr_section.find_all("div", class_="paragraph"):
            job = {
                "title":            "",
                "company":          "",
                "start_date":       "",
                "end_date":         "",
                "responsibilities": [],
            }
            title_el = para.find(class_=re.compile(r"jobtitle"))
            if title_el:
                job["title"] = title_el.get_text(strip=True)

            company_el = para.find(class_=re.compile(r"companyname"))
            if company_el:
                job["company"] = company_el.get_text(strip=True)

            dates = para.find_all(class_=re.compile(r"jobdates"))
            if len(dates) >= 2:
                job["start_date"] = dates[0].get_text(strip=True)
                job["end_date"]   = dates[1].get_text(strip=True)
            elif len(dates) == 1:
                job["start_date"] = dates[0].get_text(strip=True)

            for li in para.find_all("li"):
                text = li.get_text(strip=True)
                if text:
                    job["responsibilities"].append(text)

            # Only append if we got something meaningful
            if job["title"] or job["responsibilities"]:
                result["experience"].append(job)

    # ── 4. Education ──────────────────────────────────────────────────────
    educ_section = soup.find(id=re.compile(r"SECTION_EDUC", re.I))
    if educ_section:
        for para in educ_section.find_all("div", class_="paragraph"):
            edu = {"degree": "", "field": "", "institution": "", "year": ""}

            degree_el = para.find(class_=re.compile(r"degree"))
            if degree_el:
                edu["degree"] = degree_el.get_text(strip=True)

            program_el = para.find(class_=re.compile(r"programline"))
            if program_el:
                edu["field"] = program_el.get_text(strip=True)

            school_el = para.find(class_=re.compile(r"companyname_educ"))
            if school_el:
                edu["institution"] = school_el.get_text(strip=True)

            year_el = para.find(class_=re.compile(r"jobdates"))
            if year_el:
                edu["year"] = year_el.get_text(strip=True)

            if any(edu.values()):
                result["education"].append(edu)

    # ── 5. Skills section ─────────────────────────────────────────────────
    skills_section = soup.find(id=re.compile(r"SECTION_SKLL|SECTION_HILT", re.I))
    if skills_section:
        text = skills_section.get_text(separator=",", strip=True)
        raw_skills = [s.strip() for s in re.split(r"[,\n•]", text) if s.strip()]
        # Remove the section title itself
        raw_skills = [s for s in raw_skills if len(s) > 1
                      and not any(alias in s.lower()
                                  for alias in ["skill", "qualification",
                                                "competenc"])]
        result["skills"] = raw_skills

    # ── 6. Languages ─────────────────────────────────────────────────────
    lang_section = soup.find(id=re.compile(r"SECTION_LANG", re.I))
    if lang_section:
        text = lang_section.get_text(separator=",", strip=True)
        langs = [l.strip() for l in re.split(r"[,/\n]", text) if l.strip()]
        langs = [l for l in langs if len(l) > 1
                 and "language" not in l.lower()]
        result["languages"] = langs

    # ── 7. Accomplishments ────────────────────────────────────────────────
    accm_section = soup.find(id=re.compile(r"SECTION_ACCM", re.I))
    if accm_section:
        result["accomplishments"] = [
            li.get_text(strip=True)
            for li in accm_section.find_all("li")
            if li.get_text(strip=True)
        ]

    # ── 8. Additional ─────────────────────────────────────────────────────
    addi_section = soup.find(id=re.compile(r"SECTION_ADDI", re.I))
    if addi_section:
        result["additional"] = addi_section.get_text(separator=" ", strip=True)

    return result


# ── Plain-text fallback parser (for non-templated resumes) ────────────────
def parse_resume_plaintext(text: str) -> dict:
    """
    Fallback parser for unstructured plain-text resumes.
    Splits on common section headings and returns raw text per section.
    """
    HEADING_RE = re.compile(
        r"^(summary|objective|experience|employment|education|skills|"
        r"certifications|languages|accomplishments|projects|references)"
        r"\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    sections = {}
    current = "header"
    buffer = []

    for line in text.splitlines():
        if HEADING_RE.match(line.strip()):
            sections[current] = "\n".join(buffer).strip()
            current = line.strip().lower()
            buffer = []
        else:
            buffer.append(line)

    sections[current] = "\n".join(buffer).strip()
    return sections
