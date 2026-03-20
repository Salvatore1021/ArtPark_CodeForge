"""
proficiency_detector.py
-----------------------
Detects proficiency level for each extracted skill by scanning the
surrounding sentence context for signal words.

No LLM. Fully rule-based + weighted scoring.

Proficiency Levels (maps to O*NET levels):
    beginner      → 0–1 year, exposure/learning
    intermediate  → 1–3 years, proficient/working knowledge
    advanced      → 3–7 years, strong/managed/designed
    expert        → 7+ years, architected/led/defined strategy
"""

import re
from collections import defaultdict


# ── Signal word banks ──────────────────────────────────────────────────────
# Each word has a weight (-1 = strong beginner signal, +3 = strong expert)

PROFICIENCY_SIGNALS = {
    "expert": {
        "words": [
            "expert", "architected", "designed", "led", "spearheaded",
            "pioneered", "defined", "established", "authority",
            "10+ years", "10 years", "extensive experience", "mastery",
            "principal", "chief", "head of", "director",
        ],
        "weight": 3,
    },
    "advanced": {
        "words": [
            "advanced", "senior", "strong", "managed", "built",
            "developed", "delivered", "owned", "responsible for",
            "5+ years", "5 years", "7 years", "proficient",
            "deep knowledge", "comprehensive", "implemented",
        ],
        "weight": 2,
    },
    "intermediate": {
        "words": [
            "intermediate", "working knowledge", "solid", "competent",
            "experienced", "3 years", "2 years", "used",
            "applied", "worked with", "familiar with", "contributed",
            "assisted", "supported",
        ],
        "weight": 1,
    },
    "beginner": {
        "words": [
            "beginner", "basic", "learning", "exposure", "introduction",
            "introductory", "fundamentals", "studied", "coursework",
            "training", "1 year", "6 months", "entry", "junior",
            "fresher", "intern",
        ],
        "weight": 0,
    },
}

# Flat lookup: signal_word -> (level, weight)
SIGNAL_LOOKUP = {}
for level, data in PROFICIENCY_SIGNALS.items():
    for word in data["words"]:
        SIGNAL_LOOKUP[word.lower()] = (level, data["weight"])

# Level to numeric score
LEVEL_SCORE = {
    "beginner":     0,
    "intermediate": 1,
    "advanced":     2,
    "expert":       3,
}
SCORE_LEVEL = {v: k for k, v in LEVEL_SCORE.items()}


def _get_context_window(text: str, skill: str, window: int = 120) -> list[str]:
    """
    Find all occurrences of 'skill' in text and return surrounding
    character windows (before + after).
    """
    windows = []
    pattern = re.compile(r"\b" + re.escape(skill) + r"\b", re.IGNORECASE)
    for match in pattern.finditer(text):
        start = max(0, match.start() - window)
        end   = min(len(text), match.end() + window)
        windows.append(text[start:end].lower())
    return windows


def _score_context(context: str) -> int:
    """
    Score a context window by scanning for signal words.
    Returns a numeric proficiency score (0–3).
    """
    scores = []
    for signal, (level, weight) in SIGNAL_LOOKUP.items():
        if re.search(r"\b" + re.escape(signal) + r"\b", context):
            scores.append(weight)

    if not scores:
        return 1  # default to intermediate if no signals found

    # Take the weighted average, biased toward highest signal
    avg = sum(scores) / len(scores)
    high = max(scores)
    # Blend: 60% highest signal + 40% average
    blended = int(round(0.6 * high + 0.4 * avg))
    return max(0, min(3, blended))


def detect_proficiency(skill: str, raw_text: str) -> dict:
    """
    Detect proficiency level for a single skill.

    Returns
    -------
    {
        "level":    "beginner" | "intermediate" | "advanced" | "expert",
        "score":    0 | 1 | 2 | 3,
        "signals":  ["managed", "5+ years"],   # what triggered the score
    }
    """
    contexts = _get_context_window(raw_text, skill)

    if not contexts:
        return {
            "level":   "intermediate",
            "score":   1,
            "signals": [],
        }

    best_score = 0
    best_context = contexts[0]
    all_signals = []

    for ctx in contexts:
        score = _score_context(ctx)
        if score > best_score:
            best_score = score
            best_context = ctx

        # Collect triggered signal words
        for signal in SIGNAL_LOOKUP:
            if re.search(r"\b" + re.escape(signal) + r"\b", ctx):
                all_signals.append(signal)

    return {
        "level":    SCORE_LEVEL.get(best_score, "intermediate"),
        "score":    best_score,
        "signals":  list(set(all_signals))[:5],
    }


def detect_all_proficiencies(skills: list[dict], raw_text: str) -> list[dict]:
    """
    Enrich a list of skill dicts with proficiency info.

    Input skills format:  [{ skill, category, method, confidence }, ...]
    Output adds:          { ..., proficiency: { level, score, signals } }
    """
    enriched = []
    for s in skills:
        prof = detect_proficiency(s["skill"], raw_text)
        enriched.append({**s, "proficiency": prof})
    return enriched
