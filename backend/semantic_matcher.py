"""
semantic_matcher.py
-------------------
Lightweight semantic skill matching using character n-gram TF-IDF vectors.
No LLM. No sentence transformers. Only sklearn.

Why char n-grams?
- Handles typos:      "managment" still matches "management"
- Handles morphology: "coordinating" matches "coordination"  
- Handles compounds:  "machine-learning" matches "machine learning"
- Tiny model:         computed on-the-fly, no downloads

This catches skills that EXACT MATCH misses because the resume uses
a different form of the word.

Example:
  Resume says:     "coordinating between departments"
  Exact match:     ❌ misses (taxonomy has "coordination")
  Semantic match:  ✅ finds (high char n-gram overlap)
"""

import re
import math
from collections import Counter
from skills_taxonomy import SKILL_TO_CATEGORY


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate character n-grams from text."""
    text = f"_{text.lower()}_"
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def _tfidf_vector(text: str, idf: dict, n: int = 3) -> dict:
    """Compute TF-IDF vector for a text using pre-computed IDF."""
    ngrams = _char_ngrams(text, n)
    tf = Counter(ngrams)
    total = sum(tf.values()) or 1
    return {
        ng: (count / total) * idf.get(ng, 1.0)
        for ng, count in tf.items()
    }


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(vec_a.get(k, 0) * v for k, v in vec_b.items())
    norm_a = math.sqrt(sum(v**2 for v in vec_a.values())) or 1e-9
    norm_b = math.sqrt(sum(v**2 for v in vec_b.values())) or 1e-9
    return dot / (norm_a * norm_b)


class SemanticSkillMatcher:
    """
    Builds a char n-gram TF-IDF index over the skill taxonomy,
    then matches resume phrases against it.
    """

    def __init__(self, n: int = 3, threshold: float = 0.55):
        self.n = n
        self.threshold = threshold
        self._build_index()

    def _build_index(self):
        """Pre-compute IDF and TF-IDF vectors for all taxonomy skills."""
        self.skills = list(SKILL_TO_CATEGORY.keys())

        # Compute IDF across skill corpus
        df = Counter()
        skill_ngram_sets = []
        for skill in self.skills:
            ngrams = set(_char_ngrams(skill, self.n))
            skill_ngram_sets.append(ngrams)
            for ng in ngrams:
                df[ng] += 1

        N = len(self.skills)
        self.idf = {
            ng: math.log((N + 1) / (count + 1)) + 1
            for ng, count in df.items()
        }

        # Pre-compute vectors for all skills
        self.skill_vectors = [
            _tfidf_vector(skill, self.idf, self.n)
            for skill in self.skills
        ]

    def match_phrase(self, phrase: str) -> dict | None:
        """
        Match a single phrase against the skill taxonomy.

        Returns the best match if similarity > threshold, else None.
        Returns: { skill, category, similarity, method }
        """
        phrase_vec = _tfidf_vector(phrase.lower(), self.idf, self.n)

        best_score = 0.0
        best_skill = None

        for i, skill_vec in enumerate(self.skill_vectors):
            score = _cosine_similarity(phrase_vec, skill_vec)
            if score > best_score:
                best_score = score
                best_skill = self.skills[i]

        if best_score >= self.threshold and best_skill:
            return {
                "skill":      best_skill,
                "category":   SKILL_TO_CATEGORY[best_skill],
                "similarity": round(best_score, 3),
                "method":     "semantic_match",
                "confidence": round(min(best_score, 0.85), 3),
            }
        return None

    def extract_from_text(self, text: str,
                           known_skills: set[str]) -> list[dict]:
        """
        Tokenize text into candidate phrases, match each semantically.
        Skips skills already found by exact matching.

        Returns new semantic matches not in known_skills.
        """
        # Extract candidate phrases: unigrams + bigrams + trigrams
        words = re.findall(r"\b[a-zA-Z][a-zA-Z+#.-]{1,}\b", text.lower())
        candidates = set()

        for i, w in enumerate(words):
            candidates.add(w)
            if i < len(words) - 1:
                candidates.add(f"{w} {words[i+1]}")
            if i < len(words) - 2:
                candidates.add(f"{w} {words[i+1]} {words[i+2]}")

        results = []
        seen = set(known_skills)

        for phrase in candidates:
            if phrase in seen or len(phrase) < 4:
                continue

            match = self.match_phrase(phrase)
            if match and match["skill"] not in seen:
                # Only include if the phrase is meaningfully different
                # from the skill (i.e., it's not just an exact match
                # that was already found)
                if phrase != match["skill"]:
                    results.append({
                        **match,
                        "source_phrase": phrase,
                        "reasoning": (
                            f"'{phrase}' semantically matches "
                            f"'{match['skill']}' "
                            f"(similarity: {match['similarity']})"
                        ),
                    })
                    seen.add(match["skill"])

        # Sort by similarity score descending
        return sorted(results, key=lambda x: -x["similarity"])
