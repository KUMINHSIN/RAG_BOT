from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _load_rules_cached(path_str: str) -> list[dict]:
    path = Path(path_str)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        return []

    return [r for r in rules if isinstance(r, dict)]


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms if term)


def _rule_matches(question_lower: str, rule: dict) -> bool:
    all_of_groups = rule.get("all_of_groups", [])
    any_of_terms = rule.get("any_of_terms", [])

    if all_of_groups:
        if not isinstance(all_of_groups, list):
            return False
        for group in all_of_groups:
            if not isinstance(group, list) or not _contains_any(question_lower, group):
                return False

    if any_of_terms:
        if not isinstance(any_of_terms, list) or not _contains_any(question_lower, any_of_terms):
            return False

    return bool(all_of_groups or any_of_terms)


def expand_query_for_retrieval(
    question: str,
    rules_path: Path,
    enabled: bool,
) -> str:
    q = question.strip()
    if not enabled or not q:
        return q

    rules = _load_rules_cached(str(rules_path.resolve()))
    if not rules:
        return q

    question_lower = q.lower()
    expansions: list[str] = []

    for rule in rules:
        if not _rule_matches(question_lower, rule):
            continue

        for phrase in rule.get("expansions", []):
            if isinstance(phrase, str) and phrase and phrase not in expansions:
                expansions.append(phrase)

    if not expansions:
        return q

    return q + "\n\n檢索輔助關鍵詞: " + "；".join(expansions)
