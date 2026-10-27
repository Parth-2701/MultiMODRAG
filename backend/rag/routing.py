# backend/rag/routing.py
import re
from typing import Dict, Any

CLASSES = ["needle", "summary", "temporal", "aggregate", "clip_request"]

RULES = [
    (
        "summary",
        re.compile(
            r"\b(summar\w+|overview|what is (this|the) video about|main points?|tl;?dr|recap|brief|high[- ]level)\b",
            re.I,
        ),
    ),
    (
        "aggregate",
        re.compile(
            r"\b(how many|count|how often|number of times|every time|frequency|total occurrences?)\b",
            re.I,
        ),
    ),
    (
        "temporal",
        re.compile(
            r"\b(after|before|then|next|first|last|sequence|order|timeline|at the (start|end)|chronolog\w+)\b",
            re.I,
        ),
    ),
    (
        "clip_request",
        re.compile(
            r"\b(show|play|watch|clip|let me see|can i see|video moment)\b",
            re.I,
        ),
    ),
]

POLICY: Dict[str, Dict[str, Any]] = {
    "needle": {
        "k_spans": 6,
        "use_summary": False,
        "order": "score",
        "warn_approximate": False,
    },
    "temporal": {
        "k_spans": 14,
        "use_summary": False,
        "order": "chronological",
        "warn_approximate": False,
    },
    "aggregate": {
        "k_spans": 20,
        "use_summary": False,
        "order": "chronological",
        "warn_approximate": True,
    },
    "summary": {
        "k_spans": 4,
        "use_summary": True,
        "order": "chronological",
        "warn_approximate": False,
    },
    "clip_request": {
        "k_spans": 6,
        "use_summary": False,
        "order": "score",
        "warn_approximate": False,
    },
}


def route(question: str) -> str:
    """
    Classifies a query into one of 5 retrieval classes in <1ms using compiled regex.
    Fails open to 'needle' if no pattern matches or query is empty.
    """
    if not question or not question.strip():
        return "needle"
    for cls, pattern in RULES:
        if pattern.search(question):
            return cls
    return "needle"


def get_policy(intent_class: str) -> Dict[str, Any]:
    """Returns the retrieval and context policy for the specified intent class."""
    return POLICY.get(intent_class, POLICY["needle"])
