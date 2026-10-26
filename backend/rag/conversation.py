# backend/rag/conversation.py
import os
import re
from collections import defaultdict, deque
from typing import Optional, List, Tuple, Any

MAX_TURNS = int(os.environ.get("MEMORY_TURNS", 6))
COND_MODEL = os.environ.get("CONDENSE_MODEL", "gemini-2.5-flash")

# Session and video scoped in-memory conversation storage
# Key: (session_id, video_title) -> deque of (role, text)
_mem: dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS * 2))

ANAPHORA = re.compile(
    r"\b(it|that|this|they|he|she|them|then|again|earlier|before|after|same|there|more|his|her|their|its)\b",
    re.I,
)


def add(session: Optional[str], title: Optional[str], role: str, text: str) -> None:
    """Appends a turn to the session-and-video-scoped conversation memory."""
    s = session or "default"
    t = title or ""
    _mem[(s, t)].append((role, (text or "")[:600]))


def history(session: Optional[str], title: Optional[str]) -> List[Tuple[str, str]]:
    """Returns the list of turns for the specified session and video."""
    s = session or "default"
    t = title or ""
    return list(_mem[(s, t)])


def history_text(h: List[Tuple[str, str]], max_turns: int = 6) -> str:
    """Formats the recent turns for prompt injection or condensation."""
    recent = h[-max_turns:]
    return "\n".join(f"{r.capitalize()}: {t}" for r, t in recent)


def clear(session: Optional[str] = None, title: Optional[str] = None) -> None:
    """Clears conversation memory. If session and title are omitted, clears all."""
    if session is None and title is None:
        _mem.clear()
        return

    keys_to_del = [
        k for k in _mem
        if (session is None or k[0] == session) and (title is None or k[1] == title)
    ]
    for k in keys_to_del:
        del _mem[k]


async def condense_query(client: Any, session: Optional[str], title: Optional[str], question: str) -> str:
    """
    Rewrites a follow-up question containing pronouns or anaphora into a standalone query.
    Fails open (returns original question) if no history, question is already standalone,
    or if LLM call fails.
    """
    if not question or not question.strip():
        return question

    h = history(session, title)
    if not h:
        return question

    words = question.strip().split()
    # If query is reasonably detailed and contains no anaphora, it's already standalone
    if len(words) > 8 and not ANAPHORA.search(question):
        return question

    hist_str = history_text(h)
    prompt = f"""Rewrite the final question as a standalone search query for a video index.
Resolve pronouns and references using the conversation. Keep names, numbers and technical terms verbatim.
Output only the query.

Conversation:
{hist_str}

Final question: {question}"""

    try:
        if client is None:
            return question

        # Check for async google-genai client
        if hasattr(client, "aio") and hasattr(client.aio, "models"):
            out = await client.aio.models.generate_content(
                model=COND_MODEL,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 64},
            )
        elif hasattr(client, "models"):
            out = client.models.generate_content(
                model=COND_MODEL,
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 64},
            )
        else:
            return question

        raw_q = getattr(out, "text", "") or ""
        q = raw_q.strip().strip('"').strip("'").strip()
        if 3 <= len(q) <= 300:
            return q
        return question
    except Exception as e:
        # Fail open: never break retrieval on condensation error
        print(f"[conversation] Condensation fallback to original query: {e}")
        return question
