"""
Unit tests for FIX-07: Conversation memory and follow-up query condensation.
Verifies scoped storage, bounding, anaphora detection, and condensation fail-open.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.rag.conversation import (
    add,
    history,
    history_text,
    clear,
    condense_query,
    ANAPHORA,
    MAX_TURNS,
)


@pytest.fixture(autouse=True)
def clean_memory():
    clear()
    yield
    clear()


def test_scoped_memory_isolation():
    # Turns in (sess1, videoA) must not appear in (sess2, videoA) or (sess1, videoB)
    add("sess1", "videoA.mp4", "user", "Hello video A")
    add("sess1", "videoA.mp4", "assistant", "Answer video A")

    add("sess2", "videoA.mp4", "user", "Session 2 asking")
    add("sess1", "videoB.mp4", "user", "Video B asking")

    h_s1_vA = history("sess1", "videoA.mp4")
    assert len(h_s1_vA) == 2
    assert h_s1_vA[0] == ("user", "Hello video A")
    assert h_s1_vA[1] == ("assistant", "Answer video A")

    h_s2_vA = history("sess2", "videoA.mp4")
    assert len(h_s2_vA) == 1
    assert h_s2_vA[0] == ("user", "Session 2 asking")

    h_s1_vB = history("sess1", "videoB.mp4")
    assert len(h_s1_vB) == 1
    assert h_s1_vB[0] == ("user", "Video B asking")


def test_memory_bounded_turns():
    # Adding more turns than MAX_TURNS * 2 drops oldest turns
    limit = MAX_TURNS * 2
    for i in range(limit + 10):
        add("s", "v", "user", f"msg {i}")

    mem = history("s", "v")
    assert len(mem) == limit
    # The oldest retained message should be index 10
    assert mem[0] == ("user", "msg 10")
    assert mem[-1] == ("user", f"msg {limit + 9}")


def test_history_text_formatting():
    add("s", "v", "user", "What is the title?")
    add("s", "v", "assistant", "The title is Maurya.")
    add("s", "v", "user", "Who is in it?")

    formatted = history_text(history("s", "v"), max_turns=2)
    assert "Assistant: The title is Maurya." in formatted
    assert "User: Who is in it?" in formatted
    assert "What is the title?" not in formatted  # older turn omitted by max_turns


def test_clear_memory():
    add("s1", "v1", "user", "test 1")
    add("s2", "v2", "user", "test 2")

    clear("s1", "v1")
    assert len(history("s1", "v1")) == 0
    assert len(history("s2", "v2")) == 1

    clear()
    assert len(history("s2", "v2")) == 0


def test_anaphora_regex():
    assert ANAPHORA.search("What did he say?") is not None
    assert ANAPHORA.search("Tell me about that scene") is not None
    assert ANAPHORA.search("Show it again") is not None
    assert ANAPHORA.search("What happened after?") is not None
    assert ANAPHORA.search("Calculate the orbital velocity of Mars") is None


@pytest.mark.anyio
async def test_condense_query_passthrough_when_no_history():
    q = "What did he say?"
    # No history for ("s", "v")
    res = await condense_query(None, "s", "v", q)
    assert res == q


@pytest.mark.anyio
async def test_condense_query_passthrough_when_already_standalone():
    add("s", "v", "user", "Who was Dhananand?")
    add("s", "v", "assistant", "He was the Nanda king.")

    # Long query (>8 words) with no pronouns is already standalone
    standalone = "Please explain the primary defense strategy used by Chandragupta Maurya"
    res = await condense_query(None, "s", "v", standalone)
    assert res == standalone


@pytest.mark.anyio
async def test_condense_query_with_mock_client():
    add("s", "v", "user", "Who is Pandit Vishnu Gupta?")
    add("s", "v", "assistant", "Pandit Vishnu Gupta is Chanakya, royal advisor.")

    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = MagicMock()

    mock_response = MagicMock()
    mock_response.text = '"What did Pandit Vishnu Gupta say to the king?"'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    res = await condense_query(mock_client, "s", "v", "What did he say to the king?")
    assert res == "What did Pandit Vishnu Gupta say to the king?"


@pytest.mark.anyio
async def test_condense_query_fails_open_on_error():
    add("s", "v", "user", "Who is the speaker?")
    add("s", "v", "assistant", "The speaker is Dr. Smith.")

    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API quota exhausted"))

    q = "What does he research?"
    res = await condense_query(mock_client, "s", "v", q)
    # Fails open: returns original query
    assert res == q
