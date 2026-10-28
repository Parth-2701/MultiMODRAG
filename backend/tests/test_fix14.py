"""Unit tests for FIX-14: Stop re-embedding every frame on each server start.

Verifies:
1. Index version file creation and validation.
2. Version drift detection and warning (without re-embedding).
3. Non-destructive 'ignore' policy in schema.py for embedding indexes.
4. Robust index cleanup logic targeting 'video_rag.frames'.
"""

import os
import inspect
from unittest.mock import MagicMock, patch
import pytest

from backend.rag.index_version import (
    INDEX_VERSION,
    check_or_update_index_version,
)
from backend.cleanup_indices import cleanup_legacy_indices
import backend.schema as schema


def test_index_version_creates_file(tmp_path):
    """When version file does not exist, check_or_update_index_version creates it and returns True."""
    vfile = str(tmp_path / "test_version.txt")
    assert not os.path.exists(vfile)

    res = check_or_update_index_version(version_file=vfile, expected_version=INDEX_VERSION)
    assert res is True
    assert os.path.exists(vfile)

    with open(vfile, "r", encoding="utf-8") as f:
        content = f.read().strip()
    assert content == INDEX_VERSION


def test_index_version_matching_file(tmp_path):
    """When version file matches expected version, returns True."""
    vfile = str(tmp_path / "test_version.txt")
    with open(vfile, "w", encoding="utf-8") as f:
        f.write(INDEX_VERSION)

    res = check_or_update_index_version(version_file=vfile, expected_version=INDEX_VERSION)
    assert res is True


def test_index_version_drift_warning(tmp_path, capsys):
    """When version differs, prints warning, does not overwrite, and returns False."""
    vfile = str(tmp_path / "test_version.txt")
    old_version = "gemini-embedding-001/old_model/v0"
    with open(vfile, "w", encoding="utf-8") as f:
        f.write(old_version)

    res = check_or_update_index_version(version_file=vfile, expected_version=INDEX_VERSION)
    assert res is False

    captured = capsys.readouterr()
    assert "Embedding config changed" in captured.out
    assert "reset_db.py" in captured.out
    assert "NOT re-embedding automatically on restart" in captured.out

    # Persisted file is not overwritten
    with open(vfile, "r", encoding="utf-8") as f:
        assert f.read().strip() == old_version


def test_schema_embedding_index_uses_ignore_policy():
    """Verify that schema.setup_db uses if_exists='ignore' on embedding indexes."""
    src = inspect.getsource(schema.setup_db)

    # Must contain if_exists='ignore' for combined_caption
    assert "if_exists='ignore'" in src
    # Must NOT use 'replace' on any add_embedding_index call
    assert "if_exists='replace'" not in src


def test_cleanup_indices_targets_namespaced_table():
    """Verify cleanup_legacy_indices targets video_rag.frames and safely handles drops."""
    mock_frames = MagicMock()
    mock_pxt = MagicMock()
    mock_pxt.get_table.return_value = mock_frames

    with patch("backend.cleanup_indices.pxt", mock_pxt):
        cleanup_legacy_indices("video_rag.frames")
        mock_pxt.get_table.assert_called_once_with("video_rag.frames")
        # Should attempt to drop legacy index names
        assert mock_frames.drop_embedding_index.call_count >= 1
