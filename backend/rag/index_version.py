"""backend/rag/index_version.py: Version tracking for vector index configurations (FIX-14).

Ensures that:
1. Server restarts do not destructively rebuild indexes or re-embed frames.
2. If embedding model/dimension configurations change, the system emits an actionable
   warning requiring manual reset rather than silently issuing massive API calls on restart.
"""

import os

INDEX_VERSION = "gemini-embedding-001/combined_caption/v1"
DEFAULT_VERSION_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "index_version.txt"))


def check_or_update_index_version(
    version_file: str = DEFAULT_VERSION_FILE,
    expected_version: str = INDEX_VERSION,
) -> bool:
    """Checks whether the persisted index version matches the current code version.

    Returns:
        True: if the version file was created or matched the expected version.
        False: if a version mismatch was detected (prints warning to run reset_db.py).
    """
    version_file = os.path.abspath(version_file)
    os.makedirs(os.path.dirname(version_file), exist_ok=True)

    if not os.path.exists(version_file):
        try:
            with open(version_file, "w", encoding="utf-8") as f:
                f.write(expected_version.strip())
        except Exception as e:
            print(f"[index_version] Warning: Could not write index version file: {e}")
        return True

    try:
        with open(version_file, "r", encoding="utf-8") as f:
            stored_version = f.read().strip()
    except Exception as e:
        print(f"[index_version] Warning: Could not read index version file: {e}")
        return True

    if stored_version != expected_version.strip():
        print(
            f"[warn] Embedding config changed from {stored_version!r} to {expected_version.strip()!r}. "
            f"Run `python backend/reset_db.py` and re-ingest. NOT re-embedding automatically on restart."
        )
        return False

    return True
