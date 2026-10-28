"""backend/cleanup_indices.py: Maintenance script to drop legacy or duplicate indexes (FIX-14)."""

import pixeltable as pxt


def cleanup_legacy_indices(table_name: str = "video_rag.frames"):
    try:
        frames = pxt.get_table(table_name)
    except Exception as e:
        print(f"[cleanup] Could not access table {table_name!r}: {e}")
        return

    # Drop legacy un-named or duplicate indexes from earlier iterations
    for idx_name in ("idx2", "idx3", "idx4", "clean_caption_idx"):
        try:
            frames.drop_embedding_index(idx_name=idx_name)
            print(f"[ok] Dropped index {idx_name}")
        except Exception as e:
            print(f"[skip] {idx_name}: {e}")


if __name__ == "__main__":
    cleanup_legacy_indices()
