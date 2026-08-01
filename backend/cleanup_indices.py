import pixeltable as pxt

frames = pxt.get_table('frames')

# Drop all three duplicate indexes on clean_caption, then schema.py will
# recreate exactly one with an explicit, stable name on the next run.
for idx_name in ('idx2', 'idx3', 'idx4'):
    try:
        frames.drop_embedding_index(idx_name=idx_name)
        print(f"✅ Dropped {idx_name}")
    except Exception as e:
        print(f"⏭️  {idx_name}: {e}")