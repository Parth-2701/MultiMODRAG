import pixeltable as pxt

def inspect_database():
    print("🔗 Connecting to Pixeltable...")
    
    # 1. Grab the table
    try:
        frames = pxt.get_table('video_rag.frames')
    except Exception as e:
        print("❌ Database not found. Did you run test_pipeline.py?")
        return

    # 2. Print the Schema (Shows all your computed columns and types)
    print("\n📋 DATABASE SCHEMA:")
    print("-" * 60)
    print(frames)
    
    # 3. Fetch the actual data as a Pandas DataFrame for a clean visual check
    print("\n📊 ACTUAL SAVED DATA (Top 5 Frames):")
    print("-" * 60)
    
    # We select the timestamp (pos) and your clean_caption
    df = frames.select(frames.pos, frames.clean_caption).limit(5).collect().to_pandas()
    
    # Print the dataframe without the row index for cleaner output
    print(df.to_string(index=False))
    
    print("\n✅ If you see your captions above, the embeddings are successfully saved in the background!")

if __name__ == "__main__":
    inspect_database()