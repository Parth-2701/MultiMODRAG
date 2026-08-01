import os
import urllib.request
import pixeltable as pxt

# Importing schema.py automatically runs the setup and creates the tables/views
import schema

def run_test():
    schema.setup_db()
    video_path = "vid.mp4"
    if not os.path.exists(video_path):
        print("📥 Downloading a 10-second sample video...")
        url = "https://www.w3schools.com/html/mov_bbb.mp4"  # Classic Big Buck Bunny clip
        urllib.request.urlretrieve(url, video_path)
        print("✅ Download complete.")

    print("\n🔗 Connecting to Pixeltable database...")
    videos = pxt.get_table('video_rag.videos')
    frames_view = pxt.get_table('video_rag.frames')

    # Guard against duplicate inserts: re-running this script shouldn't add
    # the same video (and its frames) again.
    existing = videos.where(videos.title == "Solar System").count()
    if existing > 0:
        print(f"\n⏭️  'Solar System' already present ({existing} row(s)) — skipping insert.")
    else:
        print("\n🚀 Inserting video into the database...")
        print("⏳ Please wait! Pixeltable is extracting frames, running Florence-2 captioning locally,")
        print("   cleaning the text, and creating embeddings...")

        # This single insert command kicks off the entire declarative DAG!
        videos.insert([{"video": video_path, "title": "Solar System"}])

    print("\n✅ Processing complete! Here are the extracted frames and cleaned captions:")
    print("-" * 50)

    # caption_text is a plain string column now (Florence-2 output),
    # not a nested Gemini response — no more candidates[0].content.parts[0].text.
    results = frames_view.select(
        pos=frames_view.pos,
        caption_text=frames_view.caption_text,
        clean_caption=frames_view.clean_caption,
        caption_text_err=frames_view.caption_text.errormsg,
    ).collect()

    for r in results:
        print("pos:", r["pos"])
        if r["caption_text"] is not None:
            print("--- Caption ---")
            print(r["caption_text"])
            print("--- Cleaned Caption ---")
            print(r["clean_caption"])
        else:
            print("Caption is None — ERROR DETAILS:")
            print("caption_text error:", r["caption_text_err"])
        print("-" * 50)

if __name__ == "__main__":
    run_test()