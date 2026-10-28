# mcp_server/server.py: FastMCP microservice for cutting video clips (FIX-15)
import os
import subprocess
import uuid
from fastmcp import FastMCP

# We name our server and initialize it. This is the entity Gemini will talk to.
mcp = FastMCP("Video Editor Microservice")

# Root storage folder structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "storage"))
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
CLIPS_DIR = os.path.join(STORAGE_DIR, "clips")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)


def seconds_to_hhmmss(seconds: int | float) -> str:
    """Formats numeric seconds into HH:MM:SS.ss for human-readable logging."""
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    millis = int(round((total_seconds - int(total_seconds)) * 100))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:02d}"


@mcp.tool()
def extract_video_clip(video_filename: str, start_time: float, duration: float = 5.0) -> str:
    """Cuts a specific segment from a stored video file using FFmpeg with fast keyframe seeking.

    Use this tool when a user asks to see a clip, scene, or moment from a video.

    Args:
        video_filename: The name of the source video file (e.g., 'sample.mp4').
        start_time: The start time in seconds where the clip should begin (e.g., 14.5).
        duration: The duration of the clip in seconds (default is 5.0, range 1.0-30.0).

    Returns:
        A string representing the relative path to the newly created video clip (e.g., 'clips/clip_abc123.mp4').
    """
    clean_filename = os.path.basename(video_filename)
    start = max(0.0, float(start_time))
    dur = min(30.0, max(1.0, float(duration)))

    print(f"[mcp] Clip requested: '{clean_filename}' at {start:.2f}s ({seconds_to_hhmmss(start)}) for {dur:.2f}s")

    # 1. Verify the source video exists
    input_path = os.path.join(VIDEOS_DIR, clean_filename)
    if not os.path.exists(input_path):
        return f"Error: The video file {clean_filename} was not found in the videos directory."

    # 2. Generate a unique filename for the new clip
    clip_id = str(uuid.uuid4())[:8]
    output_filename = f"clip_{clip_id}.mp4"
    output_path = os.path.join(CLIPS_DIR, output_filename)

    # 3. Fast seeking: -ss BEFORE -i for rapid keyframe seek, +faststart for browser streaming
    command = [
        "ffmpeg",
        "-ss", f"{start:.2f}",
        "-i", input_path,
        "-t", f"{dur:.2f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y",
        output_path,
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[mcp] Successfully generated {output_filename}")
        return f"clips/{output_filename}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="replace")
        print(f"[mcp] FFmpeg Error: {error_msg}")
        return f"Error: FFmpeg failed to extract the clip. Details: {error_msg}"
    except Exception as e:
        print(f"[mcp] System Error: {e}")
        return f"Error: An unexpected system error occurred: {str(e)}"


if __name__ == "__main__":
    print("[mcp] Starting Video Editor FastMCP Microservice on port 9090...")
    mcp.run(transport="sse", host="0.0.0.0", port=9090)
