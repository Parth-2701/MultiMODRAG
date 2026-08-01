#server.py
import os
import subprocess
import uuid
from fastmcp import FastMCP

# We name our server and initialize it. This is the entity Gemini will talk to.
mcp = FastMCP("Video Editor Microservice")

# We need to make sure the MCP server knows exactly where the root storage folder is.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "storage"))
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
CLIPS_DIR = os.path.join(STORAGE_DIR, "clips")

@mcp.tool()
def extract_video_clip(video_filename: str, start_time: float, duration: float = 5.0) -> str:
    import math

    def seconds_to_hhmmss(seconds: int | float) -> str:
        total_seconds = math.ceil(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def format_to_str(seconds: int | float) -> str:
        return str(int(math.ceil(seconds)))
    """
    Cuts a specific segment from a video file using FFmpeg. 
    Use this tool when a user asks to see a clip, scene, or moment from a video.
    
    Args:
        video_filename: The name of the source video file (e.g., 'test_bunny.mp4').
        start_time: The start time in seconds where the clip should begin (e.g., 14.5).
        duration: The duration of the clip in seconds (default is 5.0).
        
    Returns:
        A string representing the relative path to the newly created video clip.
    """
    print(f"🎬 FAST-MCP: Gemini requested a {duration}s clip from '{video_filename}' at {start_time}s")
    
    # 1. Verify the source video exists
    input_path = os.path.join(VIDEOS_DIR, video_filename)
    if not os.path.exists(input_path):
        return f"Error: The video file {video_filename} was not found in the videos directory."

    # 2. Generate a unique filename for the new clip
    clip_id = str(uuid.uuid4())[:8]
    output_filename = f"clip_{clip_id}.mp4"
    output_path = os.path.join(CLIPS_DIR, output_filename)

    command = [
        "ffmpeg",
        "-i", input_path,
        "-ss", seconds_to_hhmmss(start_time),
        "-t", format_to_str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_path
        ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ FAST-MCP: Successfully generated {output_filename}")
        return f"clips/{output_filename}"
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        print(f"❌ FAST-MCP: FFmpeg Error: {error_msg}")
        return f"Error: FFmpeg failed to extract the clip. Details: {error_msg}"
    except Exception as e:
        print(f"❌ FAST-MCP: System Error: {e}")
        return f"Error: An unexpected system error occurred: {str(e)}"

if __name__ == "__main__":
    print("🚀 Starting Video Editor FastMCP Microservice on port 9090...")
    # We use SSE (Server-Sent Events) transport to expose this over a standard HTTP port.
    # This keeps our microservice completely decoupled from the main FastAPI backend.
    mcp.run(transport='sse', port=9090)