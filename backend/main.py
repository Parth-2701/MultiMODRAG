from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import shutil
import os
import uuid
from google import genai
from google.genai import types

# Import our database schema
import schema

# MCP client (for the video-clip-extraction tool server on port 9090)
from mcp import ClientSession
from mcp.client.sse import sse_client

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # schema.py patches subprocess.run's pg_ctl timeout to tolerate this
    # machine's ~30-35s AV file-lock delay on Postgres cold start.
    # See schema.py for details.
    print("Initializing Pixeltable schema on startup...")
    schema.setup_db()
    print("✅ Schema ready.")
    yield

app = FastAPI(title="Video RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "storage"))
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
CLIPS_DIR = os.path.join(STORAGE_DIR, "clips")

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# --- google.genai SDK: create a Client instead of calling configure() ---
api_key = os.environ.get("GEMINI_API_KEY")
genai_client: Optional[genai.Client] = None
if api_key and api_key != "your-gemini-api-key-here":
    genai_client = genai.Client(api_key=api_key)
else:
    print("⚠️  GEMINI_API_KEY not set — chat synthesis will be unavailable.")

task_store = {}
chat_memory = []

class ChatRequest(BaseModel):
    message: str
    video_path: Optional[str] = None
    image_base64: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    clip_path: Optional[str] = None


def process_video_background(task_id: str, file_path: str, title: str):
    try:
        task_store[task_id] = "processing"
        import pixeltable as pxt
        videos = pxt.get_table('video_rag.videos')

        # Guard against duplicate inserts: re-uploading the same filename
        # (e.g. repeated Swagger "Try it out" clicks) shouldn't reprocess
        # and re-add the same video's frames again.
        existing = videos.where(videos.title == title).count()
        if existing > 0:
            print(f"⏭️  '{title}' already processed ({existing} row(s)) — skipping.")
            task_store[task_id] = "completed"
            return

        videos.insert(video=file_path, title=title)

        # combined_caption is a plain (non-computed) column — see schema.py
        # for why. It must be explicitly backfilled here via ordinary Python,
        # after captioning/transcription have finished as part of insert()
        # above, or it stays null forever.
        schema.backfill_combined_captions(title)

        task_store[task_id] = "completed"
    except Exception as e:
        print(f"Error processing video: {e}")
        task_store[task_id] = "failed"


@app.post("/api/v1/video/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported.")
    local_path = os.path.join(VIDEOS_DIR, file.filename)
    with open(local_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task_id = str(uuid.uuid4())
    task_store[task_id] = "pending"
    return_path = f"videos/{file.filename}"
    background_tasks.add_task(process_video_background, task_id, local_path, file.filename)
    return {"task_id": task_id, "file_path": return_path, "status": "Ingestion started"}

@app.get("/api/v1/video/task-status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": task_store[task_id]}

@app.get("/api/v1/video/media/{file_path:path}")
async def serve_media(file_path: str):
    clean_path = file_path.replace("shared_media/", "").replace("storage/", "")
    full_path = os.path.join(STORAGE_DIR, clean_path)
    if not os.path.exists(full_path):
        if os.path.exists(os.path.join(VIDEOS_DIR, clean_path)):
            full_path = os.path.join(VIDEOS_DIR, clean_path)
        elif os.path.exists(os.path.join(CLIPS_DIR, clean_path)):
            full_path = os.path.join(CLIPS_DIR, clean_path)
        else:
            raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(full_path, media_type="video/mp4")


def _clean_schema_for_gemini(schema: dict) -> dict:
    """Gemini's function-calling parameter schema is a restricted subset of
    JSON Schema (closer to OpenAPI) and rejects the whole request if it sees
    fields outside that subset. MCP's tool.inputSchema is standard JSON
    Schema and commonly includes fields like 'additionalProperties' that
    Gemini doesn't recognize — strip those out recursively before sending.
    """
    if not isinstance(schema, dict):
        return schema

    UNSUPPORTED_KEYS = {
        'additionalProperties', 'additional_properties', '$schema',
        'title', 'examples', 'const', '$id', 'default',
    }

    cleaned = {}
    for key, value in schema.items():
        if key in UNSUPPORTED_KEYS:
            continue
        if key == 'properties' and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _clean_schema_for_gemini(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == 'items' and isinstance(value, dict):
            cleaned[key] = _clean_schema_for_gemini(value)
        elif isinstance(value, dict):
            cleaned[key] = _clean_schema_for_gemini(value)
        else:
            cleaned[key] = value
    return cleaned

@app.get("/api/v1/video/list")
async def list_videos():
    """Returns every video already stored in Pixeltable, so the frontend can
    rebuild its video tray after a page refresh instead of losing track of
    videos that were already uploaded and processed in a previous session.
 
    Note: status is reported as "completed" for every row here, since
    task_store is in-memory and doesn't persist across server restarts —
    if a video's row exists in Pixeltable at all, its ingestion pipeline
    already ran to completion (or is currently running from a still-live
    task_store entry, in which case that entry's status takes precedence
    on the frontend once polling picks it up again)."""
    import pixeltable as pxt
    try:
        videos = pxt.get_table('video_rag.videos')
    except pxt.exceptions.Error:
        return {"videos": []}
 
    rows = videos.select(videos.title, videos.video).collect()
    result = []
    for r in rows:
        filename = os.path.basename(r['video'])
        result.append({
            "title": r['title'],
            "file_path": f"videos/{filename}",
            "status": "completed",
        })
    return {"videos": result}
 
# --- API ENDPOINTS: CHAT (TRUE HYBRID SEARCH + MCP TOOLS) ---
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    user_msg = request.message
    chat_memory.append({"role": "user", "content": user_msg})

    import pixeltable as pxt
    try:
        frames = pxt.get_table('video_rag.frames')
    except pxt.exceptions.Error:
        raise HTTPException(status_code=500, detail="Database not initialized. Upload a video first.")

    # Scope the search to the specific video the user is asking about.
    # Without this, the search runs across EVERY video ever uploaded — so
    # leftover frames from earlier test videos can outrank/replace the
    # video actually being asked about, especially early on when the newly
    # uploaded video's frames may still be null/processing.
    frame_filter = frames.combined_caption != None
    if request.video_path:
        target_filename = os.path.basename(request.video_path)
        # frames.video is NOT the original upload path. Pixeltable copies local
        # media into its own internal cache (~/.pixeltable/media/...) and returns
        # that hashed cache path from queries — comparing it against our own
        # storage/videos/<file> path never matches. `title` is the column we
        # control (set at insert time in process_video_background) and it's
        # inherited unchanged into this view from the base `videos` table.
        frame_filter = frame_filter & (frames.title == target_filename)

    # 1. DENSE SEARCH (Vector Similarity)
    # Pool size matters: capping this too low means a frame can be excluded
    # before re-ranking ever sees it, even if that frame is exactly what the
    # question needs (e.g. an opening frame with key exposition, ranked low
    # on raw embedding similarity to a vague query wording). For typical
    # short demo videos (a couple minutes at most), pulling every frame as a
    # re-ranking candidate is cheap and maximizes recall.
    total_frame_count = frames.where(frame_filter).count()
    pool_size = max(total_frame_count, 15)

    dense_results = (
        frames.where(frame_filter)
        .order_by(frames.combined_caption.similarity(user_msg))
        .select(frames.pos, frames.combined_caption, frames.video, frames.title)
        .limit(pool_size)
        .collect()
    )

    if len(dense_results) == 0:
        still_processing = any(status == "processing" for status in task_store.values())
        if still_processing:
            return ChatResponse(
                message="Your video is still being processed (frame extraction + captioning "
                        "can take a minute or two on CPU) — try again shortly.",
                clip_path=None,
            )
        if request.video_path:
            return ChatResponse(
                message=f"I couldn't find any processed frames for "
                        f"'{os.path.basename(request.video_path)}'. Make sure it finished "
                        f"uploading and processing (check task-status) before chatting about it.",
                clip_path=None,
            )
        return ChatResponse(message="I couldn't find any processed videos. Upload one first.", clip_path=None)
    # 2. HYBRID RE-RANKING (Reciprocal Rank Fusion - RRF)
    # We score the pool based on exact keyword overlap (Sparse / BM25 proxy)
    query_terms = set(user_msg.lower().replace("?", "").replace(".", "").split())

    # frames.pos is the frame's sequential INDEX in the sampled sequence, not
    # a real timestamp — schema.FRAME_SAMPLE_FPS is the single source of
    # truth for the sampling rate, imported directly so this file can't drift
    # out of sync with it (the previous hardcoded copy here had already
    # drifted: comment said 0.5, actual value was 1.0).
    FRAME_SAMPLE_FPS = schema.FRAME_SAMPLE_FPS

    ranked_frames = []
    for i, row in enumerate(dense_results):
        caption_text = row['combined_caption'].lower()
        real_timestamp_seconds = row['pos'] / FRAME_SAMPLE_FPS

        # Dense Rank (1 is best)
        dense_rank = i + 1

        # Sparse Score (How many exact words match)
        keyword_hits = sum(1 for term in query_terms if term in caption_text and len(term) > 3)
        # Convert score to a pseudo-rank (higher hits = better rank)
        sparse_rank = 15 - keyword_hits

        # The RRF Formula: 1 / (k + rank) where k is traditionally 60
        k = 60
        rrf_score = (1 / (k + dense_rank)) + (1 / (k + sparse_rank))

        ranked_frames.append({
            "pos": row['pos'],
            "timestamp_seconds": real_timestamp_seconds,
            "caption": row['combined_caption'],
            "video_path": row['video'],  # source video path, needed for FFmpeg clip extraction
            "source_filename": row['title'],
            "score": rrf_score
        })

    # Sort by the new Hybrid RRF score
    ranked_frames.sort(key=lambda x: x["score"], reverse=True)

    # 3. Take Top 3 for the LLM
    top_10_frames = ranked_frames[:10]

    context_str = ""
    top_timestamp = top_10_frames[0]['timestamp_seconds'] if top_10_frames else None

    # Filename from the Pixeltable file path (e.g. 'storage/videos/vid.mp4' -> 'vid.mp4')
    source_video_name = top_10_frames[0]['source_filename'] if top_10_frames else "unknown.mp4"

    for frame in top_10_frames:
        context_str += f"- At {frame['timestamp_seconds']:.1f} seconds: {frame['caption']}\n"

    # Always fetch the FULL verbatim transcript as a separate, unconditional
    # piece of context — not just whatever audio happened to get attached to
    # the top-ranked frames. Names, dates, and expository facts are often
    # spoken exactly once, early in a video, and never repeated; if that one
    # frame doesn't rank into the top-10 for a given query (a ranking
    # imperfection, not a data problem — the audio IS correctly attached to
    # that frame's combined_caption), the model would otherwise have no way
    # to know the fact/name existed at all. This guarantees it always does,
    # regardless of how retrieval ranks any individual frame.
    full_transcript_text = ""
    try:
        videos_tbl = pxt.get_table('video_rag.videos')
        transcript_rows = (
            videos_tbl.where(videos_tbl.title == source_video_name)
            .select(videos_tbl.transcript)
            .collect()
        )
        if len(transcript_rows) > 0 and isinstance(transcript_rows[0]['transcript'], dict):
            full_transcript_text = (transcript_rows[0]['transcript'].get('text') or "").strip()
    except Exception as e:
        print(f"⚠️  Could not fetch full transcript for context: {e}")

    # DEBUG: confirm what was actually retrieved before handing off to Gemini
    print("🔍 Top 10 retrieved frames:")
    for frame in top_10_frames:
        print(f"    pos={frame['pos']} timestamp={frame['timestamp_seconds']:.1f}s caption={frame['caption'][:80]!r}")

    # 4. LLM Synthesis WITH MCP TOOL CALLING
    final_clip_path: Optional[str] = None

    if genai_client is None:
        ai_response = (
            f"I found relevant scenes at {top_timestamp}s, but text generation is "
            f"unavailable (GEMINI_API_KEY not configured)."
        )
        chat_memory.append({"role": "assistant", "content": ai_response})
        return ChatResponse(message=ai_response, clip_path=None)

    prompt = f"""
    You are a helpful video analysis assistant.
    The user asked: "{user_msg}"

    Full spoken narration of this video (verbatim, in order — use this for
    names, dates, and factual/narrative details, since it's the complete
    audio track regardless of which specific frames were matched below):
    {full_transcript_text if full_transcript_text else "(no speech detected in this video)"}

    Using a Hybrid Search, I also found these specific visual scenes in the
    source video '{source_video_name}' (use these for visual details and to
    pick a timestamp if the user wants a clip):
    {context_str}

    Provide a friendly response using BOTH the full narration above and the
    scene descriptions. Prefer specific names/dates from the narration over
    vague descriptions when they're available.
    If the user's request implies they want to *see* the video, watch the clip, or see a
    specific moment, YOU MUST use the extract_video_clip tool using '{source_video_name}'
    and the relevant timestamp.
    """

    try:
        # Connect to the FastMCP microservice (video clip extraction tool)
        print("🔗 Connecting to FastMCP server on port 9090...")
        # sse_client has no default timeout, so a server that's up but not
        # responding (or a dropped connection) can hang this forever with no
        # error. Give the SSE handshake and the read stream explicit timeouts
        # (seconds) so a stuck connection fails loudly instead of silently.
        async with sse_client("http://localhost:9090/sse", timeout=90, sse_read_timeout=60) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Fetch available tools from the MCP server
                mcp_tools = await session.list_tools()

                # Convert MCP tool schemas into a format Gemini understands.
                # tool.inputSchema is cleaned first — see _clean_schema_for_gemini.
                gemini_tools = []
                for tool in mcp_tools.tools:
                    gemini_tools.append({
                        "function_declarations": [{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": _clean_schema_for_gemini(tool.inputSchema)
                        }]
                    })

                print(f"🛠️ Fetched {len(mcp_tools.tools)} tools from FastMCP.")

                # Call Gemini, providing the tools.
                # DEBUG: this call previously gave no feedback if it hung — bound
                # it with wait_for so a stall raises asyncio.TimeoutError instead
                # of blocking the request indefinitely.
                print("📤 Sending prompt to Gemini (with tools)...")
                response = await asyncio.wait_for(
                    genai_client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            tools=gemini_tools,
                        )
                    ),
                    timeout=30,
                )
                print("📥 Gemini responded (with tools).")

                # Check if Gemini decided to call a tool
                if response.function_calls:
                    for function_call in response.function_calls:
                        tool_name = function_call.name
                        tool_args = function_call.args
                        print(f"🤖 Gemini requested tool: {tool_name} with args: {tool_args}")
                        print(f"    (for reference, top frame's real timestamp was {top_timestamp:.1f}s)")

                        if tool_name == "extract_video_clip":
                            # Execute the tool on the MCP server!
                            result = await session.call_tool("extract_video_clip", arguments=tool_args)

                            # The MCP server returns a list of content objects (usually text)
                            tool_result_text = result.content[0].text
                            print(f"✅ FastMCP Tool Result: {tool_result_text}")

                            # If it was a success, it returns the relative path (e.g. 'clips/clip_123.mp4')
                            if not tool_result_text.startswith("Error"):
                                final_clip_path = tool_result_text

                            # Send the result back to Gemini so it can finish its thought
                            print("📤 Sending tool result back to Gemini...")
                            response = await asyncio.wait_for(
                                genai_client.aio.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=[
                                        prompt,
                                        response.candidates[0].content,
                                        types.Part.from_function_response(
                                            name=tool_name,
                                            response={"result": tool_result_text}
                                        )
                                    ],
                                    config=types.GenerateContentConfig(
                                        tools=gemini_tools,
                                    )
                                ),
                                timeout=30,
                            )
                            print("📥 Gemini responded (post-tool).")

                ai_response = response.text

    except Exception as e:
        # anyio/mcp wraps connection failures in an ExceptionGroup ("unhandled
        # errors in a TaskGroup"), which hides the real error. Unwrap it so
        # the actual cause (e.g. connection refused - no server on :9090)
        # shows up in the logs instead of a generic message.
        if hasattr(e, 'exceptions'):
            for sub_e in e.exceptions:  # type: ignore[attr-defined]
                print(f"⚠️  MCP/tool-calling sub-error: {type(sub_e).__name__}: {sub_e!r}")
        else:
            print(f"⚠️  MCP/tool-calling path failed: {type(e).__name__}: {e!r}")
        print("↩️  Falling back to plain Gemini call without tools...")
        try:
            response = await genai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            ai_response = response.text
        except Exception as inner_e:
            print(f"❌ Plain Gemini fallback also failed: {inner_e!r}")
            ai_response = f"I found relevant scenes at {top_timestamp}s, but my text generator failed."
        final_clip_path = None

    chat_memory.append({"role": "assistant", "content": ai_response})

    return ChatResponse(message=ai_response, clip_path=final_clip_path)

@app.get("/api/v1/debug/frames")
async def debug_frames():
    import pixeltable as pxt
    frames = pxt.get_table('video_rag.frames')
    rows = frames.select(
        frames.title, frames.video, frames.pos,
        frames.caption_text, frames.clean_caption,
        frames.combined_caption,
        frames.transcript.errormsg,
        frames.audio.errormsg,
    ).collect()
    return {"count": len(rows), "rows": list(rows)}

@app.get("/api/v1/chat/history")
async def get_chat_history():
    return {"history": chat_memory}

@app.delete("/api/v1/chat/memory")
async def clear_memory():
    global chat_memory
    chat_memory = []
    return {"status": "Memory cleared"}