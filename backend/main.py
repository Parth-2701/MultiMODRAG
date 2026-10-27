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
try:
    import schema
except ImportError:
    from backend import schema
try:
    from backend.rag.hybrid_retrieval import hybrid_search
except ImportError:
    from rag.hybrid_retrieval import hybrid_search
try:
    from backend.rag.transcript_rag import transcript_context
except ImportError:
    from rag.transcript_rag import transcript_context
try:
    from backend.rag.prompting import (
        build_user_content,
        SYSTEM_INSTRUCTION,
        MODEL,
        TEMP,
    )
except ImportError:
    from rag.prompting import (
        build_user_content,
        SYSTEM_INSTRUCTION,
        MODEL,
        TEMP,
    )
try:
    from backend.rag.spans import to_spans
except ImportError:
    from rag.spans import to_spans
try:
    from backend.rag.conversation import (
        add as mem_add,
        history as mem_history,
        history_text as mem_history_text,
        clear as mem_clear,
        condense_query,
    )
except ImportError:
    from rag.conversation import (
        add as mem_add,
        history as mem_history,
        history_text as mem_history_text,
        clear as mem_clear,
        condense_query,
    )
try:
    from backend.rag.routing import route, get_policy
    from backend.rag.summaries import ensure_summary, format_summary_context
except ImportError:
    from rag.routing import route, get_policy
    from rag.summaries import ensure_summary, format_summary_context

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

SUMMARIES_DIR = os.path.join(STORAGE_DIR, "summaries")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

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
    session_id: Optional[str] = None

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
    sid = request.session_id or "default"
    target_filename = os.path.basename(request.video_path) if request.video_path else None
    video_key = target_filename or ""

    # Condense query if follow-up refers to prior conversation turns (FIX-07)
    standalone_query = await condense_query(genai_client, sid, video_key, user_msg)
    if standalone_query != user_msg:
        print(f"🔄 [FIX-07] Condensed query: {user_msg!r} -> {standalone_query!r}")

    # Classify query intent and determine policy (FIX-08)
    intent_class = route(standalone_query)
    policy = get_policy(intent_class)
    print(f"🧭 [FIX-08] Intent class: {intent_class} (k_spans={policy['k_spans']}, use_summary={policy['use_summary']}, order={policy['order']})")

    import pixeltable as pxt
    try:
        frames = pxt.get_table('video_rag.frames')
    except pxt.exceptions.Error:
        raise HTTPException(status_code=500, detail="Database not initialized. Upload a video first.")

    # Scope search to specific video using standalone_query
    ranked_frames = hybrid_search(frames, target_filename, standalone_query, schema.FRAME_SAMPLE_FPS)

    if len(ranked_frames) == 0:
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

    # Convert ranked frame hits into cohesive, deduplicated temporal spans (FIX-04 & FIX-08)
    all_spans = to_spans(ranked_frames[:30], schema.FRAME_SAMPLE_FPS)
    k_spans_limit = policy.get("k_spans", 6)
    if policy.get("order") == "chronological":
        spans = sorted(all_spans[:k_spans_limit], key=lambda s: s["start"])
    else:
        spans = all_spans[:k_spans_limit]

    top_timestamp = spans[0]['start'] if spans else (ranked_frames[0]['timestamp_seconds'] if ranked_frames else None)
    source_video_name = spans[0]['source_filename'] if spans else (ranked_frames[0]['source_filename'] if ranked_frames else "unknown.mp4")

    # Fetch timestamped transcript context using dedicated transcript index (FIX-02)
    transcript_ctx = ""
    try:
        transcript_ctx = transcript_context(source_video_name, standalone_query)
    except Exception as e:
        print(f"[warn] Could not fetch transcript context: {e}")
        transcript_ctx = ""

    # Global summary generation if required by policy (FIX-08)
    summary_ctx = ""
    if policy.get("use_summary"):
        try:
            sum_data = await ensure_summary(genai_client, source_video_name)
            summary_ctx = format_summary_context(sum_data)
        except Exception as e:
            print(f"[warn] Could not build summary context: {e}")
            summary_ctx = ""

    # DEBUG: confirm what spans were generated before handing off to Gemini
    print(f"🔍 Generated {len(spans)} temporal spans (intent={intent_class}):")
    for s in spans:
        print(f"    [{s['start']:.1f}s–{s['end']:.1f}s] score={s['score']:.4f} text={s['text'][:80]!r}")

    # 4. LLM Synthesis WITH MCP TOOL CALLING
    final_clip_path: Optional[str] = None

    if genai_client is None:
        ai_response = (
            f"I found relevant scenes at {top_timestamp}s, but text generation is "
            f"unavailable (GEMINI_API_KEY not configured)."
        )
        mem_add(sid, video_key, "user", user_msg)
        mem_add(sid, video_key, "assistant", ai_response)
        chat_memory.append({"role": "user", "content": user_msg})
        chat_memory.append({"role": "assistant", "content": ai_response})
        return ChatResponse(message=ai_response, clip_path=None)

    # Scoped conversation history injection (FIX-07)
    hist_entries = mem_history(sid, video_key)
    hist_text = mem_history_text(hist_entries)

    visual_items = [(s['start'], s['text']) for s in spans]
    user_content = build_user_content(
        question=user_msg,
        visual_items=visual_items,
        speech_ctx=transcript_ctx,
        video_name=source_video_name,
        history_text=hist_text,
        summary_ctx=summary_ctx,
        warn_approximate=policy.get("warn_approximate", False),
    )

    try:
        # Connect to the FastMCP microservice (video clip extraction tool)
        print("🔗 Connecting to FastMCP server on port 9090...")
        # sse_client has no default timeout, so a server that's up but not
        # responding (or a dropped connection) can hang this forever with no
        # error. Give the SSE handshake and the read stream explicit timeouts
        # (seconds) so a stuck connection fails loudly instead of silently.
        mcp_url = os.environ.get("MCP_SERVER_URL", "http://localhost:9090/sse")
        async with sse_client(mcp_url, timeout=90, sse_read_timeout=60) as (read_stream, write_stream):
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
                call_config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=TEMP,
                    tools=gemini_tools,
                )
                response = await asyncio.wait_for(
                    genai_client.aio.models.generate_content(
                        model=MODEL,
                        contents=user_content,
                        config=call_config,
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
                                    model=MODEL,
                                    contents=[
                                        user_content,
                                        response.candidates[0].content,
                                        types.Part.from_function_response(
                                            name=tool_name,
                                            response={"result": tool_result_text}
                                        )
                                    ],
                                    config=call_config,
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
                model=MODEL,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=TEMP,
                ),
            )
            ai_response = response.text
        except Exception as inner_e:
            print(f"❌ Plain Gemini fallback also failed: {inner_e!r}")
            ai_response = f"I found relevant scenes at {top_timestamp}s, but my text generator failed."
        final_clip_path = None

    mem_add(sid, video_key, "user", user_msg)
    mem_add(sid, video_key, "assistant", ai_response)
    chat_memory.append({"role": "user", "content": user_msg})
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
async def get_chat_history(session_id: Optional[str] = None, video_path: Optional[str] = None):
    if session_id or video_path:
        sid = session_id or "default"
        title = os.path.basename(video_path) if video_path else ""
        hist = mem_history(sid, title)
        return {"history": [{"role": r, "content": c} for r, c in hist]}
    return {"history": chat_memory}

@app.delete("/api/v1/chat/memory")
async def clear_memory(session_id: Optional[str] = None, video_path: Optional[str] = None):
    global chat_memory
    chat_memory = []
    sid = session_id if session_id else (None if not video_path else "default")
    title = os.path.basename(video_path) if video_path else None
    mem_clear(sid, title)
    return {"status": "Memory cleared"}