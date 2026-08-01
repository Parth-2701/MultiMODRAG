# MultiMODRAG (VideoRAG)

A multimodal Retrieval-Augmented Generation system for video. Upload an `.mp4`, and the backend samples it into frames, captions each frame, transcribes the audio track, embeds a fused visual+audio representation, and indexes everything in [Pixeltable](https://www.pixeltable.com/). A Next.js chat UI then lets you ask natural-language questions about the video — the backend runs hybrid (dense + keyword) search over the indexed frames, feeds the results plus the full transcript to Gemini, and Gemini can call an MCP tool to cut and return an actual video clip for the moment it's describing.

## Architecture

```
┌────────────┐   upload .mp4   ┌──────────────────┐   hybrid search + transcript   ┌────────┐
│  Frontend   │ ─────────────▶ │  FastAPI backend  │ ──────────────────────────────▶ │ Gemini │
│  (Next.js)  │ ◀───────────── │  (main.py)         │ ◀────────────────────────────── │        │
└────────────┘   chat / clips  └──────────┬─────────┘   tool call: extract_video_clip └────────┘
                                            │                        │
                                            ▼                        ▼
                                    ┌───────────────┐      ┌─────────────────────┐
                                    │  Pixeltable    │      │  FastMCP tool server │
                                    │  (Postgres)    │      │  (server.py, :9090)  │
                                    │  frames+embeds │      │  ffmpeg clip cutting │
                                    └───────────────┘      └─────────────────────┘
```

## How indexing works (`backend/schema.py`)

1. **`videos` table** — stores the raw video + title. A computed column extracts the audio track, and `faster-whisper` (`base.en`, local, CPU) transcribes it once per video.
2. **`frames` view** — samples the video at `FRAME_SAMPLE_FPS` (default `1.0` fps) via Pixeltable's `frame_iterator`.
3. **Captioning** — each frame is captioned by a cascading list of Gemini models (best quality first, falling through progressively cheaper/higher-quota models on rate-limit or failure), with a **local Florence-2** (`microsoft/Florence-2-base`) model as the final offline fallback so captioning never hard-fails.
4. **Cleanup** — filler words are stripped from captions (`compress_text` / `clean_audio_text`).
5. **`combined_caption`** — the visual caption is merged with the Whisper transcript segment aligned to that frame's timestamp. This is a plain (non-computed) column, backfilled explicitly after insert — see the in-code comments in `schema.py` for the Pixeltable dependency-chain issue this works around.
6. **Embedding index** — `combined_caption` is embedded with `gemini-embedding-001` for dense semantic search.

## How chat/retrieval works (`backend/main.py`, `POST /api/v1/chat`)

1. **Dense search** — ranks all frames of the target video by embedding similarity to the user's message.
2. **Sparse re-rank** — a lightweight keyword-overlap score is computed per frame.
3. **Hybrid fusion** — the two rankings are combined with Reciprocal Rank Fusion (RRF).
4. **Context assembly** — the top 10 fused frames plus the video's _full_ verbatim transcript (not just what's attached to the top frames) are sent to Gemini, so names/dates spoken once early in the video are never lost to a ranking miss.
5. **Tool calling** — Gemini is given the `extract_video_clip` tool (served over MCP/SSE by `mcp_server/server.py` on port `9090`). If the user's question implies they want to _see_ a moment, Gemini calls the tool, which runs `ffmpeg` and returns a clip path; Gemini's final answer and the clip are returned together. If the MCP server is unreachable, the backend falls back to a plain Gemini call with no clip.

## Project structure

```
MultiMODRAG/
├── backend/                # FastAPI service — ingestion pipeline + hybrid-search chat API
│   ├── main.py              # API endpoints (upload, task status, chat, media serving)
│   ├── schema.py             # Pixeltable schema, captioning/transcription/embedding pipeline
│   ├── check_db.py           # Inspect current DB contents
│   ├── reset_db.py           # Reset/wipe the DB
│   ├── cleanup_indices.py    # Remove duplicate embedding indices
│   ├── models_checking.py    # List/verify available Gemini models for the configured key
│   ├── test_pipeline.py      # End-to-end smoke test (downloads a sample clip, runs ingestion)
│   ├── pyproject.toml / requirements.txt
│   └── uv.lock
├── frontend/                # Next.js (TypeScript) chat + upload UI
│   └── src/app/              # page.tsx (main UI), layout.tsx, globals.css
├── mcp_server/               # Standalone FastMCP tool server
│   ├── server.py              # extract_video_clip tool (ffmpeg), served over SSE on :9090
│   └── requirements.txt
└── storage/                  # Local video/clip storage (created at runtime, not committed)
    ├── videos/
    └── clips/
```

## Tech stack

| Layer                        | Technology                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| Backend API                  | FastAPI, Uvicorn                                                                           |
| Data layer                   | [Pixeltable](https://www.pixeltable.com/) (Postgres-backed, declarative multimodal tables) |
| Captioning                   | Gemini (cascading models) → local Florence-2 fallback                                      |
| Transcription                | `faster-whisper` (local, CPU, int8)                                                        |
| Embeddings                   | `gemini-embedding-001`                                                                     |
| LLM synthesis + tool calling | `google-genai` SDK, `gemini-2.5-flash`                                                     |
| Tool server                  | FastMCP over SSE, `ffmpeg` for clip extraction                                             |
| Frontend                     | Next.js 16, React 19, TypeScript, Tailwind CSS                                             |

## Prerequisites

- Python **3.14+** (per `backend/pyproject.toml`) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js 18+ and npm
- `ffmpeg` available on your `PATH` (required for clip extraction)
- A free [Google AI Studio](https://aistudio.google.com/) API key
- ~1–2 GB free RAM for the local Florence-2 + faster-whisper models

## Setup

### 1. Clone

```bash
git clone https://github.com/Parth-2701/MultiMODRAG.git
cd MultiMODRAG
```

### 2. Backend

```bash
cd backend
uv sync
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Run the API (listens on **port 8080** — required, since the frontend calls `http://localhost:8080` directly):

```bash
uv run uvicorn main:app --reload --port 8080
```

On first boot this also loads Florence-2 and faster-whisper into memory and initializes the Pixeltable/Postgres schema, so startup can take a minute.

### 3. MCP tool server (required for clip extraction)

In a separate terminal:

```bash
cd mcp_server
pip install -r requirements.txt
python server.py
```

This serves the `extract_video_clip` tool over SSE on port `9090`. Without it, chat still works but Gemini can't return video clips.

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Usage

1. Upload an `.mp4` from the frontend. Ingestion (frame extraction, captioning, transcription, embedding) runs as a background task; poll `GET /api/v1/video/task-status/{task_id}` (the UI does this for you).
2. Once processing completes, ask questions about the video in the chat panel.
3. If your question implies you want to _see_ a moment ("show me...", "can I watch..."), the response includes an inline clip cut from that timestamp.

## API reference

| Endpoint                              | Method   | Description                                                                     |
| ------------------------------------- | -------- | ------------------------------------------------------------------------------- |
| `/api/v1/video/upload`                | `POST`   | Upload an `.mp4`; kicks off background ingestion, returns a `task_id`           |
| `/api/v1/video/task-status/{task_id}` | `GET`    | Poll ingestion status (`pending` / `processing` / `completed` / `failed`)       |
| `/api/v1/video/list`                  | `GET`    | List all videos already indexed in Pixeltable                                   |
| `/api/v1/video/media/{file_path}`     | `GET`    | Serve a stored video or generated clip                                          |
| `/api/v1/chat`                        | `POST`   | `{ message, video_path? }` → hybrid search + Gemini synthesis (+ optional clip) |
| `/api/v1/chat/history`                | `GET`    | Return in-memory chat history                                                   |
| `/api/v1/chat/memory`                 | `DELETE` | Clear in-memory chat history                                                    |
| `/api/v1/debug/frames`                | `GET`    | Dump every indexed frame with its captions and any pipeline errors              |

## Testing & maintenance utilities (`backend/`)

```bash
uv run test_pipeline.py     # downloads a sample clip, runs the full ingestion pipeline, prints captions
uv run check_db.py          # inspect current Pixeltable contents
uv run reset_db.py          # wipe the database
uv run cleanup_indices.py   # remove duplicate embedding indices
uv run models_checking.py   # list Gemini models available to your API key
```

## Notes & known constraints

- Chat and task-status state (`chat_memory`, `task_store` in `main.py`) is **in-memory only** and resets on backend restart.
- Only `.mp4` uploads are accepted.
- `FRAME_SAMPLE_FPS` directly drives Gemini API usage — a 2-minute video at `1.0` fps makes ~120 captioning requests. Lower it in `schema.py` if you're on the free tier and testing multiple videos.
- Search is currently scoped per-video (via `video_path`), not across your whole library.

## Contributing

Issues and pull requests are welcome.
