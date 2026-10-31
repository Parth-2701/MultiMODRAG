# MultiMODRAG: Multimodal Video Retrieval-Augmented Generation

**Demo Video:** https://drive.google.com/file/d/1YuFa2tbaUQpfjlOQTbbrHpsQ1fEHN815/view?usp=sharing

A production-grade, multimodal Retrieval-Augmented Generation (RAG) system designed to index, search, and synthesize answers across video tracks, spoken audio, and visual frames.

MultiMODRAG ingests raw `.mp4` video files, decomposes them into synchronized visual and auditory streams, indexes them into a declarative database ([Pixeltable](https://www.pixeltable.com/)), and provides a conversational interface with sub-second hybrid retrieval, second-stage neural reranking, temporal scene clustering, and deterministic video clip cutting via the Model Context Protocol (FastMCP).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Next.js Frontend                                │
│                     (React 19 / TypeScript / Tailwind)                      │
└───────────────────────┬─────────────────────────────▲───────────────────────┘
           Upload .mp4  │                             │  Answer + Timestamped
           (multipart)  │                             │  Citations + MP4 Clip
                        ▼                             │
┌─────────────────────────────────────────────────────┴───────────────────────┐
│                           FastAPI Backend (:8080)                           │
│                                                                             │
│  Ingestion Pipeline (schema.py):                                            │
│  - 1.0 FPS frame extraction + 64-bit dHash deduplication                    │
│  - Audio extraction + Silero VAD + Faster-Whisper multilingual transcription│
│  - 3-tier vision caption cascade: Gemini -> OpenRouter -> Florence-2        │
│  - Direct visual indexing with OpenAI CLIP (vit-base-patch32)               │
│  - Fused visual-audio alignment + 768d Matryoshka document embeddings       │
│                                                                             │
│  Retrieval & Synthesis Engine (backend/rag/):                               │
│  - Multi-turn scoped conversation memory + anaphora query condensation      │
│  - Sub-millisecond intent routing (needle / summary / temporal / clip)      │
│  - Tri-modal hybrid retrieval: Dense (768d) + Sparse (BM25) + Visual (CLIP) │
│  - Reciprocal Rank Fusion (RRF, k=60)                                       │
│  - Second-stage neural reranking (Gemini Flash / Cross-Encoder)             │
│  - Temporal span clustering (2.5s gap, 20s max) + Jaccard deduplication     │
│  - Dedicated speech dialogue extraction + global video chapter injection    │
│  - Anti-hallucination XML prompt assembly + multi-model chat cascade        │
└──────────┬─────────────────────────────┬──────────────────────────▲─────────┘
           │                             │                          │
           ▼                             ▼                          │ Tool Call
┌─────────────────────────┐   ┌────────────────────────┐   ┌────────┴────────┐
│   Pixeltable Database   │   │  FastMCP Video Slicer  │   │  Google Gemini  │
│  (PostgreSQL + pgvector)│   │  (server.py on :9090)  │   │   Flash/Lite    │
│  Frames, Transcripts,   │   │  FFmpeg Keyframe Seek  │   │ Cascade Models  │
│  Embeddings & HNSW Idx  │   │  Deterministic Slicing │   └─────────────────┘
└─────────────────────────┘   └────────────────────────┘
```

---

## Project Structure & Repository Layout

```
video-rag/
├── backend/                        # FastAPI core service & retrieval pipelines
│   ├── main.py                     # API routers, task management, synthesis cascade
│   ├── schema.py                   # Pixeltable declarative schema & backfill logic
│   ├── rag/                        # Modular RAG subsystem
│   │   ├── conversation.py         # Session-scoped memory & anaphora query condensation
│   │   ├── routing.py              # Query intent classification & adaptive policies
│   │   ├── visual_index.py         # OpenAI CLIP visual index retrieval & base64 decoding
│   │   ├── embeddings.py           # Task-typed (RETRIEVAL_QUERY/DOC) 768d MRL embeddings
│   │   ├── hybrid_retrieval.py     # Dense + BM25 + CLIP retrieval fused via RRF (k=60)
│   │   ├── rerank.py               # 2nd-stage cross-attention / LLM neural reranking
│   │   ├── spans.py                # Temporal span clustering & Jaccard deduplication
│   │   ├── transcript_rag.py       # Grounded Whisper speech retrieval over audio chunks
│   │   ├── summaries.py            # Global video chaptering & executive summaries
│   │   ├── clip_intent.py          # Clip intent regex & 1.0s pre-roll boundary math
│   │   ├── prompting.py            # Grounded XML prompts with anti-hallucination rules
│   │   ├── openrouter.py           # Tier-2 fallback vision client for rate-limit protection
│   │   └── index_version.py        # Schema drift & index migration tracking
│   ├── tests/                      # Pytest test suite (119 unit & regression tests)
│   ├── check_db.py                 # DB inspection utility
│   ├── reset_db.py                 # Database reset/wipe utility
│   ├── cleanup_indices.py          # Duplicate index cleanup utility
│   ├── models_checking.py          # Gemini API model verification utility
│   ├── test_pipeline.py            # End-to-end ingestion smoke test
│   ├── pyproject.toml              # Python package definitions & dependencies
│   ├── uv.lock                     # Locked dependency tree
│   └── requirements.txt            # Fallback pip dependency list
├── frontend/                       # Next.js 16 Web Application
│   ├── src/app/
│   │   ├── page.tsx                # Main split-panel UI: video player, chat, upload tray
│   │   ├── layout.tsx              # Root HTML layout & fonts
│   │   └── globals.css             # Tailwind dark-theme styling
│   ├── package.json                # Frontend scripts and Node dependencies
│   └── tsconfig.json               # TypeScript compiler configuration
├── mcp_server/                     # Standalone FastMCP microservice
│   ├── server.py                   # extract_video_clip tool over SSE on port 9090
│   ├── pyproject.toml / uv.lock    # MCP service environment configuration
│   └── requirements.txt
└── storage/                        # Local runtime media directories (git-ignored)
    ├── videos/                     # Source uploaded MP4 video files
    ├── clips/                      # Sliced MP4 video moments generated by FastMCP
    └── summaries/                  # Cached JSON global video summaries
```

---

## How Ingestion Works

Video ingestion runs asynchronously in the background so API calls return immediately. The ingestion pipeline (`backend/schema.py`) processes content in seven distinct stages:

```
[ Upload MP4 ] ──► [ Background Task ] ──► [ Idempotency Check ] ──► [ videos.insert() ]
                                                                             │
    ┌────────────────────────────────────────────────────────────────────────┴────────┐
    ▼                                                                                 ▼
[ Audio Track Processing ]                                              [ Visual Frame Processing ]
1. FFmpeg audio extraction                                              1. Sample video at 1.0 FPS
2. Silero Voice Activity Detection (VAD)                                2. Compute persistent timestamp_sec
3. Faster-Whisper multilingual transcription                             3. 64-bit dHash deduplication
4. Timestamped segment dictionary output                                4. 3-Tier Caption Cascade:
                                                                           - Google Gemini (Flash-first)
                                                                           - OpenRouter Vision Fallback
                                                                           - Local Florence-2 Base (Offline)
                                                                        5. Direct CLIP visual indexing
    │                                                                                 │
    └───────────────────────────────────────────────────┬─────────────────────────────┘
                                                        │
                                                        ▼
                                       [ Explicit Python Backfill ]
                                       1. Align speech segments to frame timestamps
                                       2. Produce fused text (clean_caption + audio)
                                       3. Batch embed 100 items/call via Gemini API
                                       4. Task-type: RETRIEVAL_DOCUMENT
                                       5. Matryoshka truncation: 768 dimensions
                                       6. L2 unit vector normalization
                                       7. Write combined_caption & combined_emb
```

1. **Upload & Task Scheduling:**
   A user uploads an `.mp4` file via `POST /api/v1/video/upload`. The file is streamed to `storage/videos/{filename}` in 64 KB buffers, an in-memory task ticket is registered (`pending`), and `process_video_background` is scheduled as a FastAPI `BackgroundTask`.

2. **Idempotency Protection:**
   The worker checks `video_rag.videos.where(videos.title == title).count()`. If the file was previously processed, it completes immediately, preventing duplicate embeddings and API burn.

3. **Audio Extraction & Whisper Speech Recognition:**
   Pixeltable triggers computed column functions on the parent table:
   - `extract_audio()` uses FFmpeg to strip the audio track to WAV.
   - `transcribe_audio()` runs `faster-whisper` (`base` model, int8 quantization, CPU) coupled with Silero VAD. It filters hallucinated loops via `no_speech_prob` gating and outputs structured segments with exact start/end timestamps.

4. **1.0 FPS Frame Extraction & Visual Deduplication:**
   The `video_rag.frames` view samples the video at 1.0 FPS. Before dispatching remote vision requests, a 64-bit difference hash (`dHash`) checks for visual change against the previous frame. Static presentation slides or unchanged camera shots reuse prior captions, saving over 60% of external API calls.

5. **3-Tier Vision Caption Cascade:**
   To guarantee ingestion never fails from rate limits or outages, frames pass through a circuit-breaker cascade:
   - **Tier 1 (Google Gemini):** Evaluates high-rate-limit models first (`gemini-2.5-flash`, `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`) using structured JSON extraction for dense visual details and on-screen OCR text.
   - **Tier 2 (OpenRouter Free Vision):** Fails over to external free multimodal models if Gemini keys encounter temporary cooldowns.
   - **Tier 3 (Local Florence-2 Base):** Runs `microsoft/Florence-2-base` locally on CPU as the final offline fallback.

6. **Direct CLIP Visual Indexing:**
   The raw `frame` image column is indexed using OpenAI CLIP (`openai/clip-vit-base-patch32`) via Pixeltable's internal vector index (`frame_clip_idx`). This builds an Approximate Nearest Neighbor (ANN) index over raw pixels, enabling zero-shot image-to-image and text-to-image searches.

7. **Temporal Alignment & Batch Document Embeddings:**
   `schema.backfill_combined_captions()` merges spoken dialogue from the Whisper transcript with visual frame captions using a $\pm 0.5$s temporal window:
   $$\text{combined\_caption} = \text{visual\_caption} + \text{" Spoken audio: "} + \text{audio\_text}$$
   Captions are embedded in batches of 100 using Google's embedding model configured with `task_type="RETRIEVAL_DOCUMENT"`, truncated to 768 dimensions using Matryoshka Representation Learning (MRL), and normalized to unit length ($\|v\|_2 = 1.0$).

---

## How Querying & Retrieval Works

When a user submits a question, MultiMODRAG executes a multi-stage retrieval and synthesis pipeline (`backend/main.py` + `backend/rag/*`):

```
[ User Query ] ──► [ Scoped Conversational Memory ] ──► [ Query Condensation ]
                                                              │
                                                              ▼
                                                   [ Intent Router (<1ms) ]
                                                   needle | summary | temporal | clip
                                                              │
    ┌─────────────────────────────────────────────────────────┴───────────────────────┐
    ▼                                                         ▼                       ▼
[ Dense Semantic Search ]                         [ Sparse BM25 Search ]    [ Visual CLIP Search ]
Gemini 768d Task-Typed Vector                     Rank-BM25 with Snowball   Text-to-Image / Image-to-Image
(Task: RETRIEVAL_QUERY)                           Stemming & Stopwords      over raw frame pixels
    │                                                         │                       │
    └─────────────────────────────────────────────────────────┼───────────────────────┘
                                                              │
                                                              ▼
                                             [ Reciprocal Rank Fusion (RRF: k=60) ]
                                             Merges top 30 candidates across modalities
                                                              │
                                                              ▼
                                             [ 2nd-Stage Neural Reranker ]
                                             Gemini Flash Cross-Attention (0-3 score)
                                                              │
                                                              ▼
                                             [ Temporal Span Clustering ]
                                             - Merges frames (GAP <= 2.5s, MAX <= 20s)
                                             - Jaccard similarity deduplication (>= 0.8)
                                                              │
    ┌─────────────────────────────────────────────────────────┴───────────────────────┐
    ▼                                                                                 ▼
[ Dedicated Speech Context ]                                              [ FastMCP Video Slicer ]
45s overlapping Whisper chunks                                            - 1.0s pre-roll for keyframes
Dense + BM25 hybrid search                                                - FFmpeg -ss before -i (:9090)
    │                                                                                 │
    └─────────────────────────────────────────────────────────┬───────────────────────┘
                                                              │
                                                              ▼
                                             [ Grounded XML Prompt Synthesis ]
                                             - <speech_evidence> & <visual_evidence>
                                             - Strict [MM:SS] citation enforcement
                                             - Chat cascade fallback against 429 errors
```

1. **Conversational Memory & Anaphora Query Condensation:**
   Conversation turns are isolated by `(session_id, video_title)`. If a follow-up query contains pronouns or temporal references (e.g., _"What did he do next?"_), `condense_query` passes recent conversational turns to Gemini Flash to rewrite it into a self-contained query (_"What did Chandragupta do after meeting Chanakya?"_). Queries longer than 8 words without pronouns bypass condensation to minimize latency.

2. **Sub-Millisecond Intent Routing:**
   `route()` uses compiled regular expressions to classify queries in $<0.05$ms into five intent classes:
   - `needle`: Precision lookup (top 6 spans, sorted by score).
   - `temporal`: Sequential timelines (top 14 spans, sorted chronologically).
   - `aggregate`: Counting events (top 20 spans, sorted chronologically, with an approximate sampling notice).
   - `summary`: Global recap (triggers injection of pre-computed video chapters).
   - `clip_request`: Video extraction (determines cutting boundaries).

3. **Tri-Modal Hybrid Search:**
   `hybrid_search()` runs three independent retrievers:
   - **Dense:** Converts query to a 768d vector using `task_type="RETRIEVAL_QUERY"` and measures cosine similarity against `combined_emb`.
   - **Sparse:** Runs `rank-bm25` with Snowball English word stemming over `combined_caption`.
   - **Visual:** Decodes optional base64 query images or queries the CLIP index using text to rank frames directly by pixel similarity.

4. **Reciprocal Rank Fusion (RRF):**
   Raw cosine similarities and BM25 scores cannot be directly added. RRF combines candidate rankings without score calibration issues using smoothing constant $k=60$:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{Dense}, \text{BM25}, \text{Visual}\}} \frac{1}{60 + \text{rank}_m(d)}$$

5. **Second-Stage Neural Cross-Attention Reranking:**
   Bi-encoders evaluate queries and documents in isolation. The top 30 RRF candidates pass through `rerank()`, which presents candidates directly to Gemini Flash to score relevance on a 0–3 scale with stable tie-breaking.

6. **Temporal Span Clustering & Scene Deduplication:**
   Individual 1-second frame hits are grouped into continuous video scenes (`to_spans`). Frames separated by $\le 2.5$s are merged up to a 20.0s window. Jaccard token similarity ($\ge 0.8$) suppresses near-identical visual captions across static scenes.

7. **Dedicated Speech Retrieval:**
   `transcript_context()` queries 45-second sliding transcript segments with 1-segment overlap. For silent or music-only videos, it injects explicit status markers (`(this video has no detectable speech)`), preventing the LLM from hallucinating dialogue.

8. **Deterministic FastMCP Clip Cutting:**
   If a user asks to see a moment, `calculate_clip_boundaries()` applies a 1.0s pre-roll and 2.0s post-padding to guarantee clean keyframe (I-frame) seeks and prevent clipped speech. It calls `extract_video_clip` over SSE on the FastMCP service (`mcp_server/server.py`), executing FFmpeg with `-movflags +faststart` for instant browser streaming.

9. **Grounded XML Prompt Synthesis with Cascade Failover:**
   `build_user_content()` encapsulates evidence into XML tags (`<speech_evidence>`, `<visual_evidence>`), protecting against indirect prompt injection. The system prompt mandates `[MM:SS]` citations. If the primary model (`gemini-2.5-flash`) encounters a rate limit (HTTP 429), `_call_gemini_with_fallback()` automatically shifts the request to `gemini-3.1-flash-lite`.

---

## Tech Stack

| Layer                         | Technology                                                        | Purpose                                                                            |
| :---------------------------- | :---------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Backend Framework**         | FastAPI, Uvicorn, Pydantic v2                                     | Asynchronous REST API, background worker threads, data validation                  |
| **Database & Vector Storage** | [Pixeltable](https://www.pixeltable.com/) (PostgreSQL + pgvector) | Declarative multimodal tables, computed views, frame iterators, vector indexing    |
| **Speech Processing**         | Faster-Whisper, Silero VAD, FFmpeg                                | Local CPU int8 audio transcription, voice activity detection, hallucination gating |
| **Vision & Captioning**       | Google Gemini (Flash Cascade), Florence-2 Base                    | Multimodal frame captioning, OCR extraction, offline CPU fallback                  |
| **Visual Search Index**       | OpenAI CLIP (`vit-base-patch32`)                                  | Direct zero-shot text-to-image and image-to-image visual vector similarity         |
| **Embeddings**                | Gemini (`gemini-embedding-001` / `text-embedding-004`)            | Asymmetric 768d Matryoshka embeddings with $L_2$ unit normalization                |
| **Lexical Search**            | Rank-BM25, Snowball Stemmer                                       | Exact keyword matching, proper noun and technical acronym retrieval                |
| **LLM Synthesis**             | Google GenAI SDK (`gemini-2.5-flash`, `gemini-3.1-flash-lite`)    | Grounded multi-turn answer generation with rate-limit cascade failover             |
| **Tool Execution**            | FastMCP (Model Context Protocol), FFmpeg                          | Standalone microservice for keyframe-accurate MP4 video slicing                    |
| **Frontend UI**               | Next.js 16 (App Router), React 19, Tailwind CSS                   | Split-screen video playback, interactive chat, auto-scroll, media previews         |
| **Package Management**        | `uv` (Astral), `npm`                                              | Fast deterministic Python environment synchronization and Node packaging           |

---

## Prerequisites

Before setting up the project, ensure your host environment meets the following requirements:

1. **Python 3.14+** (per `backend/pyproject.toml`) and [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

   ```bash
   # Install uv on Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Install uv on Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Node.js 18+** and `npm` (for running the Next.js frontend).
3. **FFmpeg** installed and accessible on your system `PATH`:
   - Used by Whisper for audio extraction, OpenCV for frame slicing, and FastMCP for cutting MP4 clips.
   - Verify by running: `ffmpeg -version`.
4. **Google AI Studio API Key:**
   - Obtain a free key at [Google AI Studio](https://aistudio.google.com/).
5. **Hardware Resources:**
   - ~1–2 GB of free RAM to run local Florence-2 Base and Faster-Whisper models on CPU.

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/Parth-2701/MultiMODRAG.git
cd MultiMODRAG
```

### 2. Backend Setup

```bash
cd backend

# Synchronize virtual environment with pinned dependencies
uv sync

# Create environment file
cp .env.example .env
```

Edit `backend/.env` and add your Google AI Studio key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
FRAME_SAMPLE_FPS=1.0
EMBED_DIM=768
```

### 3. FastMCP Tool Server Setup

Open a second terminal to host the video slicing microservice:

```bash
cd mcp_server
uv sync
```

### 4. Frontend Setup

Open a third terminal for the Next.js web application:

```bash
cd frontend
npm install
```

---

## Running the Services

To run the complete system, start the three independent services in separate terminals:

### Terminal 1: FastMCP Tool Server (Port 9090)

```bash
cd mcp_server
uv run python server.py
```

_Listens on SSE transport at `http://localhost:9090/sse`._

### Terminal 2: FastAPI Backend (Port 8080)

```bash
cd backend
uv run uvicorn main:app --reload --port 8080
```

_Initializes Pixeltable, verifies embedding models, and listens at `http://localhost:8080`._

### Terminal 3: Next.js Frontend (Port 3000)

```bash
cd frontend
npm run dev
```

_Serves the user interface at `http://localhost:3000`._

---

## How to Use

1. **Access Application:** Open `http://localhost:3000` in your web browser.
2. **Upload a Video:**
   - Click **Upload Video** in the media panel and choose an `.mp4` file.
   - The UI registers the task and polls `GET /api/v1/video/task-status/{task_id}`.
   - A spinner displays processing progress (Whisper transcription $\rightarrow$ frame extraction $\rightarrow$ Gemini captioning $\rightarrow$ backfilling embeddings).
3. **Querying & Synthesis:**
   - Type any question into the chat input (e.g., _"What did the speaker say about Chandragupta's army?"_ or _"What was written on the whiteboard?"_).
   - MultiMODRAG retrieves relevant visual spans and dialogue lines, returning a response with exact timestamp citations (`[01:14]`). Clicking citations scrubs the video player directly to that second.
4. **Requesting Video Clips:**
   - Ask to see a moment (e.g., _"Show me the part where the car drives off"_ or _"Can I watch the introduction scene?"_).
   - FastMCP cuts an MP4 clip via FFmpeg and embeds an inline video player inside the chat stream.

---

## API Reference

The FastAPI backend exposes the following REST endpoints on port `8080`:

| Endpoint                               |  Method  | Request Payload                                                                 | Description                                                                                |
| :------------------------------------- | :------: | :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- |
| `/api/v1/video/upload`                 |  `POST`  | `multipart/form-data` (`file: .mp4`)                                            | Saves uploaded video, registers task ticket, schedules background ingestion                |
| `/api/v1/video/task-status/{task_id}`  |  `GET`   | URL parameter: `task_id`                                                        | Returns status: `pending`, `processing`, `completed`, or `failed`                          |
| `/api/v1/video/list`                   |  `GET`   | None                                                                            | Lists all videos currently indexed in Pixeltable with relative storage paths               |
| `/api/v1/video/media/{file_path:path}` |  `GET`   | URL path: `videos/...` or `clips/...`                                           | Streams source MP4 videos and extracted clip files to browser                              |
| `/api/v1/chat`                         |  `POST`  | `{"message": str, "video_path": str, "session_id"?: str, "image_base64"?: str}` | Executes hybrid search, neural reranking, span clustering, and LLM answer generation       |
| `/api/v1/chat/history`                 |  `GET`   | `?session_id=...&video_path=...`                                                | Returns scoped conversation turns for the active video session                             |
| `/api/v1/chat/memory`                  | `DELETE` | `?session_id=...&video_path=...`                                                | Clears conversation memory for a specific session and video                                |
| `/api/v1/debug/frames`                 |  `GET`   | None                                                                            | Diagnostics endpoint: dumps indexed frame rows, clean captions, and computed column errors |

---

## Testing & Maintenance Utilities

The `backend/` directory provides utility scripts for maintenance, database inspection, and regression testing:

```bash
# 1. Inspect Database State
# Displays row counts, computed column health, and sample transcript segments
uv run python check_db.py

# 2. End-to-End Pipeline Smoke Test
# Downloads a test MP4 clip, triggers the ingestion cascade, and prints captions
uv run python test_pipeline.py

# 3. Model & API Key Verification
# Lists all Gemini models currently accessible under the configured GEMINI_API_KEY
uv run python models_checking.py

# 4. Clean Duplicate Indices
# Scans Pixeltable tables and removes redundant or drifted embedding indices
uv run python cleanup_indices.py

# 5. Database Reset
# Drops the 'video_rag' directory and tables in Pixeltable (use with caution)
uv run python reset_db.py

# 6. Run Complete Test Suite
# Executes all 119 unit and integration tests across the 18 quality and ML fixes
uv run python -m pytest tests/ -v
```

---

## Operational Notes & Constraints

- **Rate Limits on Free Tier Keys:** Gemini free tier keys have request-per-minute (RPM) and request-per-day (RPD) caps. MultiMODRAG implements a multi-model cascade (`gemini-2.5-flash` $\rightarrow$ `gemini-3.1-flash-lite` $\rightarrow$ `gemini-3.5-flash-lite`) and visual dHash caching to ensure continuous uptime under free-tier limits.
- **Frame Rate Sampling Tradeoff:** Frame sampling rate is governed by `FRAME_SAMPLE_FPS` in `backend/schema.py` (default: `1.0` FPS). A 2-minute video produces 120 frames. For long videos (10+ minutes) on free tier keys, setting `FRAME_SAMPLE_FPS=0.5` reduces API consumption by 50% while retaining high retrieval recall.
- **In-Memory Task Store:** Ingestion task status (`task_store`) is held in application memory. If the backend is restarted during video processing, re-uploading the file safely triggers the idempotency guard and restores completed status.
