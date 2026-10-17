# MultiMODRAG — Multimodal Video RAG

A video question-answering system that combines visual understanding, speech transcription, hybrid retrieval, and LLM-based generation to answer natural-language questions about videos and extract relevant clips on demand.

## 1. Tech Stack

- **Backend:** Python, FastAPI
- **Data & indexing:** Pixeltable, PostgreSQL-backed storage
- **Visual understanding:** Gemini, with Florence-2 as a local fallback
- **Speech-to-text:** faster-whisper
- **Embeddings:** Gemini `gemini-embedding-001`
- **LLM & tool calling:** Gemini `gemini-2.5-flash`
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Tool integration:** Model Context Protocol (MCP), FastMCP
- **Video processing:** FFmpeg

## 2. End-to-End Workflow

### Step 1: Video Ingestion — `backend/main.py`

1. The user uploads an MP4 through the Next.js frontend.
2. FastAPI saves the video and creates a task ID.
3. A background task inserts the video into Pixeltable and initiates processing.
4. The frontend polls the task-status endpoint while ingestion runs.

**Why background processing?** Video transcription, frame captioning, and embedding generation can be expensive operations. Separating them from the upload request keeps the API responsive.

### Step 2: Database Schema & Frame Indexing — `backend/schema.py`

Pixeltable organizes the data into two main structures:

- **`video_rag.videos`:** Stores video records, titles, and computed audio/transcript fields.
- **`video_rag.frames`:** A view containing sampled frames and their associated metadata and captions.

The frame iterator samples video frames at **1 FPS**.

Each frame is processed through a captioning pipeline:

1. Gemini generates a detailed visual caption.
2. If remote captioning fails, the code tries other configured Gemini models.
3. Florence-2 provides a local fallback.

**Why sample frames?** Processing every frame is computationally expensive and usually redundant. Sampling reduces processing cost, although brief visual events can be missed.

### Step 3: Audio Transcription — faster-whisper

The video's audio is transcribed locally using faster-whisper with the `base.en` model, CPU execution, and int8 computation.

The transcription pipeline retains:

- The complete transcript.
- Individual speech segments.
- Start and end timestamps for each segment.

**Why faster-whisper?** It provides local speech recognition without requiring a separate hosted transcription API.

### Step 4: Multimodal Caption Construction

Visual captions and timestamped transcript segments are aligned in `schema.py`.

For each sampled frame:

1. Calculate its timestamp from the frame position and sampling rate.
2. Find transcript segments overlapping the frame's time window.
3. Clean the visual and audio text.
4. Combine the available information into `combined_caption`.

The combined-caption column is explicitly backfilled after ingestion rather than relying on the computed-column dependency chain.

**Why combine them?** Visual captions capture scenes, objects, diagrams, and visible text, while transcripts capture spoken explanations. Their combination improves the information available for retrieval.

### Step 5: Embedding Index Creation

Pixeltable creates an embedding index over `combined_caption`:

```python
frames_view.add_embedding_index(
    "combined_caption",
    idx_name="combined_caption_idx",
    embedding=embed_content.using(
        model="gemini-embedding-001"
    ),
    if_exists="replace"
)
```

The embedding model converts each combined caption into a numerical vector representing its semantic content.

At query time, Pixeltable uses these embeddings to retrieve semantically similar captions.

**Why embeddings?** They help retrieve conceptually related content even when the query and caption use different words.

## 3. Query-Time Retrieval & Generation — `backend/main.py`

### Step 6: Semantic Retrieval

When the user submits a question, the backend:

1. Filters out frames without combined captions.
2. Restricts retrieval to the selected video, when specified.
3. Orders candidates by Pixeltable's caption-similarity search.
4. Collects the candidate frames and their metadata.

Semantic search helps match the meaning of a query rather than relying exclusively on exact word overlap.

### Step 7: Keyword-Overlap Scoring

The backend calculates a lightweight lexical score for each dense-search candidate.

It lowercases the query, extracts unique terms, and counts matching terms longer than three characters in each caption.

This acts as a **BM25-style proxy**, not a full BM25 implementation.

**Why use keyword matching?** Semantic retrieval can miss the importance of exact technical terms, identifiers, or names. Lexical matching provides a complementary relevance signal.

### Step 8: Reciprocal Rank Fusion (RRF)

The code combines dense rank and keyword-derived pseudo-rank using an RRF-style score:

\[
\operatorname{score}(d)=
\frac{1}{60+r_d}+
\frac{1}{60+r_s}
\]

Here, \(r_d\) is the dense rank and \(r_s\) is the keyword-derived pseudo-rank.

The candidates are sorted by the resulting score, and the top 10 frames are selected.

**Why rank fusion?** Dense similarity and keyword overlap have different scoring characteristics. Rank-based fusion provides a simple way to combine their relevance signals.

**Implementation note:** The current lexical component is converted into a pseudo-rank rather than a genuine independently sorted keyword ranking. A full lexical retriever followed by standard RRF would be a future improvement.

### Step 9: Context Construction & Answer Generation

The backend constructs a prompt containing:

- The user's question.
- The full video transcript.
- The top 10 retrieved frames, their timestamps, and combined captions.
- Instructions for answering using the supplied evidence.

Gemini `gemini-2.5-flash` generates the final answer.

**Why provide both retrieved frames and the full transcript?** The frames provide focused visual evidence, while the transcript preserves spoken details that may not appear in the retrieved captions.

This is the core **Retrieval-Augmented Generation (RAG)** process: retrieve relevant evidence first, then generate an answer using that context.

## 4. Optional Tool Calling — MCP + FFmpeg

### Step 10: Extracting Relevant Video Clips

When a user requests a particular moment, Gemini can invoke the clip-extraction tool.

1. The backend connects to the FastMCP server using an MCP client over SSE.
2. It discovers the available tools and adapts their input schemas for Gemini.
3. Gemini returns a structured tool call with the video filename, start time, and duration.
4. The backend invokes the MCP tool.
5. The MCP server executes FFmpeg to extract the clip.
6. The tool returns the generated clip path.
7. Gemini receives the tool result and generates the final response.
8. FastAPI serves the clip to the frontend.

**Why MCP?** It provides a structured interface between the LLM and external actions. Gemini requests the operation; the backend and MCP server execute it.

**Why FFmpeg?** It provides reliable command-line video processing without requiring a custom video-cutting implementation.

## 5. Architecture Summary

```text
Video Upload
    |
    v
FastAPI Background Ingestion
    |
    v
Pixeltable Video Table
    |
    +----> Frame Sampling (1 FPS)
    |          |
    |          v
    |      Gemini Captioning
    |      Florence-2 Fallback
    |
    +----> Audio Transcription
               |
               v
          faster-whisper
    |
    v
Timestamp Alignment
    |
    v
Combined Captions
    |
    v
Gemini Embedding Index
    |
    v
User Query
    |
    v
Dense Semantic Search
    +
Keyword-Overlap Scoring
    |
    v
RRF-Style Re-ranking
    |
    v
Top 10 Frames + Full Transcript
    |
    v
Gemini Answer Generation
    |
    +----> Text Answer
    |
    +----> Optional MCP Tool Call
                  |
                  v
             FFmpeg Clip Extraction
```

## 6. Key Design Decisions & Limitations

- **Preprocessing and indexing:** Video processing is performed during ingestion so that later queries can reuse the generated captions and embeddings.
- **Hybrid retrieval:** Dense semantic search and keyword-overlap scoring complement each other, although the current lexical ranking is a lightweight approximation.
- **Multimodal context:** Visual captions and audio transcripts are combined, but retrieval uses text embeddings rather than a native joint audio-video embedding space.
- **Resilience:** Captioning has remote-model fallbacks and a local Florence-2 fallback; tool failures can fall back to text generation.
- **Current limitations:** Fixed-rate sampling can miss brief events, captioning can lose visual details, the full transcript can increase prompt size, and task/chat state is held in memory.
