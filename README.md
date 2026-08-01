# VideoRAG

A Retrieval-Augmented Generation (RAG) system for video content — index videos, extract and embed frames/segments, and query them semantically through a chat-style interface.

## ✨ Features

- Video ingestion and frame/segment extraction
- Vector-based indexing for semantic video search
- Retrieval-augmented question answering over video content
- MCP (Model Context Protocol) server for tool integration
- Web frontend for uploading videos and querying results

## 🏗️ Project Structure

```
VideoRAG/
├── backend/          # Python backend — video processing, embedding, RAG pipeline
│   ├── main.py              # Entry point
│   ├── schema.py             # Data models / schema definitions
│   ├── check_db.py           # DB inspection utility
│   ├── reset_db.py           # DB reset utility
│   ├── cleanup_indices.py    # Index maintenance
│   ├── diagnose_*.py         # Debugging utilities
│   ├── test_*.py             # Test scripts
│   └── pyproject.toml        # Python dependencies (managed with uv)
├── frontend/         # Next.js (TypeScript) frontend
│   └── src/
├── mcp_server/       # MCP server
│   └── server.py
└── storage/          # Local storage for videos and processed clips
    ├── videos/
    └── clips/
```

## 🛠️ Tech Stack

- **Backend:** Python, [uv](https://docs.astral.sh/uv/) for dependency management
- **Frontend:** Next.js, TypeScript
- **Integration:** MCP (Model Context Protocol) server
- **Model:** Gemini Flash and Embeddings
- **Vector store / DB:** Pixeltable

## 📋 Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Node.js 18+ and npm

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Parth-2701/VideoRAG.git
cd VideoRAG
```

### 2. Backend setup

```bash
cd backend
uv sync
```

Create a `.env` file inside `backend/` with the required environment variables:

```env
# Example — replace with your actual required variables
API_KEY=your_api_key_here
DB_URL=your_database_url_here
```

Run the backend:

```bash
uv run main.py
```

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### 4. MCP server (optional)

```bash
cd mcp_server
python server.py
```

## 🧪 Testing

```bash
cd backend
uv run test_pipeline.py
uv run test_upload.py
```

## 📖 Usage

Upload a video via the frontend, it gets processed and indexed by the backend, then you can query it in natural language through the chat interface.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
