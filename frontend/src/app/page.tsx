"use client"; // This tells Next.js this is a Client Component (needed for React state)

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Upload,
  Video,
  MessageSquare,
  Loader2,
  Scissors,
  CheckCircle,
  Plus,
  AlertCircle,
} from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  clipPath?: string | null;
}

// One entry per uploaded video. `id` is a stable client-side key that exists
// even before the backend has assigned a task_id (during the initial upload
// request), so the UI has something to key/track on immediately.
interface VideoItem {
  id: string;
  title: string; // original filename, also what the backend scopes chat searches by
  filePath: string | null; // e.g. "videos/test_bunny.mp4" — null until upload responds
  taskId: string | null;
  status: "uploading" | "pending" | "processing" | "completed" | "failed";
}

export default function MultimodalRagDashboard() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! Upload one or more videos and ask me anything about them.",
    },
  ]);
  const [inputText, setInputText] = useState("");
  const [isChatting, setIsChatting] = useState(false);

  // All uploaded videos, and which one is currently active (shown in the
  // player + used to scope chat queries via video_path).
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [activeVideoId, setActiveVideoId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // On mount, rebuild the video tray from whatever's already in Pixeltable —
  // otherwise refreshing the page loses track of every previously uploaded
  // video even though the backend still has them fully indexed.
  useEffect(() => {
    const loadExistingVideos = async () => {
      try {
        const response = await fetch("http://localhost:8080/api/v1/video/list");
        if (!response.ok) return;
        const data = await response.json();

        const loaded: VideoItem[] = data.videos.map(
          (v: { title: string; file_path: string; status: string }) => ({
            id: crypto.randomUUID(),
            title: v.title,
            filePath: v.file_path,
            taskId: null,
            status: v.status as VideoItem["status"],
          }),
        );

        setVideos(loaded);
        if (loaded.length > 0) {
          setActiveVideoId((prev) => prev ?? loaded[0].id);
        }
      } catch (error) {
        console.error("Failed to load existing videos:", error);
      }
    };

    loadExistingVideos();
  }, []);

  const activeVideo = videos.find((v) => v.id === activeVideoId) ?? null;

  const updateVideo = (id: string, patch: Partial<VideoItem>) => {
    setVideos((prev) =>
      prev.map((v) => (v.id === id ? { ...v, ...patch } : v)),
    );
  };

  const pollTaskStatus = (id: string, taskId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(
          `http://localhost:8080/api/v1/video/task-status/${taskId}`,
        );
        if (!response.ok) throw new Error("Failed to fetch status");

        const data = await response.json();
        updateVideo(id, { status: data.status });

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error("Polling Error:", error);
        updateVideo(id, { status: "failed" });
        clearInterval(pollInterval);
      }
    }, 3000);
  };

  const uploadSingleFile = async (file: File) => {
    if (!file.name.endsWith(".mp4")) {
      alert(`Skipping "${file.name}" — only .mp4 files are supported.`);
      return;
    }

    const id = crypto.randomUUID();

    // Add a placeholder entry immediately so the UI shows it uploading,
    // rather than waiting for the network round-trip to show anything.
    setVideos((prev) => [
      ...prev,
      {
        id,
        title: file.name,
        filePath: null,
        taskId: null,
        status: "uploading",
      },
    ]);
    // First video uploaded becomes active automatically; later ones don't
    // steal focus from whatever the user is currently looking at/chatting about.
    setActiveVideoId((prev) => prev ?? id);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://localhost:8080/api/v1/video/upload",
        { method: "POST", body: formData },
      );
      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      updateVideo(id, {
        filePath: data.file_path,
        taskId: data.task_id,
        status: "pending",
      });
      pollTaskStatus(id, data.task_id);
    } catch (error) {
      console.error("Upload Error:", error);
      updateVideo(id, { status: "failed" });
      alert(`Failed to upload "${file.name}".`);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    // Upload every selected file — supports adding several videos at once,
    // as well as just adding one more to an existing set.
    await Promise.all(Array.from(files).map(uploadSingleFile));
    e.target.value = ""; // reset so re-selecting the same file again still fires onChange
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || !activeVideo?.filePath) return;

    const newUserMessage = inputText;
    setInputText("");
    setMessages((prev) => [...prev, { role: "user", content: newUserMessage }]);
    setIsChatting(true);

    try {
      const response = await fetch("http://localhost:8080/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: newUserMessage,
          video_path: activeVideo.filePath,
        }),
      });

      if (!response.ok) throw new Error("Chat request failed");

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.message, clipPath: data.clip_path },
      ]);
    } catch (error) {
      console.error("Chat Error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, I encountered an error communicating with the backend.",
        },
      ]);
    } finally {
      setIsChatting(false);
    }
  };

  const statusBadge = (status: VideoItem["status"]) => {
    switch (status) {
      case "uploading":
      case "pending":
      case "processing":
        return <Loader2 size={12} className="animate-spin text-blue-400" />;
      case "completed":
        return <CheckCircle size={12} className="text-green-500" />;
      case "failed":
        return <AlertCircle size={12} className="text-red-500" />;
    }
  };

  return (
    <div className="flex h-screen bg-neutral-950 text-neutral-100 font-sans">
      {/* Left Panel: Chat Interface */}
      <div className="w-1/2 flex flex-col border-r border-neutral-800 bg-neutral-900 shadow-xl z-10">
        <div className="p-5 border-b border-neutral-800 bg-neutral-900/80 backdrop-blur flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
              <MessageSquare size={20} />
            </div>
            <h1 className="text-lg font-semibold tracking-wide">
              Video RAG Assistant
            </h1>
          </div>
          <div className="text-xs text-neutral-500 font-mono">
            Powered by Gemini & FastMCP
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white rounded-tr-none"
                    : "bg-neutral-800 text-neutral-200 rounded-tl-none border border-neutral-700"
                }`}
              >
                <p className="leading-relaxed whitespace-pre-wrap">
                  {msg.content}
                </p>
              </div>

              {msg.clipPath && (
                <div className="mt-3 ml-2 max-w-[80%] rounded-xl overflow-hidden border border-neutral-700 bg-black shadow-lg">
                  <div className="bg-neutral-800 px-3 py-1.5 text-xs font-mono text-neutral-400 flex items-center gap-2 border-b border-neutral-700">
                    <Scissors size={12} className="text-blue-400" />
                    <span>Extracted Clip via FastMCP</span>
                  </div>
                  <video
                    controls
                    className="w-full object-contain max-h-48"
                    src={`http://localhost:8080/api/v1/video/media/${msg.clipPath}`}
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
              )}
            </div>
          ))}

          {isChatting && (
            <div className="flex items-start">
              <div className="bg-neutral-800 rounded-2xl rounded-tl-none p-4 flex items-center gap-2 border border-neutral-700 text-neutral-400">
                <Loader2 size={16} className="animate-spin text-blue-500" />
                <span className="text-sm">
                  Synthesizing response & extracting tools...
                </span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-neutral-900 border-t border-neutral-800">
          <div className="relative flex items-center">
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
              placeholder={
                activeVideo?.filePath
                  ? `Ask about "${activeVideo.title}"...`
                  : "Upload a video first to start chatting!"
              }
              disabled={
                !activeVideo?.filePath || activeVideo.status !== "completed"
              }
              className="w-full bg-neutral-950 border border-neutral-700 rounded-full py-4 pl-6 pr-14 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all text-neutral-100 placeholder:text-neutral-500 disabled:opacity-50"
            />
            <button
              onClick={handleSendMessage}
              disabled={
                !inputText.trim() ||
                !activeVideo?.filePath ||
                activeVideo.status !== "completed" ||
                isChatting
              }
              className="absolute right-2 p-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-neutral-800 disabled:text-neutral-600 text-white rounded-full transition-colors"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>

      {/* Right Panel: Video Library + Player */}
      <div className="w-1/2 flex flex-col bg-neutral-950">
        <input
          type="file"
          accept=".mp4"
          multiple
          className="hidden"
          ref={fileInputRef}
          onChange={handleFileUpload}
        />

        {/* Video tray — always visible, so you can add more videos anytime */}
        <div className="p-4 border-b border-neutral-800 flex items-center gap-3 overflow-x-auto">
          {videos.map((v) => (
            <button
              key={v.id}
              onClick={() => setActiveVideoId(v.id)}
              className={`flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors ${
                v.id === activeVideoId
                  ? "bg-blue-600/20 border-blue-500 text-blue-300"
                  : "bg-neutral-900 border-neutral-800 text-neutral-400 hover:border-neutral-600"
              }`}
              title={v.title}
            >
              {statusBadge(v.status)}
              <span className="max-w-[140px] truncate">{v.title}</span>
            </button>
          ))}

          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-lg border border-dashed border-neutral-700 text-neutral-400 hover:border-blue-500 hover:text-blue-400 transition-colors text-sm"
          >
            <Plus size={16} />
            Add video
          </button>
        </div>

        {!activeVideo ? (
          <div className="flex-1 flex flex-col items-center justify-center p-12 text-center">
            <div className="w-24 h-24 bg-neutral-900 rounded-full flex items-center justify-center mb-6 shadow-xl border border-neutral-800">
              <Video size={40} className="text-neutral-500" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Workspace Empty</h2>
            <p className="text-neutral-400 max-w-md mb-8">
              Upload one or more MP4 videos. Our backend will extract frames,
              generate captions with Gemini, and index them into Pixeltable for
              Hybrid Search. Each video is searched separately.
            </p>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium transition-all"
            >
              <Upload size={20} />
              Select MP4 Video(s)
            </button>
          </div>
        ) : (
          <div className="flex-1 flex flex-col p-6 overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4 text-neutral-300 truncate">
              {activeVideo.title}
            </h3>

            <div className="rounded-2xl overflow-hidden bg-black border border-neutral-800 shadow-2xl relative">
              {activeVideo.filePath ? (
                <video
                  controls
                  className="w-full aspect-video object-contain"
                  src={`http://localhost:8080/api/v1/video/media/${activeVideo.filePath}`}
                >
                  Your browser does not support the video tag.
                </video>
              ) : (
                <div className="w-full aspect-video flex items-center justify-center text-neutral-600">
                  Uploading...
                </div>
              )}

              {(activeVideo.status === "pending" ||
                activeVideo.status === "processing") && (
                <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex flex-col items-center justify-center z-10 text-center p-6">
                  <Loader2
                    size={40}
                    className="animate-spin text-blue-500 mb-4"
                  />
                  <h4 className="text-xl font-bold text-white mb-2">
                    Analyzing Video
                  </h4>
                  <p className="text-sm text-neutral-300 max-w-xs">
                    Extracting frames, running Gemini vision models, and
                    building hybrid indexes...
                  </p>
                  <div className="mt-6 bg-neutral-900 rounded-full px-4 py-1.5 border border-neutral-700 text-xs font-mono text-blue-400">
                    Status: {activeVideo.status.toUpperCase()}
                  </div>
                </div>
              )}
            </div>

            {activeVideo.status === "completed" && (
              <div className="mt-6 bg-green-500/10 border border-green-500/20 rounded-xl p-4 flex items-start gap-3">
                <CheckCircle className="text-green-500 mt-0.5" size={20} />
                <div>
                  <h4 className="text-sm font-semibold text-green-400">
                    Processing Complete
                  </h4>
                  <p className="text-xs text-neutral-400 mt-1">
                    This video has been fully indexed. Chat queries are scoped
                    to it while it's selected — switch videos above to ask about
                    a different one.
                  </p>
                </div>
              </div>
            )}

            {activeVideo.status === "failed" && (
              <div className="mt-6 bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                <h4 className="text-sm font-semibold text-red-400">
                  Processing Failed
                </h4>
                <p className="text-xs text-neutral-400 mt-1">
                  Check the backend logs for details.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
