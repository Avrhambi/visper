# Implementation Details: Server & Web Logic

This document provides a detailed technical explanation of the Hebrew STT Engine's server (FastAPI) and web frontend implementation.

---

## 1. Backend Logic (FastAPI)

The server is built with FastAPI and designed for high-performance, asynchronous audio processing.

### A. File Transcription Endpoints
1.  **`/transcribe` (One-shot)**:
    *   Accepts a multipart file upload.
    *   Saves the file to a temporary location.
    *   Detects audio duration and assigns it to a "bucket" (`short`, `medium`, `long`, `extended`).
    *   Invokes `transcribe_chunked` on a background thread using `asyncio.to_thread` to prevent blocking the event loop.
    *   Returns the full text, segments, and performance metrics (RTF).

2.  **`/transcribe/stream` (Server-Sent Events)**:
    *   Provides real-time feedback for file uploads.
    *   Uses a `StreamingResponse` to yield JSON-encoded segments as they are decoded by Whisper.
    *   **Cancellation Mechanism**: Uses a `threading.Event` (`abort_event`). If the client disconnects (detected via `asyncio.CancelledError`), the event is set, signaling the engine to stop inference immediately.

### B. Live WebSocket Transcription (`/ws/live`)
*   Provides a persistent bi-directional connection for microphone input.
*   **Calibration Phase**: The first 0.5s of audio is used to measure the ambient noise floor (RMS threshold).
*   **VAD (Voice Activity Detection)**:
    *   Audio is buffered until 0.8s of silence is detected OR the buffer reaches 28s.
    *   Once a "complete thought" is detected, the buffer is sent to the engine for transcription.
*   **Response**: The server sends JSON messages back to the client: `{"status": "ready"}` when calibrated, or `{"text": "..."}` for transcribed segments.

### C. Resource Management & Concurrency
*   **Model Persistence**: The Whisper engine is cached in `_engine_cache` using a hash of the hardware config. Subsequent requests use the already-loaded model.
*   **Thread Safety**: `faster-whisper` is thread-safe. Multiple requests (e.g., a file upload and a live recording) share the same model weights but execute their inference loops independently.
*   **Cancellation**: The `Transcriber.transcribe` loop checks the `is_aborted()` callback after every decoded segment. If aborted, it yields control back to the server instantly.

---

## 2. Frontend Logic (Web Page)

The frontend is a single-page application (SPA) using vanilla JavaScript, CSS variables, and Lucide icons.

### A. State Management
The UI state is managed via several global variables:
*   `historyData`: Local transcript history (synced to `localStorage`).
*   `activeMode`: Tracks if the user is in `file` or `live` transcription mode.
*   `currentFileBlob`: Stores the currently selected file before transcription.
*   `currentFileReader`: Stores the active `ReadableStreamDefaultReader` for the `/transcribe/stream` request, allowing the UI to cancel the network request if the user presses "Stop".

### B. View Switching & Components
*   **Main Navigation**: `showTab(name)` toggles visibility between "Create" and "Library".
*   **Segmented Control**: A custom "Pill" toggle switches between `viewFile` (upload) and `viewLive` (mic). It uses a "glider" div for smooth sliding animations.
*   **Dropzone**: Handlers for file selection (`triggerFileUpload`, `handleFileSelection`) show a floating "File Chip" and a "Transcribe" button.

### C. Transcription & Streaming
*   **File Streaming**:
    *   Uses the Fetch API to post audio.
    *   Reads the response body as a stream: `reader.read()`.
    *   Parses SSE `data:` lines and appends text chunks to the workspace using the `appendChunk()` helper.
*   **Animated Text Rendering**:
    *   `appendChunk(text)` creates a `<span>` with the `.chunk` class.
    *   The `.chunk` class triggers a CSS animation (`fadeIn + translateY`) to make the text appear to flow naturally.
*   **Live Recording**:
    *   Uses `navigator.mediaDevices.getUserMedia` for audio capture.
    *   `AudioContext` and `ScriptProcessorNode` handle the raw 16kHz PCM stream.
    *   `AnalyserNode` provides data for the real-time VAD meter (`animateMeter`).

### D. UI/UX Design Principles
*   **8pt Grid**: All margins and paddings are multiples of 8 (e.g., `padding: 2rem` = 32px).
*   **Spatial Hierarchy**: Use of `shadow-xl` and pure white surfaces against an off-white background (`#F9FAFB`) to create depth without harsh borders.
*   **Glassmorphism**: The header uses `backdrop-filter: blur(12px)` and a semi-transparent background to stay visible while scrolling without obstructing content.
*   **Intentional Feedback**: The engine status dot pulses slowly when active, and the "Stop" button morphs its internal icon fill on hover.

---

## 3. Data Persistence
*   **Local Storage**: All transcripts, metadata, and timestamps are saved to `localStorage` under the key `stt_history_v3`.
*   **Audio Blobs**: Audio blobs for library items are stored in memory via `URL.createObjectURL` during the session. They are not persisted across page reloads to save storage space.
