"""
FastAPI server for Hebrew STT.

Install:  pip install -e ".[server]"
Run:      visper-server   (or: python server.py)

Endpoints
---------
  GET  /health              → {status, device, compute_type}
  POST /transcribe          → {text, segments, rtf}   (multipart file upload)
  POST /transcribe/stream   → SSE stream of {text, is_final} events
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import queue
import tempfile
import threading
import time

log = logging.getLogger(__name__)

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

def _register_cuda_dlls() -> None:
    """Add nvidia package DLL folders to PATH so cublas/cudnn are found at runtime."""
    import os
    import site
    for sp in site.getsitepackages():
        nvidia_path = pathlib.Path(sp) / "nvidia"
        if nvidia_path.exists():
            for dll in nvidia_path.rglob("*.dll"):
                folder = str(dll.parent)
                if folder not in os.environ.get("PATH", ""):
                    os.environ["PATH"] += f";{folder}"

_register_cuda_dlls()

app = FastAPI(title="Hebrew STT", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def _save_upload(file: UploadFile) -> pathlib.Path:
    suffix = pathlib.Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            tmp.write(chunk)
    finally:
        tmp.close()
    return pathlib.Path(tmp.name)


@app.on_event("startup")
async def warmup():
    async def _load():
        try:
            from local_stt_he.api import _get_config, _get_router
            cfg = _get_config("medium")
            await asyncio.to_thread(lambda: _get_router(cfg).get("he"))
        except Exception as e:
            log.warning("Warmup failed: %s", e)
    asyncio.create_task(_load())


@app.get("/health")
def health():
    try:
        from local_stt_he.benchmark import get_best_config
        from local_stt_he.api import _router
        cfg = get_best_config("medium")
        model = _router._active_model_id if _router else None
        return {
            "status": "ok",
            "device": cfg.get("device"),
            "compute_type": cfg.get("compute_type"),
            "model": model,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...), language: str = Form("he"), initial_prompt: str = Form("")):
    tmp_path = await _save_upload(file)
    try:
        import soundfile as sf
        from local_stt_he.api import transcribe_chunked

        try:
            audio_duration = sf.info(str(tmp_path)).duration
        except Exception:
            audio_duration = None

        from local_stt_he.api import _bucket_for_duration
        bucket = _bucket_for_duration(audio_duration)

        segments: list = []
        t0 = time.monotonic()
        text = await asyncio.to_thread(
            transcribe_chunked, str(tmp_path), segments.append, bucket, None, language,
            initial_prompt or None,
        )
        elapsed = time.monotonic() - t0

        rtf = round(elapsed / audio_duration, 3) if audio_duration else None
        print(f"[transcribe] audio={audio_duration or 0:.1f}s  duration={elapsed:.1f}s  RTF={rtf}", flush=True)
        return {"text": text, "segments": segments, "rtf": rtf, "elapsed": round(elapsed, 1)}
    except Exception as e:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/transcribe/stream")
async def transcribe_stream(request: Request, file: UploadFile = File(...), language: str = Form("he"), initial_prompt: str = Form("")):
    tmp_path = await _save_upload(file)
    q: queue.Queue = queue.Queue()
    _sentinel = object()
    abort_event = threading.Event()

    def _run() -> None:
        try:
            import soundfile as sf
            from local_stt_he.api import transcribe_chunked
            try:
                audio_duration = sf.info(str(tmp_path)).duration
            except Exception:
                audio_duration = None
            from local_stt_he.api import _bucket_for_duration
            bucket = _bucket_for_duration(audio_duration)

            from local_stt_he.postprocess import normalize_text
            _norm = lambda t: normalize_text(t, language)
            t0 = time.monotonic()
            full_text = transcribe_chunked(
                str(tmp_path),
                lambda seg: q.put({**seg, "text": _norm(seg["text"])}),
                bucket,
                is_aborted=lambda: abort_event.is_set(),
                language=language,
                initial_prompt=initial_prompt or None,
            )
            elapsed = time.monotonic() - t0
            if not abort_event.is_set():
                # Emit final agent-post-processed text so the client can update the library entry
                q.put({"final_text": full_text})
                rtf = round(elapsed / audio_duration, 3) if audio_duration else None
                print(f"[stream]     audio={audio_duration or 0:.1f}s  duration={elapsed:.1f}s  RTF={rtf}", flush=True)
        except Exception as e:
            if not abort_event.is_set():
                q.put({"error": str(e)})
        finally:
            q.put(_sentinel)

    threading.Thread(target=_run, daemon=True).start()

    async def _generate():
        loop = asyncio.get_event_loop()

        async def _disconnect_watcher():
            while not abort_event.is_set():
                await asyncio.sleep(0.3)
                if await request.is_disconnected():
                    print("[server]     Client disconnected, aborting stream...", flush=True)
                    abort_event.set()
                    q.put(_sentinel)  # unblock q.get so generator can exit
                    return

        watcher = asyncio.create_task(_disconnect_watcher())
        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _sentinel:
                    break
                if abort_event.is_set():
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            abort_event.set()
            raise
        finally:
            abort_event.set()
            watcher.cancel()
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    from local_stt_he.api import _get_config, _get_router

    language       = websocket.query_params.get("language", "he")
    initial_prompt = websocket.query_params.get("initial_prompt", "") or None
    cfg    = _get_config("streaming")
    engine = _get_router(cfg).get(language)

    _SAMPLE_RATE          = 16000
    _BLOCK_SIZE           = 512
    _VAD_SILENCE_FRAMES   = int(0.5 * _SAMPLE_RATE / _BLOCK_SIZE)  # 500ms
    _MAX_FRAMES           = int(8.0 * _SAMPLE_RATE / _BLOCK_SIZE)

    cal_blocks: list  = []
    rms_threshold     = None
    audio_buf         = np.array([], dtype=np.float32)
    silence_count     = 0
    audio_offset      = 0  # cumulative samples emitted so far

    def _transcribe(chunk: np.ndarray, prompt: str = None) -> tuple:
        result = engine.transcribe(chunk, bucket="streaming", language=language, initial_prompt=prompt)
        return result.text.strip(), result.segments or []

    async def _emit(chunk: np.ndarray) -> None:
        nonlocal audio_offset
        offset_sec    = audio_offset / _SAMPLE_RATE
        audio_offset += len(chunk)

        text, segments = await asyncio.to_thread(_transcribe, chunk, initial_prompt)
        if text:
            shifted = [
                {"start": round(s["start"] + offset_sec, 2),
                 "end":   round(s["end"]   + offset_sec, 2),
                 "text":  s["text"]}
                for s in segments
            ]
            print(f"[ws/live]    {text}", flush=True)
            await websocket.send_json({"text": text, "segments": shifted})

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("text") == "stop":
                break
            data  = msg.get("bytes") or b""
            if not data:
                continue
            block = np.frombuffer(data, dtype=np.float32).copy()

            if rms_threshold is None:
                cal_blocks.append(block)
                if sum(len(b) for b in cal_blocks) >= _SAMPLE_RATE // 2:  # 0.5s
                    # use minimum block RMS so speech during cal doesn't skew threshold
                    block_rms = [float(np.sqrt(np.mean(b ** 2))) for b in cal_blocks]
                    rms_threshold = max(min(block_rms) * 2.0, 1e-4)
                    print(f"[ws/live]    calibrated noise floor: {rms_threshold:.5f}", flush=True)
                    await websocket.send_json({"status": "ready"})
                    # include cal audio in buffer in case user was already speaking
                    audio_buf = np.concatenate(cal_blocks)
                continue

            audio_buf = np.concatenate([audio_buf, block])
            rms = float(np.sqrt(np.mean(block ** 2)))
            silence_count = silence_count + 1 if rms < rms_threshold else 0

            buf_frames = len(audio_buf) / _BLOCK_SIZE
            if (silence_count >= _VAD_SILENCE_FRAMES or buf_frames >= _MAX_FRAMES) \
                    and buf_frames > _VAD_SILENCE_FRAMES:
                chunk, audio_buf, silence_count = audio_buf.copy(), np.array([], dtype=np.float32), 0
                await _emit(chunk)

    except Exception:
        pass

    # flush remaining audio then close gracefully
    if rms_threshold is not None and len(audio_buf) > _BLOCK_SIZE * 2:
        try:
            await _emit(audio_buf)
        except Exception:
            pass
    try:
        await websocket.close()
    except Exception:
        pass


def main() -> None:
    import uvicorn

    class _NoHealthLog(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "GET /health" not in record.getMessage()

    logging.getLogger("uvicorn.access").addFilter(_NoHealthLog())

    uvicorn.run("local_stt_he.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
