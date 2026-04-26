"""
FastAPI server for Hebrew STT.

Install:  pip install -e ".[server]"
Run:      stt-server   (or: python server.py)

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
import pathlib
import queue
import tempfile
import threading
import time

log = logging.getLogger(__name__)

from fastapi import FastAPI, File, HTTPException, UploadFile
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
            from local_stt_he.api import _get_engine
            from local_stt_he.benchmark import get_best_config
            cfg = get_best_config("medium")
            await asyncio.to_thread(_get_engine, cfg)
        except Exception as e:
            log.warning("Warmup failed: %s", e)
    asyncio.create_task(_load())


@app.get("/health")
def health():
    try:
        from local_stt_he.benchmark import get_best_config
        cfg = get_best_config("medium")
        return {
            "status": "ok",
            "device": cfg.get("device"),
            "compute_type": cfg.get("compute_type"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    tmp_path = await _save_upload(file)
    try:
        import soundfile as sf
        from local_stt_he.api import transcribe_chunked

        try:
            audio_duration = sf.info(str(tmp_path)).duration
        except Exception:
            audio_duration = None

        if audio_duration is None:
            bucket = "auto"
        elif audio_duration < 10:
            bucket = "short"
        elif audio_duration < 30:
            bucket = "medium"
        elif audio_duration < 60:
            bucket = "long"
        else:
            bucket = "extended"

        segments: list = []
        t0 = time.monotonic()
        text = await asyncio.to_thread(
            transcribe_chunked, str(tmp_path), segments.append, bucket
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
async def transcribe_stream(file: UploadFile = File(...)):
    tmp_path = await _save_upload(file)
    q: queue.Queue = queue.Queue()
    _sentinel = object()

    def _run() -> None:
        try:
            import soundfile as sf
            from local_stt_he.api import transcribe_chunked
            try:
                audio_duration = sf.info(str(tmp_path)).duration
            except Exception:
                audio_duration = None
            if audio_duration is None:
                bucket = "auto"
            elif audio_duration < 10:
                bucket = "short"
            elif audio_duration < 30:
                bucket = "medium"
            elif audio_duration < 60:
                bucket = "long"
            else:
                bucket = "extended"
            t0 = time.monotonic()
            transcribe_chunked(str(tmp_path), lambda seg: q.put(seg), bucket)
            elapsed = time.monotonic() - t0
            rtf = round(elapsed / audio_duration, 3) if audio_duration else None
            print(f"[stream]     audio={audio_duration or 0:.1f}s  duration={elapsed:.1f}s  RTF={rtf}", flush=True)
        except Exception as e:
            q.put({"error": str(e)})
        finally:
            q.put(_sentinel)

    threading.Thread(target=_run, daemon=True).start()

    async def _generate():
        loop = asyncio.get_event_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _sentinel:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(_generate(), media_type="text/event-stream")


def main() -> None:
    import uvicorn
    uvicorn.run("local_stt_he.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
