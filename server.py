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
import pathlib
import queue
import tempfile
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Hebrew STT", version="0.2.0")


async def _save_upload(file: UploadFile) -> pathlib.Path:
    suffix = pathlib.Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(await file.read())
    tmp.close()
    return pathlib.Path(tmp.name)


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

        segments: list = []
        t0 = time.monotonic()
        text = await asyncio.to_thread(
            transcribe_chunked, str(tmp_path), segments.append
        )
        elapsed = time.monotonic() - t0

        rtf = round(elapsed / audio_duration, 3) if audio_duration else None
        return {"text": text, "segments": segments, "rtf": rtf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/transcribe/stream")
async def transcribe_stream(file: UploadFile = File(...)):
    tmp_path = await _save_upload(file)
    q: queue.Queue = queue.Queue()
    _sentinel = object()

    def on_transcript(text: str, is_final: bool) -> None:
        q.put({"text": text, "is_final": is_final})

    def _run() -> None:
        try:
            from local_stt_he.api import stream_transcribe
            stream_transcribe(on_transcript, str(tmp_path))
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
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
