"""
FastAPI server for Visper.

Install:  pip install -e ".[server]"
Run:      visper-server            (binds 127.0.0.1:8000, serves the web UI at /)
          visper-server --host 0.0.0.0    (expose on the LAN — opt-in)

Endpoints
---------
  GET  /                     → the web UI (web/index.html)
  GET  /health               → {status, device, compute_type}  (503 on error)
  POST /transcribe           → {text, segments, rtf}   (multipart file upload)
  POST /transcribe/stream    → SSE stream of {text, is_final} events
  WS   /ws/live              → live streaming transcription
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

_WEB_DIR = pathlib.Path(__file__).parent / "web"
# Upload ceiling — refuse a file larger than this before writing it all to disk.
_MAX_UPLOAD_BYTES = int(os.environ.get("VISPER_MAX_UPLOAD_MB", "200")) * 1024 * 1024

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

from visper import __version__ as _visper_version


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup: warm the model in a background thread so the first request is fast.
    async def _load():
        try:
            from visper.api import _get_router
            from visper.benchmark import get_best_config

            def _blocking_warmup():
                # Never run the (multi-minute) benchmark as a side effect of
                # starting the server — use the heuristic config. If the user
                # has already run `visper-benchmark`, its result is used instead.
                cfg = get_best_config("medium", auto_benchmark=False)
                _get_router(cfg).get("he")

            await asyncio.to_thread(_blocking_warmup)
        except Exception as e:
            log.warning("Warmup failed: %s", e)

    warmup_task = asyncio.create_task(_load())
    try:
        yield
    finally:
        warmup_task.cancel()
        with contextlib.suppress(BaseException):
            await warmup_task
        with contextlib.suppress(Exception):
            from visper.api import _router
            if _router is not None:
                _router.unload()


app = FastAPI(title="Visper", version=_visper_version, lifespan=_lifespan)

# The UI is served from this same origin, so no cross-origin access is needed by
# default. Allow only explicit localhost dev origins (a separate Vite/live-server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get(
        "VISPER_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173",
    ).split(",") if o],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def index():
    page = _WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(page)


if (_WEB_DIR / "vendor").is_dir():
    app.mount("/vendor", StaticFiles(directory=_WEB_DIR / "vendor"), name="vendor")


async def _save_upload(file: UploadFile) -> pathlib.Path:
    suffix = pathlib.Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    total = 0
    try:
        while chunk := await file.read(1 << 20):  # 1 MB chunks
            total += len(chunk)
            if total > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            tmp.write(chunk)
    except BaseException:
        tmp.close()
        pathlib.Path(tmp.name).unlink(missing_ok=True)
        raise
    finally:
        tmp.close()
    return pathlib.Path(tmp.name)


@app.get("/health")
def health():
    try:
        from visper.benchmark import get_best_config
        from visper.api import _router
        cfg = get_best_config("medium", auto_benchmark=False)
        model = _router._active_model_id if _router else None
        device = cfg.get("device")
        # The Hebrew fine-tune (ivrit-ai CT2) is transcription-only. Hebrew
        # translation is still offered when either a translate-capable base
        # model is in play (MLX turbo) or the two-stage he->en MT path is
        # available (visper/translate.py) — else the toggle is hidden.
        from visper.translate import he_en_supported, _model_present
        no_translate = [] if (device == "mlx" or he_en_supported()) else ["he"]
        # he->en works the moment it's advertised, but stage 2 (the dedicated MT
        # model) is fetched on first use — until then a he translate request
        # falls back to Whisper's weaker native translate. The UI uses this to
        # show a "first use downloads ~210 MB" affordance.
        he_en_pending_download = (
            device != "mlx" and "he" not in no_translate and not _model_present()
        )
        return {
            "status": "ok",
            "device": device,
            "compute_type": cfg.get("compute_type"),
            "model": model,
            "no_translate": no_translate,
            "he_en_pending_download": he_en_pending_download,
        }
    except Exception as e:
        log.exception("Health check failed")
        return JSONResponse(status_code=503, content={"status": "error", "error": str(e)})


@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...), language: str = Form("he"), initial_prompt: str = Form(""), translate: str = Form("0")):
    tmp_path = await _save_upload(file)
    task = "translate" if translate in ("1", "true", "yes") else "transcribe"
    try:
        import soundfile as sf
        from visper.api import transcribe

        try:
            audio_duration = sf.info(str(tmp_path)).duration
        except Exception:
            audio_duration = None

        from visper.api import _bucket_for_duration
        bucket = _bucket_for_duration(audio_duration)

        segments: list = []
        t0 = time.monotonic()
        text = await asyncio.to_thread(
            transcribe, str(tmp_path), on_segment=segments.append, bucket=bucket,
            language=language, initial_prompt=initial_prompt or None, task=task,
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
async def transcribe_stream(request: Request, file: UploadFile = File(...), language: str = Form("he"), initial_prompt: str = Form(""), translate: str = Form("0")):
    tmp_path = await _save_upload(file)
    task = "translate" if translate in ("1", "true", "yes") else "transcribe"
    q: queue.Queue = queue.Queue()
    _sentinel = object()
    abort_event = threading.Event()

    def _run() -> None:
        try:
            import soundfile as sf
            from visper.api import transcribe
            try:
                audio_duration = sf.info(str(tmp_path)).duration
            except Exception:
                audio_duration = None
            from visper.api import _bucket_for_duration
            bucket = _bucket_for_duration(audio_duration)

            from visper.postprocess import normalize_text
            _norm_lang = "en" if task == "translate" else language
            _norm = lambda t: normalize_text(t, _norm_lang)
            t0 = time.monotonic()
            full_text = transcribe(
                str(tmp_path),
                on_segment=lambda seg: q.put({**seg, "text": _norm(seg["text"])}),
                bucket=bucket,
                is_aborted=lambda: abort_event.is_set(),
                language=language,
                initial_prompt=initial_prompt or None,
                task=task,
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
            # Sole owner of the temp file: this thread is the only reader and
            # always runs to completion (abort_event makes transcribe()
            # bail at the next segment). _generate() can't clean up reliably —
            # the client may drop the connection before iterating the response.
            tmp_path.unlink(missing_ok=True)

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
            abort_event.set()   # tells _run to stop; _run unlinks the temp file
            watcher.cancel()

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    await websocket.accept()
    from visper.api import _get_config, _get_router

    language       = websocket.query_params.get("language", "he")
    initial_prompt = websocket.query_params.get("initial_prompt", "") or None
    _translate     = websocket.query_params.get("translate", "0")
    ws_task        = "translate" if _translate in ("1", "true", "yes") else "transcribe"
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

    def _transcribe(chunk: np.ndarray, prompt: "str | None" = None) -> tuple:
        result = engine.transcribe(chunk, bucket="streaming", language=language, initial_prompt=prompt, task=ws_task)
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

    parser = argparse.ArgumentParser(description="Visper web UI + transcription API")
    parser.add_argument("--host", default=os.environ.get("VISPER_HOST", "127.0.0.1"),
                        help="Bind address. Default 127.0.0.1 (local only). "
                             "Pass 0.0.0.0 to expose on the LAN.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VISPER_PORT", "8000")))
    args = parser.parse_args()

    class _NoHealthLog(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "GET /health" not in record.getMessage()

    logging.getLogger("uvicorn.access").addFilter(_NoHealthLog())

    if args.host == "0.0.0.0":
        log.warning("Binding 0.0.0.0 — the API and web UI are reachable from the network.")

    uvicorn.run("visper.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
