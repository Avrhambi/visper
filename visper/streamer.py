"""
visper/streamer.py
----------------
VAD-gated live chunked transcription pipeline.

Two-thread architecture:
  Producer: accumulates audio blocks, detects speech/silence, emits chunks.
  Consumer: calls Transcriber.transcribe(), invokes on_transcript callback.

Supports microphone live mode and file streaming mode.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from visper.constants import SAMPLE_RATE, CHANNELS, BLOCK_SIZE, DTYPE

ROOT = Path(__file__).parent.parent

# Chunk size limits
MAX_CHUNK_S = 28.0   # hard max before forced emit (Whisper limit ~30s)
_QUEUE_MAXSIZE = 4


class LiveStreamer:
    def __init__(
        self,
        on_transcript: Callable[[str, bool], None],
        config: Optional[dict] = None,
        source: Optional[Union[str, Path]] = None,
        initial_prompt: Optional[str] = None,
    ):
        """
        on_transcript(text, is_final) — called from consumer thread.
        config=None → auto-selects via benchmark.get_best_config('streaming').
        source=None → microphone mode; Path → file streaming mode.
        """
        self._on_transcript = on_transcript
        self._source = Path(source) if source else None

        if config is None:
            from visper.benchmark import get_best_config
            config = get_best_config("streaming")
        self._config = config

        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._idle_timer: Optional[threading.Timer] = None
        self._memory_timer: Optional[threading.Timer] = None

        self._transcriber = None
        self._dropped_chunks = 0
        self._consecutive_drops = 0
        self._segments_transcribed = 0
        self._total_audio_duration = 0.0
        self._total_elapsed = 0.0

        self._running = False
        self._prev_text: str = ""  # last chunk's raw transcription for overlap dedup
        self._language: str = config.get("language", "he") if config else "he"
        self._initial_prompt: Optional[str] = initial_prompt or None

        # Graceful degradation under queue pressure
        self._pressure_mode: bool = False
        self._clean_chunks: int = 0  # consecutive chunks processed without drops

        # Load VAD settings
        self._vad_min_silence_ms = 300
        self._max_chunk_s = MAX_CHUNK_S
        self._stream_flush_on_silence = True
        self._idle_unload_seconds = 0
        self._load_config()

    def _load_config(self):
        # Defaults first so every attribute is always set — even if the read fails.
        self._vad_min_silence_ms = 300
        self._max_chunk_s = MAX_CHUNK_S
        self._stream_flush_on_silence = True
        self._noise_calibration_seconds = 1.5
        self._overlap_samples = int(2.0 * SAMPLE_RATE)
        try:
            from visper._config import load_config
            cfg = load_config()
            self._vad_min_silence_ms = cfg.get("vad_min_silence_ms", self._vad_min_silence_ms)
            self._max_chunk_s = cfg.get("max_chunk_seconds", self._max_chunk_s)
            self._stream_flush_on_silence = cfg.get("stream_flush_on_silence", True)
            self._noise_calibration_seconds = cfg.get("noise_calibration_seconds", 1.5)
            overlap_s = float(cfg.get("overlap_seconds", 2.0))
            self._overlap_samples = int(overlap_s * SAMPLE_RATE) if overlap_s > 0 else 0
        except Exception:
            pass
        from visper.resource import get_idle_unload_seconds
        self._idle_unload_seconds = get_idle_unload_seconds()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_model_loaded(self) -> bool:
        return self._transcriber is not None and self._transcriber._backend is not None

    @property
    def buffer_duration(self) -> float:
        return getattr(self, "_buffer_duration", 0.0)

    @property
    def stats(self) -> dict:
        return {
            "segments": self._segments_transcribed,
            "dropped_chunks": self._dropped_chunks,
            "total_audio_duration": round(self._total_audio_duration, 2),
            "avg_rtf": round(
                self._total_elapsed / self._total_audio_duration, 3
            ) if self._total_audio_duration > 0 else 0.0,
        }

    def start(self) -> None:
        from visper.api import _get_router
        self._transcriber = _get_router(self._config).get(self._language)
        self._stop_event.clear()
        self._running = True

        self._consumer_thread = threading.Thread(target=self._consumer, daemon=True)
        self._consumer_thread.start()

        if self._source:
            self._producer_thread = threading.Thread(
                target=self._producer_file, args=(self._source,), daemon=True
            )
        else:
            self._producer_thread = threading.Thread(target=self._producer_mic, daemon=True)

        self._producer_thread.start()

        # Start memory monitor
        self._schedule_memory_check()

    def stop(self) -> None:
        self._stop_event.set()
        self._flush_buffer()
        if self._producer_thread:
            self._producer_thread.join(timeout=5)
        # Signal consumer to exit
        self._queue.put(None)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=10)
        if self._idle_timer:
            self._idle_timer.cancel()
        if self._memory_timer:
            self._memory_timer.cancel()
        self._running = False

    def flush(self) -> None:
        """Force-transcribe whatever is currently in the buffer."""
        self._flush_buffer()

    def _flush_buffer(self) -> None:
        buf = getattr(self, "_audio_buffer", [])
        if buf:
            chunk = np.concatenate(buf)
            self._audio_buffer = []
            self._buffer_duration = 0.0
            self._enqueue_chunk(chunk, is_final=True)

    # ------------------------------------------------------------------
    # Noise floor calibration
    # ------------------------------------------------------------------

    def _calibrate_noise_floor(self) -> float:
        """
        Record a short burst of ambient audio and derive a silence RMS threshold.
        Returns max(0.01, measured_rms * 1.5) so the threshold sits 50% above floor.
        Returns 0.01 if calibration is disabled or fails.
        """
        if not getattr(self, "_noise_calibration_seconds", 0):
            return 0.01
        try:
            import sounddevice as sd
            n_samples = int(self._noise_calibration_seconds * SAMPLE_RATE)
            print(f"[STT] Calibrating noise floor ({self._noise_calibration_seconds:.1f}s)...",
                  file=sys.stderr)
            recording = sd.rec(n_samples, samplerate=SAMPLE_RATE,
                               channels=CHANNELS, dtype=DTYPE)
            sd.wait()
            rms = float(np.sqrt(np.mean(recording ** 2)))
            threshold = max(0.01, rms * 1.5)
            print(f"[STT] Noise calibration: RMS {rms:.4f} → silence threshold {threshold:.4f}",
                  file=sys.stderr)
            return threshold
        except Exception as e:
            print(f"[STT] Noise calibration failed ({e}), using default threshold 0.01",
                  file=sys.stderr)
            return 0.01

    # ------------------------------------------------------------------
    # Producer — microphone mode
    # ------------------------------------------------------------------

    def _producer_mic(self) -> None:
        import sounddevice as sd

        self._audio_buffer: list = []
        self._buffer_duration: float = 0.0
        silence_frames = 0
        silence_threshold_frames = int((self._vad_min_silence_ms / 1000) * SAMPLE_RATE / BLOCK_SIZE)
        max_frames = int(self._max_chunk_s * SAMPLE_RATE / BLOCK_SIZE)
        rms_silence_threshold = self._calibrate_noise_floor()

        def callback(indata, frames, time_info, status):
            if self._stop_event.is_set():
                raise sd.CallbackStop()
            chunk = indata[:, 0].copy()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_silence = rms < rms_silence_threshold

            nonlocal silence_frames
            self._audio_buffer.append(chunk)
            self._buffer_duration += frames / SAMPLE_RATE

            if is_silence:
                silence_frames += 1
            else:
                silence_frames = 0

            # Emit on silence gate or max chunk
            should_emit = False
            is_final = True
            if self._stream_flush_on_silence and silence_frames >= silence_threshold_frames and len(self._audio_buffer) > silence_threshold_frames:
                should_emit = True
                is_final = True
            elif len(self._audio_buffer) >= max_frames:
                should_emit = True
                is_final = False  # forced emit mid-speech

            if should_emit and self._audio_buffer:
                audio_chunk = np.concatenate(self._audio_buffer)
                # Keep overlap tail so next chunk starts with shared audio context
                if self._overlap_samples > 0 and len(audio_chunk) > self._overlap_samples:
                    overlap_tail = audio_chunk[-self._overlap_samples:]
                    self._audio_buffer = [overlap_tail]
                    self._buffer_duration = self._overlap_samples / SAMPLE_RATE
                else:
                    self._audio_buffer = []
                    self._buffer_duration = 0.0
                silence_frames = 0
                self._enqueue_chunk(audio_chunk, is_final=is_final)

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=callback,
            ):
                while not self._stop_event.is_set():
                    time.sleep(0.05)
        except Exception as e:
            if not self._stop_event.is_set():
                print(f"[STT] Microphone error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Producer — file streaming mode
    # ------------------------------------------------------------------

    def _producer_file(self, path: Path) -> None:
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            if sr != SAMPLE_RATE:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

            total_frames = len(data)
            chunk_frames = int(self._max_chunk_s * SAMPLE_RATE)
            pos = 0
            while pos < total_frames and not self._stop_event.is_set():
                end = min(pos + chunk_frames, total_frames)
                chunk = data[pos:end]
                is_final = end >= total_frames
                self._enqueue_chunk(chunk, is_final=is_final)
                # Advance by chunk minus overlap so next chunk shares the tail
                new_pos = end - self._overlap_samples
                advance = new_pos - pos
                time.sleep(max(0, advance) / SAMPLE_RATE)  # real-time for new audio only
                pos = new_pos
                if pos >= total_frames:
                    break
        except Exception as e:
            print(f"[STT] File streaming error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Enqueue with backpressure
    # ------------------------------------------------------------------

    def _enqueue_chunk(self, chunk: np.ndarray, is_final: bool) -> None:
        item = (chunk, is_final)
        try:
            self._queue.put_nowait(item)
            self._consecutive_drops = 0
        except queue.Full:
            # Drop oldest chunk
            try:
                dropped_chunk, _ = self._queue.get_nowait()
                dropped_dur = len(dropped_chunk) / SAMPLE_RATE
                self._dropped_chunks += 1
                self._consecutive_drops += 1
                print(f"[STT] Warning — dropped {dropped_dur:.1f}s of audio (consumer overloaded)",
                      file=sys.stderr)
                if self._consecutive_drops >= 3:
                    print("[STT] Warning — repeated audio drops. "
                          "Consider setting resource_profile=minimal or reducing beam_size in config.yaml.",
                          file=sys.stderr)
                self._queue.put_nowait(item)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Overlap deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_overlap(prev_text: str, current_text: str) -> str:
        """
        Strip words at the start of current_text that duplicate the tail of prev_text.
        These arise from the sliding window overlap: the last N seconds of the previous
        chunk are re-included in the current chunk, so the model transcribes them twice.

        Strategy: word-level suffix/prefix match — find the longest suffix of prev_text
        that equals a prefix of current_text (up to 10 words), strip that prefix.
        Comparison is done on clean words (punctuation stripped) to tolerate minor
        differences in how Whisper renders boundary punctuation.
        """
        if not prev_text or not current_text or not prev_text.strip():
            return current_text

        import re

        def clean(w: str) -> str:
            return re.sub(r'[^\w]', '', w)

        prev_words = prev_text.split()
        curr_words = current_text.split()
        prev_clean = [clean(w) for w in prev_words]
        curr_clean = [clean(w) for w in curr_words]

        max_check = min(10, len(prev_clean), len(curr_clean))
        best = 0
        for n in range(1, max_check + 1):
            if prev_clean[-n:] == curr_clean[:n]:
                best = n

        if best == 0:
            return current_text

        return ' '.join(curr_words[best:]).strip()

    # ------------------------------------------------------------------
    # Consumer thread
    # ------------------------------------------------------------------

    def _consumer(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                break

            chunk, is_final = item

            if self._transcriber is None or self._transcriber._backend is None:
                # Model was idle-unloaded — reload
                self._reload_model()

            # Check queue pressure: engage fast tier when ≥2 consecutive drops
            if self._consecutive_drops >= 2 and not self._pressure_mode:
                self._pressure_mode = True
                self._clean_chunks = 0
                print(
                    "[STT] Queue pressure detected — switching to fast tier (beam=1) "
                    "to reduce latency. Consider setting resource_profile=minimal.",
                    file=sys.stderr,
                )

            try:
                if self._pressure_mode:
                    from visper.params import get_params_for_tier
                    fast_params = get_params_for_tier("fast", "streaming", self._transcriber._config)
                    result = self._transcriber.transcribe(chunk, bucket="streaming",
                                                          _tier_override=fast_params,
                                                          language=self._language,
                                                          initial_prompt=self._initial_prompt)
                else:
                    result = self._transcriber.transcribe(chunk, bucket="streaming",
                                                          language=self._language,
                                                          initial_prompt=self._initial_prompt)

                self._segments_transcribed += 1
                self._total_audio_duration += result.audio_duration
                self._total_elapsed += result.elapsed

                # Track clean chunks to detect pressure recovery
                if self._consecutive_drops == 0:
                    self._clean_chunks += 1
                else:
                    self._clean_chunks = 0

                if self._pressure_mode and self._clean_chunks >= 10:
                    self._pressure_mode = False
                    print("[STT] Queue pressure cleared — restoring accuracy tier.", file=sys.stderr)

                if result.text:
                    text = self._deduplicate_overlap(self._prev_text, result.text)
                    self._prev_text = result.text  # store raw for next chunk comparison
                    if text:
                        self._reset_idle_timer()
                        self._on_transcript(text, is_final)
            except Exception as e:
                print(f"[STT] Transcription error: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Idle unload
    # ------------------------------------------------------------------

    def _reset_idle_timer(self) -> None:
        if self._idle_unload_seconds <= 0:
            return
        if self._idle_timer:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self._idle_unload_seconds, self._idle_unload)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_unload(self) -> None:
        if self._transcriber:
            self._transcriber.unload()
            print("[STT] Idle — model unloaded to free memory", file=sys.stderr)

    def _reload_model(self) -> None:
        from visper.api import _get_router
        print("[STT] Speech detected — reloading model...", file=sys.stderr)
        self._transcriber = _get_router(self._config).get("he")

    # ------------------------------------------------------------------
    # Memory monitor
    # ------------------------------------------------------------------

    def _schedule_memory_check(self) -> None:
        if self._stop_event.is_set():
            return
        self._memory_timer = threading.Timer(30.0, self._check_memory)
        self._memory_timer.daemon = True
        self._memory_timer.start()

    def _check_memory(self) -> None:
        if self._stop_event.is_set():
            return
        from visper.resource import check_memory_during_session
        action = check_memory_during_session()
        if action == "demote" and self._transcriber:
            self._transcriber.unload()
            new_config = dict(self._config)
            new_config["device"] = "cpu"
            new_config["compute_type"] = "int8"
            self._config = new_config
            self._reload_model()
        self._schedule_memory_check()
