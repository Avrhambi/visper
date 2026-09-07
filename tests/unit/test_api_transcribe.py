"""The public api.transcribe() forwards every capability through to the engine
(merged from the old transcribe() + transcribe_chunked() in 2.0.0) and keeps
its args after `source` keyword-only."""
import inspect

import pytest

import visper.api as api


class _StubEngine:
    def __init__(self):
        self.calls: list = []

    def transcribe(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return type("R", (), {"text": "ok"})()


@pytest.fixture(autouse=True)
def _fake_pipeline(monkeypatch):
    engine = _StubEngine()
    monkeypatch.setattr(api, "_get_config", lambda bucket: {"device": "cpu"})
    monkeypatch.setattr(api, "_resolve_bucket", lambda source, bucket: bucket if bucket != "auto" else "medium")
    monkeypatch.setattr(api, "_get_router", lambda cfg: type("Router", (), {"get": staticmethod(lambda lang: engine)})())
    return engine


def test_defaults_forwarded(_fake_pipeline):
    assert api.transcribe("x.wav") == "ok"
    source, kw = _fake_pipeline.calls[0]
    assert source == "x.wav"
    assert kw["on_segment"] is None
    assert kw["language"] == "he"
    assert kw["task"] == "transcribe"
    assert kw["initial_prompt"] is None
    assert kw["is_aborted"] is None
    assert kw["bucket"] == "medium"


def test_all_capabilities_forwarded(_fake_pipeline):
    cb = lambda seg: None
    abort = lambda: False
    api.transcribe(
        "x.wav", on_segment=cb, bucket="long", is_aborted=abort,
        language="en", initial_prompt="hi", task="translate",
    )
    _, kw = _fake_pipeline.calls[0]
    assert kw["on_segment"] is cb
    assert kw["bucket"] == "long"
    assert kw["is_aborted"] is abort
    assert kw["language"] == "en"
    assert kw["initial_prompt"] == "hi"
    assert kw["task"] == "translate"


def test_args_after_source_are_keyword_only(_fake_pipeline):
    with pytest.raises(TypeError):
        api.transcribe("x.wav", lambda seg: None)  # on_segment must be keyword


def test_transcribe_chunked_is_gone():
    assert not hasattr(api, "transcribe_chunked")


def test_engine_defaults_match_api_forwarding():
    """api.transcribe() always forwards on_segment / initial_prompt / task, so
    a plain call is behaviour-preserving only while the engine's own defaults
    equal the forwarded values. If someone changes a default in
    Transcriber.transcribe, this fails loudly instead of drifting silently."""
    from visper.transcriber import Transcriber

    params = inspect.signature(Transcriber.transcribe).parameters
    assert params["on_segment"].default is None
    assert params["initial_prompt"].default is None
    assert params["task"].default == "transcribe"
