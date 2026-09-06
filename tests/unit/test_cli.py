"""CLI entry-point helpers."""
import io
import sys

from visper._cli import _utf8_console


def test_utf8_console_is_idempotent_and_safe(monkeypatch):
    # A plain StringIO has no .reconfigure — must be swallowed, not raised.
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    _utf8_console()
    _utf8_console()
    print("→ — ─")  # would UnicodeEncodeError on a cp125x console


def test_utf8_console_reconfigures_a_buffered_stream(monkeypatch):
    calls = []

    class FakeStream:
        def reconfigure(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())
    _utf8_console()
    assert calls == [
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
    ]
