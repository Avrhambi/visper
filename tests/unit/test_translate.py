"""visper.translate — model acquisition + degradation (no model download needed)."""
import io
import tarfile

import pytest

from visper import translate


@pytest.fixture(autouse=True)
def _reset():
    translate.reset_cache()
    yield
    translate.reset_cache()


def _make_tar(members: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_safe_extract_rejects_path_traversal(tmp_path):
    tar_bytes = _make_tar({"../escape.txt": b"x", "ok.txt": b"y"})
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
        with pytest.raises(RuntimeError, match="unsafe path"):
            translate._safe_extract(tar, tmp_path)


def test_safe_extract_allows_normal_members(tmp_path):
    tar_bytes = _make_tar({"opus-mt-tc-big-he-en-ct2/model.bin": b"weights"})
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
        translate._safe_extract(tar, tmp_path)
    assert (tmp_path / "opus-mt-tc-big-he-en-ct2" / "model.bin").read_bytes() == b"weights"


def test_ensure_model_noops_when_present(monkeypatch):
    calls = []
    monkeypatch.setattr(translate, "_model_present", lambda: True)
    monkeypatch.setattr(translate.urllib.request, "urlopen",
                        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError))
    assert translate.ensure_model() is True
    assert calls == []  # no network touched


def test_get_translator_returns_none_and_caches_failure(monkeypatch):
    attempts = []

    def _boom():
        attempts.append(1)
        raise RuntimeError("no model here")

    monkeypatch.setattr(translate, "ensure_model", _boom)

    assert translate.get_hebrew_english_translator() is None
    assert translate.get_hebrew_english_translator() is None  # still None
    assert len(attempts) == 1  # failure cached, not retried on every call


def test_checksum_mismatch_is_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(translate, "_model_present", lambda: False)
    monkeypatch.setattr(translate, "_MODEL_DIR", tmp_path / "models" / "m")

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=-1):
            if not hasattr(self, "_done"):
                self._done = True
                return b"not the real asset"
            return b""

    monkeypatch.setattr(translate.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.delenv(translate._ASSET_URL_ENV, raising=False)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        translate.ensure_model()


@pytest.mark.skipif(not translate._model_present(),
                    reason="he->en model not installed locally")
def test_translate_roundtrip_and_blank_passthrough():
    mt = translate.get_hebrew_english_translator()
    assert mt is not None
    out = mt.translate(["שלום עולם", "", "   "])
    assert out[1] == "" and out[2] == "   "
    assert out[0] and out[0] != "שלום עולם"  # produced *some* English
