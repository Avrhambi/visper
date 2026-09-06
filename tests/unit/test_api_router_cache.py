"""_get_router must rebuild when the hardware config changes, not pin the
first one it ever saw (warmup's heuristic config vs. the post-benchmark real one)."""
import pytest

import visper.api as api
import visper.model_router as mr


@pytest.fixture(autouse=True)
def _fake_router(monkeypatch):
    built = []

    class FakeRouter:
        def __init__(self, hw):
            self.hw = hw
            self.unloaded = False
            built.append(self)

        def unload(self):
            self.unloaded = True

    monkeypatch.setattr(mr, "ModelRouter", FakeRouter)
    monkeypatch.setattr(api, "_router", None)
    monkeypatch.setattr(api, "_router_hw", ())
    return built


CPU = {"device": "cpu", "compute_type": "int8"}
CUDA = {"device": "cuda", "compute_type": "int8_float32"}


def test_same_config_reuses_instance():
    a = api._get_router(CPU)
    b = api._get_router(dict(CPU))
    assert a is b


def test_config_change_rebuilds_and_unloads_old(_fake_router):
    old = api._get_router(CPU)
    new = api._get_router(CUDA)
    assert new is not old
    assert old.unloaded is True
    assert new.hw == CUDA


def test_venv_path_change_rebuilds():
    a = api._get_router(CPU)
    b = api._get_router({**CPU, "venv_path": "/x/.venvs/cpu"})
    assert a is not b


def test_reset_caches_clears_everything(monkeypatch):
    monkeypatch.setattr(api, "_config_cache", {"medium": CPU})
    r = api._get_router(CPU)
    api.reset_caches()
    assert api._router is None
    assert api._config_cache == {}
    assert r.unloaded is True
