"""Duration -> bucket boundaries, and parity between the two implementations
(visper.api and visper.benchmark) that both classify audio length."""
import pytest

from visper.api import _bucket_for_duration
from visper.benchmark import _bucket_for


@pytest.mark.parametrize("duration, expected", [
    (0.0, "short"),
    (5.0, "short"),
    (9.99, "short"),
    (10.0, "medium"),
    (20.0, "medium"),
    (29.99, "medium"),
    (30.0, "long"),
    (45.0, "long"),
    (59.99, "long"),
    (60.0, "extended"),
    (120.0, "extended"),
])
def test_bucket_boundaries(duration, expected):
    assert _bucket_for_duration(duration) == expected
    assert _bucket_for(duration) == expected


def test_none_duration_is_medium():
    assert _bucket_for_duration(None) == "medium"
