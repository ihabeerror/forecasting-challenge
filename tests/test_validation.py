import pytest
from cwfc.validation import expanding_origins


def test_expanding_origins_end_at_last_row():
    assert expanding_origins(110, 24, 60, 6) == [62, 68, 74, 80, 86]
    assert expanding_origins(30, 24, 10, 6) == []
    with pytest.raises(ValueError):
        expanding_origins(110, 24, 0, 6)
