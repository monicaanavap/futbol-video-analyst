import pytest

from futbol_video_analyst.video import VideoInspectionError, _parse_frame_rate


def test_parses_fractional_frame_rate() -> None:
    assert _parse_frame_rate("30000/1001") == 30000 / 1001


def test_rejects_zero_frame_rate_denominator() -> None:
    with pytest.raises(VideoInspectionError):
        _parse_frame_rate("30/0")
