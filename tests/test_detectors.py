from futbol_video_analyst.detectors import CornerCandidateDetector, select_motion_timestamp
from futbol_video_analyst.domain import Match, MatchStatus, ReviewStatus, VisualSignal


def make_match() -> Match:
    return Match(
        id="match-id",
        title="Test",
        video_path="/tmp/test.mp4",
        duration_seconds=90,
        width=1920,
        height=1080,
        fps=30,
        codec="h264",
        status=MatchStatus.READY,
        created_at="2026-08-18 00:00:00",
    )


def signal(
    timestamp: float,
    players: int = 4,
    ball: int = 1,
    line_ratio: float = 0.02,
    change_score: float = 0.05,
    green_ratio: float = 0.7,
) -> VisualSignal:
    return VisualSignal(
        id=f"signal-{timestamp}",
        match_id="match-id",
        timestamp_seconds=timestamp,
        green_ratio=green_ratio,
        brightness=0.5,
        change_score=change_score,
        likely_field=True,
        player_candidates=players,
        ball_candidates=ball,
        line_ratio=line_ratio,
    )


def test_groups_nearby_corner_signals_and_uses_best_confidence() -> None:
    candidates = CornerCandidateDetector().detect(
        make_match(), [signal(20, players=3), signal(22, players=6), signal(60)]
    )

    assert [candidate.peak_seconds for candidate in candidates] == [22, 60]
    assert candidates[0].confidence > candidates[1].confidence
    assert all(candidate.source == "detector" for candidate in candidates)


def test_requires_ball_players_and_lines() -> None:
    candidates = CornerCandidateDetector().detect(
        make_match(), [signal(20, ball=0), signal(40, players=1)]
    )

    assert candidates == []


def test_refines_coarse_timestamp_to_strongest_field_motion() -> None:
    timestamp = select_motion_timestamp(
        440,
        [
            (434.5, 0.018, 0.7),
            (435.0, 0.075, 0.68),
            (436.0, 0.03, 0.7),
            (438.0, 0.4, 0.1),
        ],
    )

    assert timestamp == 435.0


def test_keeps_coarse_timestamp_without_usable_motion() -> None:
    assert select_motion_timestamp(440, [(435.0, 0.004, 0.7)]) == 440


def test_review_calibration_keeps_confirmed_like_signal_and_filters_rejected() -> None:
    positive_one = signal(10, players=10, ball=70, line_ratio=0.08, change_score=0.12)
    positive_two = signal(20, players=14, ball=95, line_ratio=0.08, change_score=0.15)
    rejected = [
        signal(30 + index, players=25, ball=140, line_ratio=0.14, change_score=0.17)
        for index in range(5)
    ]
    examples = [
        (positive_one, ReviewStatus.CONFIRMED),
        (positive_two, ReviewStatus.CONFIRMED),
        *((item, ReviewStatus.REJECTED) for item in rejected),
    ]

    accepted = CornerCandidateDetector().detect(make_match(), [positive_one], examples)
    filtered = CornerCandidateDetector().detect(make_match(), [rejected[0]], examples)

    assert len(accepted) == 1
    assert accepted[0].notes == "Candidato automático calibrado con revisiones locales"
    assert filtered == []
