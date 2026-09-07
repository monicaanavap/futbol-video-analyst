from futbol_video_analyst.spotting_evaluation import TimelineEvent, evaluate


def event(match: str, event_type: str, timestamp: float) -> TimelineEvent:
    return TimelineEvent(match, event_type, timestamp)


def test_evaluates_one_to_one_full_match_spots() -> None:
    truth = [event("a", "corner", 10), event("a", "corner", 30), event("b", "corner", 5)]
    predictions = [
        event("a", "corner", 11),
        event("a", "corner", 12),
        event("a", "corner", 40),
        event("b", "corner", 4),
    ]

    report = evaluate(truth, predictions, tolerance_seconds=2)
    corner = report["classes"]["corner"]

    assert corner["true_positive"] == 2
    assert corner["false_positive"] == 2
    assert corner["false_negative"] == 1
    assert corner["precision"] == 0.5
    assert corner["recall"] == 2 / 3
    assert corner["median_timestamp_error_seconds"] == 1


def test_reports_errors_per_90_minutes_when_durations_are_known() -> None:
    report = evaluate(
        [event("a", "corner", 10)],
        [],
        match_durations={"a": 90 * 60},
    )

    assert report["classes"]["corner"]["missed_events_per_90_minutes"] == 1
