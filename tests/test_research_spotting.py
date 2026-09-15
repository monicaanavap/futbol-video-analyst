from futbol_video_analyst.domain import EventType
from futbol_video_analyst.research_spotting import prefer_goals
from futbol_video_analyst.spotting_evaluation import TimelineEvent


def test_prefers_goal_over_nearby_shot_attempt() -> None:
    predictions = [
        TimelineEvent("match", EventType.SHOT_ATTEMPT, 807.0, 0.97),
        TimelineEvent("match", EventType.GOAL, 809.0, 0.96),
        TimelineEvent("match", EventType.SHOT_ATTEMPT, 830.0, 0.98),
    ]

    filtered = prefer_goals(predictions)

    assert [(item.event_type, item.timestamp_seconds) for item in filtered] == [
        (EventType.GOAL, 809.0),
        (EventType.SHOT_ATTEMPT, 830.0),
    ]
