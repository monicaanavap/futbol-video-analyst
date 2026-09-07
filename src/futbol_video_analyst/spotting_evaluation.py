import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimelineEvent:
    match_id: str
    event_type: str
    timestamp_seconds: float
    confidence: float = 1.0


def load_events(path: Path) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        timestamp = record.get("timestamp_seconds", record.get("peak_seconds"))
        event_type = record.get("event_type", record.get("label", record.get("type")))
        if timestamp is None or event_type is None or "match_id" not in record:
            raise ValueError(f"Invalid event on line {line_number} of {path}")
        events.append(
            TimelineEvent(
                match_id=str(record["match_id"]),
                event_type=str(event_type),
                timestamp_seconds=float(timestamp),
                confidence=float(record.get("confidence", 1.0)),
            )
        )
    return events


def _match_timeline(
    truth: list[TimelineEvent], predictions: list[TimelineEvent], tolerance: float
) -> tuple[int, int, int, list[float]]:
    expected = sorted(truth, key=lambda item: item.timestamp_seconds)
    proposed = sorted(predictions, key=lambda item: item.timestamp_seconds)
    truth_index = prediction_index = 0
    errors: list[float] = []
    false_positive = false_negative = 0
    while truth_index < len(expected) and prediction_index < len(proposed):
        delta = proposed[prediction_index].timestamp_seconds - expected[truth_index].timestamp_seconds
        if delta < -tolerance:
            false_positive += 1
            prediction_index += 1
        elif delta > tolerance:
            false_negative += 1
            truth_index += 1
        else:
            errors.append(abs(delta))
            truth_index += 1
            prediction_index += 1
    false_negative += len(expected) - truth_index
    false_positive += len(proposed) - prediction_index
    return len(errors), false_positive, false_negative, errors


def evaluate(
    truth: list[TimelineEvent],
    predictions: list[TimelineEvent],
    tolerance_seconds: float = 2.0,
    match_durations: dict[str, float] | None = None,
) -> dict[str, Any]:
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds cannot be negative")
    grouped_truth: dict[tuple[str, str], list[TimelineEvent]] = defaultdict(list)
    grouped_predictions: dict[tuple[str, str], list[TimelineEvent]] = defaultdict(list)
    for event in truth:
        grouped_truth[(event.match_id, event.event_type)].append(event)
    for event in predictions:
        grouped_predictions[(event.match_id, event.event_type)].append(event)

    classes = sorted({event.event_type for event in [*truth, *predictions]})
    match_ids = {event.match_id for event in [*truth, *predictions]}
    reports: dict[str, Any] = {}
    for event_type in classes:
        true_positive = false_positive = false_negative = 0
        errors: list[float] = []
        for match_id in match_ids:
            matched, extra, missed, match_errors = _match_timeline(
                grouped_truth[(match_id, event_type)],
                grouped_predictions[(match_id, event_type)],
                tolerance_seconds,
            )
            true_positive += matched
            false_positive += extra
            false_negative += missed
            errors.extend(match_errors)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        reports[event_type] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "median_timestamp_error_seconds": median(errors) if errors else None,
            "p90_timestamp_error_seconds": float(np.percentile(errors, 90)) if errors else None,
        }

    total_minutes = (
        sum(match_durations.get(match_id, 0.0) for match_id in match_ids) / 60
        if match_durations
        else None
    )
    if total_minutes:
        for report in reports.values():
            report["false_tags_per_90_minutes"] = report["false_positive"] * 90 / total_minutes
            report["missed_events_per_90_minutes"] = report["false_negative"] * 90 / total_minutes
    return {
        "tolerance_seconds": tolerance_seconds,
        "matches": len(match_ids),
        "classes": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-match action spotting predictions")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate(
        load_events(arguments.truth),
        load_events(arguments.predictions),
        arguments.tolerance,
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
