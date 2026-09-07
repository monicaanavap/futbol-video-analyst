import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from futbol_video_analyst.action_spotting import ActionSpottingConfig, extract_spots
from futbol_video_analyst.analysis import (
    VisualSignalAnalyzer,
    merge_spotter_candidates,
)
from futbol_video_analyst.database import Database
from futbol_video_analyst.detectors import CornerTimestampRefiner
from futbol_video_analyst.domain import EventSource, EventType, Match, ReviewStatus
from futbol_video_analyst.research_training import (
    build_temporal_head,
    extract_video_embeddings,
    predict_timelines,
)
from futbol_video_analyst.spotting import NeuralCornerSpotter
from futbol_video_analyst.spotting_evaluation import TimelineEvent, evaluate
from futbol_video_analyst.temporal import TemporalCornerSpotter
from futbol_video_analyst.training import _device

DEFAULT_MATCH_TITLES = (
    "Clasico2009",
    "Alemania vs brasil",
    "Belgica vs usa",
    "belgica vs japon",
    "uruguay vs Inglaterra",
)


def reviewed_corner_truth(database: Database, match: Match) -> list[TimelineEvent]:
    """Return human-reviewed corners, excluding pending automatic candidates."""
    return [
        TimelineEvent(match.id, EventType.CORNER, event.peak_seconds)
        for event in database.list_events(match.id)
        if event.type is EventType.CORNER
        and event.review_status is not ReviewStatus.REJECTED
        and (
            event.source is EventSource.MANUAL
            or event.review_status is ReviewStatus.CONFIRMED
        )
    ]


def _serialize_events(path: Path, events: list[TimelineEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(event), ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


def _load_cached_events(path: Path) -> list[TimelineEvent] | None:
    if not path.is_file():
        return None
    return [
        TimelineEvent(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_predictions(match: Match, candidates: list[Any]) -> list[TimelineEvent]:
    return [
        TimelineEvent(
            match.id,
            EventType.CORNER,
            candidate.peak_seconds,
            candidate.confidence,
        )
        for candidate in candidates
    ]


class ResearchCornerSpotter:
    def __init__(
        self,
        model_path: Path,
        cache_dir: Path,
        requested_device: str,
        corner_threshold: float | None = None,
    ) -> None:
        from futbol_video_analyst.research_training import _ml

        self.model_path = model_path
        self.cache_dir = cache_dir
        self.torch, timm = _ml()
        self.device = _device(self.torch, requested_device)
        checkpoint = self.torch.load(model_path, map_location="cpu", weights_only=False)
        config_values = dict(checkpoint["config"])
        config_values["labels"] = tuple(config_values["labels"])
        self.config = ActionSpottingConfig(**config_values)
        self.thresholds = checkpoint["thresholds"]
        if corner_threshold is not None:
            self.thresholds = {**self.thresholds, "corner": corner_threshold}
        self.encoder = timm.create_model(
            "regnety_002", pretrained=True, num_classes=0, global_pool="avg"
        ).eval().to(self.device)
        self.model = build_temporal_head(
            int(checkpoint["feature_size"]),
            int(checkpoint["hidden_size"]),
            len(self.config.labels),
            architecture=checkpoint.get("temporal_architecture", "bigru"),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["temporal_head_state_dict"])
        self.model.eval()

    def spot(self, match: Match, encoder_batch_size: int, batch_size: int) -> list[TimelineEvent]:
        cache_path = self.cache_dir / f"{match.id}.npz"
        embeddings, timestamps = extract_video_embeddings(
            Path(match.video_path),
            cache_path,
            self.encoder,
            self.torch,
            self.device,
            self.config,
            encoder_batch_size,
        )
        timeline = {
            "match_id": match.id,
            "embeddings": embeddings,
            "timestamps": timestamps,
            "targets": np.zeros(
                (len(timestamps), len(self.config.labels)), dtype=np.float32
            ),
        }
        probabilities = predict_timelines(
            self.model,
            [timeline],
            self.config,
            self.torch,
            self.device,
            batch_size,
        )[match.id]
        return [
            TimelineEvent(match.id, spot.event_type, spot.timestamp_seconds, spot.confidence)
            for spot in extract_spots(
                probabilities, timestamps, self.config, self.thresholds
            )
            if spot.event_type == EventType.CORNER
        ]


def commercial_spots(
    database: Database,
    match: Match,
    action_model: Path,
    temporal_model: Path,
    requested_device: str,
) -> list[TimelineEvent]:
    signals = database.list_visual_signals(match.id)
    if not signals:
        signals = VisualSignalAnalyzer().analyze(match, lambda _progress, _samples: None)
    temporal = TemporalCornerSpotter(temporal_model).detect(match, signals) or []
    last_bucket = -1

    def report(progress: float, _windows: int) -> None:
        nonlocal last_bucket
        bucket = min(10, int(progress * 10))
        if bucket > last_bucket:
            last_bucket = bucket
            print(f"    commercial R3D: {bucket * 10}%", flush=True)

    neural = NeuralCornerSpotter(action_model, requested_device).spot(match, report) or []
    candidates = merge_spotter_candidates(temporal, neural)
    candidates = CornerTimestampRefiner().refine(match, candidates)
    return _event_predictions(match, candidates)


def select_matches(database: Database, titles: tuple[str, ...]) -> list[Match]:
    normalize = lambda value: " ".join(value.split()).casefold()
    by_title = {normalize(match.title): match for match in database.list_matches()}
    missing = [title for title in titles if normalize(title) not in by_title]
    if missing:
        raise ValueError(f"Partidos no encontrados: {', '.join(missing)}")
    selected = [by_title[normalize(title)] for title in titles]
    missing_videos = [match.title for match in selected if not Path(match.video_path).is_file()]
    if missing_videos:
        raise ValueError(f"Videos no encontrados: {', '.join(missing_videos)}")
    return selected


def run_benchmark(
    database_path: Path,
    research_model: Path,
    action_model: Path,
    temporal_model: Path,
    output_dir: Path,
    cache_dir: Path,
    titles: tuple[str, ...],
    requested_device: str,
    encoder_batch_size: int = 64,
    batch_size: int = 16,
    research_corner_threshold: float | None = None,
) -> Path:
    database = Database(database_path)
    database.initialize()
    matches = select_matches(database, titles)
    output_dir.mkdir(parents=True, exist_ok=True)
    research = ResearchCornerSpotter(
        research_model, cache_dir, requested_device, research_corner_threshold
    )
    truth: list[TimelineEvent] = []
    predictions: dict[str, list[TimelineEvent]] = {"commercial": [], "research": []}

    for index, match in enumerate(matches, 1):
        match_truth = reviewed_corner_truth(database, match)
        if not match_truth:
            raise ValueError(f"{match.title} no tiene corners revisados")
        truth.extend(match_truth)
        print(
            f"Benchmark {index}/{len(matches)}: {match.title} ({len(match_truth)} corners)",
            flush=True,
        )
        for model_name, model_predictions in predictions.items():
            cache_path = output_dir / "predictions" / f"{model_name}-{match.id}.jsonl"
            cached = _load_cached_events(cache_path)
            if cached is not None:
                print(f"  {model_name}: cache ({len(cached)} candidatos)", flush=True)
                model_predictions.extend(cached)
                continue
            if model_name == "commercial":
                spots = commercial_spots(
                    database, match, action_model, temporal_model, requested_device
                )
            else:
                spots = research.spot(match, encoder_batch_size, batch_size)
            _serialize_events(cache_path, spots)
            model_predictions.extend(spots)
            print(f"  {model_name}: {len(spots)} candidatos", flush=True)

    _serialize_events(output_dir / "truth.jsonl", truth)
    durations = {match.id: match.duration_seconds for match in matches}
    reports = {
        model_name: {
            str(tolerance): evaluate(
                truth,
                model_predictions,
                tolerance_seconds=tolerance,
                match_durations=durations,
            )
            for tolerance in (2.0, 5.0, 10.0)
        }
        for model_name, model_predictions in predictions.items()
    }
    training_titles = set(
        json.loads(action_model.with_suffix(".json").read_text(encoding="utf-8"))[
            "training_matches"
        ]
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Comparable local full-match A/B benchmark",
        "models": {
            "commercial": {
                "action_model": action_model.name,
                "temporal_model": temporal_model.name,
            },
            "research": research_model.name,
            "research_corner_threshold": research.thresholds["corner"],
        },
        "matches": [
            {
                "id": match.id,
                "title": match.title,
                "corners": len(reviewed_corner_truth(database, match)),
                "commercial_training_overlap": match.title in training_titles,
            }
            for match in matches
        ],
        "truth_definition": (
            "Confirmed corner labels plus non-rejected manually added corner labels; "
            "pending automatic candidates are excluded"
        ),
        "metrics": reports,
        "limitations": [
            "Commercial training overlap is reported per match and makes its aggregate optimistic.",
            "The research checkpoint is SoccerNet-derived and research-only.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for model_name, model_predictions in predictions.items():
        _serialize_events(output_dir / f"{model_name}.jsonl", model_predictions)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark commercial and research spotters")
    parser.add_argument("--database", type=Path, default=Path("data/futbol-video-analyst.sqlite3"))
    parser.add_argument(
        "--research-model", type=Path, default=Path("models/soccernet-research-temporal-v001.pt")
    )
    parser.add_argument("--action-model", type=Path, default=Path("models/corner-action-v006.pt"))
    parser.add_argument(
        "--temporal-model", type=Path, default=Path("models/corner-temporal-v001.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/evaluations/local-ab-v001")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/training_cache/local-regnety002-2fps-pts-v1")
    )
    parser.add_argument("--match", action="append", dest="matches")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--research-corner-threshold", type=float)
    arguments = parser.parse_args()
    run_benchmark(
        arguments.database.resolve(),
        arguments.research_model.resolve(),
        arguments.action_model.resolve(),
        arguments.temporal_model.resolve(),
        arguments.output_dir.resolve(),
        arguments.cache_dir.resolve(),
        tuple(arguments.matches or DEFAULT_MATCH_TITLES),
        arguments.device,
        arguments.encoder_batch_size,
        arguments.batch_size,
        arguments.research_corner_threshold,
    )


if __name__ == "__main__":
    main()
