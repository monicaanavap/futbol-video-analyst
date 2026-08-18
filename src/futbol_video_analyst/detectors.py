from futbol_video_analyst.domain import EventCreate, EventSource, EventType, Match, VisualSignal


class CornerCandidateDetector:
    """Find conservative corner-like sequences from experimental visual signals."""

    cooldown_seconds = 20.0

    def detect(self, match: Match, signals: list[VisualSignal]) -> list[EventCreate]:
        scored: list[tuple[VisualSignal, float]] = []
        for signal in signals:
            if (
                not signal.likely_field
                or signal.ball_candidates < 1
                or signal.player_candidates < 2
                or signal.line_ratio < 0.006
                or signal.change_score >= 0.18
            ):
                continue
            score = (
                0.2
                + min(signal.player_candidates / 6, 1) * 0.25
                + min(signal.ball_candidates, 1) * 0.25
                + min(signal.line_ratio / 0.02, 1) * 0.2
                + 0.1
            )
            scored.append((signal, min(score, 0.92)))

        selected: list[tuple[VisualSignal, float]] = []
        for signal, score in scored:
            if (
                selected
                and signal.timestamp_seconds - selected[-1][0].timestamp_seconds
                < self.cooldown_seconds
            ):
                if score > selected[-1][1]:
                    selected[-1] = (signal, score)
                continue
            selected.append((signal, score))

        return [
            EventCreate(
                type=EventType.CORNER,
                start_seconds=max(0, signal.timestamp_seconds - 8),
                peak_seconds=signal.timestamp_seconds,
                end_seconds=min(match.duration_seconds, signal.timestamp_seconds + 12),
                confidence=score,
                source=EventSource.DETECTOR,
                notes="Candidato automático: balón, jugadores y líneas visibles en una toma estable",
            )
            for signal, score in selected
        ]
