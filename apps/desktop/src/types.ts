export type EventType = "corner" | "throw_in" | "penalty" | "goal" | "shot" | "foul" | "custom";

export interface Match {
  id: string;
  title: string;
  video_path: string;
  duration_seconds: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  status: string;
  created_at: string;
}

export interface MatchEvent {
  id: string;
  match_id: string;
  detected_type: EventType | null;
  type: EventType;
  start_seconds: number;
  peak_seconds: number;
  end_seconds: number;
  confidence: number;
  source: "manual" | "detector";
  review_status: "unreviewed" | "confirmed" | "rejected";
  notes: string | null;
  created_at: string;
}

export interface EventDraft {
  type: EventType;
  start_seconds: number;
  peak_seconds: number;
  end_seconds: number;
  confidence: number;
  source: "manual";
  notes: string;
}

export interface EventUpdate {
  type: EventType;
  start_seconds: number;
  peak_seconds: number;
  end_seconds: number;
  notes: string;
}

export interface AnalysisJob {
  id: string;
  match_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: "queued" | "sampling" | "refining" | "completed" | "failed";
  progress: number;
  samples_processed: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface VisualSignal {
  id: string;
  match_id: string;
  timestamp_seconds: number;
  green_ratio: number;
  brightness: number;
  change_score: number;
  likely_field: boolean;
  player_candidates: number;
  ball_candidates: number;
  line_ratio: number;
}

export interface DatasetExport {
  path: string;
  manifest_path: string;
  clips: number;
  matches: number;
  skipped_events: number;
  label_counts: Record<string, number>;
}
