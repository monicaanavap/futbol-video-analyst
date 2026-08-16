export type EventType = "corner" | "penalty" | "goal" | "shot" | "foul" | "custom";

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
  type: EventType;
  start_seconds: number;
  peak_seconds: number;
  end_seconds: number;
  confidence: number;
  source: "manual" | "detector";
  review_status: string;
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
