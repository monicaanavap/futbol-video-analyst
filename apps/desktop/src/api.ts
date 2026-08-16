import type { EventDraft, Match, MatchEvent } from "./types";

export const API_URL = "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: "Error inesperado" }));
    throw new Error(body.detail ?? "Error inesperado");
  }
  return response.json() as Promise<T>;
}

export const api = {
  listMatches: () => request<Match[]>("/matches"),
  importMatch: (title: string, videoPath: string) =>
    request<Match>("/matches/import", {
      method: "POST",
      body: JSON.stringify({ title, video_path: videoPath }),
    }),
  listEvents: (matchId: string) => request<MatchEvent[]>(`/matches/${matchId}/events`),
  createEvent: (matchId: string, event: EventDraft) =>
    request<MatchEvent>(`/matches/${matchId}/events`, {
      method: "POST",
      body: JSON.stringify(event),
    }),
  videoUrl: (matchId: string) => `${API_URL}/matches/${matchId}/video`,
};
