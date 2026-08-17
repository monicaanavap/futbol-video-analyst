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
  health: () => request<{ status: string }>("/health"),
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
  exportClip: async (eventId: string) => {
    const response = await fetch(`${API_URL}/events/${eventId}/clip`, { method: "POST" });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: "No se pudo exportar el clip" }));
      throw new Error(body.detail ?? "No se pudo exportar el clip");
    }
    const disposition = response.headers.get("Content-Disposition") ?? "";
    const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] ?? `clip-${eventId}.mp4`;
    return { blob: await response.blob(), filename };
  },
  videoUrl: (matchId: string) => `${API_URL}/matches/${matchId}/video`,
};
