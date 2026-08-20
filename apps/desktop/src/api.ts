import type { AnalysisJob, DatasetExportJob, EventDraft, EventUpdate, Match, MatchEvent, VisualSignal } from "./types";

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
  reviewEvent: (eventId: string, reviewStatus: "confirmed" | "rejected") =>
    request<MatchEvent>(`/events/${eventId}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_status: reviewStatus }),
    }),
  createEvent: (matchId: string, event: EventDraft) =>
    request<MatchEvent>(`/matches/${matchId}/events`, {
      method: "POST",
      body: JSON.stringify(event),
    }),
  updateEvent: (eventId: string, event: EventUpdate) =>
    request<MatchEvent>(`/events/${eventId}`, {
      method: "PATCH",
      body: JSON.stringify(event),
    }),
  reclassifyEvent: (eventId: string, event: EventUpdate) =>
    request<MatchEvent>(`/events/${eventId}/reclassify`, {
      method: "PATCH",
      body: JSON.stringify(event),
    }),
  deleteEvent: async (eventId: string) => {
    const response = await fetch(`${API_URL}/events/${eventId}`, { method: "DELETE" });
    if (!response.ok) throw new Error("No se pudo eliminar la etiqueta");
  },
  exportClip: async (eventId: string) => {
    const response = await fetch(`${API_URL}/events/${eventId}/clip`, { method: "POST" });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: "No se pudo exportar el clip" }));
      throw new Error(body.detail ?? "No se pudo exportar el clip");
    }
    const disposition = response.headers.get("Content-Disposition") ?? "";
    const filename = disposition.match(/filename="?([^";]+)"?/)?.[1] ?? `clip-${eventId}.mp4`;
    const exportedPath = response.headers.get("X-Exported-Path");
    return { blob: await response.blob(), filename, exportedPath };
  },
  exportDataset: () => request<DatasetExportJob>("/dataset/export", { method: "POST" }),
  getDatasetExport: (jobId: string) => request<DatasetExportJob>(`/dataset/export/${jobId}`),
  startAnalysis: (matchId: string) =>
    request<AnalysisJob>(`/matches/${matchId}/analysis`, { method: "POST" }),
  getAnalysis: (jobId: string) => request<AnalysisJob>(`/analysis/${jobId}`),
  latestAnalysis: (matchId: string) =>
    request<AnalysisJob>(`/matches/${matchId}/analysis/latest`),
  listSignals: (matchId: string) =>
    request<VisualSignal[]>(`/matches/${matchId}/signals`),
  videoUrl: (matchId: string) => `${API_URL}/matches/${matchId}/video`,
};
