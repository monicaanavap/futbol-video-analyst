import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type { EventDraft, EventType, Match, MatchEvent } from "./types";

const eventLabels: Record<EventType, string> = {
  corner: "Corners",
  penalty: "Penales",
  goal: "Goles",
  shot: "Tiros",
  foul: "Faltas",
  custom: "Otros",
};

const eventColors: Record<EventType, string> = {
  corner: "#f3c969",
  penalty: "#ef7d90",
  goal: "#6de0a5",
  shot: "#75b7f5",
  foul: "#d79bf3",
  custom: "#a7b3ab",
};

function formatTime(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [selected, setSelected] = useState<Match | null>(null);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [filters, setFilters] = useState<Set<EventType>>(new Set(Object.keys(eventLabels) as EventType[]));
  const [showImport, setShowImport] = useState(false);
  const [showEvent, setShowEvent] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [engineState, setEngineState] = useState<"starting" | "ready" | "error">("starting");

  const connectToEngine = async () => {
    setLoading(true);
    setEngineState("starting");
    setError("");
    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        await api.health();
        const result = await api.listMatches();
        setMatches(result);
        setSelected((current) => current ?? result[0] ?? null);
        setEngineState("ready");
        setLoading(false);
        return;
      } catch {
        await new Promise((resolve) => window.setTimeout(resolve, 350));
      }
    }
    setEngineState("error");
    setError("El motor local no pudo iniciar. Puedes intentar conectarlo de nuevo.");
    setLoading(false);
  };

  useEffect(() => { void connectToEngine(); }, []);
  useEffect(() => {
    if (!selected) { setEvents([]); return; }
    void api.listEvents(selected.id).then(setEvents).catch((reason: Error) => setError(reason.message));
  }, [selected]);

  const visibleEvents = useMemo(
    () => events.filter((event) => filters.has(event.type)),
    [events, filters],
  );

  const seek = (seconds: number) => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, seconds - 5);
    void videoRef.current.play();
  };

  const toggleFilter = (type: EventType) => {
    setFilters((current) => {
      const next = new Set(current);
      next.has(type) ? next.delete(type) : next.add(type);
      return next;
    });
  };

  const handleImported = (match: Match) => {
    setMatches((current) => [match, ...current]);
    setSelected(match);
    setShowImport(false);
  };

  const handleEventCreated = (event: MatchEvent) => {
    setEvents((current) => [...current, event].sort((a, b) => a.peak_seconds - b.peak_seconds));
    setShowEvent(false);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark">F</div>
        <div><strong>Fútbol Analyst</strong><span>Análisis local · Tus videos no salen de aquí</span></div>
        <button className="primary" disabled={engineState !== "ready"} onClick={() => setShowImport(true)}>+ Importar partido</button>
      </header>

      <aside className="sidebar">
        <p className="eyebrow">PARTIDOS</p>
        {loading && <p className="muted">Iniciando motor local…</p>}
        {matches.map((match) => (
          <button
            className={`match-card ${selected?.id === match.id ? "active" : ""}`}
            key={match.id}
            onClick={() => setSelected(match)}
          >
            <span className="match-date">{new Date(`${match.created_at}Z`).toLocaleDateString("es-MX")}</span>
            <strong>{match.title}</strong>
            <span>{formatTime(match.duration_seconds)} · {match.height}p</span>
          </button>
        ))}
      </aside>

      <main className="workspace">
        {error && <div className="error-banner"><span>{error}</span><button onClick={() => void connectToEngine()}>Reintentar</button></div>}
        {!selected ? (
          <section className="empty-state">
            <div className="empty-icon">▶</div>
            <h1>Tu primer partido empieza aquí</h1>
            <p>Importa un video local para verlo, etiquetarlo y preparar su análisis.</p>
            <button className="primary" onClick={() => setShowImport(true)}>Importar video</button>
          </section>
        ) : (
          <>
            <section className="match-heading">
              <div><p className="eyebrow">PARTIDO</p><h1>{selected.title}</h1></div>
              <button className="secondary" onClick={() => setShowEvent(true)}>+ Nueva etiqueta</button>
            </section>
            <section className="video-panel">
              <video ref={videoRef} key={selected.id} controls src={api.videoUrl(selected.id)} />
              <div className="timeline" aria-label="Línea de tiempo de eventos">
                <div className="timeline-track" />
                {visibleEvents.map((event) => (
                  <button
                    key={event.id}
                    className="timeline-marker"
                    style={{ left: `${(event.peak_seconds / selected.duration_seconds) * 100}%`, background: eventColors[event.type] }}
                    title={`${eventLabels[event.type]} · ${formatTime(event.peak_seconds)}`}
                    onClick={() => seek(event.peak_seconds)}
                  />
                ))}
              </div>
            </section>
            <section className="filters">
              <span className="eyebrow">MOSTRAR</span>
              {(Object.keys(eventLabels) as EventType[]).map((type) => (
                <button className={filters.has(type) ? "filter active" : "filter"} key={type} onClick={() => toggleFilter(type)}>
                  <i style={{ background: eventColors[type] }} />{eventLabels[type]}
                  <b>{events.filter((event) => event.type === type).length}</b>
                </button>
              ))}
            </section>
            <section className="events-panel">
              <div className="section-title"><h2>Momentos del partido</h2><span>{visibleEvents.length} etiquetas</span></div>
              {visibleEvents.length === 0 ? (
                <div className="no-events"><p>No hay etiquetas con estos filtros.</p><button onClick={() => setShowEvent(true)}>Agregar una manualmente</button></div>
              ) : visibleEvents.map((event) => (
                <button className="event-row" key={event.id} onClick={() => seek(event.peak_seconds)}>
                  <span className="event-time">{formatTime(event.peak_seconds)}</span>
                  <i style={{ background: eventColors[event.type] }} />
                  <span><strong>{eventLabels[event.type]}</strong><small>{event.notes || "Etiqueta manual"}</small></span>
                  <span className="event-source">{event.source === "manual" ? "Manual" : `${Math.round(event.confidence * 100)}%`}</span>
                  <span className="play-button">▶</span>
                </button>
              ))}
            </section>
          </>
        )}
      </main>
      {showImport && <ImportDialog onClose={() => setShowImport(false)} onImported={handleImported} />}
      {showEvent && selected && <EventDialog match={selected} currentTime={videoRef.current?.currentTime ?? 0} onClose={() => setShowEvent(false)} onCreated={handleEventCreated} />}
    </div>
  );
}

function ImportDialog({ onClose, onImported }: { onClose: () => void; onImported: (match: Match) => void }) {
  const [title, setTitle] = useState("");
  const [path, setPath] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const chooseFile = async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ multiple: false, filters: [{ name: "Videos", extensions: ["mp4", "mov", "mkv", "m4v"] }] });
      if (typeof selected === "string") setPath(selected);
    } catch { setError("En Web Preview pega la ruta completa; el selector se activa en la app de escritorio."); }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault(); setSaving(true); setError("");
    try { onImported(await api.importMatch(title, path)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "No se pudo importar"); }
    finally { setSaving(false); }
  };

  return <div className="modal-backdrop"><form className="modal" onSubmit={submit}>
    <button type="button" className="close" onClick={onClose}>×</button>
    <p className="eyebrow">NUEVO PARTIDO</p><h2>Importar video local</h2><p className="muted">El archivo permanece en tu computadora.</p>
    <label>Nombre del partido<input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Halcones vs Deportivo Norte" required /></label>
    <label>Ruta del video<div className="path-input"><input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/Users/.../partido.mp4" required /><button type="button" onClick={chooseFile}>Elegir</button></div></label>
    {error && <p className="form-error">{error}</p>}
    <div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancelar</button><button className="primary" disabled={saving}>{saving ? "Leyendo…" : "Importar"}</button></div>
  </form></div>;
}

function EventDialog({ match, currentTime, onClose, onCreated }: { match: Match; currentTime: number; onClose: () => void; onCreated: (event: MatchEvent) => void }) {
  const peak = Math.round(currentTime);
  const [draft, setDraft] = useState<EventDraft>({ type: "corner", start_seconds: Math.max(0, peak - 5), peak_seconds: peak, end_seconds: Math.min(match.duration_seconds, peak + 10), confidence: 1, source: "manual", notes: "" });
  const [error, setError] = useState("");
  const submit = async (event: FormEvent) => { event.preventDefault(); try { onCreated(await api.createEvent(match.id, draft)); } catch (reason) { setError(reason instanceof Error ? reason.message : "No se pudo guardar"); } };
  return <div className="modal-backdrop"><form className="modal compact" onSubmit={submit}>
    <button type="button" className="close" onClick={onClose}>×</button><p className="eyebrow">ETIQUETA MANUAL</p><h2>Marcar momento</h2>
    <label>Tipo<select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value as EventType })}>{Object.entries(eventLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
    <div className="time-grid">{(["start_seconds", "peak_seconds", "end_seconds"] as const).map((key) => <label key={key}>{key === "start_seconds" ? "Inicio" : key === "peak_seconds" ? "Momento" : "Final"}<input type="number" min="0" max={match.duration_seconds} value={draft[key]} onChange={(e) => setDraft({ ...draft, [key]: Number(e.target.value) })} /></label>)}</div>
    <label>Notas<input value={draft.notes} onChange={(e) => setDraft({ ...draft, notes: e.target.value })} placeholder="Opcional" /></label>
    {error && <p className="form-error">{error}</p>}<div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancelar</button><button className="primary">Guardar etiqueta</button></div>
  </form></div>;
}

export default App;
