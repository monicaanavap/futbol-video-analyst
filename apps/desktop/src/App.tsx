import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type { AnalysisJob, EventDraft, EventType, EventUpdate, Match, MatchEvent, VisualSignal } from "./types";

const eventLabels: Record<EventType, string> = {
  corner: "Corners",
  throw_in: "Saques de banda",
  penalty: "Penales",
  goal: "Goles",
  shot: "Tiros",
  foul: "Faltas",
  custom: "Otros",
};

const eventColors: Record<EventType, string> = {
  corner: "#f3c969",
  throw_in: "#e6a76f",
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

function formatTimeInput(seconds: number) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  const [whole, decimal] = remainder.toFixed(2).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1").split(".");
  return `${minutes}:${whole.padStart(2, "0")}${decimal ? `.${decimal}` : ""}`;
}

function parseTimeInput(value: string) {
  const match = value.trim().match(/^(\d+):([0-5]?\d(?:\.\d{1,2})?)$/);
  if (!match) return null;
  return Number(match[1]) * 60 + Number(match[2]);
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [matches, setMatches] = useState<Match[]>([]);
  const [selected, setSelected] = useState<Match | null>(null);
  const [events, setEvents] = useState<MatchEvent[]>([]);
  const [filters, setFilters] = useState<Set<EventType>>(new Set(Object.keys(eventLabels) as EventType[]));
  const [showImport, setShowImport] = useState(false);
  const [showEvent, setShowEvent] = useState(false);
  const [showRejected, setShowRejected] = useState(false);
  const [editingEvent, setEditingEvent] = useState<MatchEvent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [exportingEvent, setExportingEvent] = useState<string | null>(null);
  const [reviewingEvent, setReviewingEvent] = useState<string | null>(null);
  const [analysisJob, setAnalysisJob] = useState<AnalysisJob | null>(null);
  const [signals, setSignals] = useState<VisualSignal[]>([]);
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
    if (!selected) { setEvents([]); setAnalysisJob(null); setSignals([]); return; }
    void api.listEvents(selected.id).then(setEvents).catch((reason: Error) => setError(reason.message));
    void api.latestAnalysis(selected.id)
      .then(async (job) => {
        setAnalysisJob(job);
        if (job.status === "completed") setSignals(await api.listSignals(selected.id));
      })
      .catch(() => { setAnalysisJob(null); setSignals([]); });
  }, [selected]);

  useEffect(() => {
    if (!analysisJob || !["queued", "running"].includes(analysisJob.status)) return;
    const timer = window.setInterval(() => {
      void api.getAnalysis(analysisJob.id).then(async (job) => {
        setAnalysisJob(job);
        if (job.status === "completed" && selected) {
          setSignals(await api.listSignals(selected.id));
          setEvents(await api.listEvents(selected.id));
          setNotice("Análisis completado; revisa los corners candidatos");
        }
        if (job.status === "failed") setError(job.error ?? "El análisis no pudo completarse");
      });
    }, 700);
    return () => window.clearInterval(timer);
  }, [analysisJob?.id, analysisJob?.status, selected]);

  const activeEvents = useMemo(
    () => events.filter((event) => showRejected || event.review_status !== "rejected"),
    [events, showRejected],
  );
  const visibleEvents = useMemo(
    () => activeEvents.filter((event) => filters.has(event.type)),
    [activeEvents, filters],
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

  const exportClip = async (event: MatchEvent) => {
    setExportingEvent(event.id);
    setError("");
    setNotice("");
    try {
      const nativeDesktop = "__TAURI_INTERNALS__" in window;
      let destination: string | null = null;
      if (nativeDesktop) {
        const { save } = await import("@tauri-apps/plugin-dialog");
        destination = await save({
          defaultPath: `${event.type}-${String(Math.floor(event.peak_seconds)).padStart(6, "0")}.mp4`,
          filters: [{ name: "Video MP4", extensions: ["mp4"] }],
        });
        if (!destination) return;
      }
      const { blob, filename, exportedPath } = await api.exportClip(event.id);
      if (nativeDesktop && destination) {
        if (!exportedPath) throw new Error("El motor no indicó dónde generó el clip");
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("save_generated_clip", { sourcePath: exportedPath, destinationPath: destination });
        setNotice(`Clip guardado en: ${destination}`);
      } else {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        setNotice(`Clip descargado: ${filename}`);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo exportar el clip");
    } finally {
      setExportingEvent(null);
    }
  };

  const reviewEvent = async (event: MatchEvent, decision: "confirmed" | "rejected") => {
    setError("");
    setReviewingEvent(event.id);
    try {
      const updated = await api.reviewEvent(event.id, decision);
      setEvents((current) => current.map((item) => item.id === updated.id ? updated : item));
      setNotice(decision === "confirmed" ? "Corner confirmado" : "Candidato descartado");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo guardar la revisión");
    } finally {
      setReviewingEvent(null);
    }
  };

  const handleEventUpdated = (updated: MatchEvent, message = "Etiqueta actualizada") => {
    setEvents((current) => current.map((event) => event.id === updated.id ? updated : event));
    setEditingEvent(null);
    setNotice(message);
  };

  const handleEventDeleted = (eventId: string) => {
    setEvents((current) => current.filter((event) => event.id !== eventId));
    setEditingEvent(null);
    setNotice("Etiqueta eliminada");
  };

  const startAnalysis = async () => {
    if (!selected) return;
    setError("");
    setSignals([]);
    try {
      setAnalysisJob(await api.startAnalysis(selected.id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "No se pudo iniciar el análisis");
    }
  };

  const fieldSamples = signals.filter((signal) => signal.likely_field).length;
  const strongChanges = signals.filter((signal) => signal.change_score >= 0.18).length;
  const averagePlayers = signals.length
    ? Math.round(signals.reduce((total, signal) => total + signal.player_candidates, 0) / signals.length)
    : 0;
  const ballSamples = signals.filter((signal) => signal.ball_candidates > 0).length;

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
        {error && <div className="error-banner"><span>{error}</span>{engineState === "error" && <button onClick={() => void connectToEngine()}>Reintentar</button>}</div>}
        {notice && <div className="notice-banner"><span>{notice}</span><button onClick={() => setNotice("")}>×</button></div>}
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
              <div className="heading-actions">
                <button className="analysis-button" disabled={analysisJob?.status === "queued" || analysisJob?.status === "running"} onClick={() => void startAnalysis()}>{analysisJob?.status === "completed" ? "Analizar de nuevo" : "Analizar partido"}</button>
                <button className="secondary" onClick={() => setShowEvent(true)}>+ Nueva etiqueta</button>
              </div>
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
            {analysisJob && <section className={`analysis-card ${analysisJob.status}`}>
              <div className="analysis-copy">
                <span className="analysis-icon">◎</span>
                <div><strong>{analysisJob.status === "completed" ? "Análisis visual listo" : analysisJob.status === "failed" ? "No se pudo analizar" : "Analizando el partido"}</strong>
                <small>{analysisJob.status === "completed" ? `${signals.length} muestras · objetos mostrados como candidatos experimentales` : analysisJob.stage === "sampling" ? "Revisando campo, luz, jugadores y balón…" : analysisJob.stage === "refining" ? "Afinando el segundo exacto de cada candidato…" : "Preparando el video…"}</small></div>
              </div>
              {analysisJob.status === "completed" ? <div className="analysis-metrics">
                <span><b>{fieldSamples}</b>campo visible</span><span><b>{strongChanges}</b>cambios fuertes</span>
                <span title="Promedio por muestra; puede incluir falsos positivos"><b>{averagePlayers}</b>jugadores candidatos</span>
                <span title="Muestras donde apareció al menos un candidato"><b>{ballSamples}</b>balón candidato</span>
              </div> : <div className="progress-wrap"><span>{Math.round(analysisJob.progress * 100)}%</span><div className="progress-track"><i style={{ width: `${analysisJob.progress * 100}%` }} /></div><small>{analysisJob.samples_processed} muestras</small></div>}
            </section>}
            <section className="filters">
              <span className="eyebrow">MOSTRAR</span>
              {(Object.keys(eventLabels) as EventType[]).map((type) => (
                <button className={filters.has(type) ? "filter active" : "filter"} key={type} onClick={() => toggleFilter(type)}>
                  <i style={{ background: eventColors[type] }} />{eventLabels[type]}
                  <b>{activeEvents.filter((event) => event.type === type).length}</b>
                </button>
              ))}
            </section>
            <section className="events-panel">
              <div className="section-title"><h2>Momentos del partido</h2><div><span>{visibleEvents.length} etiquetas</span><button onClick={() => setShowRejected((value) => !value)}>{showRejected ? "Ocultar descartados" : `Ver descartados (${events.filter((event) => event.review_status === "rejected").length})`}</button></div></div>
              {visibleEvents.length === 0 ? (
                <div className="no-events"><p>No hay etiquetas con estos filtros.</p><button onClick={() => setShowEvent(true)}>Agregar una manualmente</button></div>
              ) : visibleEvents.map((event) => (
                <div className="event-row" key={event.id}>
                  <button className="event-seek" onClick={() => seek(event.peak_seconds)}>
                    <span className="event-time">{formatTime(event.peak_seconds)}</span>
                    <i style={{ background: eventColors[event.type] }} />
                    <span><strong>{eventLabels[event.type]}</strong><small>{event.notes || "Etiqueta manual"}</small></span>
                    <span className={`event-source ${event.review_status}`}>{event.source === "manual" ? "Manual" : event.review_status === "confirmed" ? "Confirmado" : event.review_status === "rejected" ? "Descartado" : `Candidato ${Math.round(event.confidence * 100)}%`}</span>
                    <span className="play-button">▶</span>
                  </button>
                  <div className="event-actions">
                    {event.source === "detector" && event.review_status !== "confirmed" && <div className="review-actions">
                      <button disabled={reviewingEvent === event.id} className="confirm-button" onClick={() => void reviewEvent(event, "confirmed")}>{reviewingEvent === event.id ? "Guardando…" : event.review_status === "rejected" ? "Restaurar y confirmar" : "Confirmar"}</button>
                      {event.review_status === "unreviewed" && <button disabled={reviewingEvent === event.id} className="reject-button" onClick={() => void reviewEvent(event, "rejected")}>{reviewingEvent === event.id ? "Guardando…" : "Descartar"}</button>}
                    </div>}
                    <button className="edit-button" onClick={() => setEditingEvent(event)}>{event.review_status === "rejected" ? "Reclasificar" : "Editar"}</button>
                    <button className="clip-button" disabled={exportingEvent === event.id} onClick={() => void exportClip(event)}>
                      {exportingEvent === event.id ? "Exportando…" : "Exportar clip"}
                    </button>
                  </div>
                </div>
              ))}
            </section>
          </>
        )}
      </main>
      {showImport && <ImportDialog onClose={() => setShowImport(false)} onImported={handleImported} />}
      {showEvent && selected && <EventDialog match={selected} currentTime={videoRef.current?.currentTime ?? 0} onClose={() => setShowEvent(false)} onCreated={handleEventCreated} />}
      {editingEvent && selected && <EditEventDialog match={selected} event={editingEvent} onClose={() => setEditingEvent(null)} onUpdated={handleEventUpdated} onDeleted={handleEventDeleted} />}
    </div>
  );
}

function EditEventDialog({ match, event, onClose, onUpdated, onDeleted }: { match: Match; event: MatchEvent; onClose: () => void; onUpdated: (event: MatchEvent, message?: string) => void; onDeleted: (eventId: string) => void }) {
  const [draft, setDraft] = useState<EventUpdate>({ type: event.type, start_seconds: event.start_seconds, peak_seconds: event.peak_seconds, end_seconds: event.end_seconds, notes: event.notes ?? "" });
  const [startText, setStartText] = useState(formatTimeInput(event.start_seconds));
  const [peakText, setPeakText] = useState(formatTimeInput(event.peak_seconds));
  const [endText, setEndText] = useState(formatTimeInput(event.end_seconds));
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const applyPeak = () => {
    const value = parseTimeInput(peakText);
    if (value === null) { setError("Usa minuto:segundo, por ejemplo 63:19"); return; }
    const contextBefore = draft.peak_seconds - draft.start_seconds;
    const contextAfter = draft.end_seconds - draft.peak_seconds;
    const next = {
      ...draft,
      peak_seconds: value,
      start_seconds: Math.max(0, value - contextBefore),
      end_seconds: Math.min(match.duration_seconds, value + contextAfter),
    };
    setDraft(next);
    setPeakText(formatTimeInput(next.peak_seconds));
    setStartText(formatTimeInput(next.start_seconds));
    setEndText(formatTimeInput(next.end_seconds));
    setError("");
  };
  const submit = async (formEvent: FormEvent) => {
    formEvent.preventDefault(); setError("");
    const start = parseTimeInput(startText);
    const peak = parseTimeInput(peakText);
    const end = parseTimeInput(endText);
    if (start === null || peak === null || end === null) {
      setError("Usa minuto:segundo, por ejemplo 63:19"); return;
    }
    if (!(0 <= start && start <= peak && peak <= end && end <= match.duration_seconds)) {
      setError("El momento debe quedar entre el inicio y el final del clip"); return;
    }
    if (event.review_status === "rejected" && draft.type === (event.detected_type ?? event.type)) {
      setError("Para reclasificar, elige el tipo real. Si sí era corner, usa Restaurar y confirmar."); return;
    }
    setSaving(true);
    try {
      const payload = { ...draft, start_seconds: start, peak_seconds: peak, end_seconds: end };
      if (event.review_status === "rejected") {
        const updated = await api.reclassifyEvent(event.id, payload);
        onUpdated(updated, "Etiqueta reclasificada y confirmada");
      } else {
        onUpdated(await api.updateEvent(event.id, payload));
      }
    }
    catch (reason) { setError(reason instanceof Error ? reason.message : "No se pudo actualizar"); }
    finally { setSaving(false); }
  };
  const remove = async () => {
    if (!window.confirm("¿Eliminar esta etiqueta definitivamente?")) return;
    setSaving(true); setError("");
    try { await api.deleteEvent(event.id); onDeleted(event.id); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "No se pudo eliminar"); setSaving(false); }
  };
  return <div className="modal-backdrop"><form className="modal compact" noValidate onSubmit={submit}>
    <button type="button" className="close" onClick={onClose}>×</button><p className="eyebrow">{event.review_status === "rejected" ? "RECLASIFICAR DESCARTADO" : "EDITAR ETIQUETA"}</p><h2>{event.review_status === "rejected" ? "¿Qué evento fue realmente?" : "Corregir momento"}</h2>
    <label>Tipo<select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value as EventType })}>{Object.entries(eventLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
    <label>Momento clave (minuto:segundo)<input inputMode="decimal" value={peakText} onChange={(e) => setPeakText(e.target.value)} onBlur={applyPeak} placeholder="63:19" /><small>Ejemplo: 7:15 o 63:19.5</small></label>
    <details className="advanced-options"><summary>Ajustar duración del clip</summary><div className="time-grid">
      <label>Inicio (min:seg)<input inputMode="decimal" value={startText} onChange={(e) => setStartText(e.target.value)} placeholder="63:11" /></label>
      <label>Final (min:seg)<input inputMode="decimal" value={endText} onChange={(e) => setEndText(e.target.value)} placeholder="63:31" /></label>
    </div></details>
    <label>Notas<input value={draft.notes} onChange={(e) => setDraft({ ...draft, notes: e.target.value })} placeholder="Opcional" /></label>
    {error && <p className="form-error" role="alert">{error}</p>}<div className="modal-actions split"><button type="button" className="danger-button" disabled={saving} onClick={() => void remove()}>Eliminar</button><span /><button type="button" className="secondary" onClick={onClose}>Cancelar</button><button type="submit" className="primary" disabled={saving}>{saving ? "Guardando…" : event.review_status === "rejected" ? "Guardar y confirmar tipo" : "Guardar cambios"}</button></div>
  </form></div>;
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
  const [startText, setStartText] = useState(formatTimeInput(Math.max(0, peak - 5)));
  const [peakText, setPeakText] = useState(formatTimeInput(peak));
  const [endText, setEndText] = useState(formatTimeInput(Math.min(match.duration_seconds, peak + 10)));
  const [error, setError] = useState("");
  const applyPeak = () => {
    const value = parseTimeInput(peakText);
    if (value === null) { setError("Usa minuto:segundo, por ejemplo 7:15"); return; }
    const start = Math.max(0, value - 5);
    const end = Math.min(match.duration_seconds, value + 10);
    setDraft({ ...draft, peak_seconds: value, start_seconds: start, end_seconds: end });
    setPeakText(formatTimeInput(value)); setStartText(formatTimeInput(start)); setEndText(formatTimeInput(end)); setError("");
  };
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setError("");
    const start = parseTimeInput(startText); const moment = parseTimeInput(peakText); const end = parseTimeInput(endText);
    if (start === null || moment === null || end === null) { setError("Usa minuto:segundo, por ejemplo 7:15"); return; }
    if (!(0 <= start && start <= moment && moment <= end && end <= match.duration_seconds)) { setError("El momento debe quedar entre el inicio y el final del clip"); return; }
    try { onCreated(await api.createEvent(match.id, { ...draft, start_seconds: start, peak_seconds: moment, end_seconds: end })); } catch (reason) { setError(reason instanceof Error ? reason.message : "No se pudo guardar"); }
  };
  return <div className="modal-backdrop"><form className="modal compact" noValidate onSubmit={submit}>
    <button type="button" className="close" onClick={onClose}>×</button><p className="eyebrow">ETIQUETA MANUAL</p><h2>Marcar momento</h2>
    <label>Tipo<select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value as EventType })}>{Object.entries(eventLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
    <label>Momento clave (minuto:segundo)<input inputMode="decimal" value={peakText} onChange={(e) => setPeakText(e.target.value)} onBlur={applyPeak} placeholder="7:15" /><small>Usamos el momento actual del reproductor. Ejemplo: 7:15.</small></label>
    <details className="advanced-options"><summary>Ajustar duración del clip</summary><p>Solo cambia estos valores si quieres más o menos contexto.</p><div className="time-grid">
      <label>Inicio (min:seg)<input inputMode="decimal" value={startText} onChange={(e) => setStartText(e.target.value)} placeholder="7:10" /></label>
      <label>Final (min:seg)<input inputMode="decimal" value={endText} onChange={(e) => setEndText(e.target.value)} placeholder="7:25" /></label>
    </div></details>
    <label>Notas<input value={draft.notes} onChange={(e) => setDraft({ ...draft, notes: e.target.value })} placeholder="Opcional" /></label>
    {error && <p className="form-error" role="alert">{error}</p>}<div className="modal-actions"><button type="button" className="secondary" onClick={onClose}>Cancelar</button><button type="submit" className="primary">Guardar etiqueta</button></div>
  </form></div>;
}

export default App;
