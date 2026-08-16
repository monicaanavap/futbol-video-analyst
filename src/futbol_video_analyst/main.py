from fastapi import FastAPI

app = FastAPI(
    title="Futbol Video Analyst API",
    description="API para procesar partidos y generar clips etiquetados.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
