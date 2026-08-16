# Futbol Video Analyst

Herramienta para convertir el video completo de un partido de futbol en clips
etiquetados de momentos clave: tiros de esquina, penales, goles, tiros, faltas y
otras acciones utiles para cuerpos tecnicos.

## Estado

Este repositorio contiene el setup inicial del MVP. La primera meta es construir
un flujo reproducible:

1. Subir o registrar un video.
2. Extraer metadatos y preparar el archivo.
3. Detectar candidatos a eventos.
4. Recortar clips alrededor de cada evento.
5. Permitir que una persona revise y corrija las etiquetas.

## Requisitos locales

- Python 3.11 o superior
- FFmpeg y FFprobe

## Inicio rapido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn futbol_video_analyst.main:app --reload
```

Visita `http://127.0.0.1:8000/docs` para probar la API.

## Flujo disponible

El motor local ya permite:

1. Importar un video por su ruta local y leer sus metadatos con FFprobe.
2. Guardar el partido en SQLite sin copiar ni subir el video.
3. Crear etiquetas manuales ligadas a segundos del partido.
4. Consultar todas las etiquetas o filtrarlas por tipo.

Ejemplo para importar un partido:

```bash
curl -X POST http://127.0.0.1:8000/matches/import \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Local vs Visitante",
    "video_path": "/ruta/absoluta/al/partido.mp4"
  }'
```

La base local se crea en `data/futbol-video-analyst.sqlite3`. Los videos permanecen
en su ubicación original.

## Pruebas

```bash
pytest
```

## Alcance sugerido del primer MVP

Empezaremos con un evento que pueda medirse bien: tiros de esquina. Cada etiqueta
conservará su intervalo, momento principal, confianza, origen y estado de revisión.
La interfaz mostrará las etiquetas sobre el video completo y permitirá filtrarlas.

Consulta [docs/architecture.md](docs/architecture.md) para la propuesta tecnica.
