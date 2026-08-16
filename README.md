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

## Pruebas

```bash
pytest
```

## Alcance sugerido del primer MVP

Empezaremos con uno o dos eventos que puedan medirse bien, por ejemplo tiros de
esquina y penales. Cada clip conservara el minuto del partido, la confianza del
detector y un estado de revision humana. La deteccion de goles, formaciones,
posesion y seguimiento de jugadores puede añadirse por etapas.

Consulta [docs/architecture.md](docs/architecture.md) para la propuesta tecnica.
