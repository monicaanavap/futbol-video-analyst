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

## Interfaz local

En otra terminal, inicia la interfaz React:

```bash
cd apps/desktop
npm install
npm run dev
```

Abre `http://127.0.0.1:1420`. La interfaz permite importar partidos, reproducir
el video original, ver etiquetas en la línea de tiempo, filtrarlas por tipo y
crear nuevas etiquetas manuales.

Para ejecutar la ventana nativa durante el desarrollo:

```bash
cd apps/desktop
npm run tauri dev
```

La ventana nativa inicia automáticamente el motor Python desde `.venv`, espera a
que esté disponible y lo detiene cuando la aplicación termina. Si el motor ya
está ejecutándose en el puerto `8000`, la app reutiliza esa instancia.

En Web Preview todavía se deben iniciar por separado la API y `npm run dev`, ya
que ahí no participa el proceso nativo de Tauri. Una etapa posterior convertirá
el motor en un binario auxiliar para distribuirlo sin requerir Python instalado.

## Flujo disponible

El motor local ya permite:

1. Importar un video por su ruta local y leer sus metadatos con FFprobe.
2. Guardar el partido en SQLite sin copiar ni subir el video.
3. Crear etiquetas manuales ligadas a segundos del partido.
4. Consultar todas las etiquetas o filtrarlas por tipo.
5. Exportar bajo demanda un clip MP4 desde el inicio hasta el final de una etiqueta.
6. Ejecutar un primer análisis visual local con progreso y señales por muestra.
7. Proponer corners candidatos, confirmarlos o descartarlos desde la línea de tiempo.
8. Editar tipo, momento, intervalo y notas de cualquier etiqueta, o eliminarla.
9. Preparar un dataset local con clips agrupados por partido y categoría.

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
en su ubicación original. Los clips solicitados se generan en `data/clips`; por
defecto incluyen cinco segundos antes y diez después del momento clave. La opción
`Ajustar duración del clip` permite cambiar ese intervalo.

El botón `Analizar partido` toma una muestra cada dos segundos y calcula, con
OpenCV, proporción de verde, luminosidad, cambio visual, líneas blancas y candidatos
a jugadores y balón. Los objetos son estimaciones geométricas experimentales, no
detecciones confirmadas: pueden contener falsos positivos y todavía no representan
eventos. Son las señales base para construir y evaluar el detector automático de
corners.

El detector temporal de corners exige que coincidan campo visible, líneas blancas,
balón y varios jugadores en una toma estable. Agrupa señales cercanas para evitar
etiquetas repetidas y muestra una confianza aproximada. Sigue siendo una heurística
experimental: cada propuesta debe revisarse antes de considerarse un corner real.
Después de encontrar una zona candidata, una segunda pasada revisa el intervalo
cada medio segundo y coloca el momento en el cambio de movimiento más probable.
Cuando existen al menos dos corners confirmados y cinco descartados, el detector
también compara las nuevas señales contra esas revisiones locales. Las etiquetas
eliminadas no participan en esta calibración.

En la app de escritorio, `Exportar clip` abre el diálogo nativo para elegir nombre
y destino y después muestra la ruta completa donde quedó guardado. En el navegador,
el mismo botón conserva la descarga normal del navegador. Las etiquetas descartadas
se pueden mostrar y restaurar si una decisión se tomó por error.

La interfaz recibe todos los tiempos como `minuto:segundo` (por ejemplo `63:19` o
`7:15.5`) y reajusta el contexto del clip al mover el momento clave. Un candidato
puede reclasificarse como penal, saque de banda, tiro
u otro evento antes de confirmarlo. La aplicación conserva internamente el tipo
original propuesto para que esa corrección también mejore el detector de corners.
Un evento descartado ofrece `Reclasificar`: guardar el tipo real y confirmarlo es
una sola operación, sin tener que restaurarlo primero como corner.

El botón `Preparar dataset` recorta en una sola operación todas las etiquetas
manuales, los candidatos confirmados y los eventos reclasificados. Los candidatos
descartados se guardan como ejemplos `negative` y los candidatos todavía sin revisar
se omiten. Cada exportación crea una carpeta nueva bajo `data/datasets`, separa los
clips por partido y categoría, y agrega `manifest.jsonl` y `summary.json`. El
identificador de partido del manifiesto permitirá dividir entrenamiento y prueba sin
mezclar clips del mismo video. La preparación corre en segundo plano y el botón muestra
su porcentaje para que una exportación larga no bloquee ni desconecte la aplicación.

## Pruebas

```bash
pytest
```

## Entrenamiento experimental

El primer entrenador local aprende `corner` contra todas las demás etiquetas usando R3D-18
preentrenado en Kinetics-400 como extractor congelado y una cabeza lineal. La
separación se hace por partido completo para evitar fuga de información.

```bash
pip install -e '.[training]'
python -m futbol_video_analyst.training \
  --dataset data/datasets/dataset-AAAAMMDD-HHMMSS-ID \
  --task corner
```

Las características se conservan en `data/training_cache` para acelerar ejecuciones
posteriores. Los pesos `.pt` quedan locales en `models` y cada versión genera también
un JSON con métricas y el partido usado para validación. Un modelo nuevo es
experimental y no se activa automáticamente dentro de la aplicación.

## Alcance sugerido del primer MVP

Empezaremos con un evento que pueda medirse bien: tiros de esquina. Cada etiqueta
conservará su intervalo, momento principal, confianza, origen y estado de revisión.
La interfaz mostrará las etiquetas sobre el video completo y permitirá filtrarlas.

Consulta [docs/architecture.md](docs/architecture.md) para la propuesta tecnica.
