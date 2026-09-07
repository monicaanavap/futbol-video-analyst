# Futbol Video Analyst

Herramienta para convertir el video completo de un partido de futbol en clips
etiquetados de momentos clave: tiros de esquina, saques de meta, tiros libres,
intentos de gol, penales, goles, faltas y
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
10. Mover un partido a la papelera y recuperarlo después con su video, etiquetas y análisis intactos.

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
puede reclasificarse como penal, saque de banda, tiro libre, intento de gol
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

El mismo entrenador acepta `--task shot_attempt` y `--task free_kick`, generando
artefactos versionados separados. Para datasets densos se puede limitar el número de
negativos por positivo con `--negative-ratio 10`. Estos modelos no se activan
automáticamente.

`--cache RUTA.npz` permite compartir embeddings entre manifiestos compatibles y
evita volver a procesar videos que ya fueron extraídos.

Las características se conservan en `data/training_cache` para acelerar ejecuciones
posteriores. Los pesos `.pt` quedan locales en `models` y cada versión genera también
un JSON con métricas y el partido usado para validación.

El modelo seleccionado se configura con `CORNER_MODEL_PATH`. Si el archivo y las
dependencias de entrenamiento están disponibles, **Analizar partido** recorre el video
completo en ventanas de cuatro segundos, evalúa cada ventana con R3D-18 y agrupa
detecciones cercanas. Si falta el modelo o PyTorch, la aplicación conserva el detector
visual anterior como respaldo.

```bash
CORNER_MODEL_ENABLED=true
CORNER_MODEL_PATH=./models/corner-spotter-v005.pt
CORNER_MODEL_DEVICE=auto
```

La activación permanece en `false` mientras el modelo no supere una evaluación de
barrido sobre un partido completo; las métricas de clips centrados no son suficientes.

## Action spotting temporal

El reemplazo experimental del clasificador de clips sigue un contrato temporal
versionado: RegNetY-200MF extrae una característica por cuadro a 2 fps y una BiGRU
produce probabilidades y ajustes temporales para `corner`, `goal_kick` y
`shot_attempt`. Cada ventana contiene 128 cuadros (64 segundos), avanza 64 cuadros
(32 segundos) y usa umbrales y supresión temporal independientes por clase. El
detector activo no se reemplaza hasta superar la evaluación de partidos completos.
La implementación usa `timm` porque `torchvision` no incluye la variante exacta
RegNetY-200MF. El código de `timm` es Apache-2.0; cualquier peso preentrenado se
audita por separado de la licencia del código y el baseline comienza sin pesos
externos por defecto.

El segundo entrenador aprende sobre la línea temporal completa en lugar de clasificar
clips aislados. Combina cada señal visual con contexto aproximado de 4, 12 y 24 segundos,
incluye juego normal como negativos, busca máximos locales y limita el resultado a 30
candidatos. La validación se mantiene separada por partido.

```bash
python -m futbol_video_analyst.temporal_training \
  --database data/futbol-video-analyst.sqlite3 \
  --validation-match Clasico2009 \
  --output models/corner-temporal-v001.json
```

El artefacto es JSON y se ejecuta con NumPy, sin requerir PyTorch durante el análisis.
Sólo debe activarse después de una prueba satisfactoria sobre un partido completo:

```bash
CORNER_TEMPORAL_MODEL_ENABLED=true
CORNER_TEMPORAL_MODEL_PATH=./models/corner-temporal-v001.json
```

La siguiente variante usa action spotting visual: concatena embeddings R3D-18 de
cinco ventanas entre ocho segundos antes y ocho después del momento evaluado. También
agrega secuencias de juego normal tomadas de los partidos completos.

```bash
python -m futbol_video_analyst.action_training \
  --dataset data/datasets/dataset-AAAAMMDD-HHMMSS-ID \
  --database data/futbol-video-analyst.sqlite3 \
  --validation-match Clasico2009 \
  --background-interval 60
```

### Snapshots y regreso a modelos anteriores

Cada entrenamiento genera un archivo versionado y no reemplaza los pesos anteriores.
Antes de activar o probar un entrenamiento nuevo, guarda además una copia exacta de
todos los modelos activos y su configuración:

```bash
python -m futbol_video_analyst.model_snapshots create --name "antes de bulk training"
python -m futbol_video_analyst.model_snapshots list
```

Para regresar a un estado anterior, usa el identificador mostrado por `list` y
reinicia la aplicación después de restaurarlo:

```bash
python -m futbol_video_analyst.model_snapshots restore ID-DEL-SNAPSHOT
```

Los artefactos quedan en `models/snapshots`. Cada copia incluye un checksum SHA-256;
la restauración se cancela si un archivo fue eliminado o alterado.

### Datos externos: SoccerTrack v2

SoccerTrack v2 se usa bajo CC BY 4.0. Los videos y anotaciones descargados permanecen
en `data/external` y no se agregan a Git. El convertidor conserva procedencia, licencia,
partido y mitad en cada ejemplo:

```bash
python -m futbol_video_analyst.soccertrack_import \
  --source data/external/soccertrack-v2 \
  --output data/datasets/soccertrack-v2-train \
  --matches 128058
```

`SHOT` y `GOAL` se convierten a `shot_attempt`; `FREE KICK` se convierte a
`free_kick`. SoccerTrack v2 no incluye etiquetas explícitas de corner ni saque de
meta, por lo que el importador no inventa esas clases.

### SoccerNet-v2 para experimentos no comerciales

SoccerNet-v2 sirve para reproducir y comparar la arquitectura temporal con muchos
partidos de transmisión. Su FAQ indica que el dataset es para investigación y no
está destinado a uso comercial. Por eso cada manifiesto queda marcado
`commercial_model_eligible: false`; no se puede activar ni distribuir como modelo
comercial. El adaptador referencia los videos existentes y no los copia:

```bash
python -m futbol_video_analyst.soccernet_import \
  --source /RUTA/A/SoccerNet \
  --output data/datasets/soccernet-v2-research \
  --plan data/external/soccernet-v2/pilot-selection.json
```

La carpeta raíz debe contener un `Labels-v2.json` por partido y los videos por mitad,
preferentemente `1_224p.mkv` y `2_224p.mkv`. Se mapean `Corner`, tiros dentro/fuera
de portería y `Goal`; SoccerNet no proporciona la clase `goal_kick` necesaria para
nuestro MVP. No guardes la contraseña del NDA en el repositorio.

Para no ocupar el disco con los 500 partidos, primero genera un piloto que conserva
los splits oficiales y prioriza partidos ricos en corners y tiros:

```bash
pip install -e '.[external-data]'
python -m futbol_video_analyst.soccernet_download plan \
  --source data/external/soccernet-v2 \
  --output data/external/soccernet-v2/pilot-selection.json \
  --train-matches 20 \
  --valid-matches 5

python -m futbol_video_analyst.soccernet_download download \
  --source data/external/soccernet-v2 \
  --plan data/external/soccernet-v2/pilot-selection.json
```

El segundo comando solicita la contraseña interactivamente y no la guarda. Descarga
solo `1_224p.mkv` y `2_224p.mkv` de los 25 partidos seleccionados. El conjunto oficial
de prueba no se descarga ni se usa durante el desarrollo del piloto.

Para una corrida mayor, crea el manifiesto con las rutas esperadas aunque los videos
todavía no estén descargados y divídelo en lotes. La contraseña puede residir en el
Llavero de macOS; nunca se escribe en el comando, los logs ni Git:

```bash
python -m futbol_video_analyst.soccernet_download batches \
  --plan data/external/soccernet-v2/research-100-selection.json \
  --output-dir data/external/soccernet-v2/research-100-batches \
  --cache-dir data/training_cache/soccernet-regnety002-2fps-pts-v3 \
  --batch-size 25

python -m futbol_video_analyst.soccernet_download download \
  --source data/external/soccernet-v2 \
  --plan data/external/soccernet-v2/research-100-batches/batch-001.json \
  --keychain-service futbol-video-analyst-soccernet

python -m futbol_video_analyst.research_preparation \
  --dataset data/datasets/soccernet-v2-research-100-v2 \
  --cache-dir data/training_cache/soccernet-regnety002-2fps-pts-v3 \
  --device mps

python -m futbol_video_analyst.soccernet_download purge \
  --source data/external/soccernet-v2 \
  --plan data/external/soccernet-v2/research-100-batches/batch-001.json \
  --cache-dir data/training_cache/soccernet-regnety002-2fps-pts-v3 \
  --confirm
```

`purge` elimina únicamente mitades incluidas en el plan y se niega a hacerlo hasta
que ambos cachés contengan timestamps y embeddings válidos. Así se pueden procesar
25-50 partidos por vez sin conservar todos los videos crudos en disco.

También existe un comando local para procesar los lotes sin utilizar Codex. Para
descargar solamente un lote:

```bash
cd /Users/monicanavapalomo/Documents/Codex/futbol-video-analyst
./scripts/soccernet-batch download data/external/soccernet-v2/research-500-batches/batch-001.json
```

Para descargar, preparar las características y borrar automáticamente los videos
crudos que ya estén protegidos por una caché válida:

```bash
./scripts/soccernet-batch all \
  data/external/soccernet-v2/research-500-batches/batch-001.json \
  data/datasets/soccernet-v2-research-500-v1
```

La herramienta obtiene la contraseña desde el servicio
`futbol-video-analyst-soccernet` del Llavero de macOS. El proceso se puede cerrar y
reanudar: las mitades ya descargadas o preparadas se reutilizan.

El evaluador compara timestamps one-to-one por clase y partido completo. Acepta
JSONL con `match_id`, `event_type` (o `label`) y `timestamp_seconds` (o
`peak_seconds`):

```bash
python -m futbol_video_analyst.spotting_evaluation \
  --truth data/evaluation/truth.jsonl \
  --predictions data/evaluation/predictions.jsonl \
  --tolerance 2 \
  --output data/evaluation/report.json
```

El reporte muestra precision, recall, F1, falsos positivos, eventos perdidos y error
temporal. La activación exige resultados en partidos completos que no participaron
en el entrenamiento.

El modelo temporal de investigación extrae y guarda características RegNetY-200MF a
2 fps, entrena una BiGRU sobre partidos del split oficial de entrenamiento, calibra
umbrales en validación y nunca activa el checkpoint automáticamente:

```bash
python -m futbol_video_analyst.research_training \
  --dataset data/datasets/soccernet-v2-research-100-v2 \
  --cache-dir data/training_cache/soccernet-regnety002-2fps-pts-v3 \
  --epochs 20 \
  --architecture tcn \
  --device mps
```

`--architecture bigru` conserva la línea base recurrente. `--architecture tcn`
emplea bloques convolucionales temporales dilatados y bidireccionales, adecuados
para el análisis offline porque aprovechan contexto anterior y posterior al evento.

Los presentation timestamps del contenedor son el reloj canónico; nunca se infiere
el tiempo únicamente a partir del número de cuadro. Esto conserva la alineación ante
videos con frame rate variable o saltos internos.

Los artefactos `soccernet-research-temporal-v*.pt` quedan marcados
`research_only`. La caché por mitad permite repetir la BiGRU sin volver a descargar
ni decodificar los videos.

El benchmark A/B local ejecuta el pipeline comercial y el checkpoint de investigación
sobre los mismos partidos completos. Guarda cada predicción por separado para poder
reanudarlo y reporta métricas a ±2, ±5 y ±10 segundos:

La calibración de revisión asistida de `v005` usa un umbral de corner de `0.93`.
Alcanzó 90.14% de recall en cinco partidos locales completos; el valor y su linaje
están congelados en `models/soccernet-research-temporal-v005-review.json`. Este perfil
sirve únicamente para proponer bloques que una persona debe verificar.

En desarrollo se habilita con `RESEARCH_ASSISTED_ENABLED=true`. La aplicación muestra
un botón separado, **Revisión asistida v005**, que ejecuta el maestro localmente y abre
automáticamente bloques de cinco candidatos. El análisis comercial conserva su propio
botón y sus modelos. Los ejemplos exportados desde v005 incluyen `annotation_tier`,
`teacher_model` y `commercial_eligibility` en el manifiesto para impedir que se pierda
su procedencia durante la preparación del futuro modelo comercial.

Las opciones y restricciones para importar grabaciones mediante proveedores externos
se documentan en `docs/licensed-video-ingestion.md`.

```bash
python -m futbol_video_analyst.model_benchmark \
  --research-model models/soccernet-research-temporal-v002.pt \
  --output-dir data/evaluations/local-ab-research-v002 \
  --device mps
```

El reporte indica qué partidos ya habían participado en el entrenamiento comercial;
esas filas sirven como prueba de regresión, pero no como estimación independiente de
generalización.

## Alcance sugerido del primer MVP

Empezaremos con un evento que pueda medirse bien: tiros de esquina. Cada etiqueta
conservará su intervalo, momento principal, confianza, origen y estado de revisión.
La interfaz mostrará las etiquetas sobre el video completo y permitirá filtrarlas.

Consulta [docs/architecture.md](docs/architecture.md) para la propuesta tecnica.
