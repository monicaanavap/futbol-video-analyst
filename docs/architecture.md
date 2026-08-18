# Arquitectura inicial

## Componentes

- **Aplicación de escritorio:** reproduce el video y filtra etiquetas en la línea de tiempo.
- **API local:** registra partidos y entrega eventos únicamente por `localhost`.
- **Procesador de video:** usa FFmpeg para inspeccionar, normalizar y recortar.
- **Detector de eventos:** combina vision, audio, OCR y reglas temporales.
- **Almacenamiento:** SQLite conserva metadatos y referencias a videos locales.
- **Interfaz de revision:** permite confirmar, corregir o descartar eventos.

## Implementación de escritorio

La interfaz utiliza React y TypeScript. Tauri crea la ventana nativa, proporciona
el selector de archivos y administra el ciclo de vida del motor FastAPI en
`127.0.0.1:8000`; no se permite acceso desde orígenes remotos. React reintenta la
conexión mientras el proceso local termina de arrancar y ofrece recuperación si
el proceso falla.

El endpoint de video solo sirve rutas registradas previamente en SQLite. Esto
permite reproducción y búsqueda temporal sin duplicar el archivo original.

Los clips son derivados opcionales: FFmpeg recodifica únicamente el intervalo
`start_seconds`–`end_seconds` cuando el usuario solicita una exportación. No se
generan clips para todas las etiquetas automáticamente, evitando procesamiento y
uso de disco innecesarios.

## Pipeline visual inicial

El análisis corre en un único worker local para no saturar la computadora del
coach. Los trabajos y su progreso se guardan en SQLite. OpenCV toma una muestra
cada dos segundos, reduce su tamaño y calcula:

- proporción de pixeles compatibles con césped;
- luminosidad media;
- diferencia visual respecto a la muestra anterior;
- una clasificación preliminar de campo visible.
- candidatos a jugadores mediante componentes verticales no verdes;
- candidatos a balón mediante regiones blancas pequeñas y circulares;
- proporción de líneas o regiones blancas.

Las señales se guardan con su timestamp y pueden recuperarse después de cerrar la
aplicación. Los candidatos usan heurísticas abiertas y pueden producir falsos
positivos. El detector temporal combina estas señales para proponer etiquetas que
siempre requieren revisión humana.

## Detector temporal de corners

La primera regla temporal propone un corner cuando una muestra estable combina
campo visible, líneas, balón y concentración mínima de jugadores. Las propuestas
cercanas se agrupan y se conserva la de mayor confianza. Se guardan como eventos
de origen `detector` y estado `unreviewed`; la interfaz permite confirmarlas o
descartarlas. Un análisis posterior reemplaza solamente propuestas no revisadas y
respeta etiquetas manuales y decisiones humanas anteriores.

Antes de guardar una propuesta, una segunda pasada muestrea el intervalo cercano
cada 500 ms. Descarta cortes de cámara y tomas sin campo, y usa el cambio visual
más fuerte restante como instante probable del saque. Si no encuentra movimiento
confiable conserva el timestamp aproximado original. Por ahora aplica dos segundos
entre el inicio de movimiento y el contacto probable con el balón, calibrados con
el primer corner real revisado (`7:13` de movimiento, `7:15` de saque).

La revisión admite corregir el tipo de evento (por ejemplo, convertir un corner
candidato en tiro), notas e intervalo antes de confirmar, así como eliminar una
etiqueta. Los descartes permanecen en SQLite para auditoría y pueden restaurarse.
La exportación nativa limita la copia a archivos generados dentro de `data/clips`
y exige un destino MP4 elegido mediante el diálogo de Tauri.

## Flujo de datos

```text
Video local -> validación -> análisis -> eventos candidatos
            -> línea de tiempo -> revisión humana -> exportación opcional
```

## Estrategia del MVP

Un evento no debe depender de una sola señal. Por ejemplo, un tiro de esquina
puede combinar posicion del balon, vista cercana de la esquina, marcador/OCR y
cambios en el audio. El sistema debe guardar las señales y la confianza para que
el cuerpo tecnico pueda verificar el resultado rapidamente.

La aplicación procesa todo en la computadora del coach. El video no se copia a la
base ni se envía por red. Una cola persistente local permitirá recuperar trabajos
interrumpidos sin necesitar Redis ni infraestructura en la nube.
