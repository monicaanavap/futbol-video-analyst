# Arquitectura inicial

## Componentes

- **Aplicación de escritorio:** reproduce el video y filtra etiquetas en la línea de tiempo.
- **API local:** registra partidos y entrega eventos únicamente por `localhost`.
- **Procesador de video:** usa FFmpeg para inspeccionar, normalizar y recortar.
- **Detector de eventos:** combina vision, audio, OCR y reglas temporales.
- **Almacenamiento:** SQLite conserva metadatos y referencias a videos locales.
- **Interfaz de revision:** permite confirmar, corregir o descartar eventos.

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
