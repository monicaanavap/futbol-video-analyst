# Arquitectura inicial

## Componentes

- **API:** recibe partidos, crea trabajos y entrega eventos y clips.
- **Procesador de video:** usa FFmpeg para inspeccionar, normalizar y recortar.
- **Detector de eventos:** combina vision, audio, OCR y reglas temporales.
- **Almacenamiento:** conserva videos originales, clips y metadatos.
- **Interfaz de revision:** permite confirmar, corregir o descartar eventos.

## Flujo de datos

```text
Video -> validacion -> normalizacion -> deteccion -> eventos candidatos
      -> recorte de clips -> revision humana -> exportacion
```

## Estrategia del MVP

Un evento no debe depender de una sola señal. Por ejemplo, un tiro de esquina
puede combinar posicion del balon, vista cercana de la esquina, marcador/OCR y
cambios en el audio. El sistema debe guardar las señales y la confianza para que
el cuerpo tecnico pueda verificar el resultado rapidamente.

En la primera iteracion conviene procesar los archivos en una sola maquina y
guardar metadatos en SQLite. Cuando el volumen lo justifique se pueden separar
los trabajos en una cola y mover los videos a almacenamiento de objetos.
