# Instrucciones para este proyecto

## Sobre el proyecto
Este es el proyecto final de mi pregrado en Física. Un proyecto de investigación. En este, vamos a trabajar tanto en la redacción del documento como en el código y procedimientos.

## Reglas de redacción
- Escribe siempre en español neutro. Traduciendo términos como Binary a Binaria.
- Usa un tono académico como el que se usa en astronomía
- Evita muletillas y el uso de "--".
- Formato de fechas: DD/MM/AAAA
- Revisar rigurosamente el uso de comas: evitar tanto el mal uso como la omisión de comas necesarias (ej. comas antes de "es decir", "sin embargo", enumeraciones, etc.)
- Para referenciar imágenes, ecuaciones y tablas en LaTeX: la oración que las menciona debe cerrar el párrafo, y el comando de inserción (\includegraphics, \begin{equation}, \begin{table}, etc.) va después, fuera del párrafo.
  Ejemplo correcto:
  "Como se aprecia en la Figura 3, la señal decae exponencialmente.

  \begin{figure}
  \includegraphics{...}
  \end{figure}"

  Ejemplo incorrecto (no seguir este):
  "Como se ve en la siguiente imagen: {inserta imagen 3}"
- Usar adecuadamente \cite{} y \citep{} para la coherencia y pulcritud de la citación.
- \citep{} cuando la cita va entre paréntesis al final de la oración (ej. "... como se ha demostrado (Autor, 2022)")
- \cite{} cuando el autor es parte de la oración (ej. "Autor (2022) demostró que...")
- Fíjate que al usar plantillas que usen títulos por defecto, estos no queden en inglés u otro idioma distinto al español.

## Estructura de archivos
- `Propuesta`: proyecto LaTeX de la propuesta de proyecto final.
- `Tesis`: proyecto LaTeX del documento del proyecto final.
- `Articulos`: PDFs de artículos que se usen o puedan usarse como referencias.
- `Tesis_Anteriores`: PDFs de proyectos finales anteriores dirigidos por el mismo profesor director. Son modelo de estructura y estética del documento, **no fuentes bibliográficas por defecto**. Solo citar de aquí si aportan algo genuinamente relevante.

## Convenciones no negociables
- Nunca inventar datos o cifras.
- Citar fuentes cuando se mencionen datos externos.
- A la hora de buscar fuentes, priorizar que estas hayan sido publicadas del 2021 en adelante. De utilizar fuentes más antiguas, avisar y explicar por qué se utilizaron. 

## Código
- Lenguaje: [Python]
- Librerías principales: [numpy, astropy, matplotlib, entropypf]
- Los scripts van en /src