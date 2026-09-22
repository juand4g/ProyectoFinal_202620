"""
Prueba exploratoria: estima el ancho tipico, en fase, de los eclipses de
binarias eclipsantes NO-contacto (tipo morfologico "NC" del catalogo, que se
aproximan a Algol/Beta Lyrae), usando curvas de luz reales bien muestreadas
(muchos puntos = mas confiables). No existe una columna de duracion de
eclipse en el catalogo de OGLE, asi que se mide directamente.

Metodo por estrella:
1. Se pliega la curva de luz en el periodo CATALOGADO (no el de Lomb-Scargle,
   para no heredar el problema de alias que se esta investigando en otras
   pruebas).
2. Se bin-nea la fase en `N_BINS` bins y se toma la mediana de magnitud por
   bin (suaviza el ruido fotometrico).
3. La linea base (fuera de eclipse) es la mediana de TODOS los puntos; la
   profundidad total es la diferencia entre el bin mas tenue y la linea
   base.
4. Un bin se marca "en eclipse" si su magnitud supera linea_base +
   0.5*profundidad (criterio de ancho a media profundidad, FWHM). Se
   agrupan bins contiguos "en eclipse" (con wraparound en fase 0/1) como
   eclipses individuales -- tipicamente 2 por estrella (primario y
   secundario).
5. Se descartan estrellas cuya profundidad total sea muy pequena (eclipse
   no resuelto con confianza sobre el ruido fotometrico).

Reaprovecha el cache ampliado de explorar_asimetria_alias_x2.py solo para
elegir candidatas por numero de puntos; no depende de sus resultados de
alias.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_AMPLIADO = BASE_DIR / "Lomb-Scargle" / "cache_binarias_no_c_asimetria.csv"

N_CANDIDATAS = 40  # top-N por numero de puntos, entre las que se filtra por profundidad
N_BINS = 100  # mas grueso que un primer intento con 300: con ~700 puntos/estrella,
              # 300 bins dejaba ~2 puntos/bin y fragmentaba los eclipses en ruido
VENTANA_SUAVIZADO = 5  # bins; suavizado circular tipo media movil
MAX_ECLIPSES_POR_ESTRELLA = 2  # primario + secundario; el resto se descarta como ruido
MIN_BINS_GRUPO = 2  # grupos mas angostos que esto se ignoran (ruido de un solo bin)
PROFUNDIDAD_MINIMA = 0.05  # mag; descarta eclipses no resueltos sobre el ruido


def cargar_fotometria(id_estrella):
    t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
    return t, mag


def perfil_binneado(fase, mag, n_bins=N_BINS):
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(fase, bordes) - 1, 0, n_bins - 1)
    perfil = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = mag[idx == b]
        if vals.size > 0:
            perfil[b] = np.median(vals)
    return perfil, bordes


def suavizar_circular(perfil, ventana=VENTANA_SUAVIZADO):
    """Media movil circular (fase 0 y 1 son el mismo punto). Interpola
    primero los bins vacios (NaN) para que el suavizado no se rompa."""
    n = len(perfil)
    idx = np.arange(n)
    valido = ~np.isnan(perfil)
    if valido.sum() < n:
        perfil = perfil.copy()
        perfil[~valido] = np.interp(idx[~valido], idx[valido], perfil[valido], period=n)
    pad = ventana // 2
    extendido = np.pad(perfil, pad, mode="wrap")
    kernel = np.ones(ventana) / ventana
    return np.convolve(extendido, kernel, mode="valid")


def anchos_de_eclipse(perfil, bordes):
    """Suaviza el perfil binneado, identifica grupos contiguos de bins 'en
    eclipse' (mag > linea_base + 0.5*profundidad, con wraparound en fase
    0/1), descarta grupos de un solo bin (ruido) y se queda con los
    `MAX_ECLIPSES_POR_ESTRELLA` mas anchos (primario y secundario)."""
    validos = ~np.isnan(perfil)
    if validos.sum() < N_BINS * 0.5:
        return [], np.nan, np.nan

    suave = suavizar_circular(perfil)
    linea_base = np.median(suave)
    profundidad = np.percentile(suave, 98) - linea_base
    if profundidad < PROFUNDIDAD_MINIMA:
        return [], linea_base, profundidad

    umbral = linea_base + 0.5 * profundidad
    en_eclipse = suave > umbral

    n = len(en_eclipse)
    visitado = np.zeros(n, dtype=bool)
    grupos = []
    ancho_bin = bordes[1] - bordes[0]
    for i in range(n):
        if en_eclipse[i] and not visitado[i]:
            j, cuenta = i, 0
            while en_eclipse[j % n] and not visitado[j % n] and cuenta < n:
                visitado[j % n] = True
                j += 1
                cuenta += 1
            k = i - 1
            while en_eclipse[k % n] and not visitado[k % n] and cuenta < n:
                visitado[k % n] = True
                k -= 1
                cuenta += 1
            if cuenta >= MIN_BINS_GRUPO:
                grupos.append(cuenta * ancho_bin)

    grupos = sorted(grupos, reverse=True)[:MAX_ECLIPSES_POR_ESTRELLA]
    return grupos, linea_base, profundidad


def main():
    cache = pd.read_csv(CACHE_AMPLIADO)
    candidatas = cache[cache["tipo_morf"] == "NC"].sort_values("n_puntos", ascending=False).head(N_CANDIDATAS)

    filas = []
    for _, fila in candidatas.iterrows():
        t, mag = cargar_fotometria(fila["id"])
        periodo = fila["periodo_catalogo"]
        fase = np.mod((t - t[0]) / periodo, 1.0)
        perfil, bordes = perfil_binneado(fase, mag)
        anchos, linea_base, profundidad = anchos_de_eclipse(perfil, bordes)
        filas.append({
            "id": fila["id"], "n_puntos": fila["n_puntos"], "periodo": periodo,
            "profundidad_mag": profundidad, "n_eclipses_detectados": len(anchos),
            "ancho_mayor": anchos[0] if len(anchos) >= 1 else np.nan,
            "ancho_menor": anchos[1] if len(anchos) >= 2 else np.nan,
            "ancho_total": sum(anchos) if anchos else np.nan,
        })

    df = pd.DataFrame(filas)
    usable = df.dropna(subset=["ancho_total"])
    descartadas = df[df["ancho_total"].isna()]

    print(f"Candidatas evaluadas: {len(df)} (top {N_CANDIDATAS} NC por num. de puntos)")
    print(f"Descartadas por profundidad < {PROFUNDIDAD_MINIMA} mag: {len(descartadas)}")
    print(f"Usables: {len(usable)}\n")
    print(usable[["id", "n_puntos", "periodo", "profundidad_mag", "n_eclipses_detectados",
                   "ancho_mayor", "ancho_menor", "ancho_total"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nAncho de eclipse individual (mayor de los detectados por estrella), en fase:")
    print(f"  mediana={usable['ancho_mayor'].median():.4f}  media={usable['ancho_mayor'].mean():.4f}  "
          f"min={usable['ancho_mayor'].min():.4f}  max={usable['ancho_mayor'].max():.4f}")

    con_dos = usable.dropna(subset=["ancho_menor"])
    print(f"\nDe {len(con_dos)} estrellas con 2 eclipses detectados, ancho del eclipse secundario (menor):")
    print(f"  mediana={con_dos['ancho_menor'].median():.4f}  media={con_dos['ancho_menor'].mean():.4f}")

    print(f"\nFraccion TOTAL de la fase ocupada por eclipses (suma de todos los detectados por estrella):")
    print(f"  mediana={usable['ancho_total'].median():.4f}  media={usable['ancho_total'].mean():.4f}")


if __name__ == "__main__":
    main()
