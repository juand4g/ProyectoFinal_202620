"""
Barrido del numero de particiones en FASE (L) de entropypf para binarias,
con K=7 fijo (eje de magnitud, sin tocar), para ver si un L mas fino que el
7 por defecto -- acorde al ancho tipico de eclipse medido en
estimar_ancho_eclipse_fase.py (mediana ~0.06-0.08 en fase) -- mejora la tasa
de aciertos de minima entropia en binarias.

Reaprovecha la muestra oficial de 500 binarias de caracterizacion_entropia.py
(mismos IDs, mismo periodo catalogado, mismo enmascaramiento de alias diurno)
y su resultado ya cacheado para L=7 (no se recalcula: ese es el valor oficial
de la tesis). Solo se corre el metodo para L en L_VALORES, reutilizando
directamente calcular_periodo_entropia_binaria del modulo oficial (se le
cambia el L y K globales antes de cada llamada) para no duplicar la logica
de enmascaramiento de alias diurno.

Cachea los resultados de cada L por separado, de forma append-safe: si se
interrumpe, retoma donde iba.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import caracterizacion_entropia as ce

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_OFICIAL = BASE_DIR / "Entropia" / "cache_resultados_entropia.csv"
CACHE_SWEEP = BASE_DIR / "Entropia" / "cache_sweep_L_binarias.csv"

K_FIJO = 7  # eje de magnitud: no se toca en todo el barrido
L_VALORES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
UMBRAL = 0.01

COLUMNAS = ["L", "K", "id", "periodo_catalogo", "periodo_entropia",
            "error_relativo", "periodo_entropia_alias", "error_relativo_alias"]


def main():
    oficial = pd.read_csv(CACHE_OFICIAL)
    binarias_oficial = oficial[oficial["tipo"] == "Binaria"].dropna(subset=["periodo_catalogo"])
    ids = binarias_oficial["id"].tolist()
    periodos_cat = dict(zip(binarias_oficial["id"], binarias_oficial["periodo_catalogo"]))

    if CACHE_SWEEP.exists():
        cache = pd.read_csv(CACHE_SWEEP)
    else:
        cache = pd.DataFrame(columns=COLUMNAS)
    procesados = set(zip(cache["L"], cache["id"]))

    for L in L_VALORES:
        ce.L, ce.K = L, K_FIJO  # calcular_periodo_entropia_binaria usa estos globales
        filas_nuevas = []
        for id_estrella in tqdm(ids, desc=f"L={L}", unit="estrella"):
            if (L, id_estrella) in procesados:
                continue
            try:
                t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
            except (ValueError, OSError):
                continue
            if t.size < 10:
                continue

            periodo_catalogo = periodos_cat[id_estrella]
            periodo_entropia, _ = ce.calcular_periodo_entropia_binaria(
                t, mag, ce.P_MIN_BINARIAS, ce.P_MAX_BINARIAS
            )
            error_relativo = abs(periodo_entropia - periodo_catalogo) / periodo_catalogo
            periodo_alias, error_alias = ce.mejor_alias(periodo_entropia, periodo_catalogo)

            filas_nuevas.append({
                "L": L, "K": K_FIJO, "id": id_estrella,
                "periodo_catalogo": periodo_catalogo,
                "periodo_entropia": periodo_entropia,
                "error_relativo": error_relativo,
                "periodo_entropia_alias": periodo_alias,
                "error_relativo_alias": error_alias,
            })

        if filas_nuevas:
            cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
            cache.to_csv(CACHE_SWEEP, index=False)  # progreso guardado tras cada L

    print(f"\nResumen de aciertos por L (K={K_FIJO} fijo en todo el barrido; umbral {UMBRAL:.0%}).")
    print(f"L=7 es el valor oficial de la tesis (cache_resultados_entropia.csv, no recalculado).\n")

    filas_resumen = []
    l7 = binarias_oficial.dropna(subset=["error_relativo"])
    filas_resumen.append({
        "L": 7, "K": 7, "n": len(l7),
        "acierto_directo_pct": 100 * (l7["error_relativo"] < UMBRAL).mean(),
        "acierto_alias_pct": 100 * (l7["error_relativo_alias"] < UMBRAL).mean(),
        "mediana_error": l7["error_relativo"].median(),
    })
    for L in L_VALORES:
        sub = cache[cache["L"] == L].dropna(subset=["error_relativo"])
        filas_resumen.append({
            "L": L, "K": K_FIJO, "n": len(sub),
            "acierto_directo_pct": 100 * (sub["error_relativo"] < UMBRAL).mean(),
            "acierto_alias_pct": 100 * (sub["error_relativo_alias"] < UMBRAL).mean(),
            "mediana_error": sub["error_relativo"].median(),
        })

    resumen = pd.DataFrame(filas_resumen).sort_values("L")
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
