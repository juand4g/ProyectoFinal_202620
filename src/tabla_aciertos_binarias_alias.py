"""
Tabla adicional (aparte de las tablas principales de caracterizacion) que,
solo para las binarias eclipsantes, descompone el porcentaje de aciertos de
forma ACUMULATIVA a medida que se aceptan ordenes de alias cada vez mas
altos: 1x unicamente, luego 1x-2x, luego 1x-2x-3x, y asi hasta 1x-6x (el
mismo rango de multiplos ahora graficado en los diagramas de frecuencia
recuperada vs. catalogada). Se calcula para Lomb-Scargle y para minima
entropia.

Reaprovecha integramente los caches ya generados por
caracterizacion_lomb_scargle.py y caracterizacion_entropia.py (periodo bruto,
sin corregir, y periodo catalogado): no se vuelve a buscar ningun periodo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_LS = BASE_DIR / "Lomb-Scargle" / "cache_resultados_ls.csv"
CACHE_ENTROPIA = BASE_DIR / "Entropia" / "cache_resultados_entropia.csv"
OUT_CSV = BASE_DIR / "Combinacion_LS_Entropia" / "tabla_aciertos_binarias_por_alias.csv"

UMBRAL = 0.01  # mismo umbral del 1% usado en las tablas principales de caracterizacion
ORDEN_MAX = 6


def tasa_acumulada_por_alias(periodo_bruto, periodo_catalogo, orden_max=ORDEN_MAX, umbral=UMBRAL):
    """Para cada orden n de 1 a `orden_max`, calcula la fraccion de estrellas
    para las que al menos uno de los candidatos {P*m, P/m : m=1..n} cae
    dentro de `umbral` del periodo catalogado. Es acumulativa (no decreciente
    en n) porque en cada paso solo se agregan candidatos nuevos."""
    p = periodo_bruto.to_numpy(dtype=float)
    pc = periodo_catalogo.to_numpy(dtype=float)
    mejor_error = np.full(p.shape, np.inf)
    tasas = []
    for n in range(1, orden_max + 1):
        candidatos = [p * n] if n == 1 else [p * n, p / n]
        for cand in candidatos:
            mejor_error = np.minimum(mejor_error, np.abs(cand - pc) / pc)
        tasas.append(float((mejor_error < umbral).mean()))
    return tasas


def main():
    cache_ls = pd.read_csv(CACHE_LS)
    cache_entropia = pd.read_csv(CACHE_ENTROPIA)

    bin_ls = cache_ls[cache_ls["tipo"] == "Binaria"].dropna(subset=["periodo_ls", "periodo_catalogo"])
    bin_entropia = cache_entropia[cache_entropia["tipo"] == "Binaria"].dropna(
        subset=["periodo_entropia", "periodo_catalogo"]
    )

    tasas_ls = tasa_acumulada_por_alias(bin_ls["periodo_ls"], bin_ls["periodo_catalogo"])
    tasas_entropia = tasa_acumulada_por_alias(bin_entropia["periodo_entropia"], bin_entropia["periodo_catalogo"])

    ordenes = list(range(1, ORDEN_MAX + 1))
    etiquetas = ["1x" if n == 1 else "1x" + "".join(f"-{m}x" for m in range(2, n + 1)) for n in ordenes]

    tabla = pd.DataFrame({
        "orden_alias": etiquetas,
        "aciertos_ls_pct": [100 * t for t in tasas_ls],
        "aciertos_entropia_pct": [100 * t for t in tasas_entropia],
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUT_CSV, index=False)

    print(f"\nAciertos acumulados por orden de alias, binarias "
          f"(n={len(bin_ls)} Lomb-Scargle, n={len(bin_entropia)} minima entropia, umbral {UMBRAL:.0%}):")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nGuardado en {OUT_CSV}")


if __name__ == "__main__":
    main()
