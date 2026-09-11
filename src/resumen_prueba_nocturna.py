"""
Genera un resumen en texto plano (.txt) de los resultados de la prueba
nocturna de busqueda de periodo sobre binarias eclipsantes
(ver prueba_nocturna_binarias.py y Prueba_Nocturna_Binarias/cache_binarias_nocturno.csv).

Incluye, por metodo: mediana/media de error relativo (con y sin correccion
de alias), porcentajes de aciertos bajo distintos umbrales, distribucion de
multiplicadores de alias necesarios, y una comparacion directa de
"producto" y "rank_product" contra Lomb-Scargle solo (mejora / empata / empeora).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "Prueba_Nocturna_Binarias"
CACHE_CSV = OUT_DIR / "cache_binarias_nocturno.csv"
REPORTE_TXT = OUT_DIR / "resumen_prueba_nocturna.txt"

METODOS = ["Lomb-Scargle", "1 - entropia", "producto", "rank_product"]
UMBRALES = [0.01, 0.05, 0.10, 0.20]


def formato_pct(x):
    return f"{100 * x:.2f}%"


def main():
    cache = pd.read_csv(CACHE_CSV)

    n_estrellas = cache["id"].nunique()
    n_filas = len(cache)
    metadatos = cache.drop_duplicates("id")

    lineas = []
    lineas.append("RESUMEN - PRUEBA NOCTURNA DE BUSQUEDA DE PERIODO EN BINARIAS ECLIPSANTES")
    lineas.append("=" * 78)
    lineas.append(f"Fuente: {CACHE_CSV}")
    lineas.append("")
    lineas.append(f"Numero de binarias analizadas: {n_estrellas}")
    lineas.append(f"Numero de filas en el cache (estrellas x metodos): {n_filas}")
    lineas.append(f"Numero de puntos fotometricos por curva -- mediana: {metadatos['n_puntos'].median():.0f}, "
                  f"minimo: {metadatos['n_puntos'].min():.0f}, maximo: {metadatos['n_puntos'].max():.0f}")
    lineas.append(f"Baseline temporal (dias) -- mediana: {metadatos['baseline_d'].median():.1f}, "
                  f"minimo: {metadatos['baseline_d'].min():.1f}, maximo: {metadatos['baseline_d'].max():.1f}")
    lineas.append("")

    lineas.append("-" * 78)
    lineas.append("ERROR RELATIVO POR METODO (respecto al periodo catalogado)")
    lineas.append("-" * 78)
    for con_alias, columna, etiqueta in [
        (False, "error_relativo", "SIN correccion de alias"),
        (True, "error_relativo_alias", "CON correccion de alias"),
    ]:
        lineas.append("")
        lineas.append(f"* {etiqueta}")
        for metodo in METODOS:
            datos = cache.loc[cache["metodo"] == metodo, columna]
            lineas.append(f"  - {metodo:<14s} mediana={datos.median():.5f}  media={datos.mean():.5f}  "
                          f"p25={datos.quantile(0.25):.5f}  p75={datos.quantile(0.75):.5f}")
            umbral_str = "   ".join(
                f"<{formato_pct(u)}: {formato_pct((datos < u).mean())}" for u in UMBRALES
            )
            lineas.append(f"    {' ' * 14} {umbral_str}")

    lineas.append("")
    lineas.append("-" * 78)
    lineas.append("DISTRIBUCION DEL MULTIPLICADOR DE ALIAS NECESARIO, POR METODO")
    lineas.append("-" * 78)
    lineas.append("(fraccion de estrellas cuyo periodo encontrado hubo que multiplicar por")
    lineas.append(" el factor indicado para igualar el periodo catalogado)")
    multiplicadores = sorted(cache["multiplicador_alias"].unique())
    for metodo in METODOS:
        lineas.append("")
        lineas.append(f"* {metodo}")
        grupo = cache.loc[cache["metodo"] == metodo, "multiplicador_alias"]
        for m in multiplicadores:
            frac = np.isclose(grupo, m).mean()
            lineas.append(f"    x{m:<10.4f} {formato_pct(frac)}")

    lineas.append("")
    lineas.append("-" * 78)
    lineas.append("COMPARACION DIRECTA CONTRA LOMB-SCARGLE SOLO (por estrella)")
    lineas.append("-" * 78)
    lineas.append("(usando error_relativo_alias; 'mejora' = error estrictamente menor que LS,")
    lineas.append(" 'empeora' = error estrictamente mayor, 'empata' = error igual)")
    tabla_ls = cache[cache["metodo"] == "Lomb-Scargle"].set_index("id")["error_relativo_alias"]
    for metodo in ["1 - entropia", "producto", "rank_product"]:
        tabla_m = cache[cache["metodo"] == metodo].set_index("id")["error_relativo_alias"]
        comunes = tabla_ls.index.intersection(tabla_m.index)
        mejora = (tabla_m.loc[comunes] < tabla_ls.loc[comunes]).mean()
        empeora = (tabla_m.loc[comunes] > tabla_ls.loc[comunes]).mean()
        empata = 1.0 - mejora - empeora
        lineas.append("")
        lineas.append(f"* {metodo} vs. Lomb-Scargle  (n={len(comunes)} binarias)")
        lineas.append(f"    mejora:  {formato_pct(mejora)}")
        lineas.append(f"    empeora: {formato_pct(empeora)}")
        lineas.append(f"    empata:  {formato_pct(empata)}")

    lineas.append("")
    lineas.append("-" * 78)
    lineas.append("RANKING GENERAL (mediana de error relativo con alias, menor a mayor)")
    lineas.append("-" * 78)
    ranking = cache.groupby("metodo")["error_relativo_alias"].median().sort_values()
    for i, (metodo, valor) in enumerate(ranking.items(), start=1):
        lineas.append(f"  {i}. {metodo:<14s} mediana = {valor:.5f}")

    REPORTE_TXT.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Reporte escrito en: {REPORTE_TXT}")


if __name__ == "__main__":
    main()
