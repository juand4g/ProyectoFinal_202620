"""
Etapa 1 del metodo hibrido (busqueda global: rank_product de Lomb-Scargle y
entropia condicional) aplicada a Cefeidas Clasicas y RR Lyrae de OGLE.

Pregunta: la busqueda global se diseno sobre binarias eclipsantes. En
estrellas pulsantes, donde LS ya funciona bien, ¿rank_product condicional
empeora a Lomb-Scargle solo?

Se usa EXACTAMENTE la misma configuracion que en las binarias (funcion
periodos_base de refinamiento_rank_product_enmascarado_bic.py): grilla
uniforme de 100 000 frecuencias entre 1/100 y 1/0.1 d^-1, LS con errores,
entropia L=10, K=7, sin mascara para la condicional. No se usa ningun rango
por tipo de estrella: es la misma busqueda general. Solo se evalua P0 (sin el
prior x2 ni el refinamiento BIC, que suponen binaria eclipsante); el
"siempre x2" se reporta para cuantificar el dano del prior en pulsantes.

Muestra: N_POR_TIPO estrellas por tipo, semilla 2026. Periodo de referencia:
P_1 de OGLE. Cache append-safe en Etapa1_Pulsantes/.
"""

import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from refinamiento_rank_product_enmascarado_bic import METODOS, periodos_base
from refinamiento_rank_product_bic import N_PROCESOS, P_MAX, P_MIN
from refinamiento_subarmonicos_bic import MIN_PUNTOS, N_MAX, RANDOM_SEED, acierto

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection"
OUT_DIR = BASE_DIR / "Etapa1_Pulsantes"
CACHE_CSV = OUT_DIR / "cache_etapa1_pulsantes.csv"
RESUMEN_TXT = OUT_DIR / "resumen_etapa1_pulsantes.txt"

N_POR_TIPO = 500
TIPOS = {
    "Cefeida": (DATA_DIR / "classical_cefeids_phot", DATA_DIR / "classical_cefeids.txt"),
    "RR Lyrae": (DATA_DIR / "rrlyrae_phot", DATA_DIR / "rrlyrae.txt"),
}
# Solo se reportan en el resumen: LS, entropia condicional y rank_product
# condicional (los tratamientos con mascara oficial ya se descartaron).
REPORTADOS = ["ls", "ent_c", "rp_c"]


def cargar_catalogo(path_txt):
    df = pd.read_csv(path_txt, sep="\t", comment="#").set_index("ID")
    if "Mode" not in df.columns:
        df["Mode"] = "-"
    return df[["Mode", "P_1"]]


def analizar_estrella(tipo, id_estrella, modo, periodo_catalogo, ruta):
    warnings.filterwarnings("ignore")
    try:
        t, mag, err = np.loadtxt(ruta, unpack=True)
    except (ValueError, OSError):
        return None
    if t.size < MIN_PUNTOS:
        return None
    err = np.where(err > 0, err, np.median(err[err > 0]))
    fila = {"tipo": tipo, "id": id_estrella, "modo": modo, "n_puntos": t.size,
            "baseline_d": t.max() - t.min(), "periodo_catalogo": periodo_catalogo}
    for metodo, p0 in periodos_base(t, mag, err).items():
        fila[f"periodo_{metodo}"] = p0
    return fila


def muestra_tipo(tipo):
    carpeta, catalogo_txt = TIPOS[tipo]
    catalogo = cargar_catalogo(catalogo_txt)
    disponibles = sorted(p.stem for p in carpeta.glob("*.dat"))
    disponibles = [i for i in disponibles if i in catalogo.index]
    random.seed(RANDOM_SEED)
    ids = random.sample(disponibles, N_POR_TIPO)
    return [(tipo, i, str(catalogo.at[i, "Mode"]), float(catalogo.at[i, "P_1"]), carpeta / f"{i}.dat")
            for i in ids]


def resumen(cache):
    lineas = []
    w = lineas.append
    w("RESUMEN - ETAPA 1 DEL METODO HIBRIDO EN CEFEIDAS Y RR LYRAE")
    w("=" * 78)
    w(f"Misma busqueda global que en binarias: {P_MIN}-{P_MAX} d, 100 000 frecuencias uniformes, "
      f"entropia L=10, K=7. Semilla {RANDOM_SEED}. Acierto = error relativo < 1%.")
    w("")
    for tipo, g in cache.groupby("tipo", sort=False):
        P = g["periodo_catalogo"].to_numpy()
        w(f"{tipo}: {len(g)} estrellas (P_cat {P.min():.3f}-{P.max():.3f} d; "
          f"{np.mean(P > P_MAX):.1%} con P > {P_MAX:.0f} d, fuera de la grilla)")
        w(f"  {'Metodo':20s}{'directo':>9}{'x1/2':>8}{'x2':>8}{'x1/3':>8}{'siempre x2':>12}{'oraculo x1-x6':>15}")
        ok = {}
        for m in REPORTADOS:
            p0 = g[f"periodo_{m}"].to_numpy()
            ok[m] = acierto(p0, P)
            oraculo = np.zeros(len(g), dtype=bool)
            for n in range(1, N_MAX + 1):
                oraculo |= acierto(n * p0, P)
            w(f"  {METODOS[m]:20s}{ok[m].mean():9.1%}{acierto(p0, P / 2).mean():8.1%}"
              f"{acierto(p0, 2 * P).mean():8.1%}{acierto(p0, P / 3).mean():8.1%}"
              f"{acierto(2 * p0, P).mean():12.1%}{oraculo.mean():15.1%}")
        a, b = ok["ls"], ok["rp_c"]
        n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
        w(f"  LS vs rank_prod condic.: ambos {np.mean(a & b):.1%}, solo LS {n10}, "
          f"solo rank_prod {n01}, ninguno {np.mean(~a & ~b):.1%}; McNemar exacto p = {mcnemar(n01, n10):.2g}")
        if g["modo"].nunique() > 1:
            w("  Por modo de pulsacion (directo):")
            for modo, gm in g.groupby("modo"):
                if len(gm) < 5:
                    continue
                Pm = gm["periodo_catalogo"].to_numpy()
                w(f"    {modo:4s} n={len(gm):4d}   " + "   ".join(
                    f"{m} {acierto(gm[f'periodo_{m}'].to_numpy(), Pm).mean():6.1%}" for m in REPORTADOS))
        cortes = [0, 1, 3, 10, 30, 100, 1e4] if tipo == "Cefeida" else [0, 0.35, 0.45, 0.55, 0.65, 2]
        w("  Por periodo catalogado (directo):")
        df = pd.DataFrame({m: ok[m] for m in REPORTADOS} | {"rango": pd.cut(P, cortes)})
        for rango, gr in df.groupby("rango", observed=True):
            w(f"    {str(rango):14s} n={len(gr):4d}   " + "   ".join(f"{m} {gr[m].mean():6.1%}" for m in REPORTADOS))
        w("")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def mcnemar(n01, n10):
    from scipy.stats import binomtest
    n = n01 + n10
    return 1.0 if n == 0 else binomtest(min(n01, n10), n, 0.5).pvalue


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tareas = [x for tipo in TIPOS for x in muestra_tipo(tipo)]
    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame()
    procesados = set(cache["id"]) if not cache.empty else set()
    pendientes = [x for x in tareas if x[1] not in procesados]

    filas_nuevas = []
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
        futuros = [pool.submit(analizar_estrella, *x) for x in pendientes]
        for k, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros),
                                        desc="Etapa 1 en pulsantes", unit="estrella")):
            fila = futuro.result()
            if fila is not None:
                filas_nuevas.append(fila)
            if filas_nuevas and (k + 1) % 50 == 0:
                cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
                cache.to_csv(CACHE_CSV, index=False)
                filas_nuevas = []
    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    ids = {x[1] for x in tareas}
    resumen(cache[cache["id"].isin(ids)].reset_index(drop=True))


if __name__ == "__main__":
    main()
