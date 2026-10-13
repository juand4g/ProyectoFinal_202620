"""
Comparacion con los metodos individuales en las pruebas del Capitulo 7
(objetivo especifico 4): Lomb-Scargle, minima entropia (convencional con la
mascara oficial y condicional) y producto de rangos, sobre las curvas
sinteticas y TESS. Para las binarias se aplica a cada busqueda la misma regla
hibrida (x2 + BIC, H0 = 4), con decision de rama perfecta, como en la
Tabla de aciertos del Capitulo 5. Los periodos de cada metodo ya estan en las
caches; aqui solo se calculan los BIC de n = 1..6 sobre el P0 de LS y de la
entropia condicional. Cache en Robustez_Sinteticas/cache_bic_individuales_*.csv,
resumen en Robustez_Sinteticas/resumen_metodos_individuales.txt.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from bic_multiplos_hasta_10 import leer, n_hibrido_hasta
from prueba_tess import CACHE_CSV as CACHE_TESS, MUESTRA_CSV as MUESTRA_TESS, archivo
from refinamiento_rank_product_bic import N_PROCESOS
from refinamiento_subarmonicos_bic import acierto, bic_fourier
from robustez_sinteticas import CACHE_CSV as CACHE_SINT, OUT_DIR, curva_sintetica

H0 = 4
BASES = ["ls", "ent_c"]
METODOS = {"ls": "Lomb-Scargle", "ent": "Entropia (mascara oficial)", "ent_c": "Entropia condicional",
           "rp_c": "Producto de rangos"}
RESUMEN_TXT = OUT_DIR / "resumen_metodos_individuales.txt"

warnings.filterwarnings("ignore")


def bics(t, mag, err, periodos):
    base = t.max() - t.min()
    return {f"bic_{m}_h{H0}_n{n}": (bic_fourier(t, mag, err, 1.0 / (n * p), n * H0) if n * p <= base else np.nan)
            for m, p in periodos.items() for n in range(1, 7)}


def tarea_sint(spec):
    warnings.filterwarnings("ignore")
    return {"id": spec["id"]} | bics(*curva_sintetica(spec), {m: spec[f"periodo_{m}"] for m in BASES})


def tarea_tess(fila):
    warnings.filterwarnings("ignore")
    return {"id": fila["nombre"]} | bics(*leer(archivo(fila["nombre"])), {m: fila[f"periodo_{m}"] for m in BASES})


def con_cache(nombre, tareas, funcion, clave):
    ruta = OUT_DIR / f"cache_bic_individuales_{nombre}.csv"
    cache = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["id"])
    pendientes = [x for x in tareas if x[clave] not in set(cache["id"])]
    if pendientes:
        with ProcessPoolExecutor(N_PROCESOS) as pool:
            filas = list(pool.map(funcion, pendientes, chunksize=20))
        cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
        cache.to_csv(ruta, index=False)
    return cache


def aciertos(d, P):
    """{metodo: (directo, siempre x2, regla hibrida o None)}."""
    out = {}
    for m in METODOS:
        p = d[f"periodo_{m}"].to_numpy()
        hib = None
        if f"bic_{m}_h{H0}_n2" in d:
            hib = acierto(n_hibrido_hasta(d, H0, 6, f"bic_{m}") * p, P)
        out[m] = (acierto(p, P), acierto(2 * p, P), hib)
    return out


def mcnemar(a, b):
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    p = 1.0 if n01 + n10 == 0 else binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return f"solo {n10} vs solo {n01}, p={p:.2g}"


def bloque(w, titulo, d, P, grupos):
    ok = aciertos(d, P)
    w.append(titulo)
    w.append(f"  {'Grupo':22s}{'n':>6}  " + "".join(f"{METODOS[m][:18]:>34s}" for m in METODOS))
    w.append(f"  {'':28s}" + "".join(f"{'dir / x2 / hib':>34s}" for _ in METODOS))
    for g, s in grupos:
        celdas = []
        for m in METODOS:
            a, b, h = ok[m]
            celdas.append(f"{a[s].mean():6.1%} / {b[s].mean():6.1%} / " + (f"{h[s].mean():6.1%}" if h is not None else "   -  "))
        w.append(f"  {g:22s}{s.sum():6d}  " + "".join(f"{c:>34s}" for c in celdas))
    return ok


def main():
    w = ["COMPARACION CON LOS METODOS INDIVIDUALES EN SINTETICAS Y TESS (regla hibrida H0=4, decision perfecta)",
         "=" * 78]
    sint = pd.read_csv(CACHE_SINT)
    cache = con_cache("sinteticas", sint.to_dict("records"), tarea_sint, "id")
    s = sint.merge(cache, on="id", how="left")
    P = s["periodo_catalogo"].to_numpy()
    cl = s["clase"].to_numpy()
    ok = bloque(w, "Sinteticas, todas las configuraciones", s, P, [(c, cl == c) for c in ["Binaria", "Cefeida", "RR Lyrae"]])
    for snr, nobj, nom in [(50, 0, "SNR 50, todos los puntos"), (4, 0, "SNR 4, todos los puntos"), (4, 100, "SNR 4, N = 100")]:
        m = ((s["snr"] == snr) & (s["n_objetivo"] == nobj)).to_numpy()
        bloque(w, f"Sinteticas, {nom}", s, P, [(c, (cl == c) & m) for c in ["Binaria", "Cefeida", "RR Lyrae"]])
    bin_ = cl == "Binaria"
    w.append(f"  McNemar binarias, hibrida rp_c vs LS: {mcnemar(ok['ls'][2][bin_], ok['rp_c'][2][bin_])}; "
             f"vs ent_c: {mcnemar(ok['ent_c'][2][bin_], ok['rp_c'][2][bin_])}")
    pul = ~bin_
    w.append(f"  McNemar pulsantes, directo rp_c vs LS: {mcnemar(ok['ls'][0][pul], ok['rp_c'][0][pul])}; "
             f"vs ent_c: {mcnemar(ok['ent_c'][0][pul], ok['rp_c'][0][pul])}")
    w.append("")

    tess = pd.read_csv(CACHE_TESS)
    tess = tess[tess["nombre"].isin(set(pd.read_csv(MUESTRA_TESS)["nombre"]))].reset_index(drop=True)
    cache = con_cache("tess", tess.to_dict("records"), tarea_tess, "nombre")
    t = tess.merge(cache.rename(columns={"id": "nombre"}), on="nombre", how="left")
    R = t["periodo_referencia"].to_numpy()
    g = t["grupo"].to_numpy()
    grupos = [(x, g == x) for x in ["Binaria separada", "Binaria intermedia", "Binaria contacto/ELL", "Cefeida", "RR Lyrae"]]
    grupos.append(("Todas las binarias", (t["clase"] == "Binaria").to_numpy()))
    ok = bloque(w, "TESS (error < 1%)", t, R, grupos)
    b = (t["clase"] == "Binaria").to_numpy()
    w.append(f"  McNemar binarias, hibrida rp_c vs LS: {mcnemar(ok['ls'][2][b], ok['rp_c'][2][b])}; "
             f"vs ent_c: {mcnemar(ok['ent_c'][2][b], ok['rp_c'][2][b])}")
    w.append(f"  McNemar pulsantes, directo rp_c vs LS: {mcnemar(ok['ls'][0][~b], ok['rp_c'][0][~b])}; "
             f"vs ent_c: {mcnemar(ok['ent_c'][0][~b], ok['rp_c'][0][~b])}")
    texto = "\n".join(w)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
