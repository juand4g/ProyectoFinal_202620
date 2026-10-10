"""
Regla hibrida con candidatos n = 1..10 (en vez de 1..6), a pedido del
10/10/2026. Solo calcula el BIC de n = 7..10 sobre el P0 ya guardado
(rank_product condicional): no recalcula ninguna busqueda global.

Muestras: 3000 binarias OGLE (todos los H0 del barrido), 3000 pulsantes
OGLE (para el metodo unificado: H0 = 4), 5760 curvas sinteticas y 720
estrellas de TESS (H0 = 4). Cache append-safe por muestra en
Refinamiento_Subarmonicos/cache_bic_n7_10_<muestra>.csv.
Resumen en Refinamiento_Subarmonicos/resumen_bic_hasta_10.txt.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from decision_binaria_pulsante import CACHE_CSV as CACHE_DECISION
from etapa1_cefeidas_rrlyrae import TIPOS
from refinamiento_rank_product_bic import N_PROCESOS
from refinamiento_rank_product_enmascarado_bic import CACHE_CSV as CACHE_HIB
from refinamiento_subarmonicos_bic import (
    H0_VALORES, MUESTRA_CSV, N_MUESTRA_BASE, OUT_DIR, PHOT_DIR, acierto, bic_fourier,
)
from robustez_sinteticas import (
    CACHE_CSV as CACHE_SINT, CACHE_SKEW_OGLE, UMBRAL_SKEW, curva_sintetica, evaluar,
)

N_NUEVOS = range(7, 11)
H0 = 4
RESUMEN_TXT = OUT_DIR / "resumen_bic_hasta_10.txt"
TESS_DIR = Path(__file__).resolve().parent.parent / "TESS_Prueba"

warnings.filterwarnings("ignore")


def bics(t, mag, err, p0, h0_valores, prefijo):
    base = t.max() - t.min()
    return {f"{prefijo}_h{h}_n{n}": (bic_fourier(t, mag, err, 1.0 / (n * p0), n * h) if n * p0 <= base else np.nan)
            for h in h0_valores for n in N_NUEVOS}


def leer(ruta):
    t, mag, err = np.loadtxt(ruta, unpack=True)
    return t, mag, np.where(err > 0, err, np.median(err[err > 0]))


def tarea_archivo(args):
    clave, ruta, p0, h0s = args
    warnings.filterwarnings("ignore")
    return {"id": clave} | bics(*leer(ruta), p0, h0s, "bic_rp_c")


def tarea_sintetica(spec):
    warnings.filterwarnings("ignore")
    return {"id": spec["id"]} | bics(*curva_sintetica(spec), spec["periodo_rp_c"], [H0], "bic_rp_c")


def con_cache(nombre, tareas, funcion):
    ruta = OUT_DIR / f"cache_bic_n7_10_{nombre}.csv"
    cache = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["id"])
    pendientes = [x for x in tareas if (x[0] if isinstance(x, tuple) else x["id"]) not in set(cache["id"])]
    if pendientes:
        with ProcessPoolExecutor(N_PROCESOS) as pool:
            filas = list(pool.map(funcion, pendientes, chunksize=20))
        cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
        cache.to_csv(ruta, index=False)
        print(f"{nombre}: {len(filas)} estrellas nuevas", flush=True)
    return cache


def n_hibrido_hasta(df, h0, n_max, prefijo="bic_rp_c"):
    """Regla de la Ec. regla-hibrida con candidatos 3..n_max."""
    cols = [f"{prefijo}_h{h0}_n{n}" for n in range(1, n_max + 1)]
    b = np.where(df[cols].isna(), np.inf, df[cols].to_numpy(dtype=float))
    n_alto = np.argmin(b[:, 2:], axis=1) + 3
    b_alto = b[np.arange(len(df)), n_alto - 1]
    n = np.where(b_alto < b[:, 1], n_alto, 2)
    return np.where(np.isinf(b[:, 1]), 1, n)


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    p = 1.0 if n01 + n10 == 0 else binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return f"pierde {n10}, gana {n01}, p={p:.2g}"


def main():
    w = []
    # ------------------------------------------------------- binarias OGLE
    ids = pd.read_csv(MUESTRA_CSV)["id"].tolist()
    hib = pd.read_csv(CACHE_HIB)
    hib = hib.set_index("id").loc[ids].reset_index()
    nuevos = con_cache("binarias", [(i, str(PHOT_DIR / f"{i}.dat"), p, H0_VALORES)
                                    for i, p in zip(hib["id"], hib["periodo_rp_c"])], tarea_archivo)
    d = hib.merge(nuevos, on="id")
    P, p0 = d["periodo_catalogo"].to_numpy(), d["periodo_rp_c"].to_numpy()
    orig = d["id"].isin(ids[:N_MUESTRA_BASE]).to_numpy()
    w.append("REGLA HIBRIDA CON CANDIDATOS HASTA n=10 vs HASTA n=6 (P0 = rank_product condicional)")
    w.append("=" * 78)
    w.append(f"1. Binarias OGLE ({len(d)}; originales {orig.sum()}, agregadas {(~orig).sum()})")
    w.append(f"  {'H0':>4}{'n<=6':>9}{'n<=10':>9}   (originales n<=6 / n<=10 | agregadas n<=6 / n<=10)")
    ok6 = ok10 = None
    for h in H0_VALORES:
        a = acierto(n_hibrido_hasta(d, h, 6) * p0, P)
        b = acierto(n_hibrido_hasta(d, h, 10) * p0, P)
        if h == H0:
            ok6, ok10, n6, n10 = a, b, n_hibrido_hasta(d, h, 6), n_hibrido_hasta(d, h, 10)
        w.append(f"  {h:4d}{a.mean():9.1%}{b.mean():9.1%}   ({a[orig].mean():.1%} / {b[orig].mean():.1%} | "
                 f"{a[~orig].mean():.1%} / {b[~orig].mean():.1%})")
    w.append(f"  H0=4, n<=10 frente a n<=6: {mcnemar(ok6, ok10)}")
    w.append("  Distribucion de n elegido (H0=4): n<=6 " + str(pd.Series(n6).value_counts().sort_index().to_dict())
             + "; n<=10 " + str(pd.Series(n10).value_counts().sort_index().to_dict()))
    nreal = np.zeros(len(d), int)
    for n in range(10, 0, -1):
        nreal[acierto(n * p0, P)] = n
    cambia = n6 != n10
    w.append(f"  Estrellas cuyo n cambia: {cambia.sum()}; de ellas n* en 7..10: {np.sum(cambia & (nreal >= 7))}, "
             f"n* en 1..6: {np.sum(cambia & (nreal >= 1) & (nreal <= 6))}, sin multiplo: {np.sum(cambia & (nreal == 0))}")
    w.append(f"  Estrellas con n* en 7..10: {np.sum(nreal >= 7)}; aciertos con n<=10: {np.sum(ok10 & (nreal >= 7))}")
    w.append("  Por periodo catalogado (n<=6 / n<=10, N):")
    rangos = pd.cut(P, [0, 1, 3, 10, 30, 100, 1e4])
    for r in rangos.categories:
        s = (rangos == r)
        w.append(f"    {str(r):16s} {ok6[s].mean():6.1%} / {ok10[s].mean():6.1%}   N={s.sum()}")

    # ------------------------------------------------------- pulsantes y unificado OGLE
    dec = pd.read_csv(CACHE_DECISION).merge(pd.read_csv(CACHE_SKEW_OGLE)[["id", "skew_corregida"]], on="id")
    carpetas = {"Binaria": PHOT_DIR} | {c: carpeta for c, (carpeta, _) in TIPOS.items()}
    pul = dec[dec["clase"] != "Binaria"]
    nuevos_p = con_cache("pulsantes", [(i, str(carpetas[c] / f"{i}.dat"), p, [H0])
                                       for i, c, p in zip(pul["id"], pul["clase"], pul["p0"])], tarea_archivo)
    # BIC n=1..6 de las pulsantes: se recalculan aqui (decision_binaria_pulsante solo guardo n_hib)
    base_p = con_cache("pulsantes_n1_6", [(i, str(carpetas[c] / f"{i}.dat"), p, [H0])
                                          for i, c, p in zip(pul["id"], pul["clase"], pul["p0"])], tarea_n1_6)
    todas = pd.concat([d[["id"] + [f"bic_rp_c_h{H0}_n{n}" for n in range(1, 11)]],
                       nuevos_p.merge(base_p, on="id")], ignore_index=True)
    u = dec.merge(todas, on="id")
    Pu, p0u = u["periodo_catalogo"].to_numpy(), u["p0"].to_numpy()
    ruta = u["skew_corregida"].fillna(-np.inf).to_numpy() > UMBRAL_SKEW
    w.append("")
    w.append(f"2. Metodo unificado OGLE ({len(u)} curvas; skew corregida > {UMBRAL_SKEW})")
    for nmax in [6, 10]:
        ok = np.where(ruta, acierto(n_hibrido_hasta(u, H0, nmax) * p0u, Pu), acierto(p0u, Pu))
        w.append(f"  n<={nmax:2d}: " + "   ".join(f"{c} {ok[(u['clase'] == c).to_numpy()].mean():6.1%}"
                                               for c in ["Binaria", "Cefeida", "RR Lyrae"]))

    # ------------------------------------------------------- sinteticas
    sint = pd.read_csv(CACHE_SINT)
    nuevos_s = con_cache("sinteticas", sint.to_dict("records"), tarea_sintetica)
    df, decisiones, _, ok_p0 = evaluar(sint)
    df = df.merge(nuevos_s, on="id", how="left")
    Ps, p0s = df["periodo_catalogo"].to_numpy(), df["p0"].to_numpy()
    es_bin = (df["clase"] == "Binaria").to_numpy()
    w.append("")
    w.append(f"3. Sinteticas ({len(df)} curvas), binarias, decision perfecta / skew corregida, por configuracion")
    w.append(f"  {'SNR':>4} {'N':>6}   perfecta n<=6  n<=10    skew corr. n<=6  n<=10")
    ok_s = {}
    for nmax in [6, 10]:
        oks = acierto(n_hibrido_hasta(df, H0, nmax) * p0s, Ps)
        ok_s[nmax] = (oks, np.where(decisiones["skew corr.>0.5"], oks, ok_p0))
    for (snr, nobj), g in df.groupby(["snr", "n_objetivo"], sort=False):
        i = g.index[es_bin[g.index]]
        w.append(f"  {snr:4d} {('todos' if nobj == 0 else nobj):>6}   {ok_s[6][0][i].mean():12.1%}{ok_s[10][0][i].mean():7.1%}"
                 f"    {ok_s[6][1][i].mean():14.1%}{ok_s[10][1][i].mean():7.1%}")
    w.append(f"  Promedio: perfecta {ok_s[6][0][es_bin].mean():.1%} -> {ok_s[10][0][es_bin].mean():.1%}; "
             f"skew corregida {ok_s[6][1][es_bin].mean():.1%} -> {ok_s[10][1][es_bin].mean():.1%}; "
             f"pulsantes (skew corr.) Cef {ok_s[6][1][(df['clase'] == 'Cefeida').to_numpy()].mean():.1%} -> "
             f"{ok_s[10][1][(df['clase'] == 'Cefeida').to_numpy()].mean():.1%}, RRL "
             f"{ok_s[6][1][(df['clase'] == 'RR Lyrae').to_numpy()].mean():.1%} -> "
             f"{ok_s[10][1][(df['clase'] == 'RR Lyrae').to_numpy()].mean():.1%}")

    # ------------------------------------------------------- TESS
    from prueba_tess import CACHE_CSV as CACHE_TESS, MUESTRA_CSV as MUESTRA_TESS, archivo
    tess = pd.read_csv(CACHE_TESS)
    tess = tess[tess["nombre"].isin(set(pd.read_csv(MUESTRA_TESS)["nombre"]))].reset_index(drop=True)
    nuevos_t = con_cache("tess", [(n, str(archivo(n)), p, [H0]) for n, p in zip(tess["nombre"], tess["periodo_rp_c"])],
                         tarea_archivo)
    tess = tess.merge(nuevos_t.rename(columns={"id": "nombre"}), on="nombre")
    ref, p0t = tess["periodo_referencia"].to_numpy(), tess["periodo_rp_c"].to_numpy()
    rt = tess["skew_corregida"].fillna(-np.inf).to_numpy() > UMBRAL_SKEW
    bt = (tess["clase"] == "Binaria").to_numpy()
    w.append("")
    w.append(f"4. TESS ({len(tess)}), acierto < 1%: rama perfecta / metodo completo")
    for nmax in [6, 10]:
        ph = n_hibrido_hasta(tess, H0, nmax) * p0t
        perf = acierto(np.where(bt, ph, p0t), ref)
        fin = acierto(np.where(rt, ph, p0t), ref)
        w.append(f"  n<={nmax:2d}: " + "   ".join(
            f"{g} {perf[(tess['grupo'] == g).to_numpy()].mean():.1%}/{fin[(tess['grupo'] == g).to_numpy()].mean():.1%}"
            for g in tess["grupo"].unique()))

    texto = "\n".join(w)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def tarea_n1_6(args):
    clave, ruta, p0, h0s = args
    warnings.filterwarnings("ignore")
    t, mag, err = leer(ruta)
    base = t.max() - t.min()
    return {"id": clave} | {f"bic_rp_c_h{H0}_n{n}": (bic_fourier(t, mag, err, 1.0 / (n * p0), n * H0)
                                                     if n * p0 <= base else np.nan) for n in range(1, 7)}


if __name__ == "__main__":
    main()
