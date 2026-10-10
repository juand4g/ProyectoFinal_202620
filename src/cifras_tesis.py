"""
Calcula, a partir de las caches, todas las cifras que citan los Capitulos 4
a 7 de la tesis (muestras ampliadas el 09/10/2026), para que el texto se
actualice de forma trazable. No recalcula ningun periodo.

Salida: Cifras_Tesis/cifras_tesis.txt (OGLE, sinteticas y TESS).

Ademas de las cifras de la muestra completa, para los Capitulos 5 y 6 se
reportan por separado la muestra original y las estrellas agregadas: H0 y el
umbral de la asimetria se fijaron sobre la original, de modo que las
agregadas son una prueba fuera de muestra de esas dos elecciones.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from caracterizacion_lomb_scargle import CACHE_CSV as CACHE_LS_C4, muestra_caracterizacion
from decision_binaria_pulsante import CACHE_CSV as CACHE_DECISION, aciertos_ruteados
from etapa1_cefeidas_rrlyrae import CACHE_CSV as CACHE_PULSANTES, N_POR_TIPO_BASE
from refinamiento_rank_product_bic import CACHE_CSV as CACHE_EXP9
from refinamiento_rank_product_enmascarado_bic import CACHE_CSV as CACHE_HIB, MASCARA, en_mascara
from refinamiento_subarmonicos_bic import (
    CACHE_CSV as CACHE_EXP8, H0_VALORES, MUESTRA_CSV, N_MAX, N_MUESTRA_BASE, N_TECHO, acierto, n_elegido,
    n_hibrido, n_real_techo,
)
from robustez_sinteticas import CACHE_SKEW_OGLE, UMBRAL_SKEW

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "Cifras_Tesis" / "cifras_tesis.txt"
CACHE_SWEEP = BASE_DIR / "Entropia" / "cache_sweep_L_binarias.csv"
lineas = []
w = lineas.append


def pct(x):
    return f"{100 * np.mean(x):.1f}%"


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"[{100 * (c - h):.1f}%, {100 * (c + h):.1f}%]"


def mcnemar(ok_ref, ok_nuevo):
    b, c = int(np.sum(ok_ref & ~ok_nuevo)), int(np.sum(~ok_ref & ok_nuevo))
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return f"b={b}, c={c}, p={p:.2g}"


def alias_diario(p, P, tol=0.005):
    """|f - f_cat| o f + f_cat cerca de un entero de ciclos por dia (k = 1, 2, 3)."""
    f, fc = 1.0 / np.asarray(p), 1.0 / np.asarray(P)
    k = np.arange(1, 4)
    return (np.abs(np.abs(f - fc)[:, None] - k[None, :]) < tol).any(axis=1) | \
           (np.abs((f + fc)[:, None] - k[None, :]) < tol).any(axis=1)


# ------------------------------------------------------------------ Cap. 4
def capitulo4():
    w("=" * 78)
    w("CAPITULO 4 - CARACTERIZACION INDIVIDUAL (muestra ampliada a 1500 por tipo)")
    muestras = muestra_caracterizacion()
    ls = pd.read_csv(CACHE_LS_C4)
    en = pd.read_csv(BASE_DIR / "Entropia" / "cache_resultados_entropia.csv")
    for nombre, cache, col in [("Lomb-Scargle", ls, "periodo_ls"), ("Minima entropia", en, "periodo_entropia")]:
        w(f"\n{nombre}:")
        for tipo, ids in muestras.items():
            g = cache[(cache["tipo"] == tipo) & cache["id"].isin(ids)].dropna(subset=["error_relativo"])
            orig = g["id"].isin(ids[:500])
            w(f"  {tipo:9s} n={len(g)} (n_total_muestra {len(ids)})  N mediano {g['n_puntos'].median():.0f}  "
              f"linea base mediana {g['baseline_d'].median() / 365.25:.1f} a  "
              f"mediana error {g['error_relativo'].median():.5f}  aciertos {pct(g['error_relativo'] < 0.01)}  "
              f"con alias {pct(g['error_relativo_alias'] < 0.01)}   | originales: aciertos "
              f"{pct(g.loc[orig, 'error_relativo'] < 0.01)}, con alias {pct(g.loc[orig, 'error_relativo_alias'] < 0.01)}")
            razon = g[col] / g["periodo_catalogo"]
            w(f"            P/P_cat ~2: {pct(np.abs(razon - 2) / 2 < 0.01)}   ~1/2: {pct(np.abs(razon - 0.5) / 0.5 < 0.01)}"
              f"   razon P5-P95 (fallos) {np.nanpercentile(razon[g['error_relativo_alias'] >= 0.01], [5, 95]) if (g['error_relativo_alias'] >= 0.01).any() else '-'}")
        b = cache[(cache["tipo"] == "Binaria") & cache["id"].isin(muestras["Binaria"])].dropna(subset=[col])
        p, pc = b[col].to_numpy(), b["periodo_catalogo"].to_numpy()
        mejor = np.full(len(b), np.inf)
        acum = []
        for n in range(1, N_TECHO + 1):
            for cand in ([p] if n == 1 else [p * n, p / n]):
                mejor = np.minimum(mejor, np.abs(cand - pc) / pc)
            acum.append(pct(mejor < 0.01))
        w(f"  Binarias, aciertos acumulados x1..x{N_TECHO}: " + "  ".join(acum))

    if CACHE_SWEEP.exists():
        sw = pd.read_csv(CACHE_SWEEP)
        ids_b = set(muestras["Binaria"])
        b7 = en[(en["tipo"] == "Binaria") & en["id"].isin(ids_b)].dropna(subset=["error_relativo"])
        w("\nBarrido de L (K=7), binarias, aciertos con alias:")
        w(f"  L=7: {pct(b7['error_relativo_alias'] < 0.01)} (n={len(b7)})")
        for L, g in sw[sw["id"].isin(ids_b)].groupby("L"):
            g = g.dropna(subset=["error_relativo"])
            w(f"  L={L}: {pct(g['error_relativo_alias'] < 0.01)} (n={len(g)})")


# ------------------------------------------------------------------ Cap. 5
def capitulo5():
    w("\n" + "=" * 78)
    w("CAPITULO 5 - METODO HIBRIDO (binarias, semilla 2026)")
    ids = pd.read_csv(MUESTRA_CSV)["id"].tolist()
    hib = pd.read_csv(CACHE_HIB)
    hib = hib[hib["id"].isin(ids)].reset_index(drop=True)
    orig = hib["id"].isin(ids[:N_MUESTRA_BASE]).to_numpy()
    P = hib["periodo_catalogo"].to_numpy()
    N = len(hib)
    w(f"Binarias analizadas: {N} (originales {orig.sum()}, agregadas {(~orig).sum()})")
    w(f"Tipos: {hib['tipo_morfologico'].value_counts().to_dict()}; N mediano {hib['n_puntos'].median():.0f}; "
      f"T mediana {hib['baseline_d'].median():.0f} d (1/T = {1 / hib['baseline_d'].median():.2e})")
    w(f"Mascara oficial cubre {MASCARA.mean():.1%} de la grilla; P_cat dentro: {en_mascara(P).mean():.1%}")

    if CACHE_EXP9.exists():
        e9 = pd.read_csv(CACHE_EXP9)
        e9 = e9[e9["id"].isin(set(hib["id"]))]
        pe = e9["periodo_ent"].to_numpy()
        diario = (np.abs(pe - 0.499) < 0.01) | (np.abs(pe - 0.997) < 0.01) | (np.abs(pe - 0.4986) < 0.01)
        w(f"Exp. 9 (entropia L=K=7 sin mascara): en ~0.499 o ~0.997 d: {diario.sum()} de {len(e9)} ({pct(diario)})")

    pe_c = hib["periodo_ent_c"].to_numpy()
    w(f"Entropia condicional: dentro de mascara {en_mascara(pe_c).mean():.1%}; directa {pct(acierto(pe_c, P))}; "
      f"converge a P/2 {pct(acierto(pe_c, P / 2))}")
    p0 = hib["periodo_rp_c"].to_numpy()
    w(f"rp_c: P0 correcto {pct(acierto(p0, P))}; 2P0 correcto {pct(acierto(2 * p0, P))}")

    metodos = {"ls": "Lomb-Scargle", "ent": "Entropia mascara oficial", "ent_c": "Entropia condicional",
               "rp_excl": "Producto de rangos mascara oficial", "rp_neut": "rp neutro",
               "rp_c": "Producto de rangos condicional"}
    ok = {}
    w(f"\nTabla (directo / siempre 2P0 / hibrida H0=4 / techo x1-x{N_MAX} / techo x1-x{N_TECHO}), todas | originales | agregadas")
    for m, nombre in metodos.items():
        pm = hib[f"periodo_{m}"].to_numpy()
        nh = n_hibrido(hib, 4, f"bic_{m}")
        ok[m] = acierto(nh * pm, P)
        nr = n_real_techo(pm, P)
        celdas = lambda s: (f"{pct(acierto(pm, P)[s])} / {pct(acierto(2 * pm, P)[s])} / {pct(ok[m][s])} / "
                            f"{pct(((nr >= 1) & (nr <= N_MAX))[s])} / {pct((nr >= 1)[s])}")
        w(f"  {nombre:36s} {celdas(slice(None))}   | {celdas(orig)}   | {celdas(~orig)}")

    k = int(ok["rp_c"].sum())
    w(f"\nHibrido: {k} de {N} ({pct(ok['rp_c'])}), Wilson {wilson(k, N)}")
    w(f"LS + regla hibrida: {pct(ok['ls'])}, Wilson {wilson(int(ok['ls'].sum()), N)}")
    siempre2 = acierto(2 * p0, P)
    w(f"rp siempre 2P0: {pct(siempre2)}, Wilson {wilson(int(siempre2.sum()), N)}")
    ls2 = acierto(2 * hib['periodo_ls'].to_numpy(), P)
    w(f"LS siempre 2P: {pct(ls2)}")
    w(f"P0 == P_LS (1%): {pct(np.abs(p0 - hib['periodo_ls']) / hib['periodo_ls'] < 0.01)}")
    w(f"McNemar hibrido vs LS+hibrida: {mcnemar(ok['ls'], ok['rp_c'])}")
    w(f"McNemar hibrido vs rp siempre 2P0: {mcnemar(siempre2, ok['rp_c'])}")
    for s, nom in [(orig, "originales"), (~orig, "agregadas")]:
        w(f"  {nom}: hibrido {pct(ok['rp_c'][s])} Wilson {wilson(int(ok['rp_c'][s].sum()), int(s.sum()))}; "
          f"LS+hib {pct(ok['ls'][s])}; McNemar vs LS+hib {mcnemar(ok['ls'][s], ok['rp_c'][s])}")
    nh = n_hibrido(hib, 4, "bic_rp_c")
    w(f"n elegido: {pd.Series(nh).value_counts().sort_index().to_dict()}")
    gana = np.sum(ok["rp_c"] & ~siempre2)
    pierde = np.sum(~ok["rp_c"] & siempre2)
    w(f"BIC vs siempre 2P0: gana {gana}, pierde {pierde}")
    w(f"2P0 > T (n=1 forzado): {np.sum(nh == 1)}")
    w("Barrido H0: " + ", ".join(f"H0={h} {pct(acierto(n_hibrido(hib, h, 'bic_rp_c') * p0, P))}" for h in H0_VALORES)
      + " | originales: " + ", ".join(f"{h} {pct(acierto(n_hibrido(hib, h, 'bic_rp_c') * p0, P)[orig])}" for h in H0_VALORES)
      + " | agregadas: " + ", ".join(f"{h} {pct(acierto(n_hibrido(hib, h, 'bic_rp_c') * p0, P)[~orig])}" for h in H0_VALORES))
    w(f"Penalizacion 2 H0 ln N con N mediano: {2 * 4 * np.log(hib['n_puntos'].median()):.0f}")

    w("\nPor periodo (hibrido / LS+hib / techo x10 rp_c / n):")
    nr = n_real_techo(p0, P)
    rangos = pd.cut(P, [0, 1, 3, 10, 30, 100, 1e4])
    for r in rangos.categories:
        s = (rangos == r)
        w(f"  {str(r):14s} n={s.sum():5d}  hib {pct(ok['rp_c'][s])} ({ok['rp_c'][s].sum()})  LS+hib {pct(ok['ls'][s])} "
          f"({ok['ls'][s].sum()})  techo10 {pct((nr >= 1)[s])}  techo6 {pct(((nr >= 1) & (nr <= N_MAX))[s])}")
    w(f"  P<3 d: {pct(P < 3)} de la muestra; P>100 d: {np.sum(P > 100)}")
    w("Por tipo morfologico:")
    for t in ["NC", "C", "ELL"]:
        s = (hib["tipo_morfologico"] == t).to_numpy()
        w(f"  {t}: n={s.sum()} hib {pct(ok['rp_c'][s])} LS+hib {pct(ok['ls'][s])}")

    falla = ~ok["rp_c"]
    w(f"\nFallos: {falla.sum()}; sin multiplo x1-x{N_TECHO} (n*=0): {np.sum(falla & (nr == 0))} "
      f"({pct(nr == 0)} de la muestra), mediana P_cat {np.median(P[nr == 0]):.1f} d, P>10 d {pct(P[nr == 0] > 10)}")
    w(f"  sin multiplo x1-x{N_MAX}: {np.sum((nr == 0) | (nr > N_MAX))}; con n* en 7..10: {np.sum(nr > N_MAX)}")
    w("  Fallos con n* disponible: " + str(pd.Series(nr[falla & (nr > 0)]).value_counts().sort_index().to_dict()))
    pf = nh * p0
    w(f"  Fallos en P/2: {np.sum(falla & acierto(pf, P / 2))}; en 2P: {np.sum(falla & acierto(pf, 2 * P))}")
    w("  Matriz n* (filas 1..10, 0) x n elegido (1..6):")
    for r in list(range(1, N_TECHO + 1)) + [0]:
        w(f"    n*={r:2d}: " + " ".join(f"{np.sum((nr == r) & (nh == c)):5d}" for c in range(1, N_MAX + 1)))
    techo10 = (nr >= 1).mean()
    w(f"Brecha LS siempre 2P ({pct(ls2)}) -> techo x10 rp_c ({100 * techo10:.1f}%): "
      f"{100 * (techo10 - ls2.mean()):.1f} pp; recupera {100 * (ok['rp_c'].mean() - ls2.mean()):.1f} pp "
      f"({100 * (ok['rp_c'].mean() - ls2.mean()) / (techo10 - ls2.mean()):.0f}%)")

    if CACHE_EXP8.exists():
        e8 = pd.read_csv(CACHE_EXP8)
        e8 = e8[e8["id"].isin(set(hib["id"]))].reset_index(drop=True)
        pls, P8 = e8["periodo_ls"].to_numpy(), e8["periodo_catalogo"].to_numpy()
        nr8 = n_real_techo(pls, P8)
        w(f"\nExp. 8 (LS 0.1-3000 d, BIC libre), n={len(e8)}: siempre 2P {pct(acierto(2 * pls, P8))}")
        for h in H0_VALORES:
            n = n_elegido(e8, h)
            w(f"  H0={h}: libre {pct(acierto(n * pls, P8))}; de n*=2 ({np.sum(nr8 == 2)}) elige n=1 en "
              f"{np.sum((nr8 == 2) & (n == 1))}; n*=3 acierta {np.sum((nr8 == 3) & (n == 3))}/{np.sum(nr8 == 3)}; "
              f"n*=5 acierta {np.sum((nr8 == 5) & (n == 5))}/{np.sum(nr8 == 5)}")


# ------------------------------------------------------------------ Cap. 6
def capitulo6():
    w("\n" + "=" * 78)
    w("CAPITULO 6 - METODO UNIFICADO (OGLE)")
    pul = pd.read_csv(CACHE_PULSANTES)
    for tipo, g in pul.groupby("tipo", sort=False):
        ids = pd.read_csv(BASE_DIR / "Etapa1_Pulsantes" / f"muestra_{tipo.replace(' ', '_')}.csv")["id"].tolist()
        g = g[g["id"].isin(ids)].reset_index(drop=True)
        orig = g["id"].isin(ids[:N_POR_TIPO_BASE]).to_numpy()
        P = g["periodo_catalogo"].to_numpy()
        ok = {m: acierto(g[f"periodo_{m}"].to_numpy(), P) for m in ["ls", "ent_c", "rp_c"]}
        w(f"\n{tipo}: n={len(g)} (originales {orig.sum()}); LS {pct(ok['ls'])}, ent_c {pct(ok['ent_c'])}, "
          f"rp_c {pct(ok['rp_c'])}, rp siempre 2P0 {pct(acierto(2 * g['periodo_rp_c'].to_numpy(), P))}")
        w(f"  originales: LS {pct(ok['ls'][orig])} rp_c {pct(ok['rp_c'][orig])}; agregadas: LS {pct(ok['ls'][~orig])} "
          f"rp_c {pct(ok['rp_c'][~orig])}")
        w(f"  McNemar LS vs rp_c: {mcnemar(ok['ls'], ok['rp_c'])}")
        w(f"  Wilson rp_c {wilson(int(ok['rp_c'].sum()), len(g))}, LS {wilson(int(ok['ls'].sum()), len(g))}")
        if tipo == "Cefeida":
            for modo in ["1", "F", "F1"]:
                s = (g["modo"].astype(str) == modo).to_numpy()
                w(f"  modo {modo}: n={s.sum()} rp_c {pct(ok['rp_c'][s])} LS {pct(ok['ls'][s])}")
            s = (g["modo"].astype(str) == "F1").to_numpy()
            r = g.loc[s, "periodo_rp_c"] / g.loc[s, "periodo_catalogo"]
            w(f"  F1: razon P/P1 mediana {r.median():.3f}; ~0.72 (2%): {np.sum(np.abs(r - 0.72) < 0.02)}")
            w(f"  modos: {g['modo'].astype(str).value_counts().to_dict()}")
        else:
            for m in ["ls", "rp_c"]:
                r = g[f"periodo_{m}"] / g["periodo_catalogo"]
                w(f"  {m}: P/P1 ~1.345: {np.sum(np.abs(r - 1.345) / 1.345 < 0.01)}")
        for m in ["ls", "rp_c"]:
            f = ~ok[m]
            w(f"  fallos {m}: {f.sum()}; alias diario {np.sum(f & alias_diario(g[f'periodo_{m}'], P))}; "
              f"N<=50 {np.sum(f & (g['n_puntos'] <= 50).to_numpy())}")

    d = pd.read_csv(CACHE_DECISION)
    sk = pd.read_csv(CACHE_SKEW_OGLE)[["id", "skew_corregida"]]
    d = d.merge(sk, on="id", how="left")
    es_bin = (d["clase"] == "Binaria").to_numpy()
    orig_ids = set(pd.read_csv(MUESTRA_CSV)["id"][:N_MUESTRA_BASE])
    for tipo in ["Cefeida", "RR Lyrae"]:
        orig_ids |= set(pd.read_csv(BASE_DIR / "Etapa1_Pulsantes" / f"muestra_{tipo.replace(' ', '_')}.csv")["id"][:N_POR_TIPO_BASE])
    orig = d["id"].isin(orig_ids).to_numpy()
    w(f"\nDecision: {len(d)} curvas ({es_bin.sum()} binarias); originales {orig.sum()}")
    g_c = d["skew_corregida"].fillna(-np.inf).to_numpy()
    for col, v in [("skew_mag", d["skew_mag"].to_numpy()), ("skew_corregida", g_c)]:
        vv = np.where(np.isfinite(v), v, -1e9)
        w(f"  AUC {col}: {roc_auc_score(es_bin, vv):.3f} (originales {roc_auc_score(es_bin[orig], vv[orig]):.3f}, "
          f"agregadas {roc_auc_score(es_bin[~orig], vv[~orig]):.3f})")
    for ind in ["sin_phi21", "asimetria"]:
        v = d[ind].to_numpy()
        m = ~np.isnan(v)
        a = roc_auc_score(es_bin[m], -v[m])
        w(f"  AUC {ind}: {max(a, 1 - a):.3f}")
    for c in ["Binaria", "Cefeida", "RR Lyrae"]:
        s = (d["clase"] == c).to_numpy()
        w(f"  {c}: mediana skew corregida {np.nanmedian(d.loc[s, 'skew_corregida']):.2f}; supera umbral {pct(g_c[s] > UMBRAL_SKEW)}"
          f" (originales {pct(g_c[s & orig] > UMBRAL_SKEW)}, agregadas {pct(g_c[s & ~orig] > UMBRAL_SKEW)})")

    reglas = {"perfecta": es_bin, "todas pulsante": np.zeros_like(es_bin), "todas binaria": np.ones_like(es_bin),
              "skew sin corregir": d["skew_mag"].to_numpy() > UMBRAL_SKEW, "skew corregida": g_c > UMBRAL_SKEW}
    w("\nTabla unificado (Bin / Cef / RRL / puls->bin / bin->puls), todas | originales | agregadas")
    for nombre, ruta in reglas.items():
        ok = aciertos_ruteados(d, ruta)
        celdas = lambda s: " / ".join([pct(ok[s & (d["clase"] == c).to_numpy()]) for c in ["Binaria", "Cefeida", "RR Lyrae"]]
                                      + [pct(ruta[s & ~es_bin]), pct(~ruta[s & es_bin])])
        w(f"  {nombre:20s} {celdas(np.ones(len(d), bool))} | {celdas(orig)} | {celdas(~orig)}")
    ruta = g_c > UMBRAL_SKEW
    g = d[es_bin & ~ruta]
    w(f"  Binarias a rama pulsante (corregida): {len(g)}; tipos {g['subtipo'].value_counts().to_dict()}; "
      f"amplitud mediana {g['amplitud'].median():.2f} (todas las binarias {d.loc[es_bin, 'amplitud'].median():.2f}); "
      f"ELL totales {np.sum(es_bin & (d['subtipo'] == 'ELL'))}")
    w("  (La fila de la regresion logistica sale de resumen_decision_forma.txt, seccion 3.)")


# ------------------------------------------------------------------ Cap. 7
def capitulo7():
    w("\n" + "=" * 78)
    w("CAPITULO 7 - Ver Robustez_Sinteticas/resumen_sinteticas.txt y TESS_Prueba/resumen_tess.txt;")
    w("aqui solo las cifras auxiliares.")
    pl = pd.read_csv(BASE_DIR / "Robustez_Sinteticas" / "plantillas.csv")
    w(f"Plantillas: {pl['clase'].value_counts().to_dict()}; binarias por tipo "
      f"{pl.loc[pl['clase'] == 'Binaria', 'subtipo'].value_counts().to_dict()}; Cefeidas por modo "
      f"{pl.loc[pl['clase'] == 'Cefeida', 'subtipo'].astype(str).value_counts().to_dict()}; amplitud mediana binarias "
      f"{pl.loc[pl['clase'] == 'Binaria', 'amplitud_plantilla'].median():.2f} mag")
    hib = pd.read_csv(CACHE_DECISION)
    w(f"Amplitud mediana binarias OGLE (cache decision): {hib.loc[hib['clase'] == 'Binaria', 'amplitud'].median():.2f}")
    tess = pd.read_csv(BASE_DIR / "TESS_Prueba" / "cache_tess.csv")
    mu = pd.read_csv(BASE_DIR / "TESS_Prueba" / "muestra_tess.csv")
    tess = tess[tess["nombre"].isin(set(mu["nombre"]))]
    w(f"TESS: {tess['grupo'].value_counts().to_dict()}; subtipos pulsantes "
      f"{tess.loc[tess['clase'] != 'Binaria'].groupby(['clase', 'subtipo']).size().to_dict()}")
    sep = tess[tess["grupo"] == "Binaria separada"]
    w(f"  Separadas con skew corregida NaN: {sep['skew_corregida'].isna().sum()} de {len(sep)}")
    con = tess[tess["grupo"] == "Binaria contacto/ELL"]
    w(f"  Contacto/ELL: P0 ~ P/2 {pct(acierto(con['periodo_rp_c'].to_numpy(), con['periodo_referencia'].to_numpy() / 2))}; "
      f"skew corregida mediana {con['skew_corregida'].median():.2f} (P25-P75 {con['skew_corregida'].quantile(.25):.2f}-"
      f"{con['skew_corregida'].quantile(.75):.2f}); a rama pulsante {pct(~(con['skew_corregida'].fillna(-np.inf) > UMBRAL_SKEW))}")
    cef = tess[(tess["clase"] == "Cefeida") & (tess["skew_corregida"] > UMBRAL_SKEW)]
    ciclos = cef["baseline_d"] / cef["periodo_referencia"]
    w(f"  Cefeidas a rama binaria: {len(cef)}; ciclos en el sector {ciclos.min():.1f}-{ciclos.max():.1f}")
    w(f"  Original (primeras por grupo) vs agregadas: ver resumen_tess.txt; agregadas = {len(tess) - 240}")


def main():
    capitulo4()
    capitulo5()
    capitulo6()
    capitulo7()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
