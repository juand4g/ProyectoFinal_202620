"""
Selector supervisado del multiplo n (P = n*P0) para binarias eclipsantes, a
partir del periodo P0 del metodo hibrido consolidado (rank_product
condicional; ver bitacora). Sustituye la regla fija "x2 por defecto + BIC
para n>=3" por un clasificador entrenado con indicadores que NO usan el
catalogo:

  - asimetria del plegado en 2*P0 y en 4*P0 (KS, forma, profundidad,
    mediana; mismas funciones que explorar_asimetria_alias_x2.py)
  - potencia de Lomb-Scargle en f0, f0/2 y f0/4
  - entropia conjunta y condicional (L=10, K=7) en P0, 2*P0 y 4*P0
  - diferencias de BIC del refinamiento (H0=4) respecto de n=2
  - concordancia con otros metodos: log2(P_entropia_condicional/P0) y
    log2(P_LS/P0), ambos ya calculados en el cache del metodo hibrido
  - propiedades de la curva: amplitud, SNR, numero de puntos, linea base/P0

Clase objetivo: n_real en 1..6 (el multiplo que reproduce el catalogo al
1%). Las estrellas sin multiplo simple (n_real=0) no se usan para entrenar
pero SI cuentan como fallo en la evaluacion. Evaluacion por validacion
cruzada estratificada de 5 particiones, repetida 10 veces con semillas
distintas; se compara contra "siempre x2" y contra el hibrido BIC sobre las
mismas 1000 binarias. Cache de indicadores append-safe.
"""

import warnings

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from entropypf import get_entropies
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from explorar_asimetria_alias_x2 import estadisticos_asimetria
from refinamiento_rank_product_enmascarado_bic import CACHE_CSV as CACHE_HIBRIDO
from refinamiento_rank_product_enmascarado_bic import K, L
from refinamiento_subarmonicos_bic import N_MAX, OUT_DIR, PHOT_DIR, acierto, n_hibrido

CACHE_INDICADORES = OUT_DIR / "cache_selector_indicadores.csv"
RESUMEN_TXT = OUT_DIR / "resumen_selector_multiplo.txt"

BASE = "rp_c"
H0 = 4
MULTIPLOS = [1, 2, 4]
N_REPETICIONES = 10
N_PARTICIONES = 5
BIC_AUSENTE = 1e4  # candidato no evaluable (n*P0 > linea base)


def entropias_en(t, mag, periodos):
    """Entropia conjunta y condicional en periodos puntuales, con el mismo
    origen de fase para ambas (ver entropias_conjunta_y_fase)."""
    i0 = int(np.argmin(np.abs(mag - np.percentile(mag, 2))))
    orden = np.r_[i0, np.delete(np.arange(t.size), i0)]
    t, mag = t[orden], mag[orden]
    conjunta = get_entropies(t, mag, periodos, L=L, K=K)
    fase = get_entropies(t, np.zeros_like(mag), periodos, L=L, K=K)
    return conjunta, conjunta - fase


def indicadores_estrella(fila):
    t, mag, err = np.loadtxt(PHOT_DIR / f"{fila['id']}.dat", unpack=True)
    err = np.where(err > 0, err, np.median(err[err > 0]))
    return {"id": fila["id"], **indicadores_curva(t, mag, err, fila[f"periodo_{BASE}"])}


def indicadores_curva(t, mag, err, p0):
    """Indicadores del selector para una curva cualquiera (OGLE o sintetica)."""
    ind = {}

    periodos = np.array([m * p0 for m in MULTIPLOS])
    potencia = LombScargle(t, mag, err).power(1.0 / periodos)
    conjunta, condicional = entropias_en(t, mag, periodos)
    for m, pot, sc, scond in zip(MULTIPLOS, potencia, conjunta, condicional):
        ind[f"ls_pot_x{m}"] = pot
        ind[f"ent_conj_x{m}"] = sc
        ind[f"ent_cond_x{m}"] = scond

    for m in [2, 4]:
        asim = estadisticos_asimetria(t, mag, m * p0)
        nombres = ["profundidad", "ks", "forma", "mediana"]
        for nombre, valor in zip(nombres, asim if asim is not None else [np.nan] * 4):
            ind[f"asim_{nombre}_x{m}"] = valor

    amplitud = np.percentile(mag, 98) - np.percentile(mag, 2)
    ind["amplitud"] = amplitud
    ind["snr"] = amplitud / np.median(err)
    return ind


def construir_tabla(hibrido, indicadores):
    df = hibrido.merge(indicadores, on="id")
    p0 = df[f"periodo_{BASE}"].to_numpy()
    X = pd.DataFrame(index=df.index)

    columnas_ind = [c for c in indicadores.columns if c != "id"]
    X[columnas_ind] = df[columnas_ind]
    for m in [2, 4]:
        X[f"ls_razon_x{m}_x1"] = df[f"ls_pot_x{m}"] / df["ls_pot_x1"]
        X[f"ent_cond_dif_x{m}_x1"] = df[f"ent_cond_x{m}"] - df["ent_cond_x1"]
        X[f"ent_conj_dif_x{m}_x1"] = df[f"ent_conj_x{m}"] - df["ent_conj_x1"]

    bic = {n: df[f"bic_{BASE}_h{H0}_n{n}"].fillna(BIC_AUSENTE) for n in range(1, N_MAX + 1)}
    for n in [1, 3, 4, 5, 6]:
        X[f"bic_dif_n{n}_n2"] = bic[n] - bic[2]
    X["n_hibrido"] = n_hibrido(df, H0, f"bic_{BASE}")

    X["log2_ent_c_sobre_p0"] = np.log2(df["periodo_ent_c"] / p0)
    X["log2_ls_sobre_p0"] = np.log2(df["periodo_ls"] / p0)
    X["log10_p0"] = np.log10(p0)
    X["log10_baseline_sobre_p0"] = np.log10(df["baseline_d"] / p0)
    X["n_puntos"] = df["n_puntos"]

    P = df["periodo_catalogo"].to_numpy()
    n_real = np.zeros(len(df), dtype=int)
    for n in range(N_MAX, 0, -1):
        n_real[acierto(n * p0, P)] = n
    return df, X, n_real


def modelos():
    return {
        "Gradient boosting": lambda s: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=s),
        "Regresion logistica": lambda s: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.5, max_iter=5000)),
    }


def validar(X, n_real, fabricar):
    """Prediccion fuera de muestra por repeticion; las n_real=0 reciben
    prediccion pero nunca entran al entrenamiento."""
    entrenables = n_real > 0
    predicciones = []
    for rep in range(N_REPETICIONES):
        pred = np.zeros(len(X), dtype=int)
        skf = StratifiedKFold(N_PARTICIONES, shuffle=True, random_state=rep)
        # las n_real=0 se reparten entre las particiones como una clase mas
        for entr, prueba in skf.split(X, n_real):
            entr = entr[entrenables[entr]]
            modelo = fabricar(rep).fit(X.iloc[entr], n_real[entr])
            pred[prueba] = modelo.predict(X.iloc[prueba])
        predicciones.append(pred)
    return np.array(predicciones)


def resumen(df, X, n_real):
    lineas = []
    w = lineas.append
    total = len(df)
    w("RESUMEN - SELECTOR SUPERVISADO DEL MULTIPLO n SOBRE P0 (rank_product condicional)")
    w("=" * 78)
    w(f"Binarias: {total}; validacion cruzada estratificada {N_PARTICIONES} particiones x "
      f"{N_REPETICIONES} repeticiones. Clases n_real: "
      + ", ".join(f"x{k}={np.sum(n_real == k)}" for k in range(0, N_MAX + 1)) + " (x0 = sin multiplo simple)")
    w("")
    n_hib = X["n_hibrido"].to_numpy()
    w("REFERENCIAS")
    w(f"  Siempre x2:                    {np.mean(n_real == 2):6.1%}")
    w(f"  Hibrido BIC (H0={H0}):            {np.mean(n_real == n_hib):6.1%}")
    w(f"  Techo oraculo x1-x{N_MAX}:           {np.mean(n_real > 0):6.1%}")
    w("")
    w("SELECTOR MULTICLASE (exactitud sobre las 1000, media +/- desv. entre repeticiones)")
    resultados = {}
    for nombre, fabricar in modelos().items():
        pred = validar(X, n_real, fabricar)
        exactitud = (pred == n_real[None, :]).mean(axis=1)
        resultados[nombre] = pred
        gana = ((pred == n_real) & (n_hib != n_real)).sum(axis=1).mean()
        pierde = ((pred != n_real) & (n_hib == n_real)).sum(axis=1).mean()
        w(f"  {nombre:20s} {exactitud.mean():6.1%} +/- {exactitud.std():.1%}   "
          f"vs. hibrido: gana {gana:.1f}, pierde {pierde:.1f} estrellas")

    mejor = max(resultados, key=lambda k: (resultados[k] == n_real).mean())
    pred = resultados[mejor][0]
    w("")
    w(f"Matriz de confusion ({mejor}, repeticion 0): filas n_real, columnas n predicho")
    w("          " + "".join(f"{f'x{k}':>6}" for k in range(1, N_MAX + 1)))
    for r in range(0, N_MAX + 1):
        m = n_real == r
        w(f"  real x{r}" + "".join(f"{np.sum(m & (pred == k)):6d}" for k in range(1, N_MAX + 1)))

    # Selector binario x1 vs x2, como se planteo originalmente
    w("")
    w("SELECTOR BINARIO x1 vs. x2 (solo estrellas con n_real en {1, 2})")
    m12 = np.isin(n_real, [1, 2])
    X12, y12 = X[m12].reset_index(drop=True), (n_real[m12] == 2).astype(int)
    for nombre, fabricar in modelos().items():
        aucs, exactitudes = [], []
        for rep in range(N_REPETICIONES):
            prob = np.zeros(len(y12))
            for entr, prueba in StratifiedKFold(N_PARTICIONES, shuffle=True, random_state=rep).split(X12, y12):
                modelo = fabricar(rep).fit(X12.iloc[entr], y12[entr])
                prob[prueba] = modelo.predict_proba(X12.iloc[prueba])[:, 1]
            aucs.append(roc_auc_score(y12, prob))
            exactitudes.append(np.mean((prob >= 0.5) == y12))
        w(f"  {nombre:20s} AUC {np.mean(aucs):.3f}   exactitud {np.mean(exactitudes):6.1%}   "
          f"(siempre x2: {y12.mean():6.1%}, n={len(y12)})")

    # Importancia por permutacion del gradient boosting entrenado con todo
    entrenables = n_real > 0
    gb = modelos()["Gradient boosting"](0).fit(X[entrenables], n_real[entrenables])
    imp = permutation_importance(gb, X[entrenables], n_real[entrenables], n_repeats=10, random_state=0)
    orden = np.argsort(imp.importances_mean)[::-1][:10]
    w("")
    w("Indicadores mas importantes (permutacion, gradient boosting, en muestra):")
    for i in orden:
        w(f"  {X.columns[i]:28s} {imp.importances_mean[i]:.4f}")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def main():
    warnings.filterwarnings("ignore")
    hibrido = pd.read_csv(CACHE_HIBRIDO)

    indicadores = pd.read_csv(CACHE_INDICADORES) if CACHE_INDICADORES.exists() else pd.DataFrame()
    procesados = set(indicadores["id"]) if not indicadores.empty else set()
    filas = [indicadores_estrella(f) for _, f in tqdm(hibrido.iterrows(), total=len(hibrido),
                                                     desc="Indicadores", unit="estrella")
             if f["id"] not in procesados]
    if filas:
        indicadores = pd.concat([indicadores, pd.DataFrame(filas)], ignore_index=True)
        indicadores.to_csv(CACHE_INDICADORES, index=False)

    df, X, n_real = construir_tabla(hibrido, indicadores)
    resumen(df, X, n_real)


if __name__ == "__main__":
    main()
