"""
Prueba exploratoria (no parte todavia de la caracterizacion formal): amplia
la muestra de binarias no-W-UMa mas alla de las 500 usadas en
caracterizacion_lomb_scargle.py, calcula el periodo de Lomb-Scargle y varios
estadisticos de asimetria de la curva de luz plegada en 2*P_LS para cada
una, y ajusta una regresion logistica que combina esos estadisticos para
predecir si toca multiplicar el periodo por 2.

Se excluyen las W UMa (tipo morfologico "C", contacto): ver conversacion
previa sobre por que ese subgrupo es un caso aparte.

Cache propio (Lomb-Scargle/cache_binarias_no_c_asimetria.csv), separado del
cache oficial de caracterizacion_lomb_scargle.py, para no alterar los
resultados ya reportados en la tesis. Reaprovecha las 500 binarias oficiales
ya calculadas (no se recalcula su Lomb-Scargle) y agrega nuevas hasta
completar N_TOTAL_OBJETIVO.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caracterizacion_lomb_scargle import (
    calcular_periodo_ls, P_MIN_BINARIAS, P_MAX_BINARIAS, SAMPLES_PER_PEAK,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_LS_OFICIAL = BASE_DIR / "Lomb-Scargle" / "cache_resultados_ls.csv"
CATALOGO = BASE_DIR / "ogle_collection" / "binaries.txt"
CACHE_PROPIO = BASE_DIR / "Lomb-Scargle" / "cache_binarias_no_c_asimetria.csv"

RANDOM_SEED_AMPLIACION = 123  # distinta de RANDOM_SEED=42 del pipeline oficial
N_TOTAL_OBJETIVO = 3000  # tamano de muestra combinado (oficiales + nuevas)
UMBRAL = 0.01
N_BINS_PERFIL = 20

COLUMNAS_CACHE = [
    "id", "tipo_morf", "n_puntos", "periodo_catalogo", "periodo_ls",
    "error_relativo", "error_relativo_x2", "necesita_x2",
    "asim_profundidad", "asim_ks", "asim_forma", "asim_mediana",
]


def plegar_en_dos_mitades(t, mag, periodo):
    fase = np.mod((t - t[0]) / periodo, 1.0)
    m1, m2 = mag[fase < 0.5], mag[fase >= 0.5]
    f1, f2 = fase[fase < 0.5], fase[fase >= 0.5] - 0.5
    return (f1, m1), (f2, m2)


def perfil_binneado(fase, mag, n_bins=N_BINS_PERFIL):
    bordes = np.linspace(0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(fase, bordes) - 1, 0, n_bins - 1)
    perfil = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = mag[idx == b]
        if vals.size > 0:
            perfil[b] = np.median(vals)
    return perfil


def estadisticos_asimetria(t, mag, periodo_candidato):
    (f1, m1), (f2, m2) = plegar_en_dos_mitades(t, mag, periodo_candidato)
    if m1.size < 5 or m2.size < 5:
        return None
    stat_profundidad = abs(np.percentile(m1, 98) - np.percentile(m2, 98))
    stat_ks, _ = ks_2samp(m1, m2)
    perfil1, perfil2 = perfil_binneado(f1, m1), perfil_binneado(f2, m2)
    validos = ~np.isnan(perfil1) & ~np.isnan(perfil2)
    stat_forma = np.sqrt(np.mean((perfil1[validos] - perfil2[validos]) ** 2)) if validos.sum() >= 4 else np.nan
    stat_mediana = abs(np.median(m1) - np.median(m2))
    return stat_profundidad, stat_ks, stat_forma, stat_mediana


def construir_lista_objetivo():
    """IDs oficiales (ya con periodo_ls calculado) + una muestra nueva
    aleatoria hasta completar N_TOTAL_OBJETIVO, todo sin W UMa (tipo 'C')."""
    cat = pd.read_csv(CATALOGO, sep="\t", comment="#").set_index("ID")
    cat_no_c = cat[cat["Type"] != "C"]

    oficial = pd.read_csv(CACHE_LS_OFICIAL)
    oficial = oficial[oficial["tipo"] == "Binaria"].dropna(subset=["periodo_ls", "periodo_catalogo"])
    oficial = oficial[oficial["id"].isin(cat_no_c.index)]
    periodos_ls_conocidos = dict(zip(oficial["id"], oficial["periodo_ls"]))
    periodos_cat_conocidos = dict(zip(oficial["id"], oficial["periodo_catalogo"]))

    archivos = {f.stem for f in DATA_DIR.glob("*.dat")}
    disponibles = sorted(set(cat_no_c.index) & archivos)

    ya_incluidos = set(oficial["id"])
    n_nuevas = max(0, N_TOTAL_OBJETIVO - len(ya_incluidos))
    candidatas_nuevas = [i for i in disponibles if i not in ya_incluidos]
    random.seed(RANDOM_SEED_AMPLIACION)
    nuevas = random.sample(candidatas_nuevas, min(n_nuevas, len(candidatas_nuevas)))

    ids_totales = list(ya_incluidos) + nuevas
    return ids_totales, periodos_ls_conocidos, periodos_cat_conocidos, cat_no_c["Type"]


def main():
    ids_totales, periodos_ls_conocidos, periodos_cat_conocidos, tipo_morf_map = construir_lista_objetivo()

    if CACHE_PROPIO.exists():
        cache = pd.read_csv(CACHE_PROPIO)
    else:
        cache = pd.DataFrame(columns=COLUMNAS_CACHE)
    procesados = set(cache["id"])

    filas_nuevas = []
    for id_estrella in tqdm(ids_totales, desc="binarias", unit="estrella"):
        if id_estrella in procesados:
            continue
        try:
            t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
        except (ValueError, OSError):
            continue
        if t.size < 10:
            continue

        if id_estrella in periodos_ls_conocidos:
            periodo_ls = periodos_ls_conocidos[id_estrella]
            periodo_catalogo = periodos_cat_conocidos[id_estrella]
        else:
            periodo_ls, _, _, _ = calcular_periodo_ls(
                t, mag, err, P_MIN_BINARIAS, P_MAX_BINARIAS, SAMPLES_PER_PEAK
            )
            periodo_catalogo = float(periodos_cat_conocidos.get(id_estrella, np.nan))
            if np.isnan(periodo_catalogo):
                periodo_catalogo = float(pd.read_csv(CATALOGO, sep="\t", comment="#").set_index("ID").loc[id_estrella, "P"])

        error_relativo = abs(periodo_ls - periodo_catalogo) / periodo_catalogo
        error_x2 = abs(2 * periodo_ls - periodo_catalogo) / periodo_catalogo
        if error_relativo < UMBRAL:
            necesita_x2 = 0
        elif error_x2 < UMBRAL:
            necesita_x2 = 1
        else:
            necesita_x2 = -1

        stats = estadisticos_asimetria(t, mag, 2 * periodo_ls)
        if stats is None:
            continue
        asim_profundidad, asim_ks, asim_forma, asim_mediana = stats

        filas_nuevas.append({
            "id": id_estrella,
            "tipo_morf": tipo_morf_map.get(id_estrella, np.nan),
            "n_puntos": int(t.size),
            "periodo_catalogo": periodo_catalogo,
            "periodo_ls": periodo_ls,
            "error_relativo": error_relativo,
            "error_relativo_x2": error_x2,
            "necesita_x2": necesita_x2,
            "asim_profundidad": asim_profundidad,
            "asim_ks": asim_ks,
            "asim_forma": asim_forma,
            "asim_mediana": asim_mediana,
        })

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_PROPIO, index=False)

    print(f"\nMuestra total (sin W UMa): {len(cache)} binarias -> "
          f"{(cache['necesita_x2']==1).sum()} necesitan x2, "
          f"{(cache['necesita_x2']==0).sum()} ya estan bien, "
          f"{(cache['necesita_x2']==-1).sum()} ninguno de los dos.")

    columnas_stat = ["asim_profundidad", "asim_ks", "asim_forma", "asim_mediana"]
    etiquetado = cache[cache["necesita_x2"] != -1].dropna(subset=columnas_stat)
    X = etiquetado[columnas_stat].to_numpy()
    y = etiquetado["necesita_x2"].to_numpy()

    print(f"\nEvaluando sobre {len(etiquetado)} binarias con etiqueta clara "
          f"({(y==1).sum()} necesitan x2, {(y==0).sum()} no).\n")

    print("AUC individual de cada estadistico (sobre la muestra ampliada):")
    for i, col in enumerate(columnas_stat):
        auc = roc_auc_score(y, X[:, i])
        print(f"  {col:<18} AUC={auc:.3f}")

    escalador = StandardScaler()
    X_esc = escalador.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    modelo = LogisticRegression(class_weight="balanced")
    proba_cv = cross_val_predict(modelo, X_esc, y, cv=cv, method="predict_proba")[:, 1]
    auc_cv = roc_auc_score(y, proba_cv)

    modelo_final = LogisticRegression(class_weight="balanced").fit(X_esc, y)
    print(f"\nRegresion logistica combinando los 4 estadisticos:")
    print(f"  AUC validado cruzado (5-fold): {auc_cv:.3f}")
    print(f"  Coeficientes (sobre variables estandarizadas, ajuste con toda la muestra):")
    for col, coef in zip(columnas_stat, modelo_final.coef_[0]):
        print(f"    {col:<18} {coef:+.3f}")


if __name__ == "__main__":
    main()
