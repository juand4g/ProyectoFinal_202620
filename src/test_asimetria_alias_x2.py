"""
Prueba exploratoria (no parte todavia de la caracterizacion formal): para
las binarias que Lomb-Scargle NO recupera directamente, se evalua si
distintos estadisticos de asimetria de la curva de luz plegada en 2*P_LS
permiten distinguir las que SI necesitan multiplicar por 2 (el periodo real
esta en 2*P_LS) de las que ya estaban bien desde un principio.

Se excluyen las W UMa (tipo morfologico "C", contacto) porque en ese
subgrupo los dos eclipses son casi identicos y el problema es, en la
practica, indistinguible por fotometria sola (ver conversacion previa).

Reaprovecha el cache de caracterizacion_lomb_scargle.py -- no vuelve a
correr Lomb-Scargle sobre nada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_LS = BASE_DIR / "Lomb-Scargle" / "cache_resultados_ls.csv"
CATALOGO = BASE_DIR / "ogle_collection" / "binaries.txt"

UMBRAL = 0.01
N_BINS_PERFIL = 20


def construir_muestra_etiquetada():
    cache = pd.read_csv(CACHE_LS)
    bin_ls = cache[cache["tipo"] == "Binaria"].dropna(subset=["periodo_ls", "periodo_catalogo"]).copy()

    cat = pd.read_csv(CATALOGO, sep="\t", comment="#").set_index("ID")
    bin_ls["tipo_morf"] = bin_ls["id"].map(cat["Type"])
    bin_ls = bin_ls[bin_ls["tipo_morf"] != "C"].copy()  # excluir W UMa

    pc = bin_ls["periodo_catalogo"].to_numpy()
    p = bin_ls["periodo_ls"].to_numpy()
    error_directo = np.abs(p - pc) / pc
    error_x2 = np.abs(2 * p - pc) / pc

    etiqueta = np.full(len(bin_ls), -1)
    etiqueta[error_directo < UMBRAL] = 0  # ya esta bien, NO necesita x2
    etiqueta[(error_directo >= UMBRAL) & (error_x2 < UMBRAL)] = 1  # necesita x2
    bin_ls["necesita_x2"] = etiqueta
    return bin_ls


def cargar_fotometria(id_estrella):
    t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
    return t, mag


def plegar_en_dos_mitades(t, mag, periodo):
    """Pliega en `periodo` y separa la fase en dos mitades [0, 0.5) y
    [0.5, 1.0), candidatas a ser el primer y segundo eclipse del periodo
    real si `periodo` = 2 * P_LS."""
    fase = np.mod((t - t[0]) / periodo, 1.0)
    m1 = mag[fase < 0.5]
    m2 = mag[fase >= 0.5]
    f1 = fase[fase < 0.5]
    f2 = fase[fase >= 0.5] - 0.5
    return (f1, m1), (f2, m2)


def perfil_binneado(fase, mag, n_bins=N_BINS_PERFIL):
    """Mediana de magnitud por bin de fase en [0, 0.5); NaN en bins vacios."""
    bordes = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(fase, bordes) - 1
    idx = np.clip(idx, 0, n_bins - 1)
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

    # 1) Asimetria de profundidad: diferencia entre el percentil 98 (proxy
    #    robusto del minimo/mas tenue) de cada mitad.
    prof1, prof2 = np.percentile(m1, 98), np.percentile(m2, 98)
    stat_profundidad = abs(prof1 - prof2)

    # 2) Estadistico KS entre las distribuciones de magnitud de cada mitad.
    stat_ks, _ = ks_2samp(m1, m2)

    # 3) Diferencia RMS entre perfiles binneados (forma de la curva, no solo
    #    profundidad).
    perfil1 = perfil_binneado(f1, m1)
    perfil2 = perfil_binneado(f2, m2)
    validos = ~np.isnan(perfil1) & ~np.isnan(perfil2)
    stat_forma = np.sqrt(np.mean((perfil1[validos] - perfil2[validos]) ** 2)) if validos.sum() >= 4 else np.nan

    # 4) Diferencia simple de medianas (asimetria "gruesa").
    stat_mediana = abs(np.median(m1) - np.median(m2))

    return {
        "asim_profundidad": stat_profundidad,
        "asim_ks": stat_ks,
        "asim_forma": stat_forma,
        "asim_mediana": stat_mediana,
    }


def main():
    muestra = construir_muestra_etiquetada()
    print(f"Muestra (sin W UMa): {len(muestra)} binarias -> "
          f"{(muestra['necesita_x2']==1).sum()} necesitan x2, "
          f"{(muestra['necesita_x2']==0).sum()} ya estan bien, "
          f"{(muestra['necesita_x2']==-1).sum()} ninguno de los dos (excluidas del AUC).")

    filas = []
    for _, fila in muestra.iterrows():
        t, mag = cargar_fotometria(fila["id"])
        stats = estadisticos_asimetria(t, mag, 2 * fila["periodo_ls"])
        if stats is None:
            continue
        stats["id"] = fila["id"]
        stats["necesita_x2"] = fila["necesita_x2"]
        filas.append(stats)

    df = pd.DataFrame(filas)
    df_valid = df[df["necesita_x2"] != -1].dropna()

    print(f"\nEvaluando discriminacion sobre {len(df_valid)} binarias con etiqueta clara "
          f"({(df_valid['necesita_x2']==1).sum()} necesitan x2, {(df_valid['necesita_x2']==0).sum()} no).\n")

    columnas_stat = ["asim_profundidad", "asim_ks", "asim_forma", "asim_mediana"]
    for col in columnas_stat:
        auc = roc_auc_score(df_valid["necesita_x2"], df_valid[col])
        mediana_0 = df_valid.loc[df_valid["necesita_x2"] == 0, col].median()
        mediana_1 = df_valid.loc[df_valid["necesita_x2"] == 1, col].median()
        print(f"{col:<18} AUC={auc:.3f}   mediana(no_necesita)={mediana_0:.4f}   mediana(necesita_x2)={mediana_1:.4f}")

    print("\nMismos estadisticos sobre las 'ninguno funciona' (etiqueta -1), para contexto:")
    df_ambiguo = df[df["necesita_x2"] == -1].dropna()
    for col in columnas_stat:
        print(f"{col:<18} mediana={df_ambiguo[col].median():.4f}  (n={len(df_ambiguo)})")


if __name__ == "__main__":
    main()
