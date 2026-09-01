"""
Prueba ampliada, enfocada solo en Binarias, de metodos para combinar
Lomb-Scargle (astropy) y minima entropia (entropypf) en una misma grilla de
FRECUENCIA -- continuacion de exploracion_combinacion_ls_entropia.py, que
con solo 6 estrellas de prueba encontro un caso (OGLE-LMC-ECL-07968) donde
el producto normalizado acertaba el periodo aunque Lomb-Scargle y minima
entropia fallaran cada uno por separado. Esta version corre sobre una
muestra mas grande de binarias para ver si ese exito se repite con alguna
frecuencia o fue casualidad de una sola estrella.

Misma convencion que el script anterior: todo el trabajo (grilla, busqueda
de picos, combinacion de curvas) ocurre en FRECUENCIA; el periodo solo
aparece (1) como entrada obligatoria de get_entropies y (2) al reportar el
resultado final de cada metodo.

Se descarta la convolucion (en la prueba anterior se vio que su forma es una
rampa triangular por efecto de borde del modo "same", no señal real) y se
fusiona "geometrica" con "producto": para encontrar el maximo da lo mismo
multiplicar dos curvas no negativas que multiplicar sus raices cuadradas
(sqrt es monotona), asi que ambas siempre eligen el mismo periodo -- solo
cambia la forma de la curva para visualizar, no la decision. Se agregan
cuatro metodos nuevos, elegidos con el usuario:
  - cociente:              LS_norm(f) / (entropia_norm(f) + eps)
  - rank_product:          percentil(LS)(f) * percentil(-entropia)(f)
  - suma_zscores:          z(LS)(f) + z(-entropia)(f)
  - emparejamiento_picos:  picos top-N de cada curva por separado; si dos
                            picos (uno de cada metodo) caen a menos de
                            TOL_FRECUENCIA uno de otro, se los declara en
                            consenso y se usa el de mayor producto entre los
                            consensos encontrados; si ninguno coincide, cae
                            de vuelta al maximo global de "producto".
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from entropypf import get_entropies
from scipy.signal import find_peaks
from scipy.stats import rankdata
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
N_MUESTRA = 40  # incluye las 2 estrellas fijas de la prueba anterior
RANDOM_SEED = 42

ESTRELLAS_FIJAS = ["OGLE-LMC-ECL-07968", "OGLE-LMC-ECL-01807"]

P_MIN, P_MAX = 0.1, 100.0  # mismo rango acotado usado para binarias en los otros scripts
N_FREQ = 50_000  # grilla UNIFORME en frecuencia (ciclos/dia)
L, K = 7, 7

EPS_COCIENTE = 0.05
N_PICOS_VOTACION = 25
DISTANCIA_PICOS_GRID = 50  # separacion minima entre picos, en pasos de grilla
TOLERANCIA_VOTACION_GRID = 50  # ventana de "coincidencia" entre picos, en pasos de grilla

N_EJEMPLOS_PLOT = 4  # cuantas estrellas graficar en detalle (de las N_MUESTRA)

DATA_DIR = Path(__file__).resolve().parent.parent / "ogle_collection"
OUT_DIR = Path(__file__).resolve().parent.parent / "Combinacion_LS_Entropia"
RESULTADOS_CSV = OUT_DIR / "resultados_binarias_ampliado.csv"
PLOTS_DIR = OUT_DIR / "graficas_binarias"


def cargar_catalogo():
    df = pd.read_csv(DATA_DIR / "binaries.txt", sep="\t", comment="#")
    return df.set_index("ID")["P"]


def normalizar(x):
    x_min, x_max = np.min(x), np.max(x)
    if x_max <= x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def mejor_alias(periodo_encontrado, periodo_catalogo):
    candidatos = [periodo_encontrado, periodo_encontrado / 2, periodo_encontrado * 2,
                  periodo_encontrado / 3, periodo_encontrado * 3]
    errores = [abs(p - periodo_catalogo) / periodo_catalogo for p in candidatos]
    i = int(np.argmin(errores))
    return candidatos[i], errores[i]


def emparejamiento_picos(frecuencia, ls_norm, bondad_entropia_norm):
    paso = frecuencia[1] - frecuencia[0]
    tolerancia = TOLERANCIA_VOTACION_GRID * paso

    picos_ls, _ = find_peaks(ls_norm, distance=DISTANCIA_PICOS_GRID)
    picos_ent, _ = find_peaks(bondad_entropia_norm, distance=DISTANCIA_PICOS_GRID)
    picos_ls = picos_ls[np.argsort(ls_norm[picos_ls])[::-1][:N_PICOS_VOTACION]]
    picos_ent = picos_ent[np.argsort(bondad_entropia_norm[picos_ent])[::-1][:N_PICOS_VOTACION]]

    consensos = []
    if len(picos_ls) and len(picos_ent):
        for i_ls in picos_ls:
            dif = np.abs(frecuencia[picos_ent] - frecuencia[i_ls])
            j = np.argmin(dif)
            if dif[j] < tolerancia:
                i_ent = picos_ent[j]
                score = ls_norm[i_ls] * bondad_entropia_norm[i_ent]
                consensos.append((score, i_ls))

    if consensos:
        consensos.sort(reverse=True)
        return consensos[0][1], True

    producto = ls_norm * bondad_entropia_norm
    return int(np.argmax(producto)), False


def analizar_estrella(id_estrella, periodo_catalogo):
    path_dat = DATA_DIR / "binaries_phot" / f"{id_estrella}.dat"
    t, mag, err = np.loadtxt(path_dat, unpack=True)

    frecuencia = np.linspace(1.0 / P_MAX, 1.0 / P_MIN, N_FREQ)

    ls_potencia = LombScargle(t, mag, err).power(frecuencia)
    entropia = get_entropies(t, mag, 1.0 / frecuencia, L=L, K=K)  # unica conversion previa a periodo: entrada obligatoria de get_entropies

    ls_norm = normalizar(ls_potencia)
    entropia_norm = normalizar(entropia)
    bondad_entropia_norm = 1.0 - entropia_norm

    curvas_indice = {}
    curvas_indice["Lomb-Scargle"] = int(np.argmax(ls_norm))
    curvas_indice["1 - entropia"] = int(np.argmax(bondad_entropia_norm))
    curvas_indice["producto"] = int(np.argmax(ls_norm * bondad_entropia_norm))
    curvas_indice["cociente"] = int(np.argmax(ls_norm / (entropia_norm + EPS_COCIENTE)))

    rango_ls = rankdata(ls_potencia) / len(ls_potencia)
    rango_ent = rankdata(-entropia) / len(entropia)
    curvas_indice["rank_product"] = int(np.argmax(rango_ls * rango_ent))

    z_ls = (ls_potencia - np.median(ls_potencia)) / np.std(ls_potencia)
    z_ent = (np.median(entropia) - entropia) / np.std(entropia)
    curvas_indice["suma_zscores"] = int(np.argmax(z_ls + z_ent))

    i_votacion, hubo_coincidencia = emparejamiento_picos(frecuencia, ls_norm, bondad_entropia_norm)
    curvas_indice["emparejamiento_picos"] = i_votacion

    filas = []
    for metodo, i_mejor in curvas_indice.items():
        periodo_encontrado = 1.0 / frecuencia[i_mejor]  # unica conversion a periodo del resultado
        error_relativo = abs(periodo_encontrado - periodo_catalogo) / periodo_catalogo
        periodo_alias, error_alias = mejor_alias(periodo_encontrado, periodo_catalogo)
        filas.append({
            "id": id_estrella, "metodo": metodo, "periodo_catalogo": periodo_catalogo,
            "periodo_encontrado": periodo_encontrado, "error_relativo": error_relativo,
            "periodo_alias": periodo_alias, "error_relativo_alias": error_alias,
            "hubo_coincidencia": hubo_coincidencia if metodo == "emparejamiento_picos" else np.nan,
        })

    return filas, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice


def graficar(id_estrella, periodo_catalogo, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice):
    periodo = 1.0 / frecuencia
    producto = ls_norm * bondad_entropia_norm

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.3), sharex=True)
    axes[0].plot(periodo, ls_norm, lw=0.7, color="steelblue", label="Lomb-Scargle")
    axes[1].plot(periodo, bondad_entropia_norm, lw=0.7, color="darkorange", label="1 - entropia")
    axes[2].plot(periodo, producto, lw=0.7, color="seagreen", label="producto")

    colores_metodo = {
        "Lomb-Scargle": "steelblue", "1 - entropia": "darkorange", "producto": "seagreen",
        "cociente": "crimson", "rank_product": "purple", "suma_zscores": "brown",
        "emparejamiento_picos": "black",
    }
    for ax in axes:
        ax.axvline(periodo_catalogo, color="black", ls="--", lw=1)
        ax.set_xscale("log")

    for metodo, i_mejor in curvas_indice.items():
        ax = axes[0] if metodo == "Lomb-Scargle" else axes[1] if metodo == "1 - entropia" else axes[2]
        ax.axvline(periodo[i_mejor], color=colores_metodo[metodo], ls=":", lw=1,
                   label=f"P {metodo} = {periodo[i_mejor]:.4f} d")

    for ax in axes:
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Periodo (d) -- eje solo para lectura, la busqueda fue en frecuencia")
    fig.suptitle(f"Binaria {id_estrella}  (P catalogo = {periodo_catalogo:.5f} d)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"combinacion_{id_estrella}.png", dpi=150)
    plt.close(fig)


def main():
    random.seed(RANDOM_SEED)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    periodos_catalogo = cargar_catalogo()
    archivos = sorted((DATA_DIR / "binaries_phot").glob("*.dat"))
    ids_disponibles = [f.stem for f in archivos if f.stem in periodos_catalogo.index]

    resto = [i for i in ids_disponibles if i not in ESTRELLAS_FIJAS]
    muestra = ESTRELLAS_FIJAS + random.sample(resto, N_MUESTRA - len(ESTRELLAS_FIJAS))

    todas_filas = []
    for i, id_estrella in enumerate(tqdm(muestra, desc="Binaria", unit="estrella")):
        periodo_catalogo = float(periodos_catalogo.loc[id_estrella])
        filas, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice = analizar_estrella(id_estrella, periodo_catalogo)
        todas_filas.extend(filas)

        if i < N_EJEMPLOS_PLOT:
            graficar(id_estrella, periodo_catalogo, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice)

    resultados = pd.DataFrame(todas_filas)
    resultados.to_csv(RESULTADOS_CSV, index=False)

    print(f"\nMuestra: {len(muestra)} binarias ({', '.join(ESTRELLAS_FIJAS)} + {N_MUESTRA - len(ESTRELLAS_FIJAS)} aleatorias, semilla {RANDOM_SEED})")
    print("\nError relativo por metodo (mediana, y tasa de acierto <1%):")
    resumen = resultados.groupby("metodo").agg(
        mediana_error=("error_relativo", "median"),
        tasa_acierto_1pct=("error_relativo", lambda s: (s < 0.01).mean()),
        mediana_error_alias=("error_relativo_alias", "median"),
        tasa_acierto_alias_1pct=("error_relativo_alias", lambda s: (s < 0.01).mean()),
    ).sort_values("mediana_error_alias")
    print(resumen.to_string(float_format=lambda x: f"{x:.4f}"))

    tasa_coincidencia = resultados.loc[resultados["metodo"] == "emparejamiento_picos", "hubo_coincidencia"].mean()
    print(f"\nemparejamiento_picos encontro consenso en {tasa_coincidencia:.1%} de las binarias (el resto cayo de vuelta al producto)")

    print(f"\nTabla completa en {RESULTADOS_CSV}")
    print(f"Graficas de ejemplo en {PLOTS_DIR}")


if __name__ == "__main__":
    main()
