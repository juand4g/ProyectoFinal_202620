"""
Exploracion (no una caracterizacion formal): pruebas creativas combinando
Lomb-Scargle (astropy) y minima entropia de Shannon (entropypf) en una
misma grilla de FRECUENCIA, para ver si la combinacion suprime los alias
propios de cada metodo (P/2 en Lomb-Scargle; alias diario/anual en minima
entropia).

Convencion de todo el archivo: se trabaja siempre en frecuencia (grilla
uniforme, busqueda de picos, combinacion de curvas). La conversion a periodo
ocurre en dos puntos nada mas, ambos inevitables o finales:
  1. get_entropies de entropypf exige periodos como entrada (es el contrato
     de esa funcion) -- se construyen como 1/frecuencia justo antes de
     llamarla, y el resultado se vuelve a indexar por frecuencia.
  2. Al reportar/graficar el resultado final (el "mejor" punto de cada
     curva), donde periodo = 1 / frecuencia.

Metodos de combinacion implementados (elegidos con el usuario):
  - producto:      LS_norm(f) * (1 - entropia_norm(f))
  - geometrica:     sqrt(LS_norm(f) * (1 - entropia_norm(f)))
  - convolucion:   convolucion discreta de las dos curvas normalizadas
                    (mode="same"); el indice de salida se interpreta de forma
                    aproximada como alineado con la grilla de frecuencia de
                    entrada -- es la combinacion mas exploratoria de las tres
                    y no se espera que supere a las otras dos.

No pretende reemplazar caracterizacion_lomb_scargle.py ni
caracterizacion_entropia.py: corre sobre un puñado de estrellas ya conocidas
de esos dos scripts (algunas "faciles", algunas "dificiles" para cada
metodo), no sobre una muestra estadistica completa.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from entropypf import get_entropies

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
N_FREQ = 50_000  # puntos de la grilla UNIFORME en frecuencia (ciclos/dia)
L, K = 7, 7  # grilla fase-magnitud de la entropia

DATA_DIR = Path(__file__).resolve().parent.parent / "ogle_collection"
OUT_DIR = Path(__file__).resolve().parent.parent / "Combinacion_LS_Entropia"
PLOTS_DIR = OUT_DIR / "graficas"
RESULTADOS_CSV = OUT_DIR / "resultados_combinacion.csv"

# Un puñado de estrellas ya vistas en caracterizacion_lomb_scargle.py /
# caracterizacion_entropia.py: una "facil" y una "dificil" por tipo, con el
# mismo rango de busqueda de periodo (dias) usado en esos scripts.
ESTRELLAS_PRUEBA = [
    dict(tipo="Binaria", id="OGLE-LMC-ECL-07968", carpeta="binaries_phot",
         p_cat=4.4259378, p_min=0.1, p_max=100.0,
         nota="dificil: minima entropia converge a alias diario/anual"),
    dict(tipo="Binaria", id="OGLE-LMC-ECL-01807", carpeta="binaries_phot",
         p_cat=4.6814202, p_min=0.1, p_max=100.0,
         nota="dificil: Lomb-Scargle tiende a P/2"),
    dict(tipo="Cefeida", id="OGLE-LMC-CEP-0275", carpeta="classical_cefeids_phot",
         p_cat=3.0778447, p_min=0.2, p_max=250.0,
         nota="facil: ambos metodos aciertan por separado"),
    dict(tipo="Cefeida", id="OGLE-SMC-CEP-4477", carpeta="classical_cefeids_phot",
         p_cat=1.1796847, p_min=0.2, p_max=250.0,
         nota="dificil: minima entropia se equivoca sola"),
    dict(tipo="RR Lyrae", id="OGLE-LMC-RRLYR-02106", carpeta="rrlyrae_phot",
         p_cat=0.5820551, p_min=0.1, p_max=1.2,
         nota="facil: ambos metodos aciertan por separado"),
    dict(tipo="RR Lyrae", id="OGLE-LMC-RRLYR-28002", carpeta="rrlyrae_phot",
         p_cat=0.3813285, p_min=0.1, p_max=1.2,
         nota="dificil: ambos metodos se equivocan solos"),
]


def normalizar(x):
    x_min, x_max = np.min(x), np.max(x)
    if x_max <= x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def mejor_alias(periodo_encontrado, periodo_catalogo):
    """Igual que en los otros dos scripts: el mejor entre P, P/2, 2P, P/3, 3P."""
    candidatos = [periodo_encontrado, periodo_encontrado / 2, periodo_encontrado * 2,
                  periodo_encontrado / 3, periodo_encontrado * 3]
    errores = [abs(p - periodo_catalogo) / periodo_catalogo for p in candidatos]
    i = int(np.argmin(errores))
    return candidatos[i], errores[i]


def combinar(ls_norm, bondad_entropia_norm):
    """Todas las curvas de entrada y salida estan indexadas por frecuencia,
    en el mismo orden que la grilla `frecuencia` de main(). `ls_norm` y
    `bondad_entropia_norm` ya estan normalizadas a [0, 1] con orientacion
    "mas alto = mejor periodo" (bondad_entropia_norm = 1 - entropia_norm)."""
    combinaciones = {}

    combinaciones["producto"] = ls_norm * bondad_entropia_norm

    combinaciones["geometrica"] = np.sqrt(ls_norm * bondad_entropia_norm)

    conv = np.convolve(ls_norm, bondad_entropia_norm, mode="same")
    combinaciones["convolucion"] = normalizar(conv)

    return combinaciones


def analizar_estrella(estrella):
    path_dat = DATA_DIR / estrella["carpeta"] / f"{estrella['id']}.dat"
    t, mag, err = np.loadtxt(path_dat, unpack=True)

    # --- grilla UNIFORME en frecuencia (ciclos/dia) -------------------------
    f_min = 1.0 / estrella["p_max"]
    f_max = 1.0 / estrella["p_min"]
    frecuencia = np.linspace(f_min, f_max, N_FREQ)

    # --- Lomb-Scargle: nativamente evaluado en frecuencia -------------------
    ls_potencia = LombScargle(t, mag, err).power(frecuencia)

    # --- minima entropia: get_entropies exige PERIODOS como entrada (unico
    # punto del calculo, aparte del reporte final, donde aparece el periodo) --
    periodos_para_entropia = 1.0 / frecuencia
    entropia = get_entropies(t, mag, periodos_para_entropia, L=L, K=K)

    ls_norm = normalizar(ls_potencia)
    entropia_norm = normalizar(entropia)
    bondad_entropia_norm = 1.0 - entropia_norm  # mas alto = mejor, igual orientacion que LS

    curvas = {"Lomb-Scargle": ls_norm, "1 - entropia": bondad_entropia_norm}
    curvas.update(combinar(ls_norm, bondad_entropia_norm))

    filas = []
    for metodo, curva in curvas.items():
        i_mejor = int(np.argmax(curva))
        periodo_encontrado = 1.0 / frecuencia[i_mejor]  # unica conversion a periodo del resultado
        error_relativo = abs(periodo_encontrado - estrella["p_cat"]) / estrella["p_cat"]
        periodo_alias, error_alias = mejor_alias(periodo_encontrado, estrella["p_cat"])
        filas.append({
            "tipo": estrella["tipo"], "id": estrella["id"], "nota": estrella["nota"],
            "metodo": metodo, "periodo_catalogo": estrella["p_cat"],
            "periodo_encontrado": periodo_encontrado, "error_relativo": error_relativo,
            "periodo_alias": periodo_alias, "error_relativo_alias": error_alias,
        })

    graficar(estrella, frecuencia, curvas, filas)
    return filas


def graficar(estrella, frecuencia, curvas, filas):
    periodo = 1.0 / frecuencia  # solo para el eje x de la grafica
    errores_por_metodo = {f["metodo"]: f["periodo_encontrado"] for f in filas}

    fig, axes = plt.subplots(len(curvas), 1, figsize=(8, 2.1 * len(curvas)), sharex=True)
    colores = {
        "Lomb-Scargle": "steelblue", "1 - entropia": "darkorange",
        "producto": "seagreen", "geometrica": "purple", "convolucion": "crimson",
    }

    for ax, (metodo, curva) in zip(axes, curvas.items()):
        ax.plot(periodo, curva, lw=0.7, color=colores.get(metodo, "black"))
        ax.axvline(estrella["p_cat"], color="black", ls="--", lw=1, label="P catalogo")
        ax.axvline(errores_por_metodo[metodo], color=colores.get(metodo, "black"), ls=":", lw=1,
                   label=f"P {metodo}")
        ax.set_xscale("log")
        ax.set_ylabel(metodo, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Periodo (d) -- eje solo para lectura, la busqueda fue en frecuencia")
    fig.suptitle(f"{estrella['tipo']} - {estrella['id']}  (P catalogo = {estrella['p_cat']:.5f} d; {estrella['nota']})",
                 fontsize=10)
    fig.tight_layout()
    nombre = f"combinacion_{estrella['tipo'].replace(' ', '_')}_{estrella['id']}.png"
    fig.savefig(PLOTS_DIR / nombre, dpi=150)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    todas_filas = []
    for estrella in ESTRELLAS_PRUEBA:
        print(f"Procesando {estrella['tipo']} {estrella['id']} ({estrella['nota']})...")
        todas_filas.extend(analizar_estrella(estrella))

    resultados = pd.DataFrame(todas_filas)
    resultados.to_csv(RESULTADOS_CSV, index=False)

    print("\nError relativo por metodo (mediana sobre las 6 estrellas de prueba):")
    print(resultados.groupby("metodo")["error_relativo"].median().sort_values().to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nError relativo con correccion de alias (P, P/2, 2P, P/3, 3P):")
    print(resultados.groupby("metodo")["error_relativo_alias"].median().sort_values().to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"\nTabla completa en {RESULTADOS_CSV}")
    print(f"Graficas en {PLOTS_DIR}")


if __name__ == "__main__":
    main()
