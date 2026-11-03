"""
Animacion de la tasa de aciertos sobre curvas sinteticas en funcion de la
fraccion de puntos de la cadencia (1% a 100%), a partir del barrido de
robustez_sinteticas.py --barrido (720 plantillas x SNR {4, 25, 50} x
FRACCIONES_BARRIDO).

Cada cuadro corresponde a una fraccion:
  - izquierda: acierto final vs. SNR para cada decision de rama (x2+BIC en la
    rama de binarias), como en unificado_sinteticas.png;
  - derecha: acierto del metodo unificado vs. fraccion, una linea por SNR,
    dibujada hasta la fraccion del cuadro.
Las curvas que no pudieron analizarse (muy pocos puntos) cuentan como fallos:
el denominador es siempre el numero de plantillas de la clase.

Salida: Robustez_Sinteticas/graficas/sinteticas_aciertos_fraccion.gif
(copiada a Tesis/img/) y un cuadro fijo con todas las fracciones,
sinteticas_aciertos_fraccion.png.
"""

import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from graficas_metodo_unificado import AGUA, AZUL, GRAF_DIR, NARANJA, TESIS_IMG, TINTA, recuadro
from refinamiento_subarmonicos_bic import acierto, n_hibrido
from robustez_sinteticas import (
    CACHE_CSV, FRACCIONES_BARRIDO, H0, SNRS, UMBRAL_LOGISTICA, UMBRAL_SKEW, con_fraccion,
    elegir_plantillas, entrenar_logistica, especificaciones, matriz_logistica,
)

warnings.filterwarnings("ignore")

NOMBRE = "sinteticas_aciertos_fraccion"
SERIES = [("oraculo", "Rama perfecta", TINTA, "--", "s"),
          ("skew corr.>0.5", "Asimetría corregida", AGUA, "-", "o"),
          ("skew>0.5", "Asimetría sin corregir", NARANJA, "-", "^"),
          ("logistica>0.7", "Regresión logística", AZUL, "-", "D")]
COLOR_SNR = {4: "#86b6ef", 25: "#2a78d6", 50: "#184f95"}
GRUPOS = [("Binarias", ["Binaria"]), ("Pulsantes", ["Cefeida", "RR Lyrae"])]


def tasas():
    """tasa[(regla, grupo, snr, fraccion)] en %, con las curvas ausentes como fallos."""
    plantillas = elegir_plantillas()
    ids = {s["id"] for s in especificaciones(plantillas, FRACCIONES_BARRIDO)}
    df = con_fraccion(pd.read_csv(CACHE_CSV))
    df = df[df["id"].isin(ids)].reset_index(drop=True)
    df["p0"] = df["periodo_rp_c"]
    p0 = df["p0"].to_numpy()
    P = df["periodo_catalogo"].to_numpy()
    ok_bin = acierto(n_hibrido(df, H0, "bic_rp_c") * p0, P)
    ok_p0 = acierto(p0, P)
    es_bin = (df["clase"] == "Binaria").to_numpy()
    decisiones = {
        "oraculo": es_bin,
        "skew>0.5": df["skew_mag"].to_numpy() > UMBRAL_SKEW,
        "skew corr.>0.5": df["skew_corregida"].fillna(-np.inf).to_numpy() > UMBRAL_SKEW,
        "logistica>0.7": entrenar_logistica().predict_proba(matriz_logistica(df))[:, 1] > UMBRAL_LOGISTICA,
    }
    n_clase = plantillas["clase"].value_counts()
    salida = {}
    for clave, *_ in SERIES:
        ok = np.where(decisiones[clave], ok_bin, ok_p0)
        for nombre, clases in GRUPOS:
            total = sum(n_clase[c] for c in clases)
            en_grupo = df["clase"].isin(clases).to_numpy()
            for snr in SNRS:
                for f in FRACCIONES_BARRIDO:
                    m = en_grupo & (df["snr"] == snr).to_numpy() & np.isclose(df["fraccion"], f)
                    salida[(clave, nombre, snr, f)] = 100 * ok[m].sum() / total
    faltan = len(ids) - len(df)
    mediana = {f: df.loc[np.isclose(df["fraccion"], f), "n_puntos"].median() for f in FRACCIONES_BARRIDO}
    return salida, mediana, faltan


def dibujar(ejes, tasa, mediana, k):
    f = FRACCIONES_BARRIDO[k]
    vistas = FRACCIONES_BARRIDO[:k + 1]
    for fila, (grupo, _) in enumerate(GRUPOS):
        ax, ax_f = ejes[fila]
        ax.clear()
        ax_f.clear()
        for clave, etiqueta, color, estilo, marcador in SERIES:
            ax.plot(SNRS, [tasa[(clave, grupo, s, f)] for s in SNRS], color=color, linestyle=estilo,
                    marker=marcador, markersize=8, linewidth=2, label=etiqueta)
        ax.set_xscale("log")
        ax.set_xticks(SNRS, [str(s) for s in SNRS])
        ax.minorticks_off()
        ax.set_ylim(0, 102)
        ax.set_ylabel("Aciertos (%)")
        ax.set_title(f"({'ac'[fila]}) {grupo}, {f:.0%} de los puntos", loc="left", fontsize=15)
        recuadro(ax)

        for s in reversed(SNRS):
            y = [tasa[("skew corr.>0.5", grupo, s, x)] for x in vistas]
            ax_f.plot([100 * x for x in vistas], y, color=COLOR_SNR[s], linewidth=2,
                      marker="o", markersize=5, label=rf"$\mathrm{{SNR}}={s}$")
            ax_f.plot(100 * f, y[-1], "o", color=COLOR_SNR[s], markersize=10)
        ax_f.set_xscale("log")
        ax_f.set_xlim(0.8, 125)
        ax_f.set_xticks([1, 2, 5, 10, 25, 50, 100], ["1", "2", "5", "10", "25", "50", "100"])
        ax_f.minorticks_off()
        ax_f.set_ylim(0, 102)
        ax_f.set_title(f"({'bd'[fila]}) {grupo}, método unificado", loc="left", fontsize=15)
        recuadro(ax_f)
        if fila == 1:
            ax.set_xlabel(r"Señal a ruido $A_{\mathrm{pp}}/\sigma$")
            ax_f.set_xlabel("Puntos de la cadencia (%)")
    ejes[0][0].legend(loc="lower right", frameon=False, fontsize=11)
    ejes[0][1].legend(loc="lower right", frameon=False, fontsize=11)
    ejes[0][0].figure.suptitle(f"{f:.0%} de los puntos de la cadencia (mediana de {mediana[f]:.0f} puntos por curva)",
                               fontsize=16)


def main():
    tasa, mediana, faltan = tasas()
    print(f"Curvas sin analizar (cuentan como fallo): {faltan}")
    for grupo, _ in GRUPOS:
        for s in SNRS:
            print(grupo, s, " ".join(f"{100 * f:g}%:{tasa[('skew corr.>0.5', grupo, s, f)]:.1f}"
                                     for f in FRACCIONES_BARRIDO))

    fig, ejes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.2, top=0.9)
    cuadros = list(range(len(FRACCIONES_BARRIDO))) + [len(FRACCIONES_BARRIDO) - 1] * 4  # pausa al final
    animacion = FuncAnimation(fig, lambda k: dibujar(ejes, tasa, mediana, k), frames=cuadros)
    GRAF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = GRAF_DIR / f"{NOMBRE}.gif"
    animacion.save(ruta, writer=PillowWriter(fps=2), dpi=90)
    dibujar(ejes, tasa, mediana, len(FRACCIONES_BARRIDO) - 1)
    fig.savefig(GRAF_DIR / f"{NOMBRE}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(ruta, TESIS_IMG / ruta.name)


if __name__ == "__main__":
    main()
