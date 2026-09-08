"""
Utilidades de graficacion compartidas entre caracterizacion_lomb_scargle.py y
caracterizacion_entropia.py: marcadores de punto de interes en el borde del
recuadro (en vez de lineas verticales que tapan el pico o el minimo), plegado
de curvas de luz en fase, y lineas diagonales de multiplos de alias en los
diagramas de frecuencia encontrada vs. frecuencia catalogada.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ESTILOS_MULTIPLO = {2: "-.", 3: ":"}

# Tamano de fuente base ~60% mayor que el default de matplotlib (10 pt), para
# que el texto de las graficas sea comparable al del cuerpo de la tesis.
BASE_FONTSIZE = 16
plt.rcParams.update({
    "font.size": BASE_FONTSIZE,
    "axes.titlesize": BASE_FONTSIZE + 2,
    "axes.labelsize": BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE - 2,
    "ytick.labelsize": BASE_FONTSIZE - 2,
    "legend.fontsize": BASE_FONTSIZE - 3,
    "figure.titlesize": BASE_FONTSIZE + 4,
})


def marcar_punto_borde(ax, x, color, tam=11):
    """Marca x en el eje horizontal con un triangulo apuntando hacia arriba
    pegado al borde inferior y otro apuntando hacia abajo pegado al borde
    superior del recuadro, en vez de una linea vertical que atraviesa toda
    la grafica y tapa el pico (Lomb-Scargle) o el minimo (entropia)."""
    trans = ax.get_xaxis_transform()
    ax.plot(x, 0.0, marker="^", color=color, markersize=tam, transform=trans,
             clip_on=False, zorder=5)
    ax.plot(x, 1.0, marker="v", color=color, markersize=tam, transform=trans,
             clip_on=False, zorder=5)


def handle_punto(color, etiqueta, tam=11):
    """Handle de leyenda a juego con marcar_punto_borde (que no acepta
    directamente un argumento label sin duplicar la entrada en la leyenda)."""
    return Line2D([0], [0], marker="^", linestyle="None", color=color,
                  markersize=tam, label=etiqueta)


def plegar_fase(t, mag, periodo):
    """Fase en [0, 1) de cada observacion al plegar la curva de luz con
    `periodo`, ordenada de menor a mayor fase."""
    fase = np.mod((t - t[0]) / periodo, 1.0)
    orden = np.argsort(fase)
    return fase[orden], mag[orden]


def panel_curva_luz(ax, t, mag, periodo, color, titulo):
    """Un solo panel de curva de luz plegada en fase (0 a 2 ciclos) con un
    periodo dado. Se usa dos veces por estrella (una vez con el periodo
    catalogado, otra con el periodo encontrado) en paneles separados, en vez
    de superponer ambas curvas en un mismo panel."""
    fase, m = plegar_fase(t, mag, periodo)
    ax.scatter(fase, m, s=8, alpha=0.6, color=color)
    ax.scatter(fase + 1, m, s=8, alpha=0.6, color=color)
    ax.invert_yaxis()
    ax.set_xlim(0, 2)
    ax.set_xlabel("Fase")
    ax.set_ylabel("Magnitud")
    # Fontsize explicito, mas pequeno que el de los titulos de eje principales:
    # estos paneles son angostos (comparten fila con otro panel) y un titulo
    # largo al tamano de fuente base desborda el ancho del panel.
    ax.set_title(titulo, fontsize=14)


def agregar_lineas_alias(ax, lims, multiplos=ESTILOS_MULTIPLO):
    """Ademas de la identidad f_encontrada = f_catalogo (dibujada aparte por
    quien llama), agrega lineas diagonales para los multiplos m\'as comunes
    del periodo (equivalentes, en frecuencia, a f = f_cat/n y f = n*f_cat)."""
    for n, estilo in multiplos.items():
        ax.plot(lims, [l * n for l in lims], color="gray", lw=0.7, ls=estilo,
                 label=f"multiplos x{n} (P/{n}, {n}P)")
        ax.plot(lims, [l / n for l in lims], color="gray", lw=0.7, ls=estilo)
