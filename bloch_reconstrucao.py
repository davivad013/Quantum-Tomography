"""
bloch_reconstrucao.py
=====================
Visualização na esfera de Bloch para estados quânticos de 1 qubit.
Mantém compatibilidade com scripts e exemplos existentes.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from quantum_tomography.visualization import (
    sigma_x,
    sigma_y,
    sigma_z,
    vetor_bloch,
    desenhar_esfera,
    desenhar_plano,
    plotar_bloch,
)

PLANOS = {
    "xz": (0, 2, "Plano x–z", (r"$|+\rangle$", r"$|-\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
    "xy": (0, 1, "Plano x–y", (r"$|+\rangle$", r"$|-\rangle$", r"$|{+i}\rangle$", r"$|{-i}\rangle$")),
    "yz": (1, 2, "Plano y–z", (r"$|{+i}\rangle$", r"$|{-i}\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
}


def desenhar_estado_3d(ax, r, cor, rotulo, marcador):
    ax.quiver(0, 0, 0, *r, color=cor, lw=2.5, arrow_length_ratio=0.10)
    ax.scatter(*r, color=cor, s=90, marker=marcador, edgecolor="k", zorder=10, label=rotulo)


__all__ = [
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "vetor_bloch",
    "desenhar_esfera",
    "desenhar_estado_3d",
    "PLANOS",
    "desenhar_plano",
    "plotar_bloch",
]
