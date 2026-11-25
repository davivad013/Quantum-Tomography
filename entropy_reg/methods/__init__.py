"""Subpacote com os métodos de otimização para regularização entrópica."""

from entropy_reg.methods.gradient_descent import gradient_descent, reconstruct_pi
from entropy_reg.methods.sgd import (
    reconstruct_spi,
    rodar_estocastico_com_historico,
    sgd,
    sgd_mem,
)

__all__ = [
    "gradient_descent",
    "reconstruct_pi",
    "sgd",
    "reconstruct_spi",
    "rodar_estocastico_com_historico",
    "sgd_mem",
]
