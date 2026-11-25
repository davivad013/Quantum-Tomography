"""Pacote de regularização entrópica para tomografia de estados quânticos."""

from entropy_reg.dual import (
    dual_cost,
    dual_gradient,
    entropia_von_neumann,
    normalizar_rho,
    rho_a_partir_de_u,
    stochastic_gradient,
)
from entropy_reg.methods.gradient_descent import gradient_descent, reconstruct_pi
from entropy_reg.methods.sgd import (
    reconstruct_spi,
    rodar_estocastico_com_historico,
    sgd,
    sgd_mem,
)
from entropy_reg.observables import (
    LETRAS,
    grad_completo,
    matriz_pauli,
    medidas_pauli,
    montar_observaveis_pauli,
    nome_pauli,
)
from entropy_reg.states import estado_gibbs, hamiltoniano_ising, rho_gibbs
from entropy_reg.utils import fmt_bytes

__all__ = [
    "LETRAS",
    "dual_cost",
    "dual_gradient",
    "entropia_von_neumann",
    "estado_gibbs",
    "fmt_bytes",
    "grad_completo",
    "gradient_descent",
    "hamiltoniano_ising",
    "matriz_pauli",
    "medidas_pauli",
    "montar_observaveis_pauli",
    "nome_pauli",
    "normalizar_rho",
    "reconstruct_pi",
    "reconstruct_spi",
    "rho_a_partir_de_u",
    "rho_gibbs",
    "rodar_estocastico_com_historico",
    "sgd",
    "sgd_mem",
    "stochastic_gradient",
]
