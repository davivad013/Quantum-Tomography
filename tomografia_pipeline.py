"""
tomografia_pipeline.py
======================
Módulo de compatibilidade e pipeline de tomografia quântica.
Encaminha as operações para a arquitetura unificada de `quantum_tomography`
preservando compatibilidade total com códigos, notebooks e scripts anteriores.
"""

from __future__ import annotations
import numpy as np

# Re-exporta definições centrais do novo pacote modular
from quantum_tomography import (
    QuantState,
    sigma_x,
    sigma_y,
    sigma_z,
    gerar_paulis,
    fidelidade,
    simula_medidas_shots,
    medidas_shots_de_estado,
    resolve_tomografia_lsq,
    parametros_para_T_geral,
    parametros_para_rho_geral,
    funcao_de_custo_geral,
    jacobiano_analitico,
    solve_tomography,
    TomographyResult,
)
from quantum_tomography.states import sqrt_psd

__all__ = [
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "QuantState",
    "gerar_paulis",
    "parametros_para_T_geral",
    "parametros_para_rho_geral",
    "funcao_de_custo_geral",
    "sqrt_psd",
    "fidelidade",
    "jacobiano_analitico",
    "simula_medidas_shots",
    "medidas_shots_de_estado",
    "resolve_tomografia_lsq",
    "solve_tomography",
    "TomographyResult",
]
