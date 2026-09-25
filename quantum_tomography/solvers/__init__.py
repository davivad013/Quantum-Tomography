r"""
quantum_tomography.solvers
==========================
Interface unificada para todos os solvers de tomografia quântica:
- Regularização Entrópica (sdplab / spacecore / JAX)
- Regularização Quadrática (sdplab / spacecore / JAX)
- SDP Linear Padrão (CVXPY / SCS)
- SDP Mínimos Quadrados Convexos (CVXPY)
- Força Bruta por Mínimos Quadrados (Cholesky T^\dagger T / SciPy)
"""

from __future__ import annotations
from typing import Sequence, Any
import numpy as np

from .base import TomographyResult
from .sdp_solver import SDPTomographySolver
from .brute_force import (
    resolve_tomografia_lsq,
    parametros_para_T_geral,
    parametros_para_rho_geral,
    funcao_de_custo_geral,
    jacobiano_analitico,
)

_SDP_SOLVER_INSTANCE: SDPTomographySolver | None = None


def get_sdp_solver() -> SDPTomographySolver:
    global _SDP_SOLVER_INSTANCE
    if _SDP_SOLVER_INSTANCE is None:
        _SDP_SOLVER_INSTANCE = SDPTomographySolver()
    return _SDP_SOLVER_INSTANCE


def solve_tomography(
    observables: Sequence[str | np.ndarray],
    q_measured: Sequence[float] | np.ndarray,
    method: str = "entropy",
    H: np.ndarray | None = None,
    true_state: Any | None = None,
    n_qubits: int | None = None,
    eps: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    verbose: int | bool = 0,
    **kwargs,
) -> TomographyResult:
    """
    Ponto de entrada unificado para resolver o problema de tomografia quântica.

    Parâmetros:
    -----------
    observables : lista de strings ('X', 'Y', 'ZZ') ou lista de matrizes densas
    q_measured  : vetor de medições q_i = Tr(Q_i * rho)
    method      : 'entropy' (ou 'sdp_entropy', 'vn') -> Regularização Entrópica MaxEnt (sdplab)
                  'quadratic' (ou 'sdp_quadratic', 'l2') -> Regularização Quadrática L2 (sdplab)
                  'cvxpy_scs' (ou 'scs', 'sdp_linear') -> SDP Linear CVXPY SCS
                  'cvxpy_lsq' (ou 'convexa_lsq') -> SDP Mínimos Quadrados Convexos CVXPY
                  'least_squares' (ou 'brute_force', 'forca_bruta') -> Parametrização T^dag T SciPy
    H           : Hamiltoniano físico opcional para tomografia com energia
    true_state  : estado quântico alvo para cálculo de fidelidade e distâncias
    eps         : parâmetro de escala da regularização (epsilon)
    tol         : tolerância de convergência do resíduo
    max_iter    : número máximo de iterações do otimizador
    """
    method_clean = method.lower().strip()
    sdp = get_sdp_solver()

    if method_clean in ("entropy", "sdp_entropy", "vn", "maxent", "entropia"):
        return sdp.solve_entropy(
            observables,
            q_measured,
            H=H,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            verbose=int(verbose),
            true_state=true_state,
        )

    if method_clean in ("quadratic", "sdp_quadratic", "l2", "frobenius", "quadratico"):
        return sdp.solve_quadratic(
            observables,
            q_measured,
            H=H,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            verbose=int(verbose),
            true_state=true_state,
        )

    if method_clean in ("cvxpy_scs", "scs", "sdp_linear", "linear"):
        return sdp.solve_cvxpy_scs(
            observables,
            q_measured,
            H=H,
            verbose=bool(verbose),
            true_state=true_state,
        )

    if method_clean in ("cvxpy_lsq", "cvxpy_least_squares", "convexa_lsq", "lsq_convexa"):
        lambda_e = kwargs.get("lambda_energy", 0.0)
        return sdp.solve_cvxpy_least_squares(
            observables,
            q_measured,
            H=H,
            lambda_energy=lambda_e,
            verbose=bool(verbose),
            true_state=true_state,
        )

    if method_clean in ("least_squares", "brute_force", "forca_bruta", "cholesky", "lsq"):
        return resolve_tomografia_lsq(
            observables,
            q_measured,
            n_qubits=n_qubits,
            estado_verdadeiro=true_state,
            H=H,
            **kwargs,
        )

    raise ValueError(
        f"Método de tomografia desconhecido: '{method}'. "
        f"Opções disponíveis: 'entropy', 'quadratic', 'cvxpy_scs', 'cvxpy_lsq', 'least_squares'."
    )


__all__ = [
    "TomographyResult",
    "SDPTomographySolver",
    "get_sdp_solver",
    "solve_tomography",
    "resolve_tomografia_lsq",
    "parametros_para_T_geral",
    "parametros_para_rho_geral",
    "funcao_de_custo_geral",
    "jacobiano_analitico",
]
