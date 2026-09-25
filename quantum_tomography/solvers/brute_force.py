r"""
quantum_tomography.solvers.brute_force
======================================
Solver de Força Bruta por Mínimos Quadrados Não-Lineares com parametrização de Cholesky
(T^\dagger T / \operatorname{Tr}(T^\dagger T)) e Jacobiano analítico exato.
"""

from __future__ import annotations
import time
from typing import Sequence, Any
import numpy as np
from scipy.optimize import least_squares

from .base import TomographyResult
from ..observables import pauli_string_to_matrix
from ..states import fidelidade, frobenius_distance, trace_distance, _symmetrize_hermitian


def parametros_para_T_geral(params: np.ndarray, n_qubits: int) -> np.ndarray:
    """Constrói a matriz triangular inferior T para N qubits (dim = 2^N)."""
    D = 2 ** n_qubits
    li, lj = np.tril_indices(D, k=-1)
    n_off = len(li)
    T = np.zeros((D, D), dtype=complex)
    T[np.arange(D), np.arange(D)] = params[:D]
    T[li, lj] = params[D : D + n_off] + 1j * params[D + n_off : D + 2 * n_off]
    return T


def parametros_para_rho_geral(params: np.ndarray, n_qubits: int) -> np.ndarray:
    r"""
    Converte os D^2 parâmetros reais na matriz densidade física correspondente:
        \rho = \frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}.
    Garante que \rho seja automaticamente hermitiana, positiva semidefinida e com traço unitário.
    """
    T = parametros_para_T_geral(params, n_qubits)
    T_dag_T = T.conj().T @ T
    traco = np.trace(T_dag_T).real
    if np.isclose(traco, 0.0):
        D = 2 ** n_qubits
        return np.eye(D, dtype=complex) / D
    return _symmetrize_hermitian(T_dag_T / traco)


def funcao_de_custo_geral(
    params: np.ndarray,
    mediadores_Q: Sequence[np.ndarray],
    dados_exp: np.ndarray,
    n_qubits: int,
    H: np.ndarray | None = None,
    target_energy: float | None = None,
    lambda_energy: float = 0.0,
) -> np.ndarray:
    """Vetor de resíduos: r_i = Tr(rho(T) * Q_i) - q_i, com suporte a termo de energia opcional."""
    rho_candidato = parametros_para_rho_geral(params, n_qubits)
    Q = np.asarray(mediadores_Q)
    previsoes = np.einsum("ij,mji->m", rho_candidato, Q).real
    residuos = previsoes - dados_exp

    if H is not None and lambda_energy > 0 and target_energy is not None:
        e_cand = float(np.trace(rho_candidato @ H).real)
        res_energy = np.sqrt(lambda_energy) * (e_cand - target_energy)
        residuos = np.append(residuos, res_energy)

    return residuos


def jacobiano_analitico(
    params: np.ndarray,
    mediadores_Q: Sequence[np.ndarray],
    dados_exp: np.ndarray,
    n_qubits: int,
    H: np.ndarray | None = None,
    target_energy: float | None = None,
    lambda_energy: float = 0.0,
) -> np.ndarray:
    r"""
    Jacobiano analítico exato do vetor de resíduos r(x) em relação aos parâmetros de Cholesky.

    Dedução Matemática Completa (Regra da Cadeia Matricial):
    -------------------------------------------------------
    1. Parametrização de Cholesky:
       Seja D = 2^N a dimensão do espaço de Hilbert. A matriz triangular inferior T(x) possui:
         - D elementos reais na diagonal principal: T_{kk} = x_k (k = 0, ..., D-1)
         - n_off = D(D-1)/2 elementos complexos sub-diagonais:
           T_{ij} = x_{D + k} + i * x_{D + n_off + k}  (com i > j).
       O vetor total de parâmetros reais livres tem dimensão D^2 = 4^N.

    2. Matriz densidade e normalização do traço:
       rho(T) = (T^\dagger T) / s,   onde s = Tr(T^\dagger T) = \sum_{i,j} |T_{ij}|^2.
       A variação do traço de normalização sob uma perturbação dT é:
         ds = Tr(dT^\dagger T + T^\dagger dT) = 2 Re Tr(T^\dagger dT).

    3. Resíduo de medição para cada observável Q_m:
       r_m(x) = f_m(T) - q_m,   com f_m(T) = Tr[rho(T) Q_m] = (1/s) Tr[Q_m T^\dagger T].

    4. Diferencial total pela regra do quociente:
       df_m = (1/s) d(Tr[Q_m T^\dagger T]) - (Tr[Q_m T^\dagger T] / s^2) ds
            = (1/s) { 2 Re Tr[Q_m T^\dagger dT] - f_m * 2 Re Tr[T^\dagger dT] }
            = (2/s) Re Tr[ (Q_m T^\dagger - f_m T^\dagger) dT ].

    5. Expressão para cada componente do gradiente via M_m = Q_m T^\dagger:
       - Para a diagonal real t_k = T_{kk}:
         \partial f_m / \partial T_{kk} = (2/s) Re[ (M_m)_{kk} - f_m (T^\dagger)_{kk} ].
       - Para a parte real do elemento subdiagonal t_{ij}^R = Re(T_{ij}) (i > j):
         dT_{ij} = dt_{ij}^R  ==>  Tr[(M_m - f_m T^\dagger) dT] = (M_m - f_m T^\dagger)_{ji}
         \partial f_m / \partial t_{ij}^R = (2/s) Re[ (M_m)_{ji} - f_m (T^\dagger)_{ji} ].
       - Para a parte imaginária do elemento subdiagonal t_{ij}^I = Im(T_{ij}) (i > j):
         dT_{ij} = i * dt_{ij}^I  ==>  Tr[(M_m - f_m T^\dagger) dT] = i * (M_m - f_m T^\dagger)_{ji}
         \partial f_m / \partial t_{ij}^I = (2/s) Re[ i * ( (M_m)_{ji} - f_m (T^\dagger)_{ji} ) ].

    6. Termo opcional de energia:
       Se um Hamiltoniano H com peso lambda_energy estiver ativo, o resíduo adicional
       r_E = \sqrt{\lambda_E} (Tr[rho H] - E_alvo) é tratado como uma linha extra do Jacobiano
       com Q_{extra} = \sqrt{\lambda_E} H.

    O resultado é a matriz Jacobiana J \in \mathbb{R}^{M \times D^2} onde J_{m, k} = \partial r_m / \partial x_k.
    """
    D = 2 ** n_qubits
    li, lj = np.tril_indices(D, k=-1)

    T = parametros_para_T_geral(params, n_qubits)
    T_dag = T.conj().T
    s = np.trace(T_dag @ T).real
    if np.isclose(s, 0.0):
        s = 1e-12

    Q_list = list(mediadores_Q)
    if H is not None and lambda_energy > 0 and target_energy is not None:
        Q_list.append(np.sqrt(lambda_energy) * H)

    Q = np.asarray(Q_list)
    rho = T_dag @ T / s
    previsoes = np.einsum("ij,mji->m", rho, Q).real
    M_all = np.einsum("mij,jk->mik", Q, T_dag)

    tr_diag = np.diagonal(M_all, axis1=1, axis2=2)
    tr_re = M_all[:, lj, li]
    tr_im = 1j * M_all[:, lj, li]

    g_diag = np.diagonal(T_dag)
    g_re = T_dag[lj, li]
    g_im = 1j * T_dag[lj, li]

    jac_diag = (2.0 / s) * (tr_diag.real - previsoes[:, None] * g_diag.real[None, :])
    jac_re = (2.0 / s) * (tr_re.real - previsoes[:, None] * g_re.real[None, :])
    jac_im = (2.0 / s) * (tr_im.real - previsoes[:, None] * g_im.real[None, :])

    return np.hstack([jac_diag, jac_re, jac_im])


def resolve_tomografia_lsq(
    mediadores_Q: Sequence[str | np.ndarray],
    q_medidos: Sequence[float] | np.ndarray,
    n_qubits: int | None = None,
    chute_inicial: np.ndarray | None = None,
    estado_verdadeiro: Any | None = None,
    usar_jacobiano_analitico: bool = True,
    H: np.ndarray | None = None,
    target_energy: float | None = None,
    lambda_energy: float = 0.0,
    seed: int | None = None,
    **kwargs_least_squares,
) -> TomographyResult:
    r"""
    Resolve a tomografia por mínimos quadrados não-lineares com a parametrização de Cholesky.

    Parâmetros:
    -----------
    mediadores_Q : lista de observáveis (strings como 'XX' ou matrizes densas)
    q_medidos    : valores esperados medidos
    n_qubits     : número de qubits (se None, infere a partir da dimensão dos observáveis)
    chute_inicial: vetor de dimensão D^2 (se None, gerado aleatoriamente)
    estado_verdadeiro: QuantState ou matriz rho para cálculo de fidelidade
    usar_jacobiano_analitico: ativa o Jacobiano analítico exato (muito mais rápido que diferenças finitas)
    """
    t0 = time.time()
    # Converte strings para matrizes se necessário
    mats = [pauli_string_to_matrix(obs) if isinstance(obs, str) else np.asarray(obs, dtype=complex) for obs in mediadores_Q]
    dim = mats[0].shape[0]

    if n_qubits is None:
        n_qubits = int(np.log2(dim))

    q_arr = np.asarray(q_medidos, dtype=float)
    num_params = dim ** 2

    if chute_inicial is None:
        rng = np.random.default_rng(seed)
        chute_inicial = rng.normal(size=num_params)

    jac = jacobiano_analitico if usar_jacobiano_analitico else "2-point"

    if "jac" in kwargs_least_squares:
        jac = kwargs_least_squares.pop("jac")

    # Para dim >= 16 (4 qubits), tr_solver='lsmr' evita falhas de convergência da SVD densa do SciPy
    if "tr_solver" not in kwargs_least_squares:
        kwargs_least_squares["tr_solver"] = "lsmr" if dim >= 16 else "exact"

    cost_history = []
    def funcao_rastreada(p, *args):
        res = funcao_de_custo_geral(p, *args)
        cost_history.append(float(0.5 * np.sum(res ** 2)))
        return res

    try:
        resultado = least_squares(
            funcao_rastreada,
            x0=chute_inicial,
            args=(mats, q_arr, n_qubits, H, target_energy, lambda_energy),
            jac=jac,
            **kwargs_least_squares,
        )
    except np.linalg.LinAlgError:
        kwargs_least_squares["tr_solver"] = "lsmr"
        resultado = least_squares(
            funcao_rastreada,
            x0=chute_inicial,
            args=(mats, q_arr, n_qubits, H, target_energy, lambda_energy),
            jac=jac,
            **kwargs_least_squares,
        )

    elapsed = time.time() - t0
    rho_rec = parametros_para_rho_geral(resultado.x, n_qubits)

    # Resíduo nas medições originais
    preds = np.array([float(np.trace(rho_rec @ Q).real) for Q in mats])
    residual = float(np.linalg.norm(preds - q_arr))

    fid, t_dist, f_dist = None, None, None
    if estado_verdadeiro is not None:
        rho_true = estado_verdadeiro.rho if hasattr(estado_verdadeiro, "rho") else np.asarray(estado_verdadeiro, dtype=complex)
        fid = fidelidade(rho_rec, rho_true)
        t_dist = trace_distance(rho_rec, rho_true)
        f_dist = frobenius_distance(rho_rec, rho_true)

    energy = float(np.trace(rho_rec @ H).real) if H is not None else None

    return TomographyResult(
        rho=rho_rec,
        method="Força Bruta (Cholesky LSQ / SciPy)",
        success=bool(resultado.success),
        status=str(resultado.message),
        num_iters=int(resultado.nfev),
        elapsed_seconds=elapsed,
        residual=residual,
        fidelity=fid,
        energy=energy,
        trace_dist=t_dist,
        frobenius_dist=f_dist,
        cost_or_loss=float(resultado.cost),
        cost_history=cost_history,
        extra={"resultado_scipy": resultado, "chute_inicial": chute_inicial, "cost_history": cost_history},
    )
