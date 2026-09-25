"""
quantum_tomography.solvers.sdp_solver
=====================================
Solvers de tomografia quântica baseados em Programação Semidefinida (SDP)
utilizando a biblioteca recomendada `sdplab` e `spacecore`, com aceleração JAX e Optax.

Métodos implementados:
1. Regularização Entrópica (von Neumann) - maximiza a entropia de von Neumann (estado de Gibbs)
2. Regularização Quadrática (L2 / Frobenius) - minimiza a norma de Frobenius (estados de baixo posto / fronteira)
3. SDP Padrão via CVXPY (SCS) - viabilidade linear sem preferência espectral
4. SDP Mínimos Quadrados Convexos via CVXPY - solução robusta e garantida para shot noise
"""

from __future__ import annotations
import time
from typing import Sequence, Any
import numpy as np

import jax
# Habilita float64 no JAX para precisão quântica (~1e-13)
jax.config.update("jax_enable_x64", True)

import optax
from spacecore import HermitianSpace, DenseCoordinateSpace, Context
from sdplab import (
    SDPProblem,
    DenseConstraintOp,
    RegularizedSDPDualFunctional as RegFunc,
    EntropyReg,
    QuadraticReg,
)
from sdplab.solvers import run_cvxpy_solver, run_regularized_solver
from sdplab.special.pauli import generate_pauli_observables
import cvxpy as cp

from .base import TomographyResult
from ..observables import pauli_string_to_matrix
from ..states import fidelidade, frobenius_distance, trace_distance, _symmetrize_hermitian


class SDPTomographySolver:
    """
    Controlador para tomografia quântica via SDP regularizada e linear.
    """

    def __init__(self, ctx: Context | None = None):
        self.ctx = ctx if ctx is not None else Context("jax", dtype="complex128")

    def _prepare_observables(self, observables: Sequence[str | np.ndarray]) -> tuple[Any, int, int]:
        """Prepara o tensor de observáveis Q no formato do backend spacecore/JAX."""
        if len(observables) == 0:
            raise ValueError("A lista de observáveis não pode ser vazia.")

        first = observables[0]
        if isinstance(first, str):
            # Lista de strings de Pauli (ex: ['X', 'Y', 'Z'] ou ['XXXI', ...])
            obs_strings = [str(s).upper().strip() for s in observables]
            n_qubits = len(obs_strings[0])
            dim = 2 ** n_qubits
            Q_raw = generate_pauli_observables(obs_strings, ctx=self.ctx)
        else:
            # Lista de matrizes numpy densas
            mats = [np.asarray(m, dtype=complex) for m in observables]
            dim = mats[0].shape[0]
            n_qubits = int(np.log2(dim)) if (dim & (dim - 1) == 0) else None
            Q_stack = np.stack(mats, axis=0)
            Q_raw = self.ctx.asarray(Q_stack)

        # DenseConstraintOp opera por Tr(T_i^T X), portanto para medir Tr(Q_i X) devemos passar Q^T
        Q = self.ctx.ops.transpose(Q_raw, (0, 2, 1))
        M = len(observables)
        return Q, dim, M

    def solve_entropy(
        self,
        observables: Sequence[str | np.ndarray],
        q_measured: Sequence[float] | np.ndarray,
        H: np.ndarray | None = None,
        eps: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 10_000,
        verbose: int = 0,
        true_state: Any | None = None,
    ) -> TomographyResult:
        r"""
        Resolve o problema regularizado com entropia de von Neumann:
            \min_\pi \varepsilon \operatorname{Tr}[\pi(\log\pi - 1)] + \operatorname{Tr}[H \pi]
            sujeito a \operatorname{Tr}[Q_i \pi] = q_i, \operatorname{Tr}\pi = 1, \pi \succeq 0.

        O dual não possui restrições e sua maximização com L-BFGS retorna o estado de Gibbs:
            \pi^* = \frac{1}{Z} \exp\left( \frac{\sum_i \alpha_i Q_i - H}{\varepsilon} \right).
        """
        t0 = time.time()
        Q, dim, M = self._prepare_observables(observables)
        dom = HermitianSpace(dim)
        cod = DenseCoordinateSpace((M,))
        A = DenseConstraintOp(Q, dom, cod, ctx=self.ctx)

        q_vec = self.ctx.asarray(np.asarray(q_measured, dtype=float))
        C = self.ctx.asarray(np.asarray(H, dtype=complex)) if H is not None else self.ctx.ops.zeros((dim, dim))

        problem = SDPProblem(C, A, q_vec)
        reg_vn = EntropyReg(dom)
        D_vn = RegFunc(problem, reg_vn).bind(eps_val=eps, normalized=True)

        info = run_regularized_solver(
            D_vn,
            method="optax",
            opt=optax.lbfgs(),
            verbose=verbose,
            max_iter=max_iter,
            tol=tol,
        )

        pi_vn = D_vn.primal_from_dual(info.dual)
        rho_rec = _symmetrize_hermitian(np.asarray(pi_vn))
        elapsed = time.time() - t0

        # Calcula o resíduo ||A pi - q||_2
        pi_ctx = self.ctx.asarray(rho_rec)
        residual = float(cod.norm(A.apply(pi_ctx) - q_vec))

        # Métricas de comparação caso estado_verdadeiro seja fornecido
        fid, t_dist, f_dist = None, None, None
        if true_state is not None:
            rho_true = true_state.rho if hasattr(true_state, "rho") else np.asarray(true_state, dtype=complex)
            fid = fidelidade(rho_rec, rho_true)
            t_dist = trace_distance(rho_rec, rho_true)
            f_dist = frobenius_distance(rho_rec, rho_true)

        energy = float(np.trace(rho_rec @ H).real) if H is not None else None

        return TomographyResult(
            rho=rho_rec,
            method="SDP Entropia (MaxEnt / sdplab)",
            success=bool(info.converged),
            status="convergido" if info.converged else "atingiu_max_iter",
            num_iters=int(info.num_iters),
            elapsed_seconds=elapsed,
            residual=residual,
            fidelity=fid,
            energy=energy,
            trace_dist=t_dist,
            frobenius_dist=f_dist,
            dual_variables=np.asarray(info.dual).real,
            extra={"info": info, "eps": eps},
        )

    def solve_quadratic(
        self,
        observables: Sequence[str | np.ndarray],
        q_measured: Sequence[float] | np.ndarray,
        H: np.ndarray | None = None,
        eps: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 10_000,
        verbose: int = 0,
        true_state: Any | None = None,
    ) -> TomographyResult:
        r"""
        Resolve a tomografia quântica regularizada por norma de Frobenius (L2 / quadrática) via Formulação Dual.

        Dedução Matemática Completa (Dualidade de Lagrange e Gradiente Dual):
        ---------------------------------------------------------------------
        1. Formulação Primal Regularizada:
           Buscamos a matriz densidade \pi \in \mathcal{H}_d que minimiza a energia e a norma de Frobenius:
               \min_{\pi \succeq 0, \operatorname{Tr}\pi = 1} \left\{ \frac{\varepsilon}{2} \|\pi\|_F^2 + \operatorname{Tr}[H \pi] \right\}
               sujeito a \quad \operatorname{Tr}[Q_i \pi] = q_i \quad (i = 1, \dots, M).
           O regularizador \frac{\varepsilon}{2} \|\pi\|_F^2 é estritamente convexo, garantindo solução primal única.

        2. Construção do Lagrangiano:
           Introduzindo multiplicadores de Lagrange \alpha \in \mathbb{R}^M para as medições,
           \theta \in \mathbb{R} para a normalização de traço, e matriz de folga Z \succeq 0 para o cone PSD:
               \mathcal{L}(\pi, \alpha, \theta, Z) = \frac{\varepsilon}{2} \operatorname{Tr}[\pi^2] + \operatorname{Tr}[H \pi]
                   - \sum_{i=1}^M \alpha_i (\operatorname{Tr}[Q_i \pi] - q_i)
                   - \theta (\operatorname{Tr}[\pi] - 1) - \operatorname{Tr}[Z \pi].

        3. Condição de Estacionariedade KKT:
           Diferenciando em relação a \pi e igualando a zero:
               \nabla_\pi \mathcal{L} = \varepsilon \pi + H - \sum_{i=1}^M \alpha_i Q_i - \theta I - Z = 0
               ==> \pi = \frac{1}{\varepsilon} \left( \sum_{i=1}^M \alpha_i Q_i - H + \theta I + Z \right).

        4. Mapeamento Primal-Dual e Corte Positivo:
           Pela condição de folga complementar KKT, \operatorname{Tr}[Z \pi] = 0 com Z \succeq 0 e \pi \succeq 0.
           Seja a matriz hermitiana S(\alpha, \theta) = \frac{1}{\varepsilon} (\sum_{i=1}^M \alpha_i Q_i - H) - \theta I.
           Para qualquer autovalor de S:
             - Se \lambda_k(S) > 0: Z tem autovalor nulo e \pi tem autovalor \lambda_k(S).
             - Se \lambda_k(S) \le 0: \pi tem autovalor zero e Z absorve o valor negativo (-\varepsilon \lambda_k).
           Conclui-se que o estado primal ótimo é dado analiticamente pela parte positiva espectral:
               \pi^*(\alpha) = \left( \frac{\sum_{i=1}^M \alpha_i Q_i - H}{\varepsilon} - \theta \right)_+,
           onde \theta(\alpha) \in \mathbb{R} é unicamente determinado pela condição unidimensional escalar \operatorname{Tr}[\pi^*(\alpha)] = 1.
           O corte espectral (\cdot)_+ zera autovalores pequenos/negativos, direcionando a solução para a fronteira do cone PSD (baixo posto).

        5. Problema Dual e Gradiente Analítico:
           Substituindo \pi^*(\alpha) no Lagrangiano, obtém-se o problema dual irrestrito:
               \max_{\alpha \in \mathbb{R}^M} \mathcal{D}_\varepsilon(\alpha) = \sum_{i=1}^M \alpha_i q_i - \varepsilon \operatorname{Tr}\left[ \psi\left( \frac{\sum_{i=1}^M \alpha_i Q_i - H}{\varepsilon} - \theta \right) \right] + \theta,
           onde \psi(s) = \frac{1}{2}\max\{s, 0\}^2.
           Pelo teorema de Danskin / regra da cadeia, o gradiente do funcional dual em relação a cada \alpha_j é:
               \frac{\partial \mathcal{D}_\varepsilon}{\partial \alpha_j}(\alpha) = q_j - \operatorname{Tr}[Q_j \pi^*(\alpha)].
           IDENTIDADE FUNDAMENTAL:
               \nabla \mathcal{D}_\varepsilon(\alpha) = q - \mathcal{A} \pi^*(\alpha) = - r_{\text{tomografia}}.
           A norma do gradiente dual \|\nabla \mathcal{D}_\varepsilon(\alpha)\|_2 é ESTRITAMENTE o resíduo das equações de medição!
           Maximizar o dual até \|\nabla \mathcal{D}_\varepsilon\| \le \text{tol} garante a recuperação exata das medições físicas.

        6. Eficiência Computacional (sdplab + JAX):
           A otimização é resolvida no espaço \mathbb{R}^M (independente da dimensão de Hilbert 2^N para a complexidade do gradiente)
           com L-BFGS em precisão float64, eliminando inversões matriciais O(d^6) e atingindo convergência superlinear em milissegundos.
        """
        t0 = time.time()
        Q, dim, M = self._prepare_observables(observables)
        dom = HermitianSpace(dim)
        cod = DenseCoordinateSpace((M,))
        A = DenseConstraintOp(Q, dom, cod, ctx=self.ctx)

        q_vec = self.ctx.asarray(np.asarray(q_measured, dtype=float))
        C = self.ctx.asarray(np.asarray(H, dtype=complex)) if H is not None else self.ctx.ops.zeros((dim, dim))

        problem = SDPProblem(C, A, q_vec)
        reg_q = QuadraticReg(dom)
        D_q = RegFunc(problem, reg_q).bind(eps_val=eps, normalized=True)

        info = run_regularized_solver(
            D_q,
            method="optax",
            opt=optax.lbfgs(),
            verbose=verbose,
            max_iter=max_iter,
            tol=tol,
        )

        pi_q = D_q.primal_from_dual(info.dual)
        rho_rec = _symmetrize_hermitian(np.asarray(pi_q))
        elapsed = time.time() - t0

        pi_ctx = self.ctx.asarray(rho_rec)
        residual = float(cod.norm(A.apply(pi_ctx) - q_vec))

        fid, t_dist, f_dist = None, None, None
        if true_state is not None:
            rho_true = true_state.rho if hasattr(true_state, "rho") else np.asarray(true_state, dtype=complex)
            fid = fidelidade(rho_rec, rho_true)
            t_dist = trace_distance(rho_rec, rho_true)
            f_dist = frobenius_distance(rho_rec, rho_true)

        energy = float(np.trace(rho_rec @ H).real) if H is not None else None

        grad_norm_history = [float(x) for x in getattr(info, "grad_norm_history", [])]
        loss_history = [float(x) for x in getattr(info, "loss_history", [])]

        return TomographyResult(
            rho=rho_rec,
            method="SDP Quadrático (L2 / sdplab)",
            success=bool(info.converged),
            status="convergido" if info.converged else "atingiu_max_iter",
            num_iters=int(info.num_iters),
            elapsed_seconds=elapsed,
            residual=residual,
            fidelity=fid,
            energy=energy,
            trace_dist=t_dist,
            frobenius_dist=f_dist,
            dual_variables=np.asarray(info.dual).real,
            cost_history=loss_history,
            grad_norm_history=grad_norm_history,
            extra={
                "info": info,
                "eps": eps,
                "grad_norm_history": grad_norm_history,
                "loss_history": loss_history,
                "cost_history": loss_history,
            },
        )

    def solve_cvxpy_scs(
        self,
        observables: Sequence[str | np.ndarray],
        q_measured: Sequence[float] | np.ndarray,
        H: np.ndarray | None = None,
        verbose: bool = False,
        true_state: Any | None = None,
    ) -> TomographyResult:
        r"""
        Resolve o problema primal de viabilidade/otimização linear via CVXPY utilizando o solver SCS.

        Formulação Matemática Primal Exata:
        -----------------------------------
        1. Problema de Programação Semidefinida (SDP) Primal:
           O estado quântico é buscado diretamente no cone das matrizes hermitianas positivas semidefinidas:
               \min_{\rho \in \mathcal{H}_d} \operatorname{Tr}[H \rho]
               sujeito a \quad \operatorname{Tr}[Q_i \rho] = q_i \quad (i = 1, \dots, M),
                              \operatorname{Tr}[\rho] = 1,
                              \rho \succeq 0.
           Para H = 0, o problema reduz-se a uma busca de viabilidade linear pura: encontrar qualquer ponto
           na interseção do hiperplano afim \mathcal{A}\rho = q com o cone convexo \mathcal{S}_+^d.

        2. Algoritmo de Divisão de Operadores (SCS / Douglas-Rachford ADMM):
           O SCS (Splitting Conic Solver) converte o problema homogêneo autocontido:
             - Projeção afim: resolve o sistema linear simétrico quase-definido (I + A^T A) x = b.
             - Projeção cônica: decompõe espectralmente a matriz candidata \rho = \sum \lambda_k |v_k><v_k|
               e projeta no cone PSD zerando autovalores negativos: \rho_+ = \sum \max(\lambda_k, 0) |v_k><v_k|.
           O custo por iteração é dominado pela diagonalização O(d^3) = O(8^N).

        3. Critérios de Parada e Limitação com Ruído:
           O solver monitora os resíduos primal ||r_p||_2 e dual ||r_d||_2.
           Se medições ruidosas levarem a dados inconsistentes (fora do cone de estados quânticos físicos),
           a interseção afim-cônica torna-se vazia e o SCS encerra com status 'infeasible',
           evidenciando a fragilidade da viabilidade linear perante ruído estatístico de shot noise.
        """
        t0 = time.time()
        # CVXPY precisa da restrição explícita de normalização Q_0 = I
        first = observables[0]
        if isinstance(first, str):
            obs_strings = [str(s).upper().strip() for s in observables]
            n_qubits = len(obs_strings[0])
            dim = 2 ** n_qubits
            full_strings = ["I" * n_qubits] + obs_strings
            Q_I_raw = generate_pauli_observables(full_strings, ctx=self.ctx)
        else:
            mats = [np.asarray(m, dtype=complex) for m in observables]
            dim = mats[0].shape[0]
            I_mat = np.eye(dim, dtype=complex)
            full_mats = [I_mat] + mats
            Q_I_raw = self.ctx.asarray(np.stack(full_mats, axis=0))

        Q_I = self.ctx.ops.transpose(Q_I_raw, (0, 2, 1))
        dom = HermitianSpace(dim)
        cod_I = DenseCoordinateSpace((len(observables) + 1,))
        A_I = DenseConstraintOp(Q_I, dom, cod_I, ctx=self.ctx)

        # q_I = [1.0, q1, q2, ...]
        q_I = self.ctx.asarray(np.array([1.0] + list(np.asarray(q_measured, dtype=float))))
        C = self.ctx.asarray(np.asarray(H, dtype=complex)) if H is not None else self.ctx.ops.zeros((dim, dim))

        problem_I = SDPProblem(C, A_I, q_I)

        try:
            pi_scs, alpha_scs, prob = run_cvxpy_solver(
                problem_I, solver="SCS", verbose=verbose, return_problem=True
            )
            rho_rec = _symmetrize_hermitian(np.asarray(pi_scs))
            status = str(prob.status)
            success = prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            nit = getattr(prob.solver_stats, "num_iters", 0) if hasattr(prob, "solver_stats") else 0
        except Exception as e:
            rho_rec = np.eye(dim, dtype=complex) / dim
            status = f"Falhou: {e}"
            success = False
            nit = 0

        elapsed = time.time() - t0

        # Resíduo
        Q_only, _, M_only = self._prepare_observables(observables)
        cod_only = DenseCoordinateSpace((M_only,))
        A_only = DenseConstraintOp(Q_only, dom, cod_only, ctx=self.ctx)
        residual = float(cod_only.norm(A_only.apply(self.ctx.asarray(rho_rec)) - self.ctx.asarray(np.asarray(q_measured, dtype=float))))

        fid, t_dist, f_dist = None, None, None
        if true_state is not None:
            rho_true = true_state.rho if hasattr(true_state, "rho") else np.asarray(true_state, dtype=complex)
            fid = fidelidade(rho_rec, rho_true)
            t_dist = trace_distance(rho_rec, rho_true)
            f_dist = frobenius_distance(rho_rec, rho_true)

        energy = float(np.trace(rho_rec @ H).real) if H is not None else None

        return TomographyResult(
            rho=rho_rec,
            method="SDP Linear CVXPY (SCS)",
            success=success,
            status=status,
            num_iters=nit,
            elapsed_seconds=elapsed,
            residual=residual,
            fidelity=fid,
            energy=energy,
            trace_dist=t_dist,
            frobenius_dist=f_dist,
        )

    def solve_cvxpy_least_squares(
        self,
        observables: Sequence[str | np.ndarray],
        q_measured: Sequence[float] | np.ndarray,
        H: np.ndarray | None = None,
        lambda_energy: float = 0.0,
        verbose: bool = False,
        true_state: Any | None = None,
    ) -> TomographyResult:
        r"""
        Tomografia SDP Convexa por Mínimos Quadrados via CVXPY (CLARABEL / SCS).

        Formulação Matemática e Robustez a Ruído:
        ------------------------------------------
        1. Formulação Convexa Primal de Mínimos Quadrados:
           Em vez de impor viabilidade linear estrita \operatorname{Tr}[Q_i \rho] = q_i, minimiza-se o erro quadrático:
               \min_{\rho \in \mathcal{H}_d} \left\{ \sum_{i=1}^M (\operatorname{Tr}[Q_i \rho] - q_i)^2 + \lambda \operatorname{Tr}[H \rho] \right\}
               sujeito a \quad \rho \succeq 0, \quad \operatorname{Tr}[\rho] = 1.

        2. Por que Mínimos Quadrados Convexos Nunca Falham (Garantia de Factibilidade):
           O conjunto de matrizes densidade válidas \mathcal{S} = \{ \rho \in \mathcal{H}_d : \rho \succeq 0, \operatorname{Tr}\rho = 1 \}
           é convexo, fechado e compacto (limitado pela norma de Frobenius \|\rho\|_F \le 1).
           Além disso, \mathcal{S} é sempre não-vazio, pois o estado totalmente misto I/d pertence ao interior relativo de \mathcal{S}.
           Pelo Teorema da Projeção em Conjuntos Convexos Fechados (Hilbert), qualquer vetor de dados ruidosos q \in \mathbb{R}^M
           possui uma projeção euclidiana ÚNICA e global sobre o espectro físico de operadores quânticos válidos.
           Portanto, mesmo sob ruído estatístico extremo (baixo N_shots), o solver nunca falha com infactibilidade,
           garantindo status 'optimal' incondicionalmente.

        3. Métodos Numéricos (CLARABEL vs SCS):
           Utiliza CLARABEL (solver de pontos interiores para cones homogêneos com método primal-dual de barreira logarítmica)
           por padrão para convergência quadrática com alta precisão, recorrendo ao SCS caso CLARABEL não esteja instalado.
        """
        t0 = time.time()
        mats = [pauli_string_to_matrix(obs) if isinstance(obs, str) else np.asarray(obs, dtype=complex) for obs in observables]
        dim = mats[0].shape[0]
        q_vals = np.asarray(q_measured, dtype=float)

        rho_var = cp.Variable((dim, dim), hermitian=True)
        preds = cp.hstack([cp.real(cp.trace(Qi @ rho_var)) for Qi in mats])

        loss = cp.sum_squares(preds - q_vals)
        if H is not None and lambda_energy > 0:
            loss = loss + lambda_energy * cp.real(cp.trace(H @ rho_var))

        constraints = [rho_var >> 0, cp.trace(rho_var) == 1]
        prob = cp.Problem(cp.Minimize(loss), constraints)

        # Tenta CLARABEL primeiro (mais rápido e preciso que SCS se disponível), depois SCS
        try:
            prob.solve(solver=cp.CLARABEL, verbose=verbose)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=verbose)

        rho_rec = _symmetrize_hermitian(rho_var.value) if rho_var.value is not None else np.eye(dim, dtype=complex) / dim
        elapsed = time.time() - t0

        preds_calc = np.array([float(np.trace(rho_rec @ Qi).real) for Qi in mats])
        residual = float(np.linalg.norm(preds_calc - q_vals))

        fid, t_dist, f_dist = None, None, None
        if true_state is not None:
            rho_true = true_state.rho if hasattr(true_state, "rho") else np.asarray(true_state, dtype=complex)
            fid = fidelidade(rho_rec, rho_true)
            t_dist = trace_distance(rho_rec, rho_true)
            f_dist = frobenius_distance(rho_rec, rho_true)

        energy = float(np.trace(rho_rec @ H).real) if H is not None else None

        return TomographyResult(
            rho=rho_rec,
            method="SDP Mínimos Quadrados Convexos (CVXPY)",
            success=prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE),
            status=str(prob.status),
            num_iters=getattr(prob.solver_stats, "num_iters", 0) if hasattr(prob, "solver_stats") else 0,
            elapsed_seconds=elapsed,
            residual=residual,
            fidelity=fid,
            energy=energy,
            trace_dist=t_dist,
            frobenius_dist=f_dist,
            extra={"objective_value": prob.value},
        )
