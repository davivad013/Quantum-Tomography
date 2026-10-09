"""
==========
Funções usadas para Tomografia de Estados Quânticos via Mínimos Quadrados
com parametrização de Cholesky e Jacobiano analítico.
"""

from __future__ import annotations
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Sequence, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import least_squares

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Matrizes de Pauli
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_DICT = {
    "I": I2,
    "X": sigma_x,
    "Y": sigma_y,
    "Z": sigma_z,
}


def pauli_string_to_matrix(p_string: str) -> np.ndarray:
    """Converte uma string de Pauli (ex: 'XZI') na sua matriz densa via produto tensorial."""
    p_string = p_string.upper().strip()
    if not p_string:
        raise ValueError("String de Pauli vazia.")
    mat = PAULI_DICT[p_string[0]]
    for ch in p_string[1:]:
        mat = np.kron(mat, PAULI_DICT[ch])
    return mat


def pauli_basis_strings(n_qubits: int, include_identity: bool = False) -> list[str]:
    """
    Retorna a lista de strings de Pauli para n qubits (4^n - 1 strings se include_identity=False).
    Exclui 'I...I' por padrão, pois a identidade entra como a restrição de normalização Tr(rho) = 1.
    """
    ident = "I" * n_qubits
    strings = []
    for combo in itertools.product("IXYZ", repeat=n_qubits):
        s = "".join(combo)
        if not include_identity and s == ident:
            continue
        strings.append(s)
    return strings


# ---------------------------------------------------------------------------
# 2. Métricas de Informação Quântica e Estados
# ---------------------------------------------------------------------------

def _symmetrize_hermitian(mat: np.ndarray) -> np.ndarray:
    """Garante simetria hermitiana exata para evitar instabilidades numéricas."""
    mat = np.asarray(mat, dtype=complex)
    return (mat + mat.conj().T) / 2.0


def sqrt_psd(mat: np.ndarray) -> np.ndarray:
    """Calcula a raiz quadrada de uma matriz positiva semidefinida via decomposição espectral."""
    mat_h = _symmetrize_hermitian(mat)
    eigvals, eigvecs = np.linalg.eigh(mat_h)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T


def fidelidade(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    r"""
    Fidelidade de Uhlmann-Jozsa entre dois estados quânticos:
    F(\rho_a, \rho_b) = (\operatorname{Tr} \sqrt{\sqrt{\rho_a} \rho_b \sqrt{\rho_a}})^2.
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)
    sa = sqrt_psd(rho_a)
    m = sa @ rho_b @ sa
    sm = sqrt_psd(m)
    tr = np.trace(sm).real
    return float(np.clip(tr ** 2, 0.0, 1.0))


def purity(rho: np.ndarray) -> float:
    r"""Pureza do estado quântico: \gamma = \operatorname{Tr}[\rho^2]."""
    rho = np.asarray(rho, dtype=complex)
    return float(np.trace(rho @ rho).real)


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    r"""Distância de traço: D(\rho_a, \rho_b) = \frac{1}{2} \|\rho_a - \rho_b\|_1."""
    diff = _symmetrize_hermitian(np.asarray(rho_a, dtype=complex) - np.asarray(rho_b, dtype=complex))
    eigvals = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigvals)))


def frobenius_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    r"""Distância de Frobenius: \|\rho_a - \rho_b\|_F."""
    diff = np.asarray(rho_a, dtype=complex) - np.asarray(rho_b, dtype=complex)
    return float(np.linalg.norm(diff))


def numerical_rank(rho: np.ndarray, tol: float = 1e-9) -> int:
    """Retorna o posto numérico (número de autovalores > tol)."""
    rho_h = _symmetrize_hermitian(rho)
    eigvals = np.linalg.eigvalsh(rho_h)
    return int((eigvals > tol).sum())


def partial_trace(
    rho: np.ndarray | QuantState,
    keep: int = 0,
    n_qubits: int | None = None,
) -> np.ndarray:
    r"""
    Calcula o traço parcial sobre todos os qubits de um sistema multipartite,
    preservando apenas o subsistema indicado por `keep`.
    """
    mat = rho.rho if isinstance(rho, QuantState) else np.asarray(rho, dtype=complex)
    dim = mat.shape[0]
    if n_qubits is None:
        n_qubits = int(round(np.log2(dim)))
    if 2 ** n_qubits != dim:
        raise ValueError(f"Dimensão {dim} não é uma potência exata de 2 (qubits indefinidos).")
    if not (0 <= keep < n_qubits):
        raise ValueError(f"Índice do qubit keep={keep} fora do intervalo válido [0, {n_qubits - 1}].")

    if n_qubits == 1:
        tr = float(np.trace(mat).real)
        return _symmetrize_hermitian(mat / tr if tr > 1e-15 else mat)

    tensor = mat.reshape([2] * (2 * n_qubits))
    chars = [chr(97 + i) for i in range(2 * n_qubits)]
    in_row = chars[:n_qubits]
    in_col = list(chars[n_qubits:])
    for i in range(n_qubits):
        if i != keep:
            in_col[i] = in_row[i]

    in_sub = "".join(in_row) + "".join(in_col)
    out_sub = in_row[keep] + chars[n_qubits + keep]
    rho_red = np.einsum(f"{in_sub}->{out_sub}", tensor)

    rho_red_h = _symmetrize_hermitian(rho_red)
    tr = float(np.trace(rho_red_h).real)
    if tr > 1e-15:
        rho_red_h = rho_red_h / tr
    return rho_red_h


class QuantState:
    """
    Representação de um estado quântico de N qubits.
    Aceita vetores de estado 1D (|psi>) ou matrizes densidade 2D (rho).
    """

    def __init__(self, estado: np.ndarray | list):
        estado = np.array(estado, dtype=complex)
        if estado.ndim == 1:
            norma = np.linalg.norm(estado)
            if norma == 0:
                raise ValueError("Vetor de estado tem norma zero.")
            psi = estado / norma
            rho = np.outer(psi, psi.conj())
        elif estado.ndim == 2:
            traco = np.trace(estado).real
            if np.isclose(traco, 0):
                raise ValueError("Matriz densidade tem traço nulo.")
            rho = estado / traco
        else:
            raise ValueError(f"Dimensão inválida para estado quântico: {estado.ndim}")

        self.rho: np.ndarray = _symmetrize_hermitian(rho)
        self.dim: int = self.rho.shape[0]
        n = math.log2(self.dim)
        self.n_qubits: int | None = int(n) if n.is_integer() else None

    def measure(self, obs: np.ndarray) -> float:
        """Valor esperado exato de um observável hermitiano: Tr(rho * obs)."""
        return float(np.trace(self.rho @ obs).real)

    def energy(self, H: np.ndarray) -> float:
        """Valor esperado de energia: Tr(rho * H)."""
        return self.measure(H)

    @property
    def purity(self) -> float:
        return purity(self.rho)

    @property
    def rank(self) -> int:
        return numerical_rank(self.rho)

    @property
    def is_pure(self) -> bool:
        return bool(np.isclose(self.purity, 1.0, atol=1e-5))

    def fidelity_with(self, other: QuantState | np.ndarray) -> float:
        target = other.rho if isinstance(other, QuantState) else other
        return fidelidade(self.rho, target)

    def bloch_vector(self) -> np.ndarray:
        """Calcula o vetor de Bloch (r_x, r_y, r_z) para 1 qubit."""
        if self.dim != 2:
            raise ValueError("Vetor de Bloch definido apenas para 1 qubit (dim=2).")
        return np.array([self.measure(sigma_x), self.measure(sigma_y), self.measure(sigma_z)], dtype=float)

    def partial_trace(self, keep: int = 0) -> QuantState:
        """Calcula o estado quântico reduzido obtido pelo traço parcial."""
        return QuantState(partial_trace(self.rho, keep=keep, n_qubits=self.n_qubits))

    def __repr__(self) -> str:
        q_str = f"{self.n_qubits} qubits" if self.n_qubits is not None else f"dim={self.dim}"
        tipo = "Puro" if self.is_pure else f"Misto (posto {self.rank})"
        return f"<QuantState ({q_str}, {tipo}, pureza={self.purity:.3f})>"


def random_pure_state(dim: int, seed: int | None = None) -> QuantState:
    """Gera um estado puro aleatório Haar-uniforme."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return QuantState(vec)


def standard_pure_state(name: str, n_qubits: int = 1) -> QuantState:
    """
    Retorna estados puros de referência comuns:
    - 1 qubit: '0', '1', '+', '-', '+i', '-i'
    - 2 qubits: 'bell', 'ghz'
    - N qubits: '0...0', 'ghz', 'w'
    """
    name_clean = name.lower().strip()
    dim = 2 ** n_qubits

    if n_qubits == 1:
        states_1q = {
            "0": [1.0, 0.0],
            "1": [0.0, 1.0],
            "+": [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
            "-": [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)],
            "+i": [1.0 / np.sqrt(2), 1j / np.sqrt(2)],
            "-i": [1.0 / np.sqrt(2), -1j / np.sqrt(2)],
        }
        if name_clean in states_1q:
            return QuantState(states_1q[name_clean])

    if name_clean in ("ghz", "bell"):
        vec = np.zeros(dim, dtype=complex)
        vec[0] = 1.0 / np.sqrt(2)
        vec[-1] = 1.0 / np.sqrt(2)
        return QuantState(vec)

    if name_clean == "w":
        vec = np.zeros(dim, dtype=complex)
        for i in range(n_qubits):
            vec[1 << i] = 1.0 / np.sqrt(n_qubits)
        return QuantState(vec)

    if name_clean in ("all_zero", "0" * n_qubits):
        vec = np.zeros(dim, dtype=complex)
        vec[0] = 1.0
        return QuantState(vec)

    raise ValueError(f"Estado padrão desconhecido: '{name}' para {n_qubits} qubits.")


def random_mixture_of_pures(dim: int, rank: int = 2, seed: int | None = None) -> QuantState:
    r"""Gera uma combinação convexa de `rank` estados puros aleatórios."""
    rng = np.random.default_rng(seed)
    if rank > dim:
        rank = dim

    G = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    Q, _ = np.linalg.qr(G)

    raw_p = rng.exponential(scale=1.0, size=rank)
    p = raw_p / np.sum(raw_p)

    rho = np.zeros((dim, dim), dtype=complex)
    for j in range(rank):
        v = Q[:, j]
        rho += p[j] * np.outer(v, v.conj())

    return QuantState(rho)


def maximally_mixed_state(dim: int) -> QuantState:
    """Gera o estado totalmente misto rho = I / d."""
    return QuantState(np.eye(dim, dtype=complex) / dim)


# ---------------------------------------------------------------------------
# 3. Medição de Observáveis e Simulação com Shots
# ---------------------------------------------------------------------------

def simula_medidas_shots(
    q_exato: Sequence[float] | np.ndarray,
    N_shots: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula N_shots medições binomiais independentes por observável (regra de Born).
    """
    q_exato = np.asarray(q_exato, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    binomial = rng.binomial
    q_ruido = np.empty_like(q_exato)

    for i, qi in enumerate(q_exato):
        p_mais = float(np.clip((1.0 + qi) / 2.0, 0.0, 1.0))
        outcomes = binomial(N_shots, p_mais)
        q_ruido[i] = 2.0 * outcomes / N_shots - 1.0

    return q_ruido


def measure_observables(
    estado: QuantState | np.ndarray,
    observables: Sequence[str | np.ndarray],
    N_shots: int | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Mede observáveis: retorna valores esperados analíticos ou com ruído binomial de shots.
    """
    rho = estado.rho if hasattr(estado, "rho") else np.asarray(estado, dtype=complex)
    mats = [pauli_string_to_matrix(obs) if isinstance(obs, str) else obs for obs in observables]
    q_exato = np.array([float(np.trace(rho @ Q).real) for Q in mats])

    if N_shots is None:
        return q_exato
    return simula_medidas_shots(q_exato, N_shots, rng=rng, seed=seed)


# ---------------------------------------------------------------------------
# 4. Estrutura de Resultado Tomográfico
# ---------------------------------------------------------------------------

@dataclass
class TomographyResult:
    """Resultado padronizado de uma reconstrução tomográfica."""
    rho: np.ndarray
    method: str
    success: bool
    status: str
    num_iters: int
    elapsed_seconds: float
    residual: float
    fidelity: float | None = None
    purity: float = 0.0
    rank: int = 0
    frobenius_norm: float = 0.0
    energy: float | None = None
    cost_or_loss: float | None = None
    trace_dist: float | None = None
    frobenius_dist: float | None = None
    dual_variables: np.ndarray | None = None
    cost_history: list[float] | None = None
    grad_norm_history: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.rho = np.asarray(self.rho, dtype=complex)
        self.purity = purity(self.rho)
        self.rank = numerical_rank(self.rho)
        self.frobenius_norm = float(np.linalg.norm(self.rho))
        if self.cost_history is None and "cost_history" in self.extra:
            self.cost_history = self.extra["cost_history"]
        if self.grad_norm_history is None and "grad_norm_history" in self.extra:
            self.grad_norm_history = self.extra["grad_norm_history"]

    def __getitem__(self, key: str) -> Any:
        alias_map = {
            "rho_reconstruido": self.rho,
            "rho": self.rho,
            "fidelidade": self.fidelity,
            "fidelity": self.fidelity,
            "tempo_s": self.elapsed_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "nfev": self.num_iters,
            "num_iters": self.num_iters,
            "status": self.status,
            "mensagem": self.status,
            "custo_final": self.cost_or_loss,
            "cost_or_loss": self.cost_or_loss,
            "residual": self.residual,
            "cost_history": self.cost_history if self.cost_history is not None else self.extra.get("cost_history", None),
            "grad_norm_history": self.grad_norm_history if self.grad_norm_history is not None else self.extra.get("grad_norm_history", None),
            "resultado": self.extra.get("resultado_scipy", None),
            "chute_inicial": self.extra.get("chute_inicial", None),
            "q_medidos": self.extra.get("q_medidos", None),
        }
        if key in alias_map:
            return alias_map[key]
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.extra:
            return self.extra[key]
        raise KeyError(f"Chave '{key}' não encontrada em TomographyResult.")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.extra or key in (
            "rho_reconstruido", "fidelidade", "tempo_s", "custo_final", "nfev", "mensagem"
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def summary(self) -> str:
        lines = [
            f"=== Resultado Tomografia: {self.method} ===",
            f"  Status        : {self.status} (Sucesso: {self.success})",
            f"  Tempo         : {self.elapsed_seconds * 1000.0:.2f} ms ({self.elapsed_seconds:.4f} s)",
            f"  Iterações     : {self.num_iters}",
            f"  Resíduo       : {self.residual:.3e}",
            f"  Pureza        : {self.purity:.6f}",
            f"  Posto numérico: {self.rank}",
        ]
        if self.fidelity is not None:
            lines.append(f"  Fidelidade    : {self.fidelity:.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        fid_str = f", Fid={self.fidelity:.4f}" if self.fidelity is not None else ""
        return (
            f"<TomographyResult ({self.method}, sucesso={self.success}, "
            f"tempo={self.elapsed_seconds:.3f}s{fid_str}, res={self.residual:.2e})>"
        )


# ---------------------------------------------------------------------------
# 5. Solver de Força Bruta por Mínimos Quadrados (Cholesky)
# ---------------------------------------------------------------------------

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
    Converte os D^2 parâmetros reais na matriz densidade física:
        \rho = \frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}.
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
    """Vetor de resíduos: r_i = Tr(rho(T) * Q_i) - q_i."""
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
    r"""Jacobiano analítico exato do vetor de resíduos em relação aos parâmetros de Cholesky."""
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
    """Resolve a tomografia por mínimos quadrados não-lineares com a parametrização de Cholesky."""
    t0 = time.time()
    mats = [pauli_string_to_matrix(obs) if isinstance(obs, str) else np.asarray(obs, dtype=complex) for obs in mediadores_Q]
    dim = mats[0].shape[0]

    if n_qubits is None:
        n_qubits = int(round(np.log2(dim)))

    q_arr = np.asarray(q_medidos, dtype=float)
    num_params = dim ** 2

    if chute_inicial is None:
        rng = np.random.default_rng(seed)
        chute_inicial = rng.normal(size=num_params)

    jac = jacobiano_analitico if usar_jacobiano_analitico else "2-point"

    if "jac" in kwargs_least_squares:
        jac = kwargs_least_squares.pop("jac")

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


def solve_tomography(
    observables: Sequence[str | np.ndarray],
    q_measured: Sequence[float] | np.ndarray,
    method: str = "least_squares",
    H: np.ndarray | None = None,
    true_state: Any | None = None,
    n_qubits: int | None = None,
    chute_inicial: np.ndarray | None = None,
    **kwargs,
) -> TomographyResult:
    """
    Ponto de entrada unificado para tomografia quântica (método padrão: least_squares).
    """
    return resolve_tomografia_lsq(
        observables,
        q_measured,
        n_qubits=n_qubits,
        chute_inicial=chute_inicial,
        estado_verdadeiro=true_state,
        H=H,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 6. Visualização e Gráficos
# ---------------------------------------------------------------------------

def vetor_bloch(rho: np.ndarray | QuantState | TomographyResult) -> np.ndarray:
    """Extrai o vetor de Bloch (rx, ry, rz) = Tr(rho * sigma_i) para 1 qubit."""
    if isinstance(rho, TomographyResult):
        mat = rho.rho
    elif isinstance(rho, QuantState):
        mat = rho.rho
    else:
        mat = np.asarray(rho, dtype=complex)
    return np.array([float(np.trace(mat @ s).real) for s in (sigma_x, sigma_y, sigma_z)])


def desenhar_esfera(ax):
    """Desenha a esfera de Bloch unitária em 3D com linhas de grade e eixos."""
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="#9ecae1", alpha=0.10, linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color="#6baed6", alpha=0.10, linewidth=0.4, rstride=4, cstride=4)

    t = np.linspace(0, 2 * np.pi, 300)
    c, s, o = np.cos(t), np.sin(t), np.zeros_like(t)
    ax.plot(c, s, o, color="gray", alpha=0.5, lw=1)
    ax.plot(c, o, s, color="gray", alpha=0.5, lw=1)
    ax.plot(o, c, s, color="gray", alpha=0.5, lw=1)

    L = 1.25
    for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        d_arr = np.array(d) * L
        ax.plot([-d_arr[0], d_arr[0]], [-d_arr[1], d_arr[1]], [-d_arr[2], d_arr[2]], color="k", alpha=0.6, lw=0.9)

    k = 1.42
    ax.text(0, 0, k, r"$|0\rangle$", ha="center", fontsize=11)
    ax.text(0, 0, -k, r"$|1\rangle$", ha="center", fontsize=11)
    ax.text(k, 0, 0, r"$|+\rangle$", ha="center", fontsize=11)
    ax.text(-k, 0, 0, r"$|-\rangle$", ha="center", fontsize=11)
    ax.text(0, k, 0, r"$|{+i}\rangle$", ha="center", fontsize=11)
    ax.text(0, -k, 0, r"$|{-i}\rangle$", ha="center", fontsize=11)


def desenhar_plano(ax, vetores, cores, rotulos, marcadores, chave, medido=False):
    """Desenha a projeção em um plano 2D."""
    planos = {
        "xz": (0, 2, "Plano x–z", (r"$|+\rangle$", r"$|-\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
        "xy": (0, 1, "Plano x–y", (r"$|+\rangle$", r"$|-\rangle$", r"$|{+i}\rangle$", r"$|{-i}\rangle$")),
        "yz": (1, 2, "Plano y–z", (r"$|{+i}\rangle$", r"$|{-i}\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
    }
    ih, iv, nome, (dir_, esq, cima, baixo) = planos[chave]

    ax.add_patch(Circle((0, 0), 1, fc="#9ecae1", ec="gray", alpha=0.15, lw=0))
    ax.add_patch(Circle((0, 0), 1, fc="none", ec="gray", lw=1.2))
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)

    for r, cor, rot, m in zip(vetores, cores, rotulos, marcadores):
        h, v = r[ih], r[iv]
        ax.annotate("", xy=(h, v), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=cor, lw=2.2))
        ax.scatter(h, v, color=cor, s=100 if m == "o" else 65, marker=m, edgecolor="k", zorder=10)

    if len(vetores) == 2:
        a, b = vetores
        ax.plot([a[ih], b[ih]], [a[iv], b[iv]], color="k", ls=":", lw=1.3)

    k = 1.18
    ax.text(k, 0, dir_, ha="left", va="center", fontsize=11)
    ax.text(-k, 0, esq, ha="right", va="center", fontsize=11)
    ax.text(0, k, cima, ha="center", va="bottom", fontsize=11)
    ax.text(0, -k, baixo, ha="center", va="top", fontsize=11)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    titulo = nome + ("  (observáveis medidos)" if medido else "")
    ax.set_title(titulo, fontsize=11, fontweight="bold" if medido else "normal")


def plotar_bloch(
    rhos: Sequence[Any],
    rotulos: Sequence[str] | None = None,
    cores: Sequence[str] | None = None,
    titulo: str = "Esfera de Bloch–Poincaré (Reconstrução Tomográfica)",
    elev: float = 20,
    azim: float = 35,
    medidos: Sequence[str] = ("x", "y"),
    salvar: str | None = None,
) -> plt.Figure:
    """Renderiza a visualização canônica da esfera de Bloch (3D + projeções xz, xy, yz)."""
    rotulos = list(rotulos) if rotulos else [f"ρ{i + 1}" for i in range(len(rhos))]
    cores = list(cores) if cores else ["#e6550d", "#31a354", "#756bb1", "#de2d26", "#3182bd"]
    marcadores = ["o", "D", "s", "^", "v"]
    vetores = [vetor_bloch(rho) for rho in rhos]

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08, left=0.04, right=0.96, top=0.92, bottom=0.12)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    desenhar_esfera(ax3d)
    for r, cor, rot, m in zip(vetores, cores, rotulos, marcadores):
        ax3d.quiver(0, 0, 0, *r, color=cor, lw=2.4, arrow_length_ratio=0.10)
        ax3d.scatter(*r, color=cor, s=90, marker=m, edgecolor="k", zorder=10, label=rot)
    if len(vetores) == 2:
        ax3d.plot(*zip(*vetores), color="k", ls=":", lw=1.5)

    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-1, 1)
    ax3d.set_axis_off()
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("Esfera de Bloch (3D)", fontsize=12)
    ax3d.legend(loc="upper left", fontsize=10)

    conj_medidos = {m.lower() for m in medidos}
    for chave, pos in (("xz", (0, 1)), ("xy", (1, 0)), ("yz", (1, 1))):
        ax = fig.add_subplot(gs[pos])
        desenhar_plano(ax, vetores, cores, rotulos, marcadores, chave, medido=(set(chave) == conj_medidos))

    linhas = [f"{rot}: r = ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})  |r| = {np.linalg.norm(r):.3f}" for rot, r in zip(rotulos, vetores)]
    if len(vetores) == 2:
        linhas.append(f"Distância entre eles: {np.linalg.norm(vetores[0] - vetores[1]):.4f}")
    fig.text(0.5, 0.02, "\n".join(linhas), ha="center", va="bottom", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    fig.suptitle(titulo, fontsize=15, y=0.97)
    if salvar:
        fig.savefig(salvar, dpi=200, bbox_inches="tight")
    return fig


def plot_fidelity_vs_shots(
    shots_list: Sequence[int],
    fidelidades_medias: dict[str, Sequence[float]],
    fidelidades_stds: dict[str, Sequence[float]] | None = None,
    taxa_infactivel: Sequence[float] | None = None,
    title: str = "Fidelidade de Reconstrução vs. Número de Shots",
    save_path: str | None = None,
) -> plt.Figure:
    """Plota fidelidade média e desvio padrão em função de N_shots com barras de erro."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    markers = ["o", "s", "^", "D"]

    for i, (metodo, fids) in enumerate(fidelidades_medias.items()):
        m = markers[i % len(markers)]
        if fidelidades_stds and metodo in fidelidades_stds:
            ax1.errorbar(shots_list, fids, yerr=fidelidades_stds[metodo], marker=m, lw=1.8, capsize=3, label=metodo)
        else:
            ax1.plot(shots_list, fids, marker=m, lw=1.8, label=metodo)

    ax1.set_xscale("log")
    ax1.set_xlabel("Número de Shots por Observável ($N_{\\mathrm{shots}}$)", fontsize=11)
    ax1.set_ylabel("Fidelidade com o Estado Alvo", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)

    if taxa_infactivel is not None:
        ax2 = ax1.twinx()
        ax2.plot(shots_list, taxa_infactivel, color="crimson", ls="--", marker="x", label="Taxa Infactibilidade (SDP Linear)")
        ax2.set_ylabel("Fração de Instâncias Infactíveis", color="crimson", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_shots_scaling(
    benchmark_data: dict[str, Any],
    title: str = "Escalabilidade Tomográfica com Shots Reais ($N_{\\text{shots}} = 10.000$)",
    save_path: str | None = None,
) -> plt.Figure:
    """Gera dois gráficos: (a) Fidelidade vs. n_qubits e (b) Iterações vs. n_qubits."""
    qubits = benchmark_data["qubits_range"]
    methods = benchmark_data["methods"]
    fids_mean = benchmark_data["fidelidades_mean"]
    fids_std = benchmark_data["fidelidades_std"]
    iters_mean = benchmark_data["iteracoes_mean"]
    iters_std = benchmark_data["iteracoes_std"]

    rotulos_metodos = {
        "least_squares": "Força Bruta (Cholesky LSQ)",
        "quadratic": "SDP Dual (Quadrático)",
        "cvxpy_lsq": "SDP Mínimos Quadrados Convexos",
        "cvxpy_scs": "SDP Primal Linear (SCS)",
    }
    cores_metodos = {
        "least_squares": "#31a354",
        "quadratic": "#e6550d",
        "cvxpy_lsq": "#3182bd",
        "cvxpy_scs": "#756bb1",
    }
    marcadores_metodos = {
        "least_squares": "o",
        "quadratic": "s",
        "cvxpy_lsq": "^",
        "cvxpy_scs": "D",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for m in methods:
        nome = rotulos_metodos.get(m, m)
        cor = cores_metodos.get(m, "#333333")
        marker = marcadores_metodos.get(m, "o")

        ax1.errorbar(
            qubits,
            fids_mean[m],
            yerr=fids_std[m],
            fmt=f"{marker}-",
            color=cor,
            capsize=4,
            capthick=1.2,
            lw=1.8,
            markersize=6,
            label=nome,
        )

        ax2.errorbar(
            qubits,
            iters_mean[m],
            yerr=iters_std[m],
            fmt=f"{marker}-",
            color=cor,
            capsize=4,
            capthick=1.2,
            lw=1.8,
            markersize=6,
            label=nome,
        )

    ax1.set_title("(a) Fidelidade vs. Número de Qubits", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Número de Qubits ($N$)", fontsize=10)
    ax1.set_ylabel("Fidelidade Média ($F$)", fontsize=10)
    ax1.set_xticks(qubits)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left")

    ax2.set_title("(b) Iterações / Avaliações vs. Número de Qubits", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Número de Qubits ($N$)", fontsize=10)
    ax2.set_ylabel("Iterações Médias (escala log)", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_xticks(qubits)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_purity_convergence(
    purity_results: dict[str, dict[str, TomographyResult]],
    title: str = "Trajetória de Convergência por Classe de Pureza",
    save_path: str | None = None,
    solver: str = "least_squares",
) -> plt.Figure:
    """Plota as curvas de custo (escala log) sobrepostas por classe de estado."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (state_name, solvers_res) in enumerate(purity_results.items()):
        res = solvers_res.get(solver)
        if res is None:
            continue

        custos = res.cost_history or res.extra.get("cost_history", [])
        if len(custos) == 0:
            continue

        ax.semilogy(
            np.arange(len(custos)),
            np.clip(custos, 1e-16, None),
            "o-",
            color=cores[idx % len(cores)],
            lw=1.8,
            markersize=3.5,
            label=f"{state_name}",
        )

    ax.set_title("Força Bruta (Cholesky LSQ)\nCusto vs. Avaliações de Função", fontsize=11, fontweight="bold")
    ax.set_xlabel("Avaliações de Função ($k$)", fontsize=10)
    ax.set_ylabel(r"Função de Custo $F(x) = \frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_purity_completeness_comparison(
    incomplete_results: dict[str, dict[str, TomographyResult]],
    complete_results: dict[str, dict[str, TomographyResult]],
    title: str = "Comparação de Completude: Medição Incompleta (M=6) vs. Completa (M=15)",
    save_path: str | None = None,
) -> plt.Figure:
    """Compara a convergência em dados incompletos (M=6) vs dados completos (M=15)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (state_name, solvers_res) in enumerate(incomplete_results.items()):
        cor = cores[idx % len(cores)]
        res_lsq = solvers_res.get("least_squares")
        if res_lsq is not None:
            c = res_lsq.cost_history or res_lsq.extra.get("cost_history", [])
            ax1.semilogy(np.arange(len(c)), np.clip(c, 1e-16, None), "o-", color=cor, lw=1.8, markersize=3.5, label=f"{state_name}")

    for idx, (state_name, solvers_res) in enumerate(complete_results.items()):
        cor = cores[idx % len(cores)]
        res_lsq = solvers_res.get("least_squares")
        if res_lsq is not None:
            c = res_lsq.cost_history or res_lsq.extra.get("cost_history", [])
            ax2.semilogy(np.arange(len(c)), np.clip(c, 1e-16, None), "o-", color=cor, lw=1.8, markersize=3.5, label=f"{state_name}")

    ax1.set_title("Medição Incompleta ($M = 6$ de 15 observáveis)\nSubdeterminado (Não-Unicidade)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Avaliações de Função", fontsize=10)
    ax1.set_ylabel(r"Custo $\frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=9, loc="upper right")

    ax2.set_title("Medição Completa ($M = 15$ observáveis de Pauli)\nDeterminismo Total (Estado Único)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Avaliações de Função", fontsize=10)
    ax2.set_ylabel(r"Custo $\frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Suíte de Experimentos
# ---------------------------------------------------------------------------

def run_uniqueness_experiment(
    q_values: tuple[float, float] = (0.7, 0.4),
    seed: int = 42,
) -> dict[str, TomographyResult]:
    """
    Demonstra a não-unicidade quando medições são incompletas:
    Para 1 qubit, mede apenas sigma_x e sigma_y (deixando sigma_z livre).
    """
    observables = ["X", "Y"]
    q = np.array(q_values, dtype=float)

    results = {}
    rng = np.random.default_rng(seed)
    chute_a = rng.normal(size=4)
    results["least_squares_a"] = solve_tomography(
        observables, q, method="least_squares", chute_inicial=chute_a, n_qubits=1
    )

    chute_b = rng.normal(size=4)
    results["least_squares_b"] = solve_tomography(
        observables, q, method="least_squares", chute_inicial=chute_b, n_qubits=1
    )

    return results


def run_noise_stability_experiment(
    n_qubits: int = 1,
    shots_list: Sequence[int] = (10, 25, 50, 100, 250, 500, 1000, 5000, 10000),
    n_reps: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Analisa a estabilidade do algoritmo de Força Bruta perante ruído de shot noise."""
    dim = 2 ** n_qubits
    state = random_pure_state(dim, seed=seed)
    observables = pauli_basis_strings(n_qubits)
    q_exato = measure_observables(state, observables)

    rng = np.random.default_rng(seed)

    fids_brute_force: list[float] = []
    stds_brute_force: list[float] = []

    for N_shots in shots_list:
        fids_bf = []
        for _ in range(n_reps):
            q_ruido = simula_medidas_shots(q_exato, N_shots, rng=rng)
            res_bf = solve_tomography(observables, q_ruido, method="least_squares", true_state=state, n_qubits=n_qubits)
            if res_bf.fidelity is not None:
                fids_bf.append(res_bf.fidelity)

        fids_brute_force.append(float(np.mean(fids_bf)) if fids_bf else 0.0)
        stds_brute_force.append(float(np.std(fids_bf)) if fids_bf else 0.0)

    return {
        "shots_list": list(shots_list),
        "fids_brute_force": fids_brute_force,
        "stds_brute_force": stds_brute_force,
    }


def run_shots_scaling_benchmark(
    qubits_range: Sequence[int] = (1, 2, 3, 4),
    n_shots: int = 10000,
    methods: Sequence[str] = ("least_squares",),
    n_reps: int = 5,
    seed: int = 42,
    max_iter_quadratic: int = 300,
) -> dict[str, Any]:
    """Benchmark empírico de escalabilidade computacional sob ruído de shots."""
    qubits_list = list(qubits_range)
    methods_list = list(methods)

    fidelidades_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    fidelidades_std: dict[str, list[float]] = {m: [] for m in methods_list}
    iteracoes_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    iteracoes_std: dict[str, list[float]] = {m: [] for m in methods_list}
    tempos_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    tempos_std: dict[str, list[float]] = {m: [] for m in methods_list}
    residuos_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    detalhes: list[dict[str, Any]] = []

    for n in qubits_list:
        dim = 2 ** n
        state_seed = seed * 100 + n
        target_state = random_pure_state(dim, seed=state_seed)
        observables = pauli_basis_strings(n)
        q_exato = measure_observables(target_state, observables)

        fids_by_method: dict[str, list[float]] = {m: [] for m in methods_list}
        iters_by_method: dict[str, list[int]] = {m: [] for m in methods_list}
        times_by_method: dict[str, list[float]] = {m: [] for m in methods_list}
        resids_by_method: dict[str, list[float]] = {m: [] for m in methods_list}

        for rep in range(n_reps):
            rep_seed = seed * 1000 + n * 100 + rep
            rng_rep = np.random.default_rng(rep_seed)
            q_shots = simula_medidas_shots(q_exato, n_shots, rng=rng_rep)

            for m in methods_list:
                t0 = time.time()
                res = solve_tomography(
                    observables,
                    q_shots,
                    method=m,
                    true_state=target_state,
                    n_qubits=n,
                )
                dur = time.time() - t0

                fid_val = float(res.fidelity) if res.fidelity is not None else 0.0
                fids_by_method[m].append(fid_val)
                iters_by_method[m].append(int(res.num_iters))
                times_by_method[m].append(float(dur))
                resids_by_method[m].append(float(res.residual))

                detalhes.append({
                    "n_qubits": n,
                    "rep": rep,
                    "method": m,
                    "fidelity": fid_val,
                    "num_iters": int(res.num_iters),
                    "time": float(dur),
                    "residual": float(res.residual),
                    "status": res.status,
                    "success": res.success,
                })

        for m in methods_list:
            fids_mean = float(np.mean(fids_by_method[m])) if fids_by_method[m] else 0.0
            fids_s = float(np.std(fids_by_method[m])) if fids_by_method[m] else 0.0
            it_mean = float(np.mean(iters_by_method[m])) if iters_by_method[m] else 0.0
            it_s = float(np.std(iters_by_method[m])) if iters_by_method[m] else 0.0
            t_mean = float(np.mean(times_by_method[m])) if times_by_method[m] else 0.0
            t_s = float(np.std(times_by_method[m])) if times_by_method[m] else 0.0
            r_mean = float(np.mean(resids_by_method[m])) if resids_by_method[m] else 0.0

            fidelidades_mean[m].append(fids_mean)
            fidelidades_std[m].append(fids_s)
            iteracoes_mean[m].append(it_mean)
            iteracoes_std[m].append(it_s)
            tempos_mean[m].append(t_mean)
            tempos_std[m].append(t_s)
            residuos_mean[m].append(r_mean)

    return {
        "qubits_range": qubits_list,
        "n_shots": n_shots,
        "n_reps": n_reps,
        "methods": methods_list,
        "fidelidades_mean": fidelidades_mean,
        "fidelidades_std": fidelidades_std,
        "iteracoes_mean": iteracoes_mean,
        "iteracoes_std": iteracoes_std,
        "tempos_mean": tempos_mean,
        "tempos_std": tempos_std,
        "residuos_mean": residuos_mean,
        "detalhes": detalhes,
    }


def run_purity_spectrum_experiment(
    n_qubits: int = 2,
    num_measurements: int | None = 6,
    seed: int = 42,
    measurement_completeness: str | bool | None = "incomplete",
    solvers: Sequence[str] = ("least_squares",),
    chute_inicial: np.ndarray | None = None,
    N_shots: int | None = 1000,
) -> dict[str, dict[str, TomographyResult]]:
    """Compara como o algoritmo se comporta perante diferentes classes de pureza."""
    dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    states_dict: dict[str, QuantState] = {
        "1. Puro (Posto 1)": random_pure_state(dim, seed=seed),
        "2. Mistura de Puros (Posto 2)": random_mixture_of_pures(dim, rank=2, seed=seed + 1),
        "3. Totalmente Misto (I/d)": maximally_mixed_state(dim),
    }

    all_paulis = pauli_basis_strings(n_qubits)
    is_complete = False
    if measurement_completeness is not None:
        if isinstance(measurement_completeness, bool):
            is_complete = measurement_completeness
        elif str(measurement_completeness).lower().strip() in ("complete", "completa", "full", "all", "15"):
            is_complete = True

    if is_complete or num_measurements is None or num_measurements >= len(all_paulis):
        observables = all_paulis
    else:
        obs_idx = rng.choice(len(all_paulis), size=num_measurements, replace=False)
        observables = [all_paulis[i] for i in sorted(obs_idx)]

    num_params = dim ** 2
    if chute_inicial is None:
        fixed_x0 = np.random.default_rng(seed + 999).normal(size=num_params)
    else:
        fixed_x0 = np.asarray(chute_inicial, dtype=float)

    experiment_results: dict[str, dict[str, TomographyResult]] = {}

    for i, (state_name, state) in enumerate(states_dict.items()):
        q_exact = measure_observables(state, observables)
        if N_shots is None:
            q_exp = q_exact
        else:
            q_exp = simula_medidas_shots(q_exact, N_shots=N_shots, seed=seed + 1000 + i)

        res_by_solver: dict[str, TomographyResult] = {}
        for s in solvers:
            res = solve_tomography(
                observables, q_exp, method=s,
                true_state=state, n_qubits=n_qubits,
                chute_inicial=fixed_x0,
            )
            res.extra["target_state"] = state
            res.extra["N_shots"] = N_shots
            res.extra["q_measured"] = q_exp
            res.extra["q_exact"] = q_exact
            res.extra["shot_noise_norm"] = float(np.linalg.norm(q_exp - q_exact))
            res_by_solver[s] = res
        experiment_results[state_name] = res_by_solver

    return experiment_results


__all__ = [
    "I2",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "PAULI_DICT",
    "pauli_string_to_matrix",
    "pauli_basis_strings",
    "fidelidade",
    "purity",
    "trace_distance",
    "frobenius_distance",
    "numerical_rank",
    "partial_trace",
    "QuantState",
    "random_pure_state",
    "standard_pure_state",
    "random_mixture_of_pures",
    "maximally_mixed_state",
    "simula_medidas_shots",
    "measure_observables",
    "TomographyResult",
    "parametros_para_T_geral",
    "parametros_para_rho_geral",
    "funcao_de_custo_geral",
    "jacobiano_analitico",
    "resolve_tomografia_lsq",
    "solve_tomography",
    "vetor_bloch",
    "desenhar_esfera",
    "desenhar_plano",
    "plotar_bloch",
    "plot_fidelity_vs_shots",
    "plot_shots_scaling",
    "plot_purity_convergence",
    "plot_purity_completeness_comparison",
    "run_uniqueness_experiment",
    "run_noise_stability_experiment",
    "run_shots_scaling_benchmark",
    "run_purity_spectrum_experiment",
]