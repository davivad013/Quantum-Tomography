"""
quantum_tomography.states
=========================
Representação de estados quânticos, geradores de famílias de estados
(puros, combinações de puros, mistos e térmicos) e métricas de informação quântica.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm, sqrtm


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


# ---------------------------------------------------------------------------
# Métricas fundamentais de informação quântica
# ---------------------------------------------------------------------------

def fidelidade(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    r"""
    Fidelidade de Uhlmann-Jozsa entre dois estados quânticos:
    F(\rho_a, \rho_b) = (\operatorname{Tr} \sqrt{\sqrt{\rho_a} \rho_b \sqrt{\rho_a}})^2.

    Retorna um float no intervalo [0, 1].
    """
    rho_a = np.asarray(rho_a, dtype=complex)
    rho_b = np.asarray(rho_b, dtype=complex)

    # Otimização se um dos estados for comprovadamente puro (posto 1)
    # F = <psi| rho |psi>
    sa = sqrt_psd(rho_a)
    m = sa @ rho_b @ sa
    sm = sqrt_psd(m)
    tr = np.trace(sm).real
    fid = float(np.clip(tr ** 2, 0.0, 1.0))
    return fid


def fidelity(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    """Alias em inglês para fidelidade."""
    return fidelidade(rho_a, rho_b)


def von_neumann_entropy(rho: np.ndarray, base: float = np.e) -> float:
    r"""
    Entropia de von Neumann:
    S(\rho) = -\operatorname{Tr}[\rho \log \rho] = -\sum_i \lambda_i \log \lambda_i.
    """
    rho_h = _symmetrize_hermitian(rho)
    eigvals = np.linalg.eigvalsh(rho_h)
    eigvals = np.clip(eigvals, 1e-300, 1.0)
    # Remove autovalores numericamente nulos
    pos_eigvals = eigvals[eigvals > 1e-15]
    if len(pos_eigvals) == 0:
        return 0.0
    s = -float(np.sum(pos_eigvals * np.log(pos_eigvals)))
    if base != np.e:
        s /= np.log(base)
    return max(0.0, s)


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

    Retorna a matriz densidade 2x2 reduzida correspondente ao qubit preservado,
    com garantia de simetria hermitiana e normalização de traço (Tr = 1).
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

    # Tensor de dimensão 2^(2N)
    tensor = mat.reshape([2] * (2 * n_qubits))
    chars = [chr(97 + i) for i in range(2 * n_qubits)]
    in_row = chars[:n_qubits]
    in_col = list(chars[n_qubits:])
    for i in range(n_qubits):
        if i != keep:
            in_col[i] = in_row[i]  # Contrai bra e ket do subsistema i

    in_sub = "".join(in_row) + "".join(in_col)
    out_sub = in_row[keep] + chars[n_qubits + keep]
    rho_red = np.einsum(f"{in_sub}->{out_sub}", tensor)

    rho_red_h = _symmetrize_hermitian(rho_red)
    tr = float(np.trace(rho_red_h).real)
    if tr > 1e-15:
        rho_red_h = rho_red_h / tr
    return rho_red_h


# ---------------------------------------------------------------------------
# Classe QuantState (Mantendo compatibilidade com códigos anteriores)
# ---------------------------------------------------------------------------

class QuantState:
    """
    Representação elegante de um estado quântico de N qubits.
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
        # Determina número de qubits se dimensão for potência de 2
        n = math.log2(self.dim)
        self.n_qubits: int | None = int(n) if n.is_integer() else None

    def measure(self, obs: np.ndarray) -> float:
        """Valor esperado exato de um observável hermitiano: Tr(rho * obs)."""
        return float(np.trace(self.rho @ obs).real)

    def energy(self, H: np.ndarray) -> float:
        """Valor esperado de energia perante um Hamiltoniano H: Tr(rho * H)."""
        return self.measure(H)

    @property
    def purity(self) -> float:
        return purity(self.rho)

    @property
    def entropy(self) -> float:
        return von_neumann_entropy(self.rho)

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
        """Calcula o vetor de Bloch (r_x, r_y, r_z) para sistemas de 1 qubit."""
        if self.dim != 2:
            raise ValueError("Vetor de Bloch definido apenas para 1 qubit (dim=2).")
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        return np.array([self.measure(sx), self.measure(sy), self.measure(sz)], dtype=float)

    def partial_trace(self, keep: int = 0) -> QuantState:
        """Calcula o estado quântico reduzido (1 qubit) obtido pelo traço parcial dos demais."""
        return QuantState(partial_trace(self.rho, keep=keep, n_qubits=self.n_qubits))

    def __repr__(self) -> str:
        q_str = f"{self.n_qubits} qubits" if self.n_qubits is not None else f"dim={self.dim}"
        tipo = "Puro" if self.is_pure else f"Misto (posto {self.rank})"
        return f"<QuantState ({q_str}, {tipo}, S={self.entropy:.3f}, pureza={self.purity:.3f})>"


# ---------------------------------------------------------------------------
# Geradores de famílias de estados
# ---------------------------------------------------------------------------

def random_pure_state(dim: int, seed: int | None = None) -> QuantState:
    """Gera um estado puro Haar-aleatório |v><v|."""
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return QuantState(vec)


def standard_pure_state(name: str, n_qubits: int = 1) -> QuantState:
    """
    Retorna estados puros de referência comuns:
    - 1 qubit: '0', '1', '+', '-', '+i', '-i'
    - 2 qubits: 'bell' (|00> + |11>)/sqrt(2), 'bell_minus', 'singlet'
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

    if name_clean == "all_zero" or name_clean == "0" * n_qubits:
        vec = np.zeros(dim, dtype=complex)
        vec[0] = 1.0
        return QuantState(vec)

    raise ValueError(f"Estado padrão desconhecido: '{name}' para {n_qubits} qubits.")


def random_mixture_of_pures(dim: int, rank: int = 2, seed: int | None = None) -> QuantState:
    r"""
    Gera uma combinação convexa de `rank` estados puros ortogonais ou aleatórios:
    \rho = \sum_{j=1}^{rank} p_j |v_j><v_j|,  \sum p_j = 1.
    """
    rng = np.random.default_rng(seed)
    if rank > dim:
        rank = dim

    # Gera autovetores via Haar / Ginibre
    G = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    Q, _ = np.linalg.qr(G)  # colunas ortonormais

    # Pesos de probabilidade aleatórios no simplex
    raw_p = rng.exponential(scale=1.0, size=rank)
    p = raw_p / np.sum(raw_p)

    rho = np.zeros((dim, dim), dtype=complex)
    for j in range(rank):
        v = Q[:, j]
        rho += p[j] * np.outer(v, v.conj())

    return QuantState(rho)


def depolarized_state(state: QuantState | np.ndarray, p: float) -> QuantState:
    r"""
    Aplica canal despolarizante ao estado:
    \mathcal{E}_p(\rho) = (1 - p) \rho + p \frac{I}{d}.
    Para p=0: estado original; para p=1: estado totalmente misto (I/d).
    """
    rho = state.rho if isinstance(state, QuantState) else np.asarray(state, dtype=complex)
    dim = rho.shape[0]
    p = float(np.clip(p, 0.0, 1.0))
    id_state = np.eye(dim, dtype=complex) / dim
    rho_depol = (1.0 - p) * rho + p * id_state
    return QuantState(rho_depol)


def maximally_mixed_state(dim: int) -> QuantState:
    """Gera o estado totalmente misto rho = I / d."""
    return QuantState(np.eye(dim, dtype=complex) / dim)


def thermal_state(H: np.ndarray, beta: float = 1.0) -> QuantState:
    r"""
    Gera o estado térmico de Gibbs associado ao Hamiltoniano H:
    \rho = \frac{e^{-\beta H}}{\operatorname{Tr}(e^{-\beta H})}.
    """
    H = _symmetrize_hermitian(H)
    # Subtrai o menor autovalor para evitar overflow numérico em expm(-beta * H)
    eig_min = float(np.linalg.eigvalsh(H)[0])
    H_shifted = H - eig_min * np.eye(H.shape[0])
    op = expm(-beta * H_shifted)
    tr = np.trace(op).real
    return QuantState(op / tr)


def gibbs_state(H: np.ndarray, beta: float = 1.0) -> QuantState:
    """
    Estado térmico rho = exp(-beta*H) / Tr[exp(-beta*H)].
    beta -> inf tende ao estado fundamental (puro); beta -> 0 tende ao
    estado totalmente misto. Use scipy.linalg.expm.
    """
    return thermal_state(H, beta=beta)


def random_state_with_matched_entropy(
    target_entropy: float,
    dim: int = 4,
    seed: int | None = None,
) -> QuantState:
    r"""
    Gera um estado de controle não-físico/aleatório com entropia de von Neumann
    aproximadamente igual a `target_entropy` (|S_ctrl - target_entropy| < 1e-6).

    Construído a partir de uma combinação convexa de autovetores aleatórios de Haar:
        \rho = U \operatorname{diag}(p(\alpha)) U^\dagger,
    onde p(\alpha) interpola entre um estado puro ([1, 0, ...]) e o estado totalmente misto (I/d),
    com \alpha determinado por busca de raiz escalar (brentq).
    """
    from scipy.optimize import brentq

    rng = np.random.default_rng(seed)
    # Autovetores aleatórios via matriz de Ginibre ortogonalizada (Haar)
    G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    U, _ = np.linalg.qr(G)

    max_s = float(np.log(dim))
    target_s = float(np.clip(target_entropy, 0.0, max_s))

    if target_s <= 1e-12:
        return QuantState(U[:, 0])

    if target_s >= max_s - 1e-12:
        return QuantState(np.eye(dim, dtype=complex) / dim)

    def f_entropy(alpha: float) -> float:
        p = (1.0 - alpha) * np.array([1.0] + [0.0] * (dim - 1)) + alpha * (np.ones(dim) / dim)
        p_pos = p[p > 1e-15]
        return -float(np.sum(p_pos * np.log(p_pos))) - target_s

    alpha_opt = float(brentq(f_entropy, 0.0, 1.0))
    p_opt = (1.0 - alpha_opt) * np.array([1.0] + [0.0] * (dim - 1)) + alpha_opt * (np.ones(dim) / dim)
    rho = U @ np.diag(p_opt) @ U.conj().T
    return QuantState(_symmetrize_hermitian(rho))
