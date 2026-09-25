"""
quantum_tomography.observables
==============================
Álgebra de Pauli, geração de bases de medição para N qubits,
subconjuntos de observáveis e simulação de medidas quânticas (exatas e com ruído de shot noise).
"""

from __future__ import annotations
import itertools
from typing import Sequence
import numpy as np

# Matrizes de Pauli de 1 qubit
I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Aliases curtos
sx, sy, sz = sigma_x, sigma_y, sigma_z

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


def gerar_paulis(n_qubits: int, include_identity: bool = False) -> list[np.ndarray]:
    """
    Gera a base completa de matrizes de Pauli para n qubits (mantendo o nome e comportamento
    exato da função original dos autores).
    """
    strings = pauli_basis_strings(n_qubits, include_identity=include_identity)
    return [pauli_string_to_matrix(s) for s in strings]


# ---------------------------------------------------------------------------
# Seleção de subconjuntos de observáveis (Medições incompletas / Não-unicidade)
# ---------------------------------------------------------------------------

def select_coordinate_paulis(n_qubits: int = 1, axes: Sequence[str] = ("X", "Y")) -> list[str]:
    """
    Gera observáveis apenas nos eixos especificados.
    Para 1 qubit e axes=('X', 'Y'): gera ['X', 'Y'], omitindo 'Z'.
    Útil para reproduzir o problema clássico da não-unicidade na esfera de Bloch.
    """
    axes_clean = [a.upper().strip() for a in axes]
    for a in axes_clean:
        if a not in ("X", "Y", "Z"):
            raise ValueError(f"Eixo desconhecido: '{a}'")
    return list(axes_clean)


def select_random_paulis(n_qubits: int, num_observables: int, seed: int | None = None) -> list[str]:
    """Seleciona aleatoriamente um subconjunto de Paulis traceless."""
    rng = np.random.default_rng(seed)
    all_paulis = pauli_basis_strings(n_qubits, include_identity=False)
    if num_observables >= len(all_paulis):
        return all_paulis
    indices = rng.choice(len(all_paulis), size=num_observables, replace=False)
    return [all_paulis[i] for i in sorted(indices)]


def select_k_local_paulis(n_qubits: int, max_weight: int = 2) -> list[str]:
    """
    Gera apenas Pauli strings com peso (número de fatores não-identidade) até `max_weight`.
    Corresponde a medições fisicamente realistas em processadores quânticos (1-local e 2-local).
    """
    all_paulis = pauli_basis_strings(n_qubits, include_identity=False)
    k_local = []
    for s in all_paulis:
        weight = sum(1 for ch in s if ch != "I")
        if 1 <= weight <= max_weight:
            k_local.append(s)
    return k_local


# ---------------------------------------------------------------------------
# Simulação de Medições Quânticas (Exatas e com Shot Noise)
# ---------------------------------------------------------------------------

def simula_medidas_shots(
    q_exato: Sequence[float] | np.ndarray,
    N_shots: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula N_shots medições binomiais independentes por observável (regra de Born, autovalores +-1).

    q_exato: vetor com valores esperados exatos em [-1, 1].
    N_shots: número de repetições da medição projetiva.
    rng: gerador numpy opcional.
    seed: semente opcional se rng não for fornecido.
    """
    q_exato = np.asarray(q_exato, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed)
    binomial = rng.binomial
    q_ruido = np.empty_like(q_exato)

    for i, qi in enumerate(q_exato):
        # Probabilidade do autovalor +1: p_+ = (1 + <Q>) / 2
        p_mais = float(np.clip((1.0 + qi) / 2.0, 0.0, 1.0))
        outcomes = binomial(N_shots, p_mais)
        # Estimador empírico: 2 * (N_+ / N) - 1
        q_ruido[i] = 2.0 * outcomes / N_shots - 1.0

    return q_ruido


def medidas_shots_de_estado(
    estado,
    mediadores_Q: Sequence[np.ndarray],
    N_shots: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Calcula valores exatos a partir do estado e aplica shot noise binomial."""
    rho = estado.rho if hasattr(estado, "rho") else np.asarray(estado, dtype=complex)
    q_exato = np.array([float(np.trace(rho @ Q).real) for Q in mediadores_Q])
    return simula_medidas_shots(q_exato, N_shots, rng=rng, seed=seed)


def measure_observables(
    estado,
    observables: Sequence[str | np.ndarray],
    N_shots: int | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Função unificada de medição:
    - Se observables contiver strings (ex: ['XX', 'ZZ']), converte automaticamente para matrizes.
    - Se N_shots for None: retorna valores esperados analíticos exatos.
    - Se N_shots for int: aplica simulação de shot noise binomial.
    """
    rho = estado.rho if hasattr(estado, "rho") else np.asarray(estado, dtype=complex)
    mats = [pauli_string_to_matrix(obs) if isinstance(obs, str) else obs for obs in observables]
    q_exato = np.array([float(np.trace(rho @ Q).real) for Q in mats])

    if N_shots is None:
        return q_exato
    return simula_medidas_shots(q_exato, N_shots, rng=rng, seed=seed)
