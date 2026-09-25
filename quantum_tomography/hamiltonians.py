"""
quantum_tomography.hamiltonians
===============================
Construção de Hamiltonianos físicos (Zeeman, Ising de Campo Transversal, Heisenberg)
e modelos de energia para tomografia quântica regularizada com termo de energia.
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from .observables import pauli_string_to_matrix, I2, sigma_x, sigma_y, sigma_z
from .states import QuantState, _symmetrize_hermitian


def single_qubit_zeeman(omega_z: float = 1.0, omega_x: float = 0.0) -> np.ndarray:
    r"""
    Hamiltoniano de Zeeman de 1 qubit em campo magnético:
    H = -\omega_z \sigma_z - \omega_x \sigma_x.
    Para \omega_x = 0 e \omega_z > 0, o estado fundamental é |0>.
    """
    H = -omega_z * sigma_z - omega_x * sigma_x
    return _symmetrize_hermitian(H)


def transverse_field_ising(
    n_qubits: int,
    J: float = 1.0,
    h: float = 1.0,
    periodic: bool = False,
) -> np.ndarray:
    r"""
    Modelo de Ising com Campo Transversal (TFIM) para N qubits:
    H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i  (com acoplamento N-1 -> 0 se periódico).
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # Termos de interação ZZ
    num_links = n_qubits if periodic and n_qubits > 2 else n_qubits - 1
    for i in range(num_links):
        j = (i + 1) % n_qubits
        chars = ["I"] * n_qubits
        chars[i] = "Z"
        chars[j] = "Z"
        p_str = "".join(chars)
        H -= J * pauli_string_to_matrix(p_str)

    # Termos de campo transversal X
    for i in range(n_qubits):
        chars = ["I"] * n_qubits
        chars[i] = "X"
        p_str = "".join(chars)
        H -= h * pauli_string_to_matrix(p_str)

    return _symmetrize_hermitian(H)


def heisenberg_xxz(
    n_qubits: int,
    J_xy: float = 1.0,
    Delta: float = 1.0,
    periodic: bool = False,
) -> np.ndarray:
    r"""
    Modelo de Heisenberg XXZ para N qubits:
    H = \sum_{i=0}^{N-2} [ J_{xy} (X_i X_{i+1} + Y_i Y_{i+1}) + \Delta Z_i Z_{i+1} ].
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    num_links = n_qubits if periodic and n_qubits > 2 else n_qubits - 1

    for i in range(num_links):
        j = (i + 1) % n_qubits
        for axis, coeff in (("X", J_xy), ("Y", J_xy), ("Z", Delta)):
            chars = ["I"] * n_qubits
            chars[i] = axis
            chars[j] = axis
            H += coeff * pauli_string_to_matrix("".join(chars))

    return _symmetrize_hermitian(H)


def random_hamiltonian(n_qubits: int, seed: int | None = None) -> np.ndarray:
    """Gera um Hamiltoniano hermitiano aleatório de N qubits (GUE)."""
    dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (G + G.conj().T) / 2.0
    return H


def ground_state(H: np.ndarray) -> tuple[float, QuantState]:
    """Retorna (energia_fundamental, estado_fundamental)."""
    H = _symmetrize_hermitian(H)
    eigvals, eigvecs = np.linalg.eigh(H)
    e0 = float(eigvals[0])
    psi0 = eigvecs[:, 0]
    return e0, QuantState(psi0)


def gibbs_state(H: np.ndarray, beta: float = 1.0) -> QuantState:
    """
    Estado térmico rho = exp(-beta*H) / Tr[exp(-beta*H)].
    beta -> inf tende ao estado fundamental (puro); beta -> 0 tende ao
    estado totalmente misto. Use scipy.linalg.expm.
    """
    H = _symmetrize_hermitian(H)
    e_min = float(np.linalg.eigvalsh(H)[0])
    # Shift para estabilidade numérica contra overflow em expm(-beta * H)
    op = expm(-beta * (H - e_min * np.eye(H.shape[0])))
    rho = op / np.trace(op).real
    return QuantState(rho)
