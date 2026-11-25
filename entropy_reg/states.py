"""Funções auxiliares para geração de Hamiltonianos e estados térmicos de Gibbs."""

import numpy as np
from scipy.linalg import expm

import quantum_tomography as qt
from entropy_reg.dual import normalizar_rho


def hamiltoniano_ising(n, g=1.0):
    """H_fis = - sum Z_i Z_{i+1} - g sum X_i (apenas para gerar o estado de Gibbs)."""
    Hf = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n - 1):
        s = ["I"] * n
        s[i] = s[i + 1] = "Z"
        Hf -= qt.pauli_string_to_matrix("".join(s))
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        Hf -= g * qt.pauli_string_to_matrix("".join(s))
    return Hf


def estado_gibbs(H, beta):
    """Estado térmico exp(-beta H)/Z retornado como QuantState."""
    return qt.QuantState(normalizar_rho(expm(-beta * H)))


def rho_gibbs(n, beta):
    """Matriz densidade normalizada do estado térmico de Gibbs da cadeia de Ising."""
    rho = expm(-beta * hamiltoniano_ising(n))
    return rho / np.trace(rho).real
