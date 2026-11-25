"""Funções auxiliares para construção de observáveis e medidas de Pauli."""

import numpy as np

import quantum_tomography as qt

LETRAS = "IXYZ"


def montar_observaveis_pauli(n_qubits):
    """Retorna os nomes e as matrizes de todos os 4^n observáveis de Pauli (com identidade)."""
    nomes = ["I" * n_qubits] + qt.pauli_basis_strings(n_qubits)
    matrizes = [qt.pauli_string_to_matrix(nome) for nome in nomes]
    return nomes, matrizes


def nome_pauli(idx, n):
    """Índice 0..4^n-1 -> string de Pauli (mesma ordem de qt.pauli_basis_strings com identidade)."""
    s = []
    for _ in range(n):
        s.append(LETRAS[idx & 3])
        idx >>= 2
    return "".join(reversed(s))


def matriz_pauli(idx, n):
    """Gera sob demanda a matriz do observável de Pauli correspondente ao índice idx."""
    return qt.pauli_string_to_matrix(nome_pauli(idx, n))


def medidas_pauli(rho, n):
    """q_i = Tr(Q_i rho) para todas as 4^n Paulis, sem guardar as matrizes."""
    q = np.empty(4**n)
    rho_T = rho.T
    for i in range(4**n):
        q[i] = np.real(np.sum(matriz_pauli(i, n) * rho_T))
    return q


def grad_completo(pi, q, n):
    """Gradiente completo: Tr(Q_i pi) - q_i para as 4^n coordenadas (em loop)."""
    g = np.empty(4**n)
    pi_T = pi.T
    for i in range(4**n):
        g[i] = np.real(np.sum(matriz_pauli(i, n) * pi_T)) - q[i]
    return g
