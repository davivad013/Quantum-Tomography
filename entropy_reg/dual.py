"""Funções do problema dual, reconstrução de estado e entropia."""

import numpy as np
from scipy.linalg import expm


def dual_cost(u, H, Q_list, q, epsilon):
    """Calcula o valor da função de custo dual g(u)."""
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    arg = operator_sum / epsilon

    exp_matrix = expm(arg)
    trace_val = np.real(np.trace(exp_matrix))

    cost = epsilon * trace_val - np.dot(u, q)
    return cost


def dual_gradient(u, H, Q_list, q, epsilon):
    """Calcula o gradiente analítico completo da função de custo dual."""
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    arg = operator_sum / epsilon

    pi_u = expm(arg)

    grad = np.zeros_like(u, dtype=float)
    for i, Q_i in enumerate(Q_list):
        grad[i] = np.real(np.trace(Q_i @ pi_u)) - q[i]

    return grad


def stochastic_gradient(u, H, Q_list, q, epsilon, rng=None, k=None):
    """Calcula aleatoriamente o gradiente em k direções sorteadas."""
    rng = np.random.default_rng() if rng is None else rng
    m = len(u)
    k = m if k is None else min(k, m)

    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    pi_u = expm(operator_sum / epsilon)

    idx = rng.choice(m, size=k, replace=False)  # direções sorteadas

    grad = np.zeros(m, dtype=float)
    for i in idx:
        grad[i] = np.real(np.sum(Q_list[i] * pi_u.T)) - q[i]

    return grad, idx


def rho_a_partir_de_u(u, H, Q_list, epsilon):
    """Reconstrói a matriz não normalizada exp((sum u_i Q_i - H) / epsilon)."""
    operador = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    return expm(operador / epsilon)


def normalizar_rho(rho):
    """Normaliza uma matriz densidade pelo seu traço."""
    tr = np.trace(rho).real
    if not np.isfinite(tr) or tr <= 0:
        raise ValueError("A exponencial não produziu um traço positivo e finito.")
    return rho / tr


def entropia_von_neumann(rho):
    """Calcula a entropia de von Neumann de uma matriz densidade."""
    rho_h = (rho + rho.conj().T) / 2
    autovalores = np.linalg.eigvalsh(rho_h)
    autovalores = np.clip(autovalores, 0.0, None)
    positivos = autovalores[autovalores > 1e-14]
    return float(-np.sum(positivos * np.log(positivos)))
