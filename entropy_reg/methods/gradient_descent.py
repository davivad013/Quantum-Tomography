"""Método de Gradient Descent para o problema dual de regularização entrópica."""

import numpy as np
from scipy.linalg import expm

from entropy_reg.dual import dual_gradient


def gradient_descent(H, Q_list, q, epsilon, max_iters, lr, tol, count_it=0):
    """Executa Gradient Descent para encontrar o mínimo global da função de custo dual."""
    u = np.zeros(len(q), dtype=float)  # Inicialização
    it = 0
    us = []
    grad_values = []
    for k in range(max_iters):
        grad = dual_gradient(u, H, Q_list, q, epsilon)
        grad_values.append([k, np.linalg.norm(grad)])

        # Passo do Gradient Descent: u_{k+1} = u_k - lr * grad
        u_new = u - lr * grad
        it += 1
        if np.linalg.norm(grad) < tol:
            it = k
            break
        us.append(u_new.copy())
        u = u_new

    return [u, it, us, grad_values]


def reconstruct_pi(H, Q_list, q, epsilon, max_iters, lr=0.01, tol=1e-6, count_it=0):
    """Reconstrói a matriz densidade ótima via Gradient Descent a partir de u ótimo."""
    u_opt = gradient_descent(H, Q_list, q, epsilon, max_iters, lr, tol, count_it)
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_opt[0], Q_list)) - H

    return [expm(operator_sum / epsilon), u_opt[0], u_opt[1], u_opt[2], u_opt[3]]
