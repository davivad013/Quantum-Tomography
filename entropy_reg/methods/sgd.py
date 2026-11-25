"""Métodos estocásticos (SGD) para o problema dual de regularização entrópica."""

import time
import numpy as np
from scipy.linalg import expm

from entropy_reg.dual import dual_gradient, stochastic_gradient
from entropy_reg.observables import grad_completo, matriz_pauli


def sgd(H, Q_list, q, epsilon, k=None, lr=0.1, n_iter=10000, tol=1e-6,
        seed=0, u0=None, check_every=50, count_it=0):
    """Executa Stochastic Gradient Descent em k coordenadas sorteadas por iteração."""
    rng = np.random.default_rng(seed)
    u = np.zeros(len(q), dtype=float) if u0 is None else np.array(u0, dtype=float)
    m = len(u)
    grad_data = []

    for t in range(1, n_iter + 1):
        grad, idx = stochastic_gradient(u, H, Q_list, q, epsilon, rng=rng, k=k)
        u -= lr * grad

        if count_it > 0 and t % count_it == 0:
            grad = dual_gradient(u, H, Q_list, q, epsilon)
            grad_data.append([t, np.linalg.norm(grad)])

        # Critério de parada com o gradiente completo (k = m), verificado periodicamente
        if t % check_every == 0:
            full_grad, _ = stochastic_gradient(u, H, Q_list, q, epsilon, rng=rng, k=m)
            if np.linalg.norm(full_grad) < tol:
                return u, t, grad_data

    return u, n_iter, grad_data


def reconstruct_spi(H, Q_list, q, epsilon, max_iters, lr=0.01, tol=1e-6, k=None,
                    seed=0, count_it=0, check_every=50):
    """Reconstrói a matriz densidade ótima via SGD a partir de u ótimo."""
    u_opt, n_it, grad_data = sgd(
        H, Q_list, q, epsilon, k=k, lr=lr, n_iter=max_iters,
        tol=tol, seed=seed, count_it=count_it, check_every=check_every
    )
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_opt, Q_list)) - H

    return [expm(operator_sum / epsilon), u_opt, n_it, grad_data]


def rodar_estocastico_com_historico(H, Q_list, q, epsilon, k, lr,
                                    max_iters=5000, tol=1e-6, seed=0,
                                    check_every=10):
    """
    Atualiza k coordenadas sorteadas. A cada check_every passos, calcula
    a norma do gradiente completo para avaliar/parar o método.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros(len(q), dtype=float)
    m = len(q)
    historico = [[0, np.linalg.norm(dual_gradient(u, H, Q_list, q, epsilon))]]
    inicio = time.perf_counter()

    for t in range(1, max_iters + 1):
        grad_parcial, _ = stochastic_gradient(
            u, H, Q_list, q, epsilon, rng=rng, k=k
        )
        if not np.all(np.isfinite(grad_parcial)):
            break
        u -= lr * grad_parcial

        if t % check_every == 0:
            # k=m faz stochastic_gradient devolver todas as componentes:
            # portanto, aqui temos o gradiente completo.
            grad_completo_val, _ = stochastic_gradient(
                u, H, Q_list, q, epsilon, rng=rng, k=m
            )
            norma = np.linalg.norm(grad_completo_val)
            historico.append([t, norma])
            if np.isfinite(norma) and norma < tol:
                break

    tempo = time.perf_counter() - inicio
    return u, t, np.asarray(historico), tempo


def sgd_mem(H, q, n, epsilon, k, lr=1.0, n_iter=10000, tol=1e-6, seed=0,
            check_every=500, count_it=500):
    """
    SGD com memória reduzida (gera matrizes de Pauli sob demanda).
    Partida normalizada: u = 0 exceto u_I = -eps*ln(2^n) (Tr(pi)=1 no início).
    Retorna u, iterações feitas, histórico [[t, ||grad||], ...] e A final.
    """
    rng = np.random.default_rng(seed)
    d, m = 2**n, 4**n
    u = np.zeros(m)
    u[0] = -epsilon * np.log(d)
    A = u[0] * np.eye(d, dtype=complex) - H
    grad_data = []

    for t in range(1, n_iter + 1):
        pi = expm(A / epsilon)
        idx = rng.choice(m, size=k, replace=False)

        dA = np.zeros_like(A)
        for i in idx:
            Qi = matriz_pauli(i, n)  # gerada e descartada
            g = np.real(np.sum(Qi * pi.T)) - q[i]
            u[i] -= lr * g
            dA += (-lr * g) * Qi
        A += dA

        if t % count_it == 0 or t % check_every == 0:
            gn = np.linalg.norm(grad_completo(expm(A / epsilon), q, n))
            if t % count_it == 0:
                grad_data.append([t, gn])
            if not np.isfinite(gn):  # divergiu: não adianta continuar
                return u, t, grad_data, A
            if t % check_every == 0 and gn < tol:
                return u, t, grad_data, A

    return u, n_iter, grad_data, A
