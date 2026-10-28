import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import least_squares


# Importa a biblioteca desenvolvida para o projeto
import quantum_tomography as qt

""" Tentando fazer um solver usando Gradient Descent """

# Definimos a função de custo dual, que é a função que queremos minimizar em relação a u.
def dual_cost(u, H, Q_list, q, epsilon):

    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    
    arg = operator_sum / epsilon

    exp_matrix = expm(arg)
    trace_val = np.real(np.trace(exp_matrix))

    cost = epsilon * trace_val - np.dot(u, q)
    return cost

"""
Como sabemos a cara do nosso problema, podemos determinar 
o gradiente de modo analítico. Dessa forma, apenas utilizamos 
a "fórmula fechada" para construir a função do gradiente, 
e então, como nosso problema é estritamente convexo, nos basta
usar o método de Gradient Descent para encontrar o mínimo global
para u. Assim, com a forma teórica que encontramos para pi star,
podemos reconstruir a matriz densidade ótima a partie de u ótimo,
então, resolvemos o problema com a regularização entrópica.
"""

# Construímos o gradiente da função de custo dual
def dual_gradient(u, H, Q_list, q, epsilon):
    
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    arg = operator_sum / epsilon
    
    pi_u = expm(arg)
    
    grad = np.zeros_like(u, dtype=float)
    for i, Q_i in enumerate(Q_list):
        grad[i] = np.real(np.trace(Q_i @ pi_u)) - q[i]
        
    return grad

# Implementamos o método de Gradient Descent para encontrar o mínimo global da função de custo dual
def gradient_descent(H, Q_list, q, epsilon, max_iters, lr, tol):

    u = np.zeros(len(q), dtype=float)  # Inicialização
    it = 0
    us = []
    grad_values = []
    for k in range(max_iters):
        grad = dual_gradient(u, H, Q_list, q, epsilon)
        grad_values.append([k, np.linalg.norm(grad)])

        # Passo do Gradient Descent: u_{k+1} = u_k - lr * grad
        u_new = u - lr * grad
        
        if np.linalg.norm(grad) < tol:
            it = k
            break
        us.append(u_new.copy())    
        u = u_new

    return [u, it, us, grad_values]

# Reconstruímos a matriz densidade ótima a partir de u ótimo
def reconstruct_pi(H, Q_list, q, epsilon, max_iters, lr=0.01, tol=1e-6):

    u_opt = gradient_descent(H, Q_list, q, epsilon, max_iters, lr, tol)
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_opt[0], Q_list)) - H

    return [expm(operator_sum / epsilon), u_opt[0], u_opt[1], u_opt[2], u_opt[3]]


""" Tentando implementar o Stochastic Method """


"""
Para implementar o stochastic method, apenas selecionamos 
aleatoriamente k direções do gradiente para atualizar u,
e então otimizá-lo. Desse modo, definimos novas funções para 
o gradiente e para o método de otimização.
"""

# Função que calcula aleatoriamente o gradiente em k direções
def stochastic_gradient(u, H, Q_list, q, epsilon, rng=None, k=None):

    rng = np.random.default_rng() if rng is None else rng
    m = len(u)
    k = m if k is None else min(k, m)

    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u, Q_list)) - H
    pi_u = expm(operator_sum / epsilon)

    idx = rng.choice(m, size=k, replace=False)   # direções sorteadas

    grad = np.zeros(m, dtype=float)
    for i in idx:
        grad[i] = np.real(np.sum(Q_list[i] * pi_u.T)) - q[i]

    return grad, idx


def sgd(H, Q_list, q, epsilon, k=None, lr=0.1, n_iter=10000, tol=1e-6,
        seed=0, u0=None, check_every=50):
    
    rng = np.random.default_rng(seed)
    u = np.zeros(len(q), dtype=float) if u0 is None else np.array(u0, dtype=float)
    m = len(u)
    us = []

    for t in range(1, n_iter + 1):
        grad, idx = stochastic_gradient(u, H, Q_list, q, epsilon, rng=rng, k=k)
        u -= lr * grad
        us.append(u.copy())

        # critério de parada com o gradiente COMPLETO (k = m), só de vez em quando
        if t % check_every == 0:
            full_grad, _ = stochastic_gradient(u, H, Q_list, q, epsilon, rng=rng, k=m)
            if np.linalg.norm(full_grad) < tol:

                return u, t, us

    return u, n_iter, us


def reconstruct_spi(H, Q_list, q, epsilon, max_iters, lr=0.01, tol=1e-6, k=None, seed=0):

    u_opt, n_it, us = sgd(H, Q_list, q, epsilon, k=k, lr=lr,
                      n_iter=max_iters, tol=tol, seed=seed)
    operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_opt, Q_list)) - H

    return [expm(operator_sum / epsilon), u_opt, n_it, us]