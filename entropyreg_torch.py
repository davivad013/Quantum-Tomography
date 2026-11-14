import time
import torch
import numpy as np

# Dispositivo padrão: GPU se disponível, senão CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_device():
    return DEVICE


def set_default_device(device):
    global DEVICE
    DEVICE = torch.device(device)


def to_tensor(x, dtype=None, device=None):
    """Converte array numpy, lista ou tensor para torch.Tensor no dispositivo especificado."""
    dev = device if device is not None else DEVICE
    if isinstance(x, torch.Tensor):
        res = x
        if dtype is not None and res.dtype != dtype:
            res = res.to(dtype=dtype)
        if res.device != dev:
            res = res.to(device=dev)
        return res

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and isinstance(x[0], (np.ndarray, torch.Tensor)):
            tensors = [to_tensor(item, dtype=dtype, device=dev) for item in x]
            return torch.stack(tensors)
        x = np.asarray(x)

    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype=dtype)
        return t.to(device=dev)

    return torch.tensor(x, dtype=dtype, device=dev)


def _prepare_inputs(u, H, Q_list, q=None, device=None):
    dev = device if device is not None else DEVICE
    u_t = to_tensor(u, dtype=torch.float64, device=dev)
    Q_t = to_tensor(Q_list, dtype=torch.complex128, device=dev)

    d = Q_t.shape[-1]
    if isinstance(H, (int, float)) and H == 0:
        H_t = torch.zeros((d, d), dtype=torch.complex128, device=dev)
    else:
        H_t = to_tensor(H, dtype=torch.complex128, device=dev)

    if q is not None:
        q_t = to_tensor(q, dtype=torch.float64, device=dev)
        return u_t, H_t, Q_t, q_t
    return u_t, H_t, Q_t


def _dual_cost_core(u_t, H_t, Q_t, q_t, epsilon):
    if Q_t.ndim == 3:
        operator_sum = torch.tensordot(u_t.to(torch.complex128), Q_t, dims=([0], [0])) - H_t
    else:
        operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_t, Q_t)) - H_t
    arg = operator_sum / epsilon
    exp_matrix = torch.linalg.matrix_exp(arg)
    trace_val = torch.real(torch.trace(exp_matrix))
    return epsilon * trace_val - torch.dot(u_t, q_t)


def _dual_gradient_core(u_t, H_t, Q_t, q_t, epsilon):
    if Q_t.ndim == 3:
        operator_sum = torch.tensordot(u_t.to(torch.complex128), Q_t, dims=([0], [0])) - H_t
        arg = operator_sum / epsilon
        pi_u = torch.linalg.matrix_exp(arg)
        traces = torch.real(torch.sum(Q_t * pi_u.T, dim=(-2, -1)))
        return traces - q_t
    else:
        operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_t, Q_t)) - H_t
        arg = operator_sum / epsilon
        pi_u = torch.linalg.matrix_exp(arg)
        grad = torch.zeros_like(u_t)
        for i, Q_i in enumerate(Q_t):
            grad[i] = torch.real(torch.trace(Q_i @ pi_u)) - q_t[i]
        return grad


def _stochastic_gradient_core(u_t, H_t, Q_t, q_t, epsilon, idx):
    dev = u_t.device
    m = len(u_t)
    if Q_t.ndim == 3:
        operator_sum = torch.tensordot(u_t.to(torch.complex128), Q_t, dims=([0], [0])) - H_t
    else:
        operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_t, Q_t)) - H_t
    pi_u = torch.linalg.matrix_exp(operator_sum / epsilon)

    grad = torch.zeros(m, dtype=torch.float64, device=dev)
    if idx is None or len(idx) == m:
        traces = torch.real(torch.sum(Q_t * pi_u.T, dim=(-2, -1)))
        grad = traces - q_t
    else:
        sub_Q = Q_t[idx]
        traces = torch.real(torch.sum(sub_Q * pi_u.T, dim=(-2, -1)))
        grad[idx] = traces - q_t[idx]

    return grad


def dual_cost(u, H, Q_list, q, epsilon, device=None):
    """Função de custo dual implementada em PyTorch na GPU."""
    u_t, H_t, Q_t, q_t = _prepare_inputs(u, H, Q_list, q, device=device)
    cost = _dual_cost_core(u_t, H_t, Q_t, q_t, epsilon)
    return float(cost.item())


def dual_gradient(u, H, Q_list, q, epsilon, device=None):
    """Gradiente da função de custo dual implementado em PyTorch na GPU."""
    u_t, H_t, Q_t, q_t = _prepare_inputs(u, H, Q_list, q, device=device)
    return _dual_gradient_core(u_t, H_t, Q_t, q_t, epsilon)


def gradient_descent(H, Q_list, q, epsilon, max_iters, lr, tol, count_it=0, device=None):
    """Gradient descent implementado em PyTorch na GPU."""
    dev = device if device is not None else DEVICE
    q_t = to_tensor(q, dtype=torch.float64, device=dev)
    Q_t = to_tensor(Q_list, dtype=torch.complex128, device=dev)
    d = Q_t.shape[-1]
    if isinstance(H, (int, float)) and H == 0:
        H_t = torch.zeros((d, d), dtype=torch.complex128, device=dev)
    else:
        H_t = to_tensor(H, dtype=torch.complex128, device=dev)

    u = torch.zeros(len(q_t), dtype=torch.float64, device=dev)
    it = 0
    us = []
    grad_values = []

    for k in range(max_iters):
        grad = _dual_gradient_core(u, H_t, Q_t, q_t, epsilon)
        norm_grad = torch.linalg.norm(grad).item()
        grad_values.append([k, norm_grad])

        u_new = u - lr * grad
        it += 1
        if norm_grad < tol:
            it = k
            break
        us.append(u_new.clone())
        u = u_new

    return [u, it, us, grad_values]


def reconstruct_pi(H, Q_list, q, epsilon, max_iters, lr=0.01, tol=1e-6, count_it=0, device=None):
    """Reconstrói a matriz densidade ótima usando PyTorch na GPU."""
    dev = device if device is not None else DEVICE
    u_opt = gradient_descent(H, Q_list, q, epsilon, max_iters, lr, tol, count_it, device=dev)

    u_t, H_t, Q_t = _prepare_inputs(u_opt[0], H, Q_list, device=dev)
    if Q_t.ndim == 3:
        operator_sum = torch.tensordot(u_t.to(torch.complex128), Q_t, dims=([0], [0])) - H_t
    else:
        operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_t, Q_t)) - H_t

    pi_star = torch.linalg.matrix_exp(operator_sum / epsilon)
    return [pi_star, u_opt[0], u_opt[1], u_opt[2], u_opt[3]]


def stochastic_gradient(u, H, Q_list, q, epsilon, rng=None, k=None, device=None):
    """Calcula o gradiente estocástico em k direções usando PyTorch na GPU."""
    dev = device if device is not None else DEVICE
    u_t, H_t, Q_t, q_t = _prepare_inputs(u, H, Q_list, q, device=dev)

    m = len(u_t)
    k = m if k is None else min(k, m)

    if isinstance(rng, np.random.Generator):
        idx_np = rng.choice(m, size=k, replace=False)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=dev)
    elif isinstance(rng, torch.Generator):
        idx = torch.randperm(m, generator=rng, device=dev)[:k]
    elif isinstance(rng, (int, np.integer)):
        gen = torch.Generator(device=dev).manual_seed(int(rng))
        idx = torch.randperm(m, generator=gen, device=dev)[:k]
    else:
        idx = torch.randperm(m, device=dev)[:k]

    grad = _stochastic_gradient_core(u_t, H_t, Q_t, q_t, epsilon, idx if k < m else None)
    return grad, idx


def sgd(H, Q_list, q, epsilon, k=None, lr=0.1, n_iter=10000, tol=1e-6,
        seed=0, u0=None, check_every=50, count_it=0, device=None):
    """Stochastic Gradient Descent implementado em PyTorch na GPU."""
    dev = device if device is not None else DEVICE
    u_t, H_t, Q_t, q_t = _prepare_inputs(
        torch.zeros(len(q), dtype=torch.float64, device=dev) if u0 is None else u0,
        H, Q_list, q, device=dev
    )
    u = u_t.clone()
    m = len(u)
    k = m if k is None else min(k, m)
    grad_data = []

    if isinstance(seed, (int, np.integer)):
        gen = torch.Generator(device=dev).manual_seed(int(seed))
    elif isinstance(seed, torch.Generator):
        gen = seed
    else:
        gen = None

    for t in range(1, n_iter + 1):
        if gen is not None:
            idx = torch.randperm(m, generator=gen, device=dev)[:k]
        else:
            idx = torch.randperm(m, device=dev)[:k]

        grad = _stochastic_gradient_core(u, H_t, Q_t, q_t, epsilon, idx if k < m else None)
        u -= lr * grad

        if count_it > 0 and t % count_it == 0:
            full_g = _dual_gradient_core(u, H_t, Q_t, q_t, epsilon)
            norm_g = torch.linalg.norm(full_g).item()
            grad_data.append([t, norm_g])

        # Critério de parada com o gradiente COMPLETO (k = m), periodicamente
        if t % check_every == 0:
            full_grad = _dual_gradient_core(u, H_t, Q_t, q_t, epsilon)
            norm_full = torch.linalg.norm(full_grad).item()
            if norm_full < tol:
                return u, t, grad_data

    return u, n_iter, grad_data


def reconstruct_spi(H, Q_list, q, epsilon, max_iters, lr=0.01, tol=1e-6, k=None, seed=0, count_it=0, device=None):
    """Reconstrói a matriz densidade com SGD em PyTorch na GPU."""
    dev = device if device is not None else DEVICE
    u_opt, n_it, grad_data = sgd(H, Q_list, q, epsilon, k=k, lr=lr,
                                 n_iter=max_iters, tol=tol, seed=seed,
                                 count_it=count_it, device=dev)

    u_t, H_t, Q_t = _prepare_inputs(u_opt, H, Q_list, device=dev)
    if Q_t.ndim == 3:
        operator_sum = torch.tensordot(u_t.to(torch.complex128), Q_t, dims=([0], [0])) - H_t
    else:
        operator_sum = sum(u_i * Q_i for u_i, Q_i in zip(u_t, Q_t)) - H_t

    pi_star = torch.linalg.matrix_exp(operator_sum / epsilon)
    return [pi_star, u_opt, n_it, grad_data]
