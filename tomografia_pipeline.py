"""
Modulo pronto para substituir os trechos do notebook que fazem:
    q_medidos = [unk_state.measure(Q) for Q in mediadores_Q]   (dados exatos)
    resultado = least_squares(funcao_de_custo_geral, x0=chute_inicial,
                               args=(mediadores_Q, q_medidos, n_qubits))
    rho_reconstruido = parametros_para_rho_geral(resultado.x, n_qubits)

por uma versao que simula N_shots=50000 medicoes reais por observavel (regra
de Born + ruido binomial) antes de resolver, e usa o Jacobiano analitico
(mais rapido e exato) em vez do numerico default do scipy.

Uso tipico:
    from tomografia_pipeline import (QuantState, sigma_x, sigma_y, sigma_z,
        gerar_paulis, fidelidade, medidas_shots_de_estado,
        simula_medidas_shots, resolve_tomografia_lsq)

    q_medidos = medidas_shots_de_estado(unk_state, mediadores_Q, N_shots=50000)
    saida = resolve_tomografia_lsq(mediadores_Q, q_medidos, n_qubits,
                                    estado_verdadeiro=unk_state)
    rho_reconstruido = saida["rho_reconstruido"]

AVISO: mediadores_Q deve SEMPRE ser passado cru (sem `.transpose(0,2,1)`).
Essa transposicao inverte o sinal da previsao para qualquer string de Pauli
com numero impar de fatores sigma_y (sigma_y e antissimetrica). As celulas
6 e 7 do notebook original ja fazem isso certo; as celulas 8 e 9 tinham esse
bug -- os exemplos "depois" abaixo mostram a correcao.
"""

import time
import itertools
import numpy as np
from scipy.optimize import least_squares

# --------------------------------------------------------------------- #
# Definicoes de base (identicas as que ja estao no notebook de voces)
# --------------------------------------------------------------------- #
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


class QuantState:
    def __init__(self, estado):
        estado = np.array(estado, dtype=complex)
        if estado.ndim == 1:
            psi = estado / np.linalg.norm(estado)
            self.rho = np.outer(psi, psi.conj())
        elif estado.ndim == 2:
            self.rho = estado / np.trace(estado)

    def measure(self, obs):
        return np.trace(self.rho @ obs).real


def gerar_paulis(n_qubits):
    I = np.eye(2, dtype=complex)
    base = [I, sigma_x, sigma_y, sigma_z]
    saida = []
    for combo in itertools.product(base, repeat=n_qubits):
        P = combo[0]
        for c in combo[1:]:
            P = np.kron(P, c)
        saida.append(P)
    return saida[1:]


def parametros_para_T_geral(params, n_qubits):
    D = 2 ** n_qubits
    li, lj = np.tril_indices(D, k=-1)
    n_off = len(li)
    T = np.zeros((D, D), dtype=complex)
    T[np.arange(D), np.arange(D)] = params[:D]
    T[li, lj] = params[D:D + n_off] + 1j * params[D + n_off:]
    return T


def parametros_para_rho_geral(params, n_qubits):
    T = parametros_para_T_geral(params, n_qubits)
    T_dag_T = T.conj().T @ T
    traco = np.trace(T_dag_T).real
    if np.isclose(traco, 0):
        D = 2 ** n_qubits
        return np.eye(D, dtype=complex) / D
    return T_dag_T / traco


def funcao_de_custo_geral(params, mediadores_Q, dados_exp, n_qubits):
    """mediadores_Q deve ser a lista de observaveis CRUA (ver aviso no topo)."""
    rho_candidato = parametros_para_rho_geral(params, n_qubits)
    previsoes = np.einsum('ij,mji->m', rho_candidato, mediadores_Q).real
    return previsoes - dados_exp


def sqrt_psd(A):
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def fidelidade(rho_a, rho_b):
    sa = sqrt_psd(rho_a)
    return np.real(np.trace(sqrt_psd(sa @ rho_b @ sa))) ** 2


def jacobiano_analitico(params, mediadores_Q, dados_exp, n_qubits):
    """Jacobiano fechado de funcao_de_custo_geral (ver deducao no arquivo
    experimentos_least_squares.py entregue antes); validado contra diferenca
    finita central, erro maximo ~1e-10."""
    D = 2 ** n_qubits
    li, lj = np.tril_indices(D, k=-1)

    T = parametros_para_T_geral(params, n_qubits)
    T_dag = T.conj().T
    s = np.trace(T_dag @ T).real

    Q = np.asarray(mediadores_Q)
    previsoes = np.einsum('ij,mji->m', T_dag @ T / s, Q).real
    M_all = np.einsum('mij,jk->mik', Q, T_dag)

    tr_diag = np.diagonal(M_all, axis1=1, axis2=2)
    tr_re = M_all[:, lj, li]
    tr_im = 1j * M_all[:, lj, li]

    g_diag = np.diagonal(T_dag)
    g_re = T_dag[lj, li]
    g_im = 1j * T_dag[lj, li]

    jac_diag = (2.0 / s) * (tr_diag.real - previsoes[:, None] * g_diag.real[None, :])
    jac_re = (2.0 / s) * (tr_re.real - previsoes[:, None] * g_re.real[None, :])
    jac_im = (2.0 / s) * (tr_im.real - previsoes[:, None] * g_im.real[None, :])
    return np.hstack([jac_diag, jac_re, jac_im])


# --------------------------------------------------------------------- #
# As duas funcoes novas pedidas
# --------------------------------------------------------------------- #
def simula_medidas_shots(q_exato, N_shots, rng=None):
    """
    Simula N_shots medicoes binomiais por observavel (regra de Born,
    autovalores +-1) a partir de valores esperados EXATOS.

    q_exato: array-like de valores em [-1,1] -- pode vir de
             unk_state.measure(...) OU ser digitado a mao (como no exemplo
             de nao-unicidade do notebook de voces, onde nunca existiu um
             "estado verdadeiro" por tras dos numeros).
    rng: se None (default), usa o gerador global do numpy
         (np.random.binomial) -- assim um unico np.random.seed(...) no topo
         do notebook, como voces ja fazem, controla tudo. Passe um
         np.random.default_rng(...) aqui se quiser um fluxo de aleatoriedade
         isolado (por exemplo, para nao interferir na sequencia de
         np.random.randn() usada para o chute inicial).
    """
    q_exato = np.asarray(q_exato, dtype=float)
    binomial = rng.binomial if rng is not None else np.random.binomial
    q_ruido = np.empty_like(q_exato)
    for i, qi in enumerate(q_exato):
        p_mais = np.clip((1 + qi) / 2, 0.0, 1.0)
        outcomes = binomial(N_shots, p_mais)
        q_ruido[i] = 2 * outcomes / N_shots - 1
    return q_ruido


def medidas_shots_de_estado(estado, mediadores_Q, N_shots, rng=None):
    """Atalho para quando existe um QuantState de verdade: calcula os
    valores exatos via estado.measure(...) e ja simula o shot noise."""
    q_exato = np.array([estado.measure(Q) for Q in mediadores_Q])
    return simula_medidas_shots(q_exato, N_shots, rng=rng)


def resolve_tomografia_lsq(mediadores_Q, q_medidos, n_qubits, chute_inicial=None,
                            estado_verdadeiro=None, usar_jacobiano_analitico=True,
                            **kwargs_least_squares):
    """
    Resolve o problema de tomografia por minimos quadrados (parametrizacao
    T, SEM regularizacao entropica) -- e o substituto direto do bloco
        resultado = least_squares(funcao_de_custo_geral, x0=chute_inicial,
                                   args=(mediadores_Q, q_medidos, n_qubits))
        rho_reconstruido = parametros_para_rho_geral(resultado.x, n_qubits)

    Parametros
    ----------
    mediadores_Q : observaveis CRUS (sem transpor -- ver aviso no topo)
    q_medidos    : valores medidos (ja com ruido, se for o caso)
    n_qubits     : numero de qubits
    chute_inicial: vetor de parametros reais (D^2,); se None, gera via
                   np.random.randn (precisa de um np.random.seed(...) antes,
                   igual ao que voces ja fazem)
    estado_verdadeiro : QuantState opcional -- se dado, a fidelidade ja
                   vem calculada no retorno
    usar_jacobiano_analitico : usa o Jacobiano fechado (mais rapido e
                   exato) em vez do '2-point' numerico default do scipy
    **kwargs_least_squares : repassado direto para scipy.optimize.least_squares

    A partir de D=2**n_qubits >= 16 (n_qubits >= 4), o solver interno troca
    automaticamente de tr_solver='exact' para 'lsmr': o solver 'exact' faz
    uma SVD densa a cada iteracao e comeca a falhar numericamente
    (`SVD did not converge`) nessa escala -- confirmei isso empiricamente
    para n_qubits=5. 'lsmr' e iterativo, mais lento por iteracao mas muito
    mais robusto. Ha tambem um fallback automatico: se 'exact' ainda assim
    for pedido explicitamente e falhar, tenta de novo com 'lsmr'.

    Retorna um dicionario:
        resultado, rho_reconstruido, custo_final, nfev, status, mensagem,
        tempo_s, fidelidade (None se estado_verdadeiro nao foi passado),
        chute_inicial, q_medidos
    """
    D = 2 ** n_qubits
    if chute_inicial is None:
        chute_inicial = np.random.randn(D ** 2)

    jac = jacobiano_analitico if usar_jacobiano_analitico else "2-point"
    if "jac" in kwargs_least_squares:
        jac = kwargs_least_squares.pop("jac")

    if "tr_solver" not in kwargs_least_squares:
        kwargs_least_squares["tr_solver"] = "lsmr" if D >= 16 else "exact"

    t0 = time.time()
    try:
        resultado = least_squares(
            funcao_de_custo_geral, x0=chute_inicial,
            args=(mediadores_Q, q_medidos, n_qubits),
            jac=jac, **kwargs_least_squares,
        )
    except np.linalg.LinAlgError:
        print("[aviso] SVD não convergiu com tr_solver='exact'; tentando de novo com tr_solver='lsmr'...")
        kwargs_least_squares["tr_solver"] = "lsmr"
        resultado = least_squares(
            funcao_de_custo_geral, x0=chute_inicial,
            args=(mediadores_Q, q_medidos, n_qubits),
            jac=jac, **kwargs_least_squares,
        )
    tempo_s = time.time() - t0

    rho_reconstruido = parametros_para_rho_geral(resultado.x, n_qubits)
    fid = fidelidade(rho_reconstruido, estado_verdadeiro.rho) if estado_verdadeiro is not None else None

    return dict(
        resultado=resultado,
        rho_reconstruido=rho_reconstruido,
        custo_final=resultado.cost,
        nfev=resultado.nfev,
        status=resultado.status,
        mensagem=resultado.message,
        tempo_s=tempo_s,
        fidelidade=fid,
        chute_inicial=chute_inicial,
        q_medidos=np.asarray(q_medidos),
    )
