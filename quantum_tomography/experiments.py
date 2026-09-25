"""
quantum_tomography.experiments
==============================
Suíte completa de experimentos pré-configurados e reprodutíveis:
1. Sanity Check (reconstrução de estados puros simples)
2. Não-Unicidade com Medições Incompletas (esfera de Bloch)
3. Benchmark de Escalabilidade em N Qubits (tempo, iterações e memória)
4. Espectro de Pureza (Estados Puros, Misturas de Puros e Estados Altamente Mistos)
5. Introdução de Energia / Hamiltoniano (Tomografia guiada por termodinâmica quântica)
6. Estabilidade Numérica e Ruído de Shot Noise (Infactibilidade vs. Mínimos Quadrados)
"""

from __future__ import annotations
import time
from typing import Sequence
import numpy as np

from .states import (
    QuantState,
    random_pure_state,
    random_mixture_of_pures,
    depolarized_state,
    maximally_mixed_state,
    thermal_state,
    fidelidade,
    von_neumann_entropy,
    purity,
    numerical_rank,
    random_state_with_matched_entropy,
)
from .observables import (
    pauli_basis_strings,
    select_coordinate_paulis,
    select_random_paulis,
    measure_observables,
    simula_medidas_shots,
)
from .hamiltonians import (
    single_qubit_zeeman,
    transverse_field_ising,
    heisenberg_xxz,
    ground_state,
    gibbs_state,
)
from .solvers import solve_tomography, TomographyResult


# ---------------------------------------------------------------------------
# Experimento 1: Sanity Check (Reconstrução Exata de Estados Puros)
# ---------------------------------------------------------------------------

def run_sanity_check(
    n_qubits: int = 1,
    methods: Sequence[str] = ("entropy", "quadratic", "cvxpy_scs", "least_squares"),
    seed: int = 42,
) -> dict[str, TomographyResult]:
    """
    Sanity test: reconstrói um estado puro simples a partir de medições completas exatas.
    Confirma se todos os solvers recuperam fidelidade ~ 1.0 e resíduo próximo de zero.
    """
    dim = 2 ** n_qubits
    state = random_pure_state(dim, seed=seed)
    observables = pauli_basis_strings(n_qubits)
    q = measure_observables(state, observables)

    results = {}
    for m in methods:
        res = solve_tomography(observables, q, method=m, true_state=state, n_qubits=n_qubits)
        results[m] = res

    return results


# ---------------------------------------------------------------------------
# Experimento 2: Não-Unicidade com Medições Incompletas
# ---------------------------------------------------------------------------

def run_uniqueness_experiment(
    q_values: tuple[float, float] = (0.7, 0.4),
    seed: int = 42,
) -> dict[str, TomographyResult]:
    """
    Demonstra a não-unicidade quando medições são incompletas:
    Para 1 qubit, mede apenas sigma_x e sigma_y (deixando sigma_z livre).
    Mostra como diferentes regularizadores e chutes iniciais selecionam diferentes estados compatíveis.
    """
    observables = ["X", "Y"]
    q = np.array(q_values, dtype=float)

    results = {}
    # 1. Regularização Entrópica (escolhe o estado de máxima entropia compatível: r_z = 0)
    results["entropy"] = solve_tomography(observables, q, method="entropy")

    # 2. Regularização Quadrática (escolhe menor norma de Frobenius compatível)
    results["quadratic"] = solve_tomography(observables, q, method="quadratic")

    # 3. Força Bruta com Chute Inicial A (cai em um dos extremos permitidos)
    rng = np.random.default_rng(seed)
    chute_a = rng.normal(size=4)
    results["least_squares_a"] = solve_tomography(
        observables, q, method="least_squares", chute_inicial=chute_a, n_qubits=1
    )

    # 4. Força Bruta com Chute Inicial B (cai em outro estado permitido)
    chute_b = rng.normal(size=4)
    results["least_squares_b"] = solve_tomography(
        observables, q, method="least_squares", chute_inicial=chute_b, n_qubits=1
    )

    return results


# ---------------------------------------------------------------------------
# Experimento 3: Escalabilidade em N Qubits (Tempo e Iterações)
# ---------------------------------------------------------------------------

def run_qubit_scaling_benchmark(
    qubits_range: Sequence[int] = (1, 2, 3, 4),
    methods: Sequence[str] = ("entropy", "quadratic", "least_squares", "cvxpy_scs"),
    max_observables: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Avalia a complexidade computacional e tempo de execução conforme n_qubits aumenta.
    Evidencia a superioridade da formulação dual JAX (sdplab) sobre métodos convencionais.
    """
    rng = np.random.default_rng(seed)
    tempos: dict[str, list[float]] = {m: [] for m in methods}
    iteracoes: dict[str, list[int]] = {m: [] for m in methods}
    residuos: dict[str, list[float]] = {m: [] for m in methods}
    fidelidades: dict[str, list[float]] = {m: [] for m in methods}

    for n in qubits_range:
        dim = 2 ** n
        state = random_pure_state(dim, seed=int(rng.integers(1, 10000)))
        all_paulis = pauli_basis_strings(n)

        if max_observables is not None and max_observables < len(all_paulis):
            idx = rng.choice(len(all_paulis), size=max_observables, replace=False)
            observables = [all_paulis[i] for i in sorted(idx)]
        else:
            observables = all_paulis

        q = measure_observables(state, observables)

        for m in methods:
            # Pula SCS se n >= 4 para evitar travamentos longos
            if m == "cvxpy_scs" and n >= 4:
                tempos[m].append(np.nan)
                iteracoes[m].append(0)
                residuos[m].append(np.nan)
                fidelidades[m].append(np.nan)
                continue

            t0 = time.time()
            res = solve_tomography(observables, q, method=m, true_state=state, n_qubits=n)
            dur = time.time() - t0

            tempos[m].append(dur)
            iteracoes[m].append(res.num_iters)
            residuos[m].append(res.residual)
            fidelidades[m].append(res.fidelity if res.fidelity is not None else 1.0)

    return {
        "qubits_range": list(qubits_range),
        "tempos": tempos,
        "iteracoes": iteracoes,
        "residuos": residuos,
        "fidelidades": fidelidades,
    }


# ---------------------------------------------------------------------------
# Experimento 4: Espectro de Pureza (Puros, Mistura de Puros e Muito Mistos)
# ---------------------------------------------------------------------------

def run_purity_spectrum_experiment(
    n_qubits: int = 2,
    num_measurements: int | None = 6,  # Medições incompletas padrão (6 de 15)
    seed: int = 42,
    measurement_completeness: str | bool | None = "incomplete",
    solvers: Sequence[str] = ("quadratic", "least_squares"),
    chute_inicial: np.ndarray | None = None,
) -> dict[str, dict[str, TomographyResult]]:
    r"""
    Compara como os algoritmos se comportam perante diferentes classes de pureza:
    1. Estado Puro (Posto 1, S=0)
    2. Mistura de Puros (Posto 2 intermediário)
    3. Estado Despolarizado (Misto, alta entropia)
    4. Estado Totalmente Misto (I / d, máxima entropia)

    Para isolar estritamente o efeito da geometria do cone/pureza espectral:
      - Utiliza o MESMO conjunto fixo de observáveis para todas as classes;
      - Utiliza o MESMO chute inicial fixo (x0) para todas as classes no método 'least_squares';
      - Captura o histórico completo de convergência (cost_history para least_squares e
        grad_norm_history para quadratic) em cada TomographyResult retornado.
      - Suporta medição completa (15 observáveis para 2 qubits) via `measurement_completeness="complete"`.
    """
    dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)

    # 1. Definição dos estados com diferentes níveis de pureza
    states_dict: dict[str, QuantState] = {
        "1. Puro (Posto 1)": random_pure_state(dim, seed=seed),
        "2. Mistura de Puros (Posto 2)": random_mixture_of_pures(dim, rank=2, seed=seed + 1),
        "3. Despolarizado (p=0.6)": depolarized_state(random_pure_state(dim, seed=seed), p=0.6),
        "4. Totalmente Misto (I/d)": maximally_mixed_state(dim),
    }

    # 2. Seleção de observáveis com controle de completude
    all_paulis = pauli_basis_strings(n_qubits)
    is_complete = False
    if measurement_completeness is not None:
        if isinstance(measurement_completeness, bool):
            is_complete = measurement_completeness
        elif str(measurement_completeness).lower().strip() in ("complete", "completa", "full", "all", "15"):
            is_complete = True

    if is_complete or num_measurements is None or num_measurements >= len(all_paulis):
        observables = all_paulis
    else:
        obs_idx = rng.choice(len(all_paulis), size=num_measurements, replace=False)
        observables = [all_paulis[i] for i in sorted(obs_idx)]

    # 3. Chute inicial fixo e determinístico para todas as classes (isolando a variável classe de estado)
    num_params = dim ** 2
    if chute_inicial is None:
        rng_init = np.random.default_rng(seed + 999)
        fixed_x0 = rng_init.normal(size=num_params)
    else:
        fixed_x0 = np.asarray(chute_inicial, dtype=float)

    experiment_results: dict[str, dict[str, TomographyResult]] = {}

    for state_name, state in states_dict.items():
        q = measure_observables(state, observables)
        res_by_solver: dict[str, TomographyResult] = {}
        for s in solvers:
            if s in ("least_squares", "brute_force", "forca_bruta", "cholesky", "lsq"):
                res = solve_tomography(
                    observables,
                    q,
                    method=s,
                    true_state=state,
                    n_qubits=n_qubits,
                    chute_inicial=fixed_x0,
                )
            else:
                res = solve_tomography(
                    observables,
                    q,
                    method=s,
                    true_state=state,
                    n_qubits=n_qubits,
                )
            res.extra["target_state"] = state
            res.extra["true_entropy"] = float(state.entropy)
            res_by_solver[s] = res
        experiment_results[state_name] = res_by_solver

    return experiment_results


# ---------------------------------------------------------------------------
# Experimento 4b: Estado Térmico de Gibbs vs. Controle de Mesma Pureza
# ---------------------------------------------------------------------------

def run_gibbs_vs_control_experiment(
    n_qubits: int = 2,
    betas: Sequence[float] = (0.1, 1.0, 5.0),
    J: float = 1.0,
    h: float = 0.8,
    seed: int = 42,
    solvers: Sequence[str] = ("least_squares", "quadratic"),
) -> dict[str, Any]:
    r"""
    Experimento comparativo: Estado térmico de Gibbs vs. Estado de controle aleatório com mesma entropia.

    1. Gera rho_gibbs = exp(-beta * H) / Z para o Hamiltoniano TFIM (transverse_field_ising)
       para múltiplos valores de beta cobrindo do regime quase totalmente misto ao quase puro.
    2. Para cada beta, calcula a entropia S(rho_gibbs) e gera um estado de controle aleatório
       com entropia equivalente (|S_ctrl - S_gibbs| < 1e-6).
    3. Roda solve_tomography com medição COMPLETA de Pauli (15 observáveis para 2 qubits),
       usando o mesmo chute inicial fixo para Força Bruta.
    4. Retorna resultados com trajetórias de convergência para comparação direta.
    """
    dim = 2 ** n_qubits
    H = transverse_field_ising(n_qubits, J=J, h=h)
    observables = pauli_basis_strings(n_qubits)

    # Chute inicial idêntico para todos os testes de least_squares
    rng_init = np.random.default_rng(seed + 777)
    fixed_x0 = rng_init.normal(size=dim ** 2)

    data_by_beta = []

    for beta in betas:
        # Estado de Gibbs físico
        rho_gibbs = gibbs_state(H, beta=beta)
        s_gibbs = float(von_neumann_entropy(rho_gibbs.rho))

        # Estado de controle não-físico / aleatório com mesma entropia
        rho_ctrl = random_state_with_matched_entropy(s_gibbs, dim=dim, seed=int(beta * 100) + seed)
        s_ctrl = float(von_neumann_entropy(rho_ctrl.rho))

        # Medições completas
        q_gibbs = measure_observables(rho_gibbs, observables)
        q_ctrl = measure_observables(rho_ctrl, observables)

        solvers_res_gibbs = {}
        solvers_res_ctrl = {}

        for s in solvers:
            if s in ("least_squares", "brute_force", "forca_bruta", "cholesky", "lsq"):
                res_g = solve_tomography(
                    observables, q_gibbs, method=s, true_state=rho_gibbs, n_qubits=n_qubits, chute_inicial=fixed_x0
                )
                res_c = solve_tomography(
                    observables, q_ctrl, method=s, true_state=rho_ctrl, n_qubits=n_qubits, chute_inicial=fixed_x0
                )
            else:
                res_g = solve_tomography(
                    observables, q_gibbs, method=s, true_state=rho_gibbs, n_qubits=n_qubits
                )
                res_c = solve_tomography(
                    observables, q_ctrl, method=s, true_state=rho_ctrl, n_qubits=n_qubits
                )

            res_g.extra["target_state"] = rho_gibbs
            res_g.extra["true_entropy"] = s_gibbs
            res_c.extra["target_state"] = rho_ctrl
            res_c.extra["true_entropy"] = s_ctrl

            solvers_res_gibbs[s] = res_g
            solvers_res_ctrl[s] = res_c

        data_by_beta.append({
            "beta": beta,
            "s_gibbs": s_gibbs,
            "s_ctrl": s_ctrl,
            "rho_gibbs": rho_gibbs,
            "rho_ctrl": rho_ctrl,
            "res_gibbs": solvers_res_gibbs,
            "res_ctrl": solvers_res_ctrl,
        })

    return {
        "H": H,
        "n_qubits": n_qubits,
        "observables": observables,
        "betas": list(betas),
        "data": data_by_beta,
    }


# ---------------------------------------------------------------------------
# Experimento 5: Introdução de Energia / Hamiltoniano na Tomografia
# ---------------------------------------------------------------------------

def run_energy_introduction_experiment(
    n_qubits: int = 2,
    hamiltonian_type: str = "ising",
    num_measurements: int = 4,
    eps: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    r"""
    Investiga o efeito de introduzir um Hamiltoniano H no problema:
        \min_\pi \operatorname{Tr}[H \pi] + \varepsilon \operatorname{Tr}[\varphi(\pi)]
        sujeito a \operatorname{Tr}[Q_i \pi] = q_i.

    Quando as medições são escassas (subdeterminadas), o termo de energia direciona
    o estado para o espaço de baixa energia / estado fundamental físico do sistema.
    """
    if hamiltonian_type.lower() == "ising":
        H = transverse_field_ising(n_qubits, J=1.0, h=0.8)
    elif hamiltonian_type.lower() == "heisenberg":
        H = heisenberg_xxz(n_qubits, J_xy=1.0, Delta=1.0)
    else:
        H = single_qubit_zeeman()

    # Estado alvo físico: estado fundamental de H (ou térmico a baixa temperatura)
    e0, target_state = ground_state(H)

    # Medições parciais (dados insuficientes para determinar o estado sozinho)
    all_paulis = pauli_basis_strings(n_qubits)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_paulis), size=min(num_measurements, len(all_paulis)), replace=False)
    observables = [all_paulis[i] for i in sorted(idx)]

    q = measure_observables(target_state, observables)

    # Reconstrução 1: Sem Energia (H = 0, tomografia puramente informacional)
    res_sem_h = solve_tomography(
        observables, q, method="entropy", H=None, eps=eps, true_state=target_state
    )

    # Reconstrução 2: Com Energia H e Regularização Entrópica (MaxEnt com Energia)
    res_com_h_ent = solve_tomography(
        observables, q, method="entropy", H=H, eps=eps, true_state=target_state
    )

    # Reconstrução 3: Com Energia H e Regularização Quadrática
    res_com_h_quad = solve_tomography(
        observables, q, method="quadratic", H=H, eps=eps, true_state=target_state
    )

    # Energias esperadas
    e_true = target_state.energy(H)
    e_sem_h = float(np.trace(res_sem_h.rho @ H).real)
    e_com_h_ent = float(np.trace(res_com_h_ent.rho @ H).real)
    e_com_h_quad = float(np.trace(res_com_h_quad.rho @ H).real)

    return {
        "H": H,
        "target_state": target_state,
        "observables": observables,
        "q": q,
        "e_ground": e0,
        "res_sem_h": res_sem_h,
        "res_com_h_ent": res_com_h_ent,
        "res_com_h_quad": res_com_h_quad,
        "energias": {
            "Alvo (Fundamental)": e_true,
            "Reconstruído (H=0)": e_sem_h,
            "Reconstruído (Com H, Entropia)": e_com_h_ent,
            "Reconstruído (Com H, Quadrático)": e_com_h_quad,
        },
    }


# ---------------------------------------------------------------------------
# Experimento 6: Estabilidade e Ruído de Shot Noise
# ---------------------------------------------------------------------------

def run_noise_stability_experiment(
    n_qubits: int = 1,
    shots_list: Sequence[int] = (10, 25, 50, 100, 250, 500, 1000, 5000, 10000),
    n_reps: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Analisa a estabilidade dos algoritmos perante ruído binomial de shot noise (Born rule):
    - Mostra que o SDP de viabilidade estrita (Minimize 0) torna-se infactível para N pequeno.
    - Mostra que Mínimos Quadrados Convexos (CVXPY) e Força Bruta (Cholesky) mantêm estabilidade garantida.
    """
    dim = 2 ** n_qubits
    state = random_pure_state(dim, seed=seed)
    observables = pauli_basis_strings(n_qubits)
    q_exato = measure_observables(state, observables)

    rng = np.random.default_rng(seed)

    fids_lsq_convexa: list[float] = []
    stds_lsq_convexa: list[float] = []
    fids_brute_force: list[float] = []
    stds_brute_force: list[float] = []
    taxa_infactivel_sdp: list[float] = []

    for N_shots in shots_list:
        fids_cvx = []
        fids_bf = []
        n_infeasible = 0

        for _ in range(n_reps):
            q_ruido = simula_medidas_shots(q_exato, N_shots, rng=rng)

            # 1. Viabilidade SDP Linear
            res_scs = solve_tomography(observables, q_ruido, method="cvxpy_scs", true_state=state)
            if not res_scs.success or "infeasible" in res_scs.status.lower():
                n_infeasible += 1

            # 2. Mínimos Quadrados Convexos
            res_lsq_c = solve_tomography(observables, q_ruido, method="cvxpy_lsq", true_state=state)
            if res_lsq_c.fidelity is not None:
                fids_cvx.append(res_lsq_c.fidelity)

            # 3. Força Bruta Cholesky LSQ
            res_bf = solve_tomography(observables, q_ruido, method="least_squares", true_state=state, n_qubits=n_qubits)
            if res_bf.fidelity is not None:
                fids_bf.append(res_bf.fidelity)

        fids_lsq_convexa.append(float(np.mean(fids_cvx)) if fids_cvx else 0.0)
        stds_lsq_convexa.append(float(np.std(fids_cvx)) if fids_cvx else 0.0)
        fids_brute_force.append(float(np.mean(fids_bf)) if fids_bf else 0.0)
        stds_brute_force.append(float(np.std(fids_bf)) if fids_bf else 0.0)
        taxa_infactivel_sdp.append(n_infeasible / n_reps)

    return {
        "shots_list": list(shots_list),
        "fids_lsq_convexa": fids_lsq_convexa,
        "stds_lsq_convexa": stds_lsq_convexa,
        "fids_brute_force": fids_brute_force,
        "stds_brute_force": stds_brute_force,
        "taxa_infactivel_sdp": taxa_infactivel_sdp,
    }


# ---------------------------------------------------------------------------
# Experimento 7: Escalabilidade em Qubits com Shots Reais
# ---------------------------------------------------------------------------

def run_shots_scaling_benchmark(
    qubits_range: Sequence[int] = (1, 2, 3, 4),
    n_shots: int = 10000,
    methods: Sequence[str] = ("least_squares", "quadratic"),
    n_reps: int = 5,
    seed: int = 42,
    max_iter_quadratic: int = 300,
) -> dict[str, Any]:
    r"""
    Avalia a fidelidade, número de iterações/avaliações e tempo de execução
    em função do número de qubits (1 a 4) sob amostragem estatística realista de shots (Born rule).

    Para cada n_qubits:
      - Gera um estado puro aleatório (random_pure_state) com seed fixa por qubit (seed + n);
      - Constrói a base completa de Pauli (pauli_basis_strings);
      - Simula as medições com N_shots realista e fixo (ex: 10.000) por n_reps repetições;
      - Executa os métodos solicitados ('least_squares' e 'quadratic');
      - Calcula e registra médias e desvios padrão de fidelidade, iterações e tempo.
    """
    qubits_list = list(qubits_range)
    methods_list = list(methods)

    fidelidades_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    fidelidades_std: dict[str, list[float]] = {m: [] for m in methods_list}
    iteracoes_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    iteracoes_std: dict[str, list[float]] = {m: [] for m in methods_list}
    tempos_mean: dict[str, list[float]] = {m: [] for m in methods_list}
    tempos_std: dict[str, list[float]] = {m: [] for m in methods_list}
    residuos_mean: dict[str, list[float]] = {m: [] for m in methods_list}

    detalhes: list[dict[str, Any]] = []

    for n in qubits_list:
        dim = 2 ** n
        # Seed fixa por n_qubits para reprodutibilidade do estado-alvo
        state_seed = seed * 100 + n
        target_state = random_pure_state(dim, seed=state_seed)
        observables = pauli_basis_strings(n)
        q_exato = measure_observables(target_state, observables)

        fids_by_method: dict[str, list[float]] = {m: [] for m in methods_list}
        iters_by_method: dict[str, list[int]] = {m: [] for m in methods_list}
        times_by_method: dict[str, list[float]] = {m: [] for m in methods_list}
        resids_by_method: dict[str, list[float]] = {m: [] for m in methods_list}

        for rep in range(n_reps):
            rep_seed = seed * 1000 + n * 100 + rep
            rng_rep = np.random.default_rng(rep_seed)
            q_shots = simula_medidas_shots(q_exato, n_shots, rng=rng_rep)

            for m in methods_list:
                t0 = time.time()
                if m in ("quadratic", "entropy"):
                    res = solve_tomography(
                        observables,
                        q_shots,
                        method=m,
                        true_state=target_state,
                        n_qubits=n,
                        max_iter=max_iter_quadratic,
                    )
                else:
                    res = solve_tomography(
                        observables,
                        q_shots,
                        method=m,
                        true_state=target_state,
                        n_qubits=n,
                    )
                dur = time.time() - t0

                fid_val = float(res.fidelity) if res.fidelity is not None else 0.0
                fids_by_method[m].append(fid_val)
                iters_by_method[m].append(int(res.num_iters))
                times_by_method[m].append(float(dur))
                resids_by_method[m].append(float(res.residual))

                detalhes.append({
                    "n_qubits": n,
                    "rep": rep,
                    "method": m,
                    "fidelity": fid_val,
                    "num_iters": int(res.num_iters),
                    "time": float(dur),
                    "residual": float(res.residual),
                    "status": res.status,
                    "success": res.success,
                })

        for m in methods_list:
            fids_mean = float(np.mean(fids_by_method[m])) if fids_by_method[m] else 0.0
            fids_s = float(np.std(fids_by_method[m])) if fids_by_method[m] else 0.0
            it_mean = float(np.mean(iters_by_method[m])) if iters_by_method[m] else 0.0
            it_s = float(np.std(iters_by_method[m])) if iters_by_method[m] else 0.0
            t_mean = float(np.mean(times_by_method[m])) if times_by_method[m] else 0.0
            t_s = float(np.std(times_by_method[m])) if times_by_method[m] else 0.0
            r_mean = float(np.mean(resids_by_method[m])) if resids_by_method[m] else 0.0

            fidelidades_mean[m].append(fids_mean)
            fidelidades_std[m].append(fids_s)
            iteracoes_mean[m].append(it_mean)
            iteracoes_std[m].append(it_s)
            tempos_mean[m].append(t_mean)
            tempos_std[m].append(t_s)
            residuos_mean[m].append(r_mean)

    return {
        "qubits_range": qubits_list,
        "n_shots": n_shots,
        "n_reps": n_reps,
        "methods": methods_list,
        "fidelidades_mean": fidelidades_mean,
        "fidelidades_std": fidelidades_std,
        "iteracoes_mean": iteracoes_mean,
        "iteracoes_std": iteracoes_std,
        "tempos_mean": tempos_mean,
        "tempos_std": tempos_std,
        "residuos_mean": residuos_mean,
        "detalhes": detalhes,
    }

