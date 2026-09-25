"""
Quantum Tomography Toolkit
==========================
Biblioteca completa e modular para Tomografia de Estados Quânticos com Programação Semidefinida (SDP)
regularizada (sdplab / spacecore / JAX) e Mínimos Quadrados de Força Bruta (Cholesky / SciPy).
"""

from .states import (
    QuantState,
    random_pure_state,
    standard_pure_state,
    random_mixture_of_pures,
    depolarized_state,
    maximally_mixed_state,
    thermal_state,
    fidelidade,
    fidelity,
    von_neumann_entropy,
    purity,
    trace_distance,
    frobenius_distance,
    numerical_rank,
    random_state_with_matched_entropy,
    partial_trace,
)

from .observables import (
    I2,
    sigma_x,
    sigma_y,
    sigma_z,
    sx,
    sy,
    sz,
    pauli_string_to_matrix,
    pauli_basis_strings,
    gerar_paulis,
    select_coordinate_paulis,
    select_random_paulis,
    select_k_local_paulis,
    simula_medidas_shots,
    medidas_shots_de_estado,
    measure_observables,
)

from .hamiltonians import (
    single_qubit_zeeman,
    transverse_field_ising,
    heisenberg_xxz,
    random_hamiltonian,
    ground_state,
    gibbs_state,
)

from .solvers import (
    TomographyResult,
    SDPTomographySolver,
    solve_tomography,
    resolve_tomografia_lsq,
    parametros_para_T_geral,
    parametros_para_rho_geral,
    funcao_de_custo_geral,
    jacobiano_analitico,
)

from .visualization import (
    vetor_bloch,
    plotar_bloch,
    plot_spectra,
    plot_qubit_scaling,
    plot_fidelity_vs_shots,
    plot_purity_convergence,
    plot_gibbs_vs_control_convergence,
    plot_purity_completeness_comparison,
    plot_shots_scaling,
)

from .experiments import (
    run_sanity_check,
    run_uniqueness_experiment,
    run_qubit_scaling_benchmark,
    run_purity_spectrum_experiment,
    run_gibbs_vs_control_experiment,
    run_energy_introduction_experiment,
    run_noise_stability_experiment,
    run_shots_scaling_benchmark,
)

__version__ = "0.2.0"

__all__ = [
    # Estados
    "QuantState",
    "random_pure_state",
    "standard_pure_state",
    "random_mixture_of_pures",
    "depolarized_state",
    "maximally_mixed_state",
    "thermal_state",
    "fidelidade",
    "fidelity",
    "von_neumann_entropy",
    "purity",
    "trace_distance",
    "frobenius_distance",
    "numerical_rank",
    "partial_trace",
    # Observáveis
    "I2",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sx",
    "sy",
    "sz",
    "pauli_string_to_matrix",
    "pauli_basis_strings",
    "gerar_paulis",
    "select_coordinate_paulis",
    "select_random_paulis",
    "select_k_local_paulis",
    "simula_medidas_shots",
    "medidas_shots_de_estado",
    "measure_observables",
    # Hamiltonianos
    "single_qubit_zeeman",
    "transverse_field_ising",
    "heisenberg_xxz",
    "random_hamiltonian",
    "ground_state",
    "gibbs_state",
    # Solvers
    "TomographyResult",
    "SDPTomographySolver",
    "solve_tomography",
    "resolve_tomografia_lsq",
    "parametros_para_T_geral",
    "parametros_para_rho_geral",
    "funcao_de_custo_geral",
    "jacobiano_analitico",
    # Visualização
    "vetor_bloch",
    "plotar_bloch",
    "plot_spectra",
    "plot_qubit_scaling",
    "plot_fidelity_vs_shots",
    "plot_purity_convergence",
    "plot_gibbs_vs_control_convergence",
    "plot_purity_completeness_comparison",
    "plot_shots_scaling",
    # Experimentos
    "run_sanity_check",
    "run_uniqueness_experiment",
    "run_qubit_scaling_benchmark",
    "run_purity_spectrum_experiment",
    "run_gibbs_vs_control_experiment",
    "run_energy_introduction_experiment",
    "run_noise_stability_experiment",
    "run_shots_scaling_benchmark",
    "random_state_with_matched_entropy",
]
