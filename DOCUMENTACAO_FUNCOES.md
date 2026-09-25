# Documentação Completa do Pacote `quantum_tomography`

Este documento apresenta a referência técnica e matemática completa de todas as classes, funções e módulos desenvolvidos no pacote `quantum_tomography`. O objetivo é permitir que qualquer pesquisador ou estudante compreenda a finalidade de cada componente, a fundamentação teórica subjacente e os detalhes de implementação adotados.

---

## Índice

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Módulo `states` (Estados Quânticos e Métricas)](#2-módulo-states)
   - [`QuantState`](#quantstate)
   - [`fidelidade` / `fidelity`](#fidelidade--fidelity)
   - [`von_neumann_entropy`](#von_neumann_entropy)
   - [`purity`](#purity)
   - [`trace_distance` e `frobenius_distance`](#trace_distance-e-frobenius_distance)
   - [`numerical_rank`](#numerical_rank)
   - [`partial_trace`](#partial_trace)
   - [`random_pure_state`](#random_pure_state)
   - [`standard_pure_state`](#standard_pure_state)
   - [`random_mixture_of_pures`](#random_mixture_of_pures)
   - [`depolarized_state`](#depolarized_state)
   - [`maximally_mixed_state`](#maximally_mixed_state)
   - [`thermal_state` e `gibbs_state`](#thermal_state-e-gibbs_state)
   - [`random_state_with_matched_entropy`](#random_state_with_matched_entropy)
   - [Funções Utilitárias: `_symmetrize_hermitian` e `sqrt_psd`](#utilitários-espectrais)
3. [Módulo `observables` (Observáveis de Pauli e Simulação de Medição)](#3-módulo-observables)
   - [Matrizes Fundamentais (`I2`, `sigma_x`, `sigma_y`, `sigma_z`)](#matrizes-de-pauli)
   - [`pauli_string_to_matrix`](#pauli_string_to_matrix)
   - [`pauli_basis_strings`](#pauli_basis_strings)
   - [`select_coordinate_paulis`, `select_random_paulis`, `select_k_local_paulis`](#seleção-de-observáveis)
   - [`measure_observables`](#measure_observables)
   - [`simula_medidas_shots` e `medidas_shots_de_estado`](#simulação-de-shot-noise)
4. [Módulo `hamiltonians` (Modelos Físicos e Termodinâmica)](#4-módulo-hamiltonians)
   - [`single_qubit_zeeman`](#single_qubit_zeeman)
   - [`transverse_field_ising`](#transverse_field_ising)
   - [`heisenberg_xxz`](#heisenberg_xxz)
   - [`random_hamiltonian`](#random_hamiltonian)
   - [`ground_state`](#ground_state)
5. [Módulo `solvers` (Algoritmos de Tomografia Quântica)](#5-módulo-solvers)
   - [`TomographyResult` (Classe de Resultado Unificado)](#tomographyresult)
   - [`solve_tomography` (Ponto de Entrada Polimórfico)](#solve_tomography)
   - [Submódulo `brute_force` (Parametrização de Cholesky)](#submódulo-brute_force)
     - [`parametros_para_T_geral`](#parametros_para_t_geral)
     - [`parametros_para_rho_geral`](#parametros_para_rho_geral)
     - [`funcao_de_custo_geral`](#funcao_de_custo_geral)
     - [`jacobiano_analitico`](#jacobiano_analitico)
     - [`resolve_tomografia_lsq`](#resolve_tomografia_lsq)
   - [Submódulo `sdp_solver` (Programação Semidefinida Primal e Dual)](#submódulo-sdp_solver)
     - [`SDPTomographySolver`](#sdptomographysolver)
     - [`solve_cvxpy_scs` (Primal Linear / SCS)](#solve_cvxpy_scs)
     - [`solve_cvxpy_least_squares` (Mínimos Quadrados Convexos)](#solve_cvxpy_least_squares)
     - [`solve_quadratic` (SDP Dual Regularizado / JAX)](#solve_quadratic)
     - [`solve_entropy` (MaxEnt / von Neumann)](#solve_entropy)
6. [Módulo `visualization` (Visualizações Gráficas)](#6-módulo-visualization)
   - [`vetor_bloch`](#vetor_bloch)
   - [`plotar_bloch`](#plotar_bloch)
   - [`plot_spectra`](#plot_spectra)
   - [`plot_qubit_scaling`](#plot_qubit_scaling)
   - [`plot_fidelity_vs_shots`](#plot_fidelity_vs_shots)
   - [`plot_purity_convergence`](#plot_purity_convergence)
   - [`plot_gibbs_vs_control_convergence`](#plot_gibbs_vs_control_convergence)
   - [`plot_purity_completeness_comparison`](#plot_purity_completeness_comparison)
   - [`plot_shots_scaling`](#plot_shots_scaling)
7. [Módulo `experiments` (Suíte Experimental Automatizada)](#7-módulo-experiments)
   - [`run_sanity_check`](#run_sanity_check)
   - [`run_uniqueness_experiment`](#run_uniqueness_experiment)
   - [`run_qubit_scaling_benchmark`](#run_qubit_scaling_benchmark)
   - [`run_purity_spectrum_experiment`](#run_purity_spectrum_experiment)
   - [`run_gibbs_vs_control_experiment`](#run_gibbs_vs_control_experiment)
   - [`run_energy_introduction_experiment`](#run_energy_introduction_experiment)
   - [`run_noise_stability_experiment`](#run_noise_stability_experiment)
   - [`run_shots_scaling_benchmark`](#run_shots_scaling_benchmark)

---

## 1. Visão Geral da Arquitetura

O problema central da **Tomografia de Estados Quânticos (QST)** consiste em reconstruir uma matriz densidade desconhecida $\rho \in \mathbb{C}^{d \times d}$ ($d = 2^N$ para $N$ qubits) a partir de um conjunto de valores esperados experimentais:
$$
q_i = \operatorname{Tr}[Q_i \rho] \quad (i = 1, \dots, M),
$$
onde $Q_i$ são operadores hermitianos (observáveis de Pauli) e $\rho$ deve satisfazer as três propriedades fundamentais de um estado quântico:
1. **Hermiticidade:** $\rho = \rho^\dagger$;
2. **Positividade Semidefinida:** $\rho \succeq 0$ (todos os autovalores $\lambda_k \ge 0$);
3. **Traço Unitário:** $\operatorname{Tr}[\rho] = 1$.

O toolkit divide-se em camadas modulares e desacopladas:
- **`states`**: Cria e manipula operadores densidade, calculando métricas de informação quântica;
- **`observables`**: Constrói bases de Pauli e simula a regra de Born estatística ($N_{\text{shots}}$);
- **`hamiltonians`**: Constrói modelos físicos da mecânica quântica e termodinâmica;
- **`solvers`**: Implementa as três abordagens computacionais centrais (Força Bruta não-linear, SDP Primal Cônico e SDP Dual Regularizado);
- **`visualization`**: Gera gráficos de publicação (Bloch 3D/2D, espectros, escalabilidade, trajetórias de convergência);
- **`experiments`**: Orquestra pipelines reprodutíveis para avaliação científica dos algoritmos.

---

## 2. Módulo `states`

Arquivo: `quantum_tomography/states.py`

### `QuantState`
- **Assinatura:** `QuantState(estado: np.ndarray | list)`
- **Para que serve:** Representa de forma orientada a objetos um estado quântico de $N$ qubits, unificando a manipulação de ket puros $|\psi\rangle \in \mathbb{C}^d$ e matrizes densidade gerais $\rho \in \mathbb{C}^{d \times d}$.
- **Como foi feito:**
  - Se a entrada for 1D ($|\psi\rangle$), normaliza o vetor pela norma euclidiana e calcula o projetor $\rho = |\psi\rangle\langle\psi| = \vec{v} \vec{v}^\dagger$.
  - Se a entrada for 2D ($\rho$), divide pelo traço $\operatorname{Tr}[\rho]$ e aplica simetrização hermiteana.
  - Expõe propriedades dinâmicas: `.rho`, `.dim`, `.n_qubits`, `.purity`, `.entropy`, `.rank`, `.is_pure` e o método `.energy(H)` para calcular $\operatorname{Tr}[\rho H]$.

### `fidelidade` / `fidelity`
- **Assinatura:** `fidelidade(rho_a: np.ndarray, rho_b: np.ndarray) -> float`
- **Para que serve:** Mede a proximidade quântica entre dois operadores densidade $\rho_a$ e $\rho_b$. Retorna $1.0$ se e somente se os estados forem idênticos e $0.0$ se forem ortogonais.
- **Como foi feito:** Implementa a fórmula analítica de Uhlmann-Jozsa:
  $$
  F(\rho_a, \rho_b) = \left( \operatorname{Tr} \sqrt{\sqrt{\rho_a} \rho_b \sqrt{\rho_a}} \right)^2.
  $$
  Calcula a raiz quadrada matricial $\sqrt{\rho_a}$ via decomposição espectral (`eigh`), multiplica por $\rho_b$, extrai a raiz do operador intermediário resultante e eleva o traço ao quadrado, com corte numérico no intervalo $[0, 1]$.

### `von_neumann_entropy`
- **Assinatura:** `von_neumann_entropy(rho: np.ndarray, base: float = np.e) -> float`
- **Para que serve:** Quantifica a desordem informacional ou a mistura do estado quântico. Vale $S = 0$ para estados puros e atinge o máximo $S = \ln(d)$ para o estado totalmente misto $I/d$.
- **Como foi feito:** Diagonaliza $\rho$ via `np.linalg.eigvalsh`, filtra autovalores numericamente positivos ($\lambda_i > 10^{-15}$) e avalia:
  $$
  S(\rho) = -\sum_{\lambda_i > 0} \lambda_i \ln \lambda_i.
  $$
  Caso uma base alternativa seja fornecida (ex: base 2 para bits/shannons), divide o resultado por $\ln(\text{base})$.

### `purity`
- **Assinatura:** `purity(rho: np.ndarray) -> float`
- **Para que serve:** Mede a pureza do estado $\gamma \in [1/d, 1.0]$.
- **Como foi feito:** Calcula o traço do quadrado da matriz:
  $$
  \gamma = \operatorname{Tr}[\rho^2] = \sum_i \lambda_i^2.
  $$

### `trace_distance` e `frobenius_distance`
- **Assinatura:** `trace_distance(rho_a, rho_b) -> float`, `frobenius_distance(rho_a, rho_b) -> float`
- **Para que serve:** Distâncias métricas entre dois operadores densidade.
- **Como foi feito:**
  - Distância de traço: $D(\rho_a, \rho_b) = \frac{1}{2} \|\rho_a - \rho_b\|_1 = \frac{1}{2} \sum_k |\lambda_k(\rho_a - \rho_b)|$. Representa o limite superior fundamental para a probabilidade de distinguir os dois estados em qualquer medição quântica.
  - Distância de Frobenius: $\|\rho_a - \rho_b\|_F = \sqrt{\operatorname{Tr}[(\rho_a - \rho_b)^2]}$, distância euclidiana direta no espaço de Hilbert-Schmidt.

### `numerical_rank`
- **Assinatura:** `numerical_rank(rho: np.ndarray, tol: float = 1e-9) -> int`
- **Para que serve:** Retorna o número de autovalores estritamente maiores que um limiar `tol`, indicando a dimensionalidade do subespaço ocupado pelo estado.

### `partial_trace`
- **Assinatura:** `partial_trace(rho: np.ndarray | QuantState, keep: int = 0, n_qubits: int | None = None) -> np.ndarray`
- **Para que serve:** Calcula o traço parcial sobre todos os qubits de um sistema multipartite de $N$ qubits, preservando exclusivamente o subsistema de 1 qubit especificado por `keep` (por padrão, o primeiro qubit `keep=0`).
- **Como foi feito:**
  - Converte a matriz $2^N \times 2^N$ em um tensor de $2N$ índices correspondentes aos operadores bra e ket de cada subsistema: $T_{i_0 i_1 \dots i_{N-1}, j_0 j_1 \dots j_{N-1}}$;
  - Utiliza `numpy.einsum` para contrair $j_k = i_k$ para todos os subsistemas $k \ne \text{keep}$, preservando os índices abertos $(i_{\text{keep}}, j_{\text{keep}})$;
  - Sincroniza a simetria hermitiana e normaliza o traço para produzir uma matriz densidade válida $2 \times 2$ pronta para visualização e cálculo do vetor de Bloch;
  - Também acessível diretamente pelo método `.partial_trace(keep=0)` da classe `QuantState`.


### `random_pure_state`
- **Assinatura:** `random_pure_state(dim: int, seed: int | None = None) -> QuantState`
- **Para que serve:** Gera um vetor de estado puro aleatório distribuído uniformemente segundo a medida invariante de Haar.
- **Como foi feito:** Sorteia partes real e imaginária de uma distribuição gaussiana padrão $\mathcal{N}(0, 1)$, normaliza o vetor resultante em $\mathbb{C}^d$ e constrói o projetor $|\psi\rangle\langle\psi|$.

### `standard_pure_state`
- **Assinatura:** `standard_pure_state(name: str, n_qubits: int = 1) -> QuantState`
- **Para que serve:** Fornece estados canônicos de teste da informação quântica.
- **Como foi feito:** Constrói os vetores para:
  - 1 qubit: `'0'`, `'1'`, `'+'` ($(|0\rangle+|1\rangle)/\sqrt{2}$), `'-'`, `'+i'`, `'-i'`;
  - $N$ qubits: `'ghz'` ou `'bell'` ($( |0\dots 0\rangle + |1\dots 1\rangle )/\sqrt{2}$), `'w'` ($\frac{1}{\sqrt{N}} \sum |0\dots 1 \dots 0\rangle$), `'all_zero'` ($|0\dots 0\rangle$).

### `random_mixture_of_pures`
- **Assinatura:** `random_mixture_of_pures(dim: int, rank: int = 2, seed: int | None = None) -> QuantState`
- **Para que serve:** Gera uma combinação convexa de $k = \text{rank}$ estados puros ortogonais:
  $$
  \rho = \sum_{j=1}^{\text{rank}} p_j |v_j\rangle\langle v_j|, \quad \sum p_j = 1, \quad p_j > 0.
  $$
- **Como foi feito:** Amostra uma matriz aleatória de Ginibre complexa de dimensões $d \times \text{rank}$, aplica decomposição $QR$ para obter colunas ortonormais $|v_j\rangle$, sorteia pesos via distribuição exponencial normalizada (amostragem uniforme no simplex de probabilidades) e soma os produtos externos ponderados.

### `depolarized_state`
- **Assinatura:** `depolarized_state(state: QuantState | np.ndarray, p: float) -> QuantState`
- **Para que serve:** Aplica o canal quântico despolarizante com parâmetro $p \in [0, 1]$:
  $$
  \mathcal{E}_p(\rho) = (1 - p)\rho + p \frac{I}{d}.
  $$
  Para $p = 0$, mantém o estado intacto; para $p = 1$, transforma-o no estado totalmente misto.

### `maximally_mixed_state`
- **Assinatura:** `maximally_mixed_state(dim: int) -> QuantState`
- **Para que serve:** Retorna $\rho = \frac{1}{d} I$, estado central da esfera de Bloch / cone PSD com entropia máxima $\ln(d)$ e informação zero.

### `thermal_state` e `gibbs_state`
- **Assinatura:** `thermal_state(H: np.ndarray, beta: float = 1.0) -> QuantState`, `gibbs_state(H: np.ndarray, beta: float = 1.0) -> QuantState`
- **Para que serve:** Calcula o estado de equilíbrio termodinâmico (estado de Gibbs canônico) associado a um Hamiltoniano $H$ sob temperatura inversa $\beta = 1/(k_B T)$:
  $$
  \rho_{\text{gibbs}} = \frac{e^{-\beta H}}{\operatorname{Tr}[e^{-\beta H}]}.
  $$
- **Como foi feito:** 
  - Subtrai o menor autovalor de $H$ ($E_{\min} = \lambda_{\min}(H)$) para produzir $H_{\text{shifted}} = H - E_{\min} I$. Isso elimina o risco de overflow numérico na exponencial matricial `scipy.linalg.expm(-beta * H_shifted)`;
  - Normaliza pelo traço escalar real $\operatorname{Tr}[e^{-\beta H_{\text{shifted}}}]$.
  - Para $\beta \to \infty$, converge continuamente ao estado fundamental $|\psi_0\rangle\langle\psi_0|$; para $\beta \to 0$, converge a $I/d$.

### `random_state_with_matched_entropy`
- **Assinatura:** `random_state_with_matched_entropy(target_entropy: float, dim: int = 4, seed: int | None = None) -> QuantState`
- **Para que serve:** Gera um estado de controle **não-físico e aleatório**, cuja entropia de von Neumann coincide com exatidão com uma entropia alvo $S^*$ (usado na Tarefa 2 para parear com o estado de Gibbs).
- **Como foi feito:** 
  - Gera uma base ortonormal aleatória $U$ via decomposição $QR$ de uma matriz gaussiana de Ginibre;
  - Constrói uma interpolação monotônica contínua nos autovalores:
    $$
    p(\alpha) = (1 - \alpha) [1, 0, \dots, 0]^T + \alpha \left[ \frac{1}{d}, \dots, \frac{1}{d} \right]^T, \quad \alpha \in [0, 1].
    $$
  - Como $S(p(\alpha))$ cresce estritamente de $0$ a $\ln(d)$, aplica o método de busca de raiz de Brent (`scipy.optimize.brentq`) para encontrar $\alpha^*$ tal que $|S(p(\alpha^*)) - S^*| < 10^{-14}$;
  - Retorna $\rho = U \operatorname{diag}(p(\alpha^*)) U^\dagger$.

### Utilitários Espectrais
- `_symmetrize_hermitian(mat)`: Executa $(M + M^\dagger)/2$ para mitigar resíduos imaginários de ponto flutuante na diagonal.
- `sqrt_psd(mat)`: Diagonaliza via `eigh`, trunca autovalores negativos causados por ruído de máquina para zero e reconstrói $U \sqrt{\Lambda} U^\dagger$.

---

## 3. Módulo `observables`

Arquivo: `quantum_tomography/observables.py`

### Matrizes de Pauli
Matrizes padrão $2 \times 2$ em base computacional:
$$
I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.
$$

### `pauli_string_to_matrix`
- **Assinatura:** `pauli_string_to_matrix(pauli_str: str) -> np.ndarray`
- **Para que serve:** Converte representações em string como `'XX'`, `'YZ'`, `'ZIZI'` no produto tensorial de Kronecker de dimensão $2^N \times 2^N$:
  $$
  Q = \sigma_{s_1} \otimes \sigma_{s_2} \otimes \dots \otimes \sigma_{s_N}.
  $$
- **Como foi feito:** Mapeia cada caractere em sua matriz $2 \times 2$ correspondente e aplica sucessivamente `np.kron`.

### `pauli_basis_strings`
- **Assinatura:** `pauli_basis_strings(n_qubits: int, include_identity: bool = False) -> list[str]`
- **Para que serve:** Gera o conjunto completo de cadeias de Pauli que formam uma base ortogonal para o espaço de operadores hermitianos com traço nulo (dimensão $4^N - 1$).
- **Como foi feito:** Usa `itertools.product('IXYZ', repeat=n_qubits)` e exclui `'I'*N` caso `include_identity=False`.

### Seleção de Observáveis
- `select_coordinate_paulis(n_qubits)`: Seleciona operadores com Pauli não-trivial em apenas um qubit e identidade nos demais (ex: $X_i, Y_i, Z_i$, totalizando $3N$ operadores).
- `select_random_paulis(n_qubits, k, seed)`: Seleciona aleatoriamente $k$ operadores distintos da base de Pauli.
- `select_k_local_paulis(n_qubits, k)`: Seleciona observáveis que atuam não-trivialmente em no máximo $k$ qubits simultaneamente.

### `measure_observables`
- **Assinatura:** `measure_observables(state: QuantState | np.ndarray, observables: Sequence[str | np.ndarray]) -> np.ndarray`
- **Para que serve:** Simula as previsões exatas da mecânica quântica (regra de Born analítica sem ruído) para uma lista de observáveis:
  $$
  q_i = \operatorname{Tr}[\rho Q_i].
  $$
- **Como foi feito:** Realiza a multiplicação matricial $\rho Q_i$ e extrai a parte real do traço.

### Simulação de Shot Noise
- `simula_medidas_shots(q_exatos: np.ndarray, N_shots: int, rng=None) -> np.ndarray`:
  Simula o ruído estatístico binomial decorrente de um número finito de repetições experimentais $N_{\text{shots}}$. Como cada observável de Pauli tem autovalores $\pm 1$, a probabilidade de registrar o resultado $+1$ no projetor $P_+ = \frac{I + Q_i}{2}$ é:
  $$
  p_+ = \frac{1 + q_i}{2}.
  $$
  Sorteia $N_+ \sim \operatorname{Binomial}(N_{\text{shots}}, p_+)$ e calcula o estimador empírico experimental:
  $$
  \hat{q}_i = \frac{N_+ - (N_{\text{shots}} - N_+)}{N_{\text{shots}}} = 2\frac{N_+}{N_{\text{shots}}} - 1.
  $$
- `medidas_shots_de_estado(state, observables, N_shots, seed)`: Encapsula a medição analítica e a perturbação binomial em uma única chamada.

---

## 4. Módulo `hamiltonians`

Arquivo: `quantum_tomography/hamiltonians.py`

### `single_qubit_zeeman`
- **Assinatura:** `single_qubit_zeeman(omega_z: float = 1.0, omega_x: float = 0.0) -> np.ndarray`
- **Para que serve:** Hamiltoniano de acoplamento com campo magnético externo:
  $$
  H = -\omega_z \sigma_z - \omega_x \sigma_x.
  $$

### `transverse_field_ising`
- **Assinatura:** `transverse_field_ising(n_qubits: int, J: float = 1.0, h: float = 1.0, periodic: bool = False) -> np.ndarray`
- **Para que serve:** Constrói o modelo de Ising de Campo Transversal (TFIM), um dos sistemas mais importantes da física da matéria condensada e informação quântica:
  $$
  H = -J \sum_{i=0}^{N-2} Z_i Z_{i+1} - h \sum_{i=0}^{N-1} X_i.
  $$
- **Como foi feito:** Constrói as matrizes de Pauli em formato Kronecker para cada par acoplado $(i, i+1)$ e para cada campo transversal no sítio $i$, somando os operadores hermitianos.

### `heisenberg_xxz`
- **Assinatura:** `heisenberg_xxz(n_qubits: int, J_xy: float = 1.0, Delta: float = 1.0, periodic: bool = False) -> np.ndarray`
- **Para que serve:** Modelo quântico de spin Heisenberg XXZ:
  $$
  H = \sum_{i} \left[ J_{xy} (X_i X_{i+1} + Y_i Y_{i+1}) + \Delta Z_i Z_{i+1} \right].
  $$

### `random_hamiltonian`
- **Assinatura:** `random_hamiltonian(n_qubits: int, seed: int | None = None) -> np.ndarray`
- **Para que serve:** Gera um Hamiltoniano aleatório do Ensemble Unitário Gaussiano (GUE): $H = (G + G^\dagger)/2$.

### `ground_state`
- **Assinatura:** `ground_state(H: np.ndarray) -> tuple[float, QuantState]`
- **Para que serve:** Extrai o estado fundamental físico $|\psi_0\rangle$ e sua energia $E_0 = \min \lambda_i(H)$ via diagonalização espectral exata com `np.linalg.eigh`.

---

## 5. Módulo `solvers`

Arquivos: `quantum_tomography/solvers/`

### `TomographyResult`
Classe dataclass que unifica os resultados de qualquer método tomográfico com campos ricos:
- `rho`: matriz densidade física reconstruída $d \times d$;
- `method`: nome legível do método executado;
- `success`: booleano indicando convergência;
- `status`: mensagem de término do otimizador;
- `num_iters`: contagem de iterações ou avaliações de função;
- `elapsed_seconds`: tempo de execução em segundos;
- `residual`: resíduo final $\|A\rho - q\|_2$;
- `fidelity`, `entropy`, `purity`, `rank`, `frobenius_norm`: métricas automáticas calculadas no `__post_init__`;
- `cost_history`: histórico de decaimento do custo para Força Bruta;
- `grad_norm_history`: histórico da norma do gradiente dual $\|\nabla \mathcal{D}\|_2$ por iteração para o solver quadrático;
- Acesso indexado compatível com dicionário (`res["fidelidade"]`, `res["tempo_s"]`, etc.) e método `.summary()`.

### `solve_tomography`
Ponto de entrada unificado com despacho dinâmico:
```python
solve_tomography(
    observables,
    q_measured,
    method="quadratic",
    H=None,
    true_state=None,
    n_qubits=None,
    eps=1.0,
    tol=1e-12,
    max_iter=10000,
    verbose=0,
    **kwargs
)
```
Mapeia strings para os métodos correspondentes:
- `'quadratic'`, `'l2'`: `SDPTomographySolver.solve_quadratic`;
- `'least_squares'`, `'cholesky'`: `resolve_tomografia_lsq`;
- `'cvxpy_scs'`, `'scs'`: `SDPTomographySolver.solve_cvxpy_scs`;
- `'cvxpy_lsq'`, `'convexa_lsq'`: `SDPTomographySolver.solve_cvxpy_least_squares`;
- `'entropy'`, `'maxent'`: `SDPTomographySolver.solve_entropy`.

---

### Submódulo `brute_force`

Arquivo: `quantum_tomography/solvers/brute_force.py`

#### `parametros_para_T_geral` e `parametros_para_rho_geral`
- **Fundamento Matemático:** A imposição de $\rho \succeq 0, \rho = \rho^\dagger$ e $\operatorname{Tr}[\rho] = 1$ é feita por construção geométrica via decomposição de Cholesky modificada:
  $$
  T = \begin{pmatrix}
  t_0 & 0 & \dots & 0 \\
  t_d + i t_{d+n_{\text{off}}} & t_1 & \dots & 0 \\
  \vdots & \vdots & \ddots & 0 \\
  \dots & \dots & \dots & t_{d-1}
  \end{pmatrix}, \quad n_{\text{off}} = \frac{d(d-1)}{2}.
  $$
  O vetor $x \in \mathbb{R}^{d^2}$ possui $d$ parâmetros diagonais reais e $2 n_{\text{off}}$ parâmetros subdiagonais (partes real e imaginária). A matriz densidade física é obtida por:
  $$
  \rho(T) = \frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}.
  $$
  Como $T^\dagger T$ é auto-adjunta e positiva semidefinida para qualquer $T \neq 0$, e a divisão pelo traço garante normalização estrita, as três restrições físicas são automaticamente satisfeitas sem multiplicadores de Lagrange.

#### `funcao_de_custo_geral`
- **Assinatura:** `funcao_de_custo_geral(params, mediadores_Q, dados_exp, n_qubits, H=None, target_energy=None, lambda_energy=0.0)`
- **Para que serve:** Avalia o vetor de resíduos $r(x) \in \mathbb{R}^M$:
  $$
  r_m(x) = \operatorname{Tr}[\rho(T(x)) Q_m] - q_m.
  $$
  Se um termo de energia for fornecido ($\lambda_E > 0$), concatena o resíduo escalar $r_E = \sqrt{\lambda_E} (\operatorname{Tr}[\rho H] - E_{\text{alvo}})$.

#### `jacobiano_analitico`
- **Assinatura:** `jacobiano_analitico(params, mediadores_Q, dados_exp, n_qubits, ...)`
- **Dedução Matemática Completa:**
  Seja $s = \operatorname{Tr}(T^\dagger T) = \sum_{ij} |T_{ij}|^2$. A previsão do observável é:
  $$
  f_m(T) = \frac{1}{s} \operatorname{Tr}[Q_m T^\dagger T].
  $$
  Pela regra da cadeia matricial do quociente:
  $$
  df_m = \frac{d(\operatorname{Tr}[Q_m T^\dagger T])}{s} - \frac{\operatorname{Tr}[Q_m T^\dagger T]}{s^2} ds.
  $$
  Como $ds = 2 \operatorname{Re}\operatorname{Tr}[T^\dagger dT]$ e $d(\operatorname{Tr}[Q_m T^\dagger T]) = 2 \operatorname{Re}\operatorname{Tr}[Q_m T^\dagger dT]$, agrupando termos:
  $$
  df_m = \frac{2}{s} \operatorname{Re}\operatorname{Tr}[(Q_m T^\dagger - f_m T^\dagger) dT].
  $$
  Definindo a matriz auxiliar $M_m = Q_m T^\dagger$:
  1. Para os parâmetros diagonais reais $t_k = T_{kk}$:
     $$
     \frac{\partial f_m}{\partial T_{kk}} = \frac{2}{s} \operatorname{Re}\left[ (M_m)_{kk} - f_m (T^\dagger)_{kk} \right].
     $$
  2. Para a parte real dos elementos subdiagonais $t_{ij}^R = \operatorname{Re}(T_{ij})$ ($i > j$):
     $$
     \frac{\partial f_m}{\partial t_{ij}^R} = \frac{2}{s} \operatorname{Re}\left[ (M_m)_{ji} - f_m (T^\dagger)_{ji} \right].
     $$
  3. Para a parte imaginária dos elementos subdiagonais $t_{ij}^I = \operatorname{Im}(T_{ij})$ ($i > j$):
     $$
     \frac{\partial f_m}{\partial t_{ij}^I} = \frac{2}{s} \operatorname{Re}\left[ i \left( (M_m)_{ji} - f_m (T^\dagger)_{ji} \right) \right].
     $$
  Essa dedução analítica fechada elimina completamente o cálculo por diferenças finitas $O(d^2)$, acelerando a convergência em ordens de magnitude.

#### `resolve_tomografia_lsq`
Resolve o problema $\min_x \frac{1}{2} \|r(x)\|_2^2$ utilizando `scipy.optimize.least_squares` com o Jacobiano analítico exato. Para dimensões elevadas ($d \ge 16$), ativa automaticamente o solver de subproblemas `tr_solver='lsmr'`, evitando falhas de memória ou singularidade da SVD densa do SciPy.

---

### Submódulo `sdp_solver`

Arquivo: `quantum_tomography/solvers/sdp_solver.py`

#### `SDPTomographySolver`
Controlador principal integrando o backend `spacecore` e compilador JAX com suporte a precisão float64 (`jax_enable_x64 = True`).

#### `solve_cvxpy_scs` (Primal Linear)
- **Formulação Matemática:**
  $$
  \min_{\rho \in \mathcal{H}_d} \operatorname{Tr}[H \rho] \quad \text{sujeito a} \quad \operatorname{Tr}[Q_i \rho] = q_i, \quad \operatorname{Tr}[\rho] = 1, \quad \rho \succeq 0.
  $$
- **Algoritmo:** Utiliza o solver cônico SCS (Splitting Conic Solver), baseado em ADMM / Douglas-Rachford. A cada iteração, realiza uma projeção afim linear e uma projeção cônica não-linear no cone semidefinido via diagonalização espectral completa truncando autovalores negativos ($\Lambda_+ = \max\{\Lambda, 0\}$).
- **Limitação:** Quando dados ruidosos caem fora do cone quântico, o hiperplano afim e o cone PSD têm interseção vazia, fazendo o SCS retornar status `infeasible`.

#### `solve_cvxpy_least_squares` (Mínimos Quadrados Convexos)
- **Formulação Matemática:**
  $$
  \min_{\rho \in \mathcal{H}_d} \sum_{i=1}^M \left( \operatorname{Tr}[Q_i \rho] - q_i \right)^2 + \lambda \operatorname{Tr}[H \rho] \quad \text{sujeito a} \quad \rho \succeq 0, \quad \operatorname{Tr}[\rho] = 1.
  $$
- **Por que Nunca Falha (Robustez a Ruído):** O conjunto de estados físicos $\mathcal{S} = \{ \rho : \rho \succeq 0, \operatorname{Tr}\rho = 1 \}$ é convexo, fechado, compacto e não-vazio (contém $I/d$). Pelo Teorema da Projeção em Conjuntos Convexos Fechados (Hilbert), qualquer vetor arbitrário de dados ruidosos $q \in \mathbb{R}^M$ possui uma projeção euclidiana global e única sobre $\mathcal{S}$. Consequentemente, o solver sempre converge com status `optimal`, sem risco de infactibilidade. Utiliza o solver de pontos interiores com barreira logarítmica CLARABEL.

#### `solve_quadratic` (SDP Dual Regularizado via JAX)
- **Formulação Primal:**
  $$
  \min_{\pi \succeq 0, \operatorname{Tr}\pi = 1} \left\{ \frac{\varepsilon}{2} \|\pi\|_F^2 + \operatorname{Tr}[H \pi] \right\} \quad \text{sujeito a} \quad \operatorname{Tr}[Q_i \pi] = q_i \quad (i = 1, \dots, M).
  $$
- **Dedução do Lagrangiano e KKT:**
  Com multiplicadores $\alpha \in \mathbb{R}^M$, $\theta \in \mathbb{R}$ e folga $Z \succeq 0$:
  $$
  \mathcal{L} = \frac{\varepsilon}{2} \operatorname{Tr}[\pi^2] + \operatorname{Tr}[H\pi] - \sum_{i=1}^M \alpha_i (\operatorname{Tr}[Q_i \pi] - q_i) - \theta (\operatorname{Tr}\pi - 1) - \operatorname{Tr}[Z\pi].
  $$
  A condição de ponto estacionário $\nabla_\pi \mathcal{L} = 0$ resulta em:
  $$
  \pi = \frac{1}{\varepsilon} \left( \sum_{i=1}^M \alpha_i Q_i - H + \theta I + Z \right).
  $$
  Pela folga complementar KKT ($\operatorname{Tr}[Z\pi] = 0$), os operadores diagonalizam simultaneamente. Nos subespaços com autovalores positivos, $Z = 0$; nos negativos, $\pi = 0$. Obtém-se o **mapeamento primal-dual exato com operador de corte positivo**:
  $$
  \pi^*(\alpha) = \left( \frac{\sum_{i=1}^M \alpha_i Q_i - H}{\varepsilon} - \theta \right)_+,
  $$
  onde $\theta$ é o escalar que satisfaz $\operatorname{Tr}[\pi^*(\alpha)] = 1$. O corte $(\dots)_+$ esparsifica os autovalores, direcionando o estado para a **fronteira do cone semidefinido** ($\lambda_k = 0$), o que favorece estados de baixo posto / alta pureza.
- **Identidade Fundamental do Gradiente Dual:**
  Substituindo $\pi^*$ no Lagrangiano e diferenciando em relação a $\alpha_j$:
  $$
  \frac{\partial \mathcal{D}_\varepsilon}{\partial \alpha_j} = q_j - \operatorname{Tr}[Q_j \pi^*(\alpha)] \implies \nabla \mathcal{D}_\varepsilon(\alpha) = q - \mathcal{A}\pi^*(\alpha).
  $$
  **A norma do gradiente dual é estritamente o resíduo das equações de medição tomográficas!**
  Portanto, anular o gradiente dual $\|\nabla \mathcal{D}_\varepsilon\| \le 10^{-12}$ equivale a satisfazer exatamente todas as condições experimentais.
- **Vantagem Computacional:** O problema dual é estritamente côncavo e irrestrito em $\mathbb{R}^M$. É otimizado via L-BFGS compilado em JAX (`optax.lbfgs()`), eliminando fatorações de matrizes de ordem $O(d^6)$ e convergindo em milissegundos mesmo para $N = 4$ qubits ($d = 16$).

---

## 6. Módulo `visualization`

Arquivo: `quantum_tomography/visualization.py`

### `vetor_bloch`
Extrai as coordenadas $r_k = \operatorname{Tr}[\rho \sigma_k]$ para $k \in \{x, y, z\}$ a partir de qualquer representação de estado de 1 qubit.

### `plotar_bloch`
- **Assinatura:** `plotar_bloch(rhos, rotulos=None, cores=None, titulo=..., salvar=None)`
- **Para que serve:** Renderiza a visualização canônica na Esfera de Bloch em uma figura $2 \times 2$ contendo a esfera tridimensional 3D e três projeções equatoriais bidimensionais ($xz$, $xy$, $yz$).
- **Uso no Projeto:** Demonstra geometricamente o fenômeno da não-unicidade tomográfica (Seção 1.3), onde dados parciais ($x, y$) deixam o eixo $z$ indeterminado, formando um segmento de reta no interior da esfera.

### `plot_spectra`
- **Assinatura:** `plot_spectra(panels, title=..., save_path=None)`
- **Para que serve:** Plota os autovalores ordenados em ordem decrescente $\lambda_i(\rho)$ em escala semi-logarítmica.
- **Uso no Projeto:** Distingue estados no interior do cone PSD (autovalores estritamente positivos finitos) de estados na fronteira do cone gerados pelo corte quadrático (autovalores numéricos nulos da ordem de $10^{-17}$).

### `plot_qubit_scaling`
Plota a curva de escalabilidade computacional de tempo de execução em escala logarítmica vs. número de qubits $N \in \{1, 2, 3, 4\}$, comparando Força Bruta, SDP Primal e SDP Dual.

### `plot_fidelity_vs_shots`
Plota a fidelidade média de reconstrução com barras de desvio padrão em função do número de repetições $N_{\text{shots}}$, exibindo em um eixo secundário a taxa de infactibilidade do SDP linear.

### `plot_purity_convergence`
- **Assinatura:** `plot_purity_convergence(purity_results, title=..., save_path=None)`
- **Para que serve:** Atende à **Tarefa 1**, plotando as curvas de custo das 4 classes de pureza (Puro, Mistura de 2 Puros, Despolarizado, Totalmente Misto) sobrepostas em dois subplots lado a lado:
  - Esquerda: Força Bruta (Custo $\frac{1}{2}\|f(x)\|^2$ vs. avaliações de função);
  - Direita: SDP Dual (Norma do Gradiente Dual $\|\nabla \mathcal{D}\|_2$ vs. iterações L-BFGS).
  - Inclui a entropia exata de von Neumann $S(\rho)$ na legenda de cada classe.

### `plot_gibbs_vs_control_convergence`
- **Assinatura:** `plot_gibbs_vs_control_convergence(gibbs_exp_results, title=..., save_path=None)`
- **Para que serve:** Atende à **Tarefa 2**, plotando curvas sobrepostas do estado térmico de Gibbs vs. estado de controle de mesma pureza para múltiplos $\beta \in [0.1, 1.0, 5.0]$ em dois subplots (Força Bruta e SDP Dual).

### `plot_purity_completeness_comparison`
- **Assinatura:** `plot_purity_completeness_comparison(incomplete_results, complete_results, title=..., save_path=None)`
- **Para que serve:** Atende à **Tarefa 4**, plotando a convergência sob medição incompleta ($M=6$) lado a lado com a medição completa ($M=15$).

### `plot_shots_scaling`
- **Assinatura:** `plot_shots_scaling(benchmark_data: dict, title: str = ..., save_path: str | None = None) -> plt.Figure`
- **Para que serve:** Atende à **Tarefa 1**, plotando em dois subplots lado a lado a escalabilidade em função do número de qubits ($N=1$ a $4$) sob ruído realista e fixo de shots ($N_{\text{shots}}=10{,}000$):
  - Subplot (a): Fidelidade média de reconstrução com barras de erro verticais ($\pm 1\sigma$);
  - Subplot (b): Número médio de iterações/avaliações (escala log) com barras de erro verticais ($\pm 1\sigma$).

---

## 7. Módulo `experiments`

Arquivo: `quantum_tomography/experiments.py`

### `run_sanity_check`
Executa a reconstrução de um estado puro aleatório sob medições completas exatas com todos os métodos implementados, verificando se todos atingem $F \approx 1.0000$ e resíduo próximo a zero.

### `run_uniqueness_experiment`
Demonstra experimentalmente a subdeterminação tomográfica medindo apenas $\sigma_x = 0.7$ e $\sigma_y = 0.4$ em 1 qubit. Mostra que a Força Bruta converge para estados distintos conforme a inicialização aleatória $x_0$, enquanto os regularizadores determinísticos selecionam soluções com propriedades espectrais específicas.

### `run_qubit_scaling_benchmark`
Executa testes de tempo e iterações para $N \in \{1, 2, 3, 4\}$ qubits. Salva métricas de tempo, resíduo e fidelidade, evidenciando o timeout do SCS para $N=4$ e a rapidez do dual JAX (~0.2s).

### `run_purity_spectrum_experiment`
- **Assinatura:** `run_purity_spectrum_experiment(n_qubits=2, num_measurements=6, seed=42, measurement_completeness="incomplete", solvers=("quadratic", "least_squares"), chute_inicial=None)`
- **Para que serve (Tarefas 1 e 4):**
  - Testa as 4 classes fundamentais de estados quânticos;
  - Isola a variável "tipo de estado" fixando o **mesmo conjunto de observáveis** e o **mesmo chute inicial $x_0$** para Força Bruta;
  - Captura as trajetórias de convergência `.cost_history` e `.grad_norm_history`;
  - Suporta medição completa ($M=15$) via `measurement_completeness="complete"`.

### `run_gibbs_vs_control_experiment`
- **Assinatura:** `run_gibbs_vs_control_experiment(n_qubits=2, betas=(0.1, 1.0, 5.0), J=1.0, h=0.8, seed=42, solvers=("least_squares", "quadratic"))`
- **Para que serve (Tarefa 2):**
  - Constrói o estado térmico $\rho_{\text{gibbs}} = e^{-\beta H} / Z$ para o Hamiltoniano TFIM;
  - Gera para cada $\beta$ um estado de controle aleatório com mesma entropia de von Neumann;
  - Executa tomografia com **medição completa de Pauli** ($M=15$) em ambos os estados com os mesmos chutes iniciais;
  - Retorna tabela de comparação e trajetórias de custo para responder diretamente à questão de facilidade relativa de reconstrução.

### `run_energy_introduction_experiment`
Investiga o efeito de incorporar a energia $H$ como custo primal com observáveis escassos ($M=4$ de 15), demonstrando que o prior físico permite recuperar o estado fundamental com fidelidade $1.0000$ onde a tomografia padrão atinge apenas $F \approx 0.32$.

### `run_noise_stability_experiment`
Executa simulações estatísticas monte carlo com ruído binomial de shot noise para $N_{\text{shots}} \in [10, 10000]$, quantificando a infactibilidade do SDP linear de viabilidade vs. a estabilidade garantida de Mínimos Quadrados.

### `run_shots_scaling_benchmark`
- **Assinatura:** `run_shots_scaling_benchmark(qubits_range=(1, 2, 3, 4), n_shots=10000, methods=("least_squares", "quadratic"), n_reps=5, seed=42, max_iter_quadratic=300)`
- **Para que serve (Tarefa 1):**
  - Conduz o benchmark sistemático de escalabilidade em qubits com ruído estatístico real da regra de Born;
  - Gera estados puros aleatórios com semente determinística fixa por qubit;
  - Utiliza base completa de Pauli ($4^N-1$ operadores);
  - Aplica $N_{\text{shots}} = 10{,}000$ consistente em todas as configurações por $\ge 5$ repetições independentes;
  - Retorna médias e desvios padrão ($\pm 1\sigma$) de fidelidade, número de iterações/avaliações e tempo de computação.

---

## 8. Tabela Síntese Comparativa dos Métodos

| Critério | Força Bruta (Cholesky $T^\dagger T$) | SDP Primal Linear (SCS) | SDP Dual Regularizado (Quadrático JAX) | SDP Mínimos Quadrados Convexos (CLARABEL) |
|---|---|---|---|---|
| **Espaço de Otimização** | $\mathbb{R}^{d^2}$ (não-linear) | Matrizes hermitianas $d \times d$ no cone PSD | $\mathbb{R}^M$ (irrestrito, côncavo) | Matrizes hermitianas $d \times d$ no cone PSD |
| **Garantia de Mínimo** | Mínimo local (não-convexo) | Ponto arbitrário do conjunto afim | Mínimo global único certificado | Mínimo global único certificado |
| **Sensibilidade a $x_0$** | Alta ($x_0$ aleatório afeta trajetória) | Baixa ($\rho_0 = 0$) | Nenhuma ($\alpha_0 = \vec{0}$) | Nenhuma ($\rho_0 = 0$) |
| **Tempo de Parada ($N=4$ qubits)** | Médio (~0.05s a 0.2s) | Inviável / Timeout (> dezenas de s) | **Ultra-rápido (~0.2s a 0.4s)** | Alto (segundos) |
| **Critério de Parada** | `gtol` / `ftol` no resíduo | Resíduos primal e dual $\le \epsilon_{\text{feas}}$ | $\|\nabla \mathcal{D}\| = \|A\pi - q\| \le 10^{-12}$ | Tolerância KKT de interior-point |
| **Robustez a Shot Noise** | Alta (mínimos quadrados) | **Baixa (infactível para shots baixos)** | Alta com $\varepsilon$ adequado | **Alta (garantia de factibilidade)** |
| **Propriedade Espectral** | Depende do chute inicial | Indeterminada | Favorece baixo posto (fronteira PSD) | Projeção euclidiana no cone |
| **Integração de Energia ($H$)** | Penalidade ad-hoc na perda | Custo linear em $\operatorname{Tr}[H\rho]$ | Termo físico natural no argumento de corte | Termo linear de energia |
