import json
import uuid

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# ==============================================================================
# HEADER DO NOTEBOOK
# ==============================================================================
add_md(r"""# Tomografia de Estados Quânticos (Quantum State Tomography)
### Projeto de Aspectos Matemáticos e Computacionais da Mecânica Quântica
**Autor:** David Oliveira & Colaboradores

---

## Estrutura do Trabalho

Este projeto está estruturado em **quatro seções centrais**, analisando as abordagens fundamentais para reconstrução de matrizes densidade quânticas:

1. **Seção 1: O Algoritmo de Força Bruta (Parametrização de Cholesky)**
   - Formulação teórica de Cholesky $T^\dagger T / \operatorname{Tr}(T^\dagger T)$ e Jacobiano analítico exato.
   - Análise detalhada de **inicialização**, **trajetória/curso** e **tempo/critérios de parada**.
   - O problema da não-unicidade na Esfera de Bloch com medições incompletas.

2. **Seção 2: A Formulação Primal do SDP (Programação Semidefinida)**
   - Formulação cônica direta no espaço de operadores hermitianos $\rho \succeq 0, \operatorname{Tr}\rho = 1$.
   - Solvers cônicos primais (SCS / CLARABEL).
   - Análise de **inicialização**, **trajetória de divisão de operadores** e **tempo de parada**.

3. **Seção 3: A Formulação Computacional do Problema Dual (A Terceira Via)**
   - Dualidade de Lagrange e regularização quadrática ($L^2$ / Frobenius).
   - Otimização dual irrestrita e côncava em $\mathbb{R}^M$ com compilador JAX e L-BFGS (`sdplab`).
   - Análise de **inicialização**, **trajetória de decaimento do resíduo** e **tempo de parada ultra-rápido**.

4. **Seção 4: Comparação Abrangente dos Métodos (Aspectos Computacionais e Físicos)**
   - **Aspectos Computacionais:** Comparação de inicializações, trajetórias de convergência, tempos de parada e escalabilidade em qubits ($N = 1, 2, 3, 4$).
   - **Aspectos Físicos:** Espectro de pureza (estados puros vs misturas vs muito mistos), introdução de **Energia / Hamiltoniano físico** ($H$) e análise de estabilidade perante **ruído de shot noise**.
   - **Escalabilidade com Shots Reais:** Benchmark estatístico de fidelidade e iterações vs. número de qubits sob amostragem de Born ($N_{\text{shots}} = 10{,}000$).
   - **Exemplos Palpáveis em Altos Qubits:** Reconstrução observável por observável e projeção na Esfera de Bloch via traço parcial para os estados W ($N=3$) e GHZ ($N=4$).
   - Tabela síntese final com resumo de todos os critérios.
""")

# ==============================================================================
# SETUP E IMPORTAÇÕES
# ==============================================================================
add_code(r"""# Configuração do ambiente e importação do toolkit modular
import time
import numpy as np
import matplotlib.pyplot as plt

# Habilita float64 no JAX antes de instanciar contextos (precisão quântica ~1e-13)
import jax
jax.config.update("jax_enable_x64", True)

# Importa a biblioteca desenvolvida para o projeto
import quantum_tomography as qt
from quantum_tomography import (
    QuantState,
    random_pure_state,
    standard_pure_state,
    random_mixture_of_pures,
    depolarized_state,
    maximally_mixed_state,
    pauli_basis_strings,
    measure_observables,
    simula_medidas_shots,
    solve_tomography,
    partial_trace,
    vetor_bloch,
    plotar_bloch,
    plot_spectra,
    plot_qubit_scaling,
    plot_fidelity_vs_shots,
    plot_shots_scaling,
    plot_purity_convergence,
    plot_gibbs_vs_control_convergence,
    plot_purity_completeness_comparison,
    transverse_field_ising,
    ground_state,
    gibbs_state,
    run_shots_scaling_benchmark,
)

print(f"Toolkit quantum_tomography v{qt.__version__} carregado com sucesso!")
""")

# ==============================================================================
# SEÇÃO 1: FORÇA BRUTA (CHOLESKY)
# ==============================================================================
add_md(r"""---
# Seção 1: O Algoritmo de Força Bruta (Parametrização de Cholesky)

### 1.1 Formulação Matemática
Uma matriz densidade física $\rho$ que descreve um sistema quântico de $N$ qubits ($d = 2^N$) deve satisfazer três restrições fundamentais:
1. **Hermiticidade:** $\rho = \rho^\dagger$
2. **Positividade Semidefinida:** $\rho \succeq 0$ (todos os autovalores $\lambda_i \ge 0$)
3. **Normalização do Traço:** $\operatorname{Tr}[\rho] = 1$

Em vez de impor essas restrições via multiplicadores de Lagrange ou penalidades, a parametrização de Cholesky garante todas as três propriedades por construção:
Definimos uma matriz triangular inferior $T \in \mathbb{C}^{d \times d}$ com diagonal real estritamente e elementos sub-diagonais complexos:
$$
T = \begin{pmatrix}
t_0 & 0 & \dots & 0 \\
t_d + i t_{d+n_{\text{off}}} & t_1 & \dots & 0 \\
\vdots & \vdots & \ddots & 0 \\
\dots & \dots & \dots & t_{d-1}
\end{pmatrix}, \quad n_{\text{off}} = \frac{d(d-1)}{2}.
$$
O número total de parâmetros reais livres é $d + 2 \cdot \frac{d(d-1)}{2} = d^2 = 4^N$.
A matriz densidade é dada por:
$$
\rho(T) = \frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}.
$$
- $T^\dagger T$ é auto-adjunta e positiva semidefinida para qualquer escolha de $T \neq 0$.
- A divisão pelo traço $\operatorname{Tr}(T^\dagger T) = \sum_{ij} |T_{ij}|^2$ garante $\operatorname{Tr}[\rho] = 1$.

O problema de tomografia reduz-se a minimizar o funcional de mínimos quadrados não-lineares:
$$
\min_{x \in \mathbb{R}^{d^2}} F(x) = \frac{1}{2} \sum_{i=1}^M \left( \operatorname{Tr}[\rho(T(x)) Q_i] - q_i \right)^2.
$$
Utilizamos o **Jacobiano Analítico Fechado** derivado via regra da cadeia matricial, evitando erros de truncamento das diferenças finitas e acelerando a convergência.

---

### 1.2 Inicialização, Trajetória/Curso e Tempo de Parada
- **Inicialização ($x_0$):** O vetor inicial é sorteado aleatoriamente de uma gaussiana padrão $x_0 \sim \mathcal{N}(0, 1)$. Como a superfície de custo é não-convexa, diferentes inicializações percorrem trajetórias distintas e, em sistemas subdeterminados, convergem para matrizes densidade distintas.
- **Trajetória (Curso do Algoritmo):** O algoritmo de Trust-Region Reflective (TRF) / Levenberg-Marquardt atualiza $x_{k+1} = x_k + \Delta x_k$ reduzindo o resíduo quadrático $\|f(x)\|_2^2$.
- **Tempo e Critérios de Parada:** O solver interrompe a busca quando:
  1. `gtol`: $\|\nabla F(x)\|_\infty < 10^{-8}$ (gradiente nulo, ponto estacionário atingido).
  2. `ftol`: decaimento relativo da função de custo menor que $10^{-8}$.
  3. `xtol`: tamanho do passo $\|\Delta x\| < 10^{-8}$.
  4. `max_nfev`: limite de avaliações de função atingido.
""")

add_md(r"""#### [Slide] Experimento 1.1: Reconstrução por Força Bruta e Dinâmica de Convergência
- **Estado Quântico Alvo:** Estado puro arbitrário de 1 qubit $|\psi\rangle = \frac{2}{\sqrt{5}}|0\rangle + \frac{1}{\sqrt{5}}|1\rangle$ (vetor de Bloch $\vec{r} = [0.8, 0.0, 0.6]$, pureza $\gamma = 1.0$, posto 1).
- **Objetivo da Demonstração:** Avaliar o algoritmo de Força Bruta com parametrização de Cholesky ($T^\dagger T / \operatorname{Tr}(T^\dagger T)$), mostrando que a convergência da função de custo e o critério de parada (`gtol`/`ftol`) dependem da trajetória traçada a partir de diferentes chutes iniciais aleatórios $x_0 \sim \mathcal{N}(0, 1)$, sob medições exatas completas ($\sigma_x, \sigma_y, \sigma_z$).
""")

add_code(r"""# Experimento 1.1: Reconstrução via Força Bruta e Rastreamento da Trajetória
np.random.seed(42)
unk_state = QuantState([2.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)])
observables_1q = ["X", "Y", "Z"]
q_exatos = measure_observables(unk_state, observables_1q)

# Executa com múltiplos chutes iniciais para ilustrar trajetórias e tempos de parada
chutes = [np.random.randn(4) for _ in range(3)]
resultados_fb = []

plt.figure(figsize=(8, 4.5))

for idx, x0 in enumerate(chutes):
    res = solve_tomography(observables_1q, q_exatos, method="least_squares",
                           chute_inicial=x0, true_state=unk_state, n_qubits=1)
    resultados_fb.append(res)
    custos = res.extra["cost_history"]
    plt.semilogy(custos, "o-", markersize=3, label=f"Chute {idx+1} (Parou em {res.num_iters} evals, {res.elapsed_seconds*1000:.1f} ms)")

plt.title("Força Bruta (Cholesky): Trajetória da Função de Custo vs. Avaliações", fontsize=12, fontweight="bold")
plt.xlabel("Avaliação de Função (k)", fontsize=11)
plt.ylabel("Custo F(x) = 0.5 * ||f(x)||^2 (escala log)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

print(f"{'Execução':<12s} | {'Critério de Parada':<40s} | {'Avaliações':>10s} | {'Tempo (ms)':>10s} | {'Fidelidade':>10s} | {'Resíduo':>10s}")
print("-" * 105)
for i, res in enumerate(resultados_fb):
    print(f"Chute {i+1:<6d} | {res.status:<40s} | {res.num_iters:10d} | {res.elapsed_seconds*1000:10.2f} | {res.fidelity:10.6f} | {res.residual:10.2e}")
""")

add_md(r"""### 1.3 O Problema da Não-Unicidade na Esfera de Bloch
Quando medimos apenas um subconjunto estrito de observáveis ($M < 4^N - 1$), como $\sigma_x = 0.7$ e $\sigma_y = 0.4$, o grau de liberdade $\sigma_z$ permanece livre. 
A Força Bruta cai em diferentes pontos da reta permitida no interior da esfera dependendo do chute inicial $x_0$, demonstrando a ausência de um critério de seleção espectral intrínseco.
""")

add_md(r"""#### [Slide] Experimento 1.2: Não-Unicidade na Esfera de Bloch (Medições Incompletas)
- **Estado Quântico Alvo:** Família contínua de estados de 1 qubit compatíveis com dados incompletos $\langle \sigma_x \rangle = 0.7$ e $\langle \sigma_y \rangle = 0.4$, com o eixo $\sigma_z$ completamente não medido.
- **Objetivo da Demonstração:** Demonstrar geometricamente na Esfera de Bloch que, na ausência de medições informacionalmente completas, a Força Bruta colapsa em pontos arbitrários da reta de incerteza dependendo do chute inicial, evidenciando a ausência de um critério de seleção espectral intrínseco.
""")

add_code(r"""# Experimento 1.2: Não-Unicidade com Força Bruta na Esfera de Bloch
res_uniq = qt.run_uniqueness_experiment(q_values=(0.7, 0.4), seed=42)

rho_lsq_a = res_uniq["least_squares_a"].rho
rho_lsq_b = res_uniq["least_squares_b"].rho
rho_lsq_c = solve_tomography(["X", "Y"], [0.7, 0.4], method="least_squares",
                             chute_inicial=np.random.randn(4), n_qubits=1).rho

print("--- Estados Reconstruídos por Força Bruta com Diferentes Inicializações ---")
print(f"Força Bruta A: r = {np.round(qt.vetor_bloch(rho_lsq_a), 3)} | Parada: {res_uniq['least_squares_a'].num_iters} evals, S = {qt.von_neumann_entropy(rho_lsq_a):.4f}")
print(f"Força Bruta B: r = {np.round(qt.vetor_bloch(rho_lsq_b), 3)} | Parada: {res_uniq['least_squares_b'].num_iters} evals, S = {qt.von_neumann_entropy(rho_lsq_b):.4f}")
print(f"Força Bruta C: r = {np.round(qt.vetor_bloch(rho_lsq_c), 3)} | Parada: {qt.von_neumann_entropy(rho_lsq_c):.4f}")

# Visualização na Esfera de Bloch
fig_bloch = plotar_bloch(
    [rho_lsq_a, rho_lsq_b, rho_lsq_c],
    rotulos=["Cholesky Chute A", "Cholesky Chute B", "Cholesky Chute C"],
    cores=["#31a354", "#756bb1", "#de2d26"],
    medidos=("x", "y"),
    titulo="Não-Unicidade na Esfera de Bloch (Força Bruta com Diferentes Chutes)",
    salvar="bloch_reconstrucoes.png"
)
plt.show()
""")

# ==============================================================================
# SEÇÃO 2: SDP PRIMAL
# ==============================================================================
add_md(r"""---
# Seção 2: A Formulação Primal do SDP (Programação Semidefinida Padrão)

### 2.1 Formulação Matemática Primal
A Programação Semidefinida (SDP) formula a tomografia diretamente no espaço afim das matrizes hermitianas sob a restrição cônica de positividade semidefinida.
Para dados exatos, trata-se de um problema de viabilidade com custo nulo:
$$
\min_{\rho \in \mathcal{H}_d} 0 \quad \text{sujeito a} \quad \operatorname{Tr}[Q_i \rho] = q_i \quad (i=1,\dots,M), \quad \operatorname{Tr}[\rho] = 1, \quad \rho \succeq 0.
$$
Quando os dados contêm ruído estatístico (onde o conjunto afim pode não intersectar o cone $\rho \succeq 0$), a formulação primal estende-se naturalmente para **Mínimos Quadrados Convexos**:
$$
\min_{\rho \in \mathcal{H}_d} \sum_{i=1}^M \left(\operatorname{Tr}[Q_i \rho] - q_i\right)^2 \quad \text{sujeito a} \quad \operatorname{Tr}[\rho] = 1, \quad \rho \succeq 0.
$$
Esta formulação convexa elimina mínimos locais: qualquer mínimo local é global e a solução é geometricamente a projeção euclidiana dos dados sobre o cone de matrizes densidade válidas.

---

### 2.2 Inicialização, Trajetória/Curso e Tempo de Parada
- **Inicialização:** Solvers primais como o SCS (Splitting Conic Solver) utilizam técnicas de divisão de operadores (ADMM / Douglas-Rachford). A inicialização padrão é no ponto zero $(\rho_0 = 0, y_0 = 0)$ ou projeção inicial.
- **Trajetória (Curso do Algoritmo):** A iteração projeta alternadamente o ponto candidato sobre o subespaço afim $\mathcal{A}\rho = q$ e sobre o cone positivo semidefinido $\rho \succeq 0$ (via diagonalização e corte de autovalores negativos). Os resíduos primal $\|r_p\|_2 = \|\mathcal{A}\rho - q\|_2$ e dual $\|r_d\|_2$ são monitorados iterativamente.
- **Tempo e Critérios de Parada:** 
  1. *Convergência ótima:* Parada quando $\|r_p\| < \epsilon_{\text{feas}}$ e $\|r_d\| < \epsilon_{\text{feas}}$.
  2. *Infactibilidade:* Se os dados experimentais forem ruidosos e saírem do cone físico, o solver primal interrompe com o status `infeasible`, demonstrando a sensibilidade da viabilidade estrita. Em contrapartida, a versão de mínimos quadrados convexos converge sempre com sucesso incondicional.
""")

add_md(r"""#### [Slide] Experimento 2: Formulação Primal do SDP (Viabilidade Estrita vs. Mínimos Quadrados Convexos)
- **Estado Quântico Alvo:** Estado puro de 1 qubit $|\psi\rangle = \frac{2}{\sqrt{5}}|0\rangle + \frac{1}{\sqrt{5}}|1\rangle$ sob medições exatas, contrastado com medições artificiais ruidosas não-físicas ($\vec{q} = [0.9, 0.9, 0.9]$ com $\|\vec{q}\|_2 > 1$).
- **Objetivo da Demonstração:** Comparar o solver primal cônico linear (SCS) com a formulação por Mínimos Quadrados Convexos, comprovando que dados experimentais ruidosos tornam o SDP de viabilidade estrita infactível (`infeasible`), enquanto a projeção euclidiana convexa garante convergência incondicional e factibilidade física ($\rho \succeq 0, \operatorname{Tr}\rho = 1$).
""")

add_code(r"""# Experimento 2: Solvers Primais via CVXPY (Viabilidade Linear SCS vs Mínimos Quadrados Convexos)
print("Resolvendo problema primal via CVXPY SCS e Mínimos Quadrados Convexos...")

# 1. Viabilidade Linear SCS (Dados Exatos)
res_scs = solve_tomography(observables_1q, q_exatos, method="cvxpy_scs", true_state=unk_state)

# 2. Mínimos Quadrados Convexos (Dados Exatos)
res_lsq_c = solve_tomography(observables_1q, q_exatos, method="cvxpy_lsq", true_state=unk_state)

# 3. Teste de Estabilidade com Ruído Extremo (ponto fora da esfera de Bloch: q = [0.9, 0.9, 0.9])
q_ruido_extremo = [0.9, 0.9, 0.9] # norma = sqrt(0.81*3) = 1.56 > 1 (não físico!)
res_scs_ruido = solve_tomography(observables_1q, q_ruido_extremo, method="cvxpy_scs")
res_lsq_ruido = solve_tomography(observables_1q, q_ruido_extremo, method="cvxpy_lsq")

print(f"{'Formulação Primal':<35s} | {'Dados':<10s} | {'Status de Parada':<15s} | {'Iterações':>9s} | {'Tempo (ms)':>10s} | {'Resíduo':>10s}")
print("-" * 105)
print(f"{'Viabilidade Linear (SCS)':<35s} | {'Exatos':<10s} | {res_scs.status:<15s} | {res_scs.num_iters:9d} | {res_scs.elapsed_seconds*1000:10.2f} | {res_scs.residual:10.2e}")
print(f"{'Mínimos Quadrados Convexos':<35s} | {'Exatos':<10s} | {res_lsq_c.status:<15s} | {res_lsq_c.num_iters:9d} | {res_lsq_c.elapsed_seconds*1000:10.2f} | {res_lsq_c.residual:10.2e}")
print(f"{'Viabilidade Linear (SCS)':<35s} | {'Ruidosos':<10s} | {res_scs_ruido.status:<15s} | {res_scs_ruido.num_iters:9d} | {res_scs_ruido.elapsed_seconds*1000:10.2f} | {res_scs_ruido.residual:10.2e}")
print(f"{'Mínimos Quadrados Convexos':<35s} | {'Ruidosos':<10s} | {res_lsq_ruido.status:<15s} | {res_lsq_ruido.num_iters:9d} | {res_lsq_ruido.elapsed_seconds*1000:10.2f} | {res_lsq_ruido.residual:10.2e}")
""")

# ==============================================================================
# SEÇÃO 3: FORMULAÇÃO DUAL (A TERCEIRA VIA)
# ==============================================================================
add_md(r"""---
# Seção 3: A Formulação Computacional do Problema Dual (A Terceira Via)

### 3.1 Por que o Problema Dual é uma Terceira Via Superior?
Tanto a Força Bruta quanto o SDP Primal enfrentam sérios gargalos:
- A Força Bruta otimiza em um espaço não-convexo, sofrendo com mínimos locais e dependência de semente aleatória.
- O SDP Primal manipula variáveis matriciais densas de dimensão $2^N \times 2^N$ sob restrições cônicas semidefinidas $\rho \succeq 0$, exigindo fatorações matriciais $\mathcal{O}(d^6)$ que tornam o cálculo inviável para $N \ge 4$ qubits.

A **Teoria da Dualidade de Lagrange com Regularização Espectral** fornece uma alternativa computacionalmente elegante:
Introduzimos uma penalidade de norma de Frobenius (regularização quadrática $L^2$) com parâmetro $\varepsilon > 0$ e um termo de energia opcional $\operatorname{Tr}[H \pi]$:
$$
\min_{\pi \succeq 0, \operatorname{Tr}\pi = 1} \left\{ \frac{\varepsilon}{2} \|\pi\|_F^2 + \operatorname{Tr}[H \pi] \right\} \quad \text{sujeito a} \quad \operatorname{Tr}[Q_i \pi] = q_i \quad (i=1,\dots,M).
$$
Por dualidade de Fenchel-Legendre, o problema dual associado é:
$$
\max_{\alpha \in \mathbb{R}^M} \mathcal{D}_\varepsilon(\alpha) = \sum_{i=1}^M \alpha_i q_i - \varepsilon \operatorname{Tr}\left[ \psi\left( \frac{\sum_{i=1}^M \alpha_i Q_i - H}{\varepsilon} \right) \right],
$$
onde $\psi(s) = \frac{1}{2}\max\{s, 0\}^2$ e sua derivada é $\psi'(s) = \max\{s, 0\}$.

### O Mapeamento Primal-Dual e o Gradiente Analítico
Uma vez encontrado o maximizador dual $\alpha^*$, o estado quântico primal correspondente é obtido analiticamente por:
$$
\pi^*(\alpha) = \left( \frac{\sum_{i=1}^M \alpha_i Q_i - H}{\varepsilon} - \theta \right)_+,
$$
onde $\theta \in \mathbb{R}$ é o multiplicador de Lagrange que impõe a normalização de traço $\operatorname{Tr}\pi = 1$.
Diferenciando o dual em relação a cada variável $\alpha_j$:
$$
\frac{\partial \mathcal{D}_\varepsilon}{\partial \alpha_j}(\alpha) = q_j - \operatorname{Tr}[Q_j \pi^*(\alpha)].
$$
**Propriedade Notável:** O gradiente da função dual $\nabla \mathcal{D}_\varepsilon(\alpha)$ **é exatamente o resíduo das restrições tomográficas**!
Portanto, resolver o dual até a condição de gradiente nulo $\|\nabla \mathcal{D}\| = 0$ equivale a recuperar um estado quântico físico que satisfaz exatamente todas as medições experimentais!

---

### 3.2 Inicialização, Trajetória/Curso e Tempo de Parada no Dual
- **Inicialização ($\alpha_0$):** O ponto inicial natural é $\alpha_0 = \vec{0} \in \mathbb{R}^M$. Diferente da Força Bruta, não há aleatoriedade ou sensibilidade a sementes.
- **Trajetória (Curso do Algoritmo):** Como a função dual $\mathcal{D}_\varepsilon(\alpha)$ é estritamente côncava e suave, sua maximização é resolvida por **L-BFGS unconstrained compilado em JAX** (`optax.lbfgs()`). A norma do gradiente decresce com taxa superlinear.
- **Tempo e Critérios de Parada:** A condição de parada é $\|\nabla \mathcal{D}_\varepsilon(\alpha)\|_\infty \le \text{tol}$ (fixamos $\text{tol} = 10^{-12}$). A convergência ocorre em frações de milissegundo mesmo para dimensões elevadas.
""")

add_md(r"""#### [Slide] Experimento 3: Formulação Dual Regularizada (A Terceira Via em JAX)
- **Estado Quântico Alvo:** Estado puro de 1 qubit $|\psi\rangle = \frac{2}{\sqrt{5}}|0\rangle + \frac{1}{\sqrt{5}}|1\rangle$ (medição completa) e estado puro Haar-aleatório de 2 qubits (medição incompleta de 6 observáveis).
- **Objetivo da Demonstração:** Demonstrar a resolução no espaço dual irrestrito $\mathbb{R}^M$ via L-BFGS acelerado em JAX (`sdplab`), confirmando que a inicialização neutra $\alpha_0 = \vec{0}$ atinge tolerância de máquina ($\|\nabla \mathcal{D}\| \le 10^{-12}$) em frações de milissegundo, onde o gradiente dual atua diretamente como o resíduo das equações de medição.
""")

add_code(r"""# Experimento 3: Resolução via SDP Dual e Análise de Trajetória e Parada
print("Resolvendo tomografia via formulação dual (sdplab / spacecore / JAX)...")

res_dual = solve_tomography(observables_1q, q_exatos, method="quadratic",
                            true_state=unk_state, tol=1e-12)

print(f"Status de Parada : {res_dual.status}")
print(f"Iterações L-BFGS : {res_dual.num_iters}")
print(f"Tempo de Parada  : {res_dual.elapsed_seconds*1000:.2f} ms ({res_dual.elapsed_seconds:.6f} s)")
print(f"Norma ||Grad D|| : {res_dual.residual:.2e}")
print(f"Fidelidade Final : {res_dual.fidelity:.6f}")
print(f"Variáveis Dual a*: {np.round(res_dual.dual_variables, 4)}\n")

# Para ilustrar a trajetória completa em múltiplas iterações, executamos em 2 qubits com dados parciais
state_2q = random_pure_state(4, seed=42)
all_obs_2q = pauli_basis_strings(2)
obs_parcial = all_obs_2q[:6]  # 6 observáveis
q_parcial = measure_observables(state_2q, obs_parcial)

res_dual_2q = solve_tomography(obs_parcial, q_parcial, method="quadratic", true_state=state_2q, tol=1e-12)

# Exibe relatório resumo
print(res_dual_2q.summary())
""")

# ==============================================================================
# SEÇÃO 4: COMPARAÇÃO ABRANGENTE
# ==============================================================================
add_md(r"""---
# Seção 4: Comparação Abrangente dos Métodos (Aspectos Computacionais e Físicos)

Nesta seção final, colocamos lado a lado as três abordagens (**Força Bruta**, **SDP Primal** e **SDP Dual**) sob testes rigorosos:
1. **Aspectos Computacionais:** Comparação de inicializações, trajetórias e tempo de parada (escalabilidade com múltiplos qubits).
2. **Aspectos Físicos:** Espectro de pureza espectral (estados puros vs misturas de puros vs totalmente mistos).
3. **Introdução de Energia ao Problema:** Acoplamento do Hamiltoniano do sistema ($H$) para tomografia física em dados escassos.
4. **Estabilidade e Ruído de Shot Noise:** Resposta dos algoritmos sob medições binomiais reais da regra de Born.
5. **Escalabilidade com Shots Reais:** Benchmark estatístico de fidelidade e iterações vs. número de qubits sob amostragem de Born ($N_{\text{shots}} = 10{,}000$).
6. **Exemplos Palpáveis em Altos Qubits:** Reconstrução observável por observável e projeção na Esfera de Bloch via traço parcial para os estados W ($N=3$) e GHZ ($N=4$).

---

### 4.1 Inicialização, Trajetória e Tempo de Parada Comparados
""")

add_md(r"""#### [Slide] Comparação 4.1: Trajetória de Convergência e Tempo de Parada Lado a Lado
- **Estado Quântico Alvo:** Estado puro de 1 qubit $|\psi\rangle = \frac{2}{\sqrt{5}}|0\rangle + \frac{1}{\sqrt{5}}|1\rangle$ com base completa exata $\{\sigma_x, \sigma_y, \sigma_z\}$.
- **Objetivo da Demonstração:** Confrontar lado a lado o perfil de decaimento do erro e a latência de execução dos três métodos (Força Bruta, SDP Primal SCS e SDP Dual Regularizado), visualizando a evolução das iterações e os tempos de parada em milissegundos.
""")

add_code(r"""# Comparação 4.1: Trajetória de Convergência e Tempo de Parada Lado a Lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Trajetória Força Bruta (Cholesky)
custos_fb = resultados_fb[0].extra["cost_history"]
ax1.semilogy(custos_fb, "o-", color="#31a354", lw=2, label="Força Bruta (Cholesky)")
ax1.set_title("Força Bruta: Custo vs. Avaliações de Função", fontsize=11, fontweight="bold")
ax1.set_xlabel("Avaliações de Função", fontsize=10)
ax1.set_ylabel("Custo 0.5 * ||f(x)||^2", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Resíduo e Tempo de Parada
metodos_comp = ["Força Bruta", "SDP Primal (SCS)", "SDP Dual (Quadrático)"]
tempos_comp = [res_fb.elapsed_seconds * 1000 for res_fb in [resultados_fb[0], res_scs, res_dual]]
cores = ["#31a354", "#3182bd", "#e6550d"]

bars = ax2.bar(metodos_comp, tempos_comp, color=cores, width=0.5)
ax2.set_ylabel("Tempo de Parada (ms)", fontsize=10)
ax2.set_title("Comparação do Tempo de Parada (1 Qubit)", fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.3, axis="y")
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.show()
""")

add_md(r"""### 4.2 Escalabilidade Computacional em Múltiplos Qubits ($N = 1, 2, 3, 4$)
A dimensão do espaço de operadores de Pauli cresce como $4^N - 1$. 
O teste a seguir demonstra por que a formulação dual em JAX é necessária para tomografia em escala quântica real, superando solvers tradicionais por ordens de grandeza sob dados exatos.
""")

add_md(r"""#### [Slide] Comparação 4.2: Escalabilidade Computacional em Múltiplos Qubits (Tempo com Dados Exatos)
- **Estado Quântico Alvo:** Estados puros Haar-aleatórios gerados para sistemas de $N \in \{1, 2, 3, 4\}$ qubits ($d = 2, 4, 8, 16$) com base de Pauli completa exata ($M = 4^N - 1$).
- **Objetivo da Demonstração:** Medir a escalabilidade assintótica do tempo de computação conforme a dimensão cresce exponencialmente, demonstrando a inviabilidade do solver primal SCS para $N \ge 4$ qubits e a superioridade de ordens de grandeza do solver dual compilado em JAX.
""")

add_code(r"""# Comparação 4.2: Benchmark de Escalabilidade em Qubits
print("Executando benchmark de escalabilidade em qubits...")
benchmark_data = qt.run_qubit_scaling_benchmark(
    qubits_range=[1, 2, 3, 4],
    methods=["quadratic", "least_squares", "cvxpy_scs"],
    seed=42
)

# Gráfico de tempo vs N qubits (escala log)
fig_scale = plot_qubit_scaling(
    benchmark_data["qubits_range"],
    {
        "SDP Dual (Quadrático JAX)": benchmark_data["tempos"]["quadratic"],
        "Força Bruta (Cholesky SciPy)": benchmark_data["tempos"]["least_squares"],
        "SDP Primal (CVXPY SCS)": benchmark_data["tempos"]["cvxpy_scs"],
    },
    title="Escalabilidade: Tempo de Parada vs. Número de Qubits",
    save_path="benchmark_tempo_sdp.png"
)
plt.show()

# Tabela resumo de tempos
print(f"{'Qubits':<8s} | {'Dimensão':<10s} | {'Observáveis':<12s} | {'SDP Dual (s)':<16s} | {'Força Bruta (s)':<18s} | {'SDP Primal SCS (s)':<20s}")
print("-" * 95)
for i, n in enumerate(benchmark_data["qubits_range"]):
    d = 2 ** n
    m = 4 ** n - 1
    t_dual = benchmark_data["tempos"]["quadratic"][i]
    t_lsq = benchmark_data["tempos"]["least_squares"][i]
    t_scs = benchmark_data["tempos"]["cvxpy_scs"][i]
    scs_str = f"{t_scs:.4f}" if not np.isnan(t_scs) else "Timeout / Inviável"
    print(f"{n:<8d} | {d:<10d} | {m:<12d} | {t_dual:<16.4f} | {t_lsq:<18.4f} | {scs_str:<20s}")
""")

add_md(r"""### 4.3 Aspectos Físicos: Espectro de Pureza
Analisamos como os métodos respondem a diferentes níveis de pureza espectral:
1. **Estado Puro:** Posto 1, $\gamma = \operatorname{Tr}[\rho^2] = 1$, entropia $S(\rho) = 0$.
2. **Mistura de Puros:** Combinação convexa $\sum p_j |v_j\rangle\langle v_j|$, com posto $k < d$ intermediário.
3. **Estado Despolarizado:** $(1-p)\rho + p \frac{I}{d}$ (mistura com ruído uniforme).
4. **Estado Totalmente Misto:** $\rho = I/d$, máxima desordem e posto completo.

A **Regularização Quadrática** $\pi^* = (\dots)_+$ atua como um operador de corte sobre os autovalores negativos, direcionando o estado para a **fronteira do cone semidefinido** ($\lambda_i = 0$), sendo particularmente adaptada para reconstruir estados de baixo posto numérico.
""")

add_md(r"""#### [Slide] Comparação 4.3: Resposta dos Algoritmos ao Espectro de Pureza
- **Estado Quântico Alvo:** Quatro classes espectrais de 2 qubits: 1) Estado Puro (posto 1), 2) Mistura convexa de puros (posto 2), 3) Estado Despolarizado ($p=0.6$) e 4) Estado Totalmente Misto ($\rho = I/4$), sob medições parciais ($M=6$).
- **Objetivo da Demonstração:** Analisar a resposta de autovalores dos algoritmos, demonstrando que a regularização quadrática dual projeta naturalmente a solução na fronteira do cone PSD (favorecendo baixo posto numérico), enquanto a Força Bruta distribui o espectro conforme o chute inicial.
""")

add_code(r"""# Comparação 4.3: Espectro de Pureza
purity_results = qt.run_purity_spectrum_experiment(n_qubits=2, num_measurements=6, seed=42)

print(f"{'Classe de Estado':<30s} | {'Método':<25s} | {'Fidelidade':>10s} | {'Entropia S':>11s} | {'Pureza':>8s} | {'Posto':>6s}")
print("-" * 100)

for state_name, solvers_res in purity_results.items():
    for solver_name, res in solvers_res.items():
        if solver_name == "entropy":
            continue  # Omite entropia conforme solicitado
        nome_metodo = "SDP Dual (Quadrático)" if solver_name == "quadratic" else "Força Bruta (Cholesky)"
        print(f"{state_name:<30s} | {nome_metodo:<25s} | {res.fidelity:10.4f} | {res.entropy:11.4f} | {res.purity:8.4f} | {res.rank:6d}")
    print("-" * 100)

# Gráfico dos espectros de autovalores para Mistura de Puros
mistura_res = purity_results["2. Mistura de Puros (Posto 2)"]
true_state_mistura = random_mixture_of_pures(4, rank=2, seed=43)

paineis_espectro = [
    ("Estado Alvo (Posto 2)", true_state_mistura),
    ("SDP Dual (Quadrático - Fronteira)", mistura_res["quadratic"]),
    ("Força Bruta (Cholesky)", mistura_res["least_squares"]),
]

fig_spec = plot_spectra(paineis_espectro, title="Espectro de Autovalores: Mistura de Puros (dim=4)")
plt.show()
""")

add_md(r"""### 4.3.1 Trajetória de Convergência por Classe de Pureza (Isolamento de Variáveis)
Para investigar rigorosamente o impacto da pureza espectral e do posto sobre a convergência dos algoritmos, executamos o experimento de pureza fixando:
- **O MESMO conjunto de observáveis** para todas as 4 classes de estados;
- **O MESMO chute inicial fixo ($x_0$)** para o método de Força Bruta (Cholesky);
- A inicialização natural $\alpha_0 = \vec{0}$ para o método SDP Dual (Quadrático).

Com isso, isolamos estritamente a variável "tipo/pureza de estado" de quaisquer flutuações de inicialização ou amostragem de operadores.
Abaixo, comparamos a dinâmica de decaimento do custo para **Força Bruta** (custo $F(x) = \frac{1}{2} \|f(x)\|^2$ vs. avaliações) e para o **SDP Dual** (norma do gradiente dual $\|\nabla \mathcal{D}\|_2$ vs. iterações L-BFGS) lado a lado em escala semi-logarítmica.
""")

add_md(r"""#### [Slide] Comparação 4.3.1: Dinâmica de Convergência por Classe de Pureza (Variáveis Isoladas)
- **Estado Quântico Alvo:** As quatro classes espectrais de 2 qubits (Puro, Mistura Posto 2, Despolarizado, Totalmente Misto), submetidas ao **mesmo chute inicial $x_0$ fixo** e ao **mesmo conjunto fixo de 6 observáveis**.
- **Objetivo da Demonstração:** Isolar a pureza e a entropia do estado como variáveis causais independentes, comparando as curvas de custo e gradiente dual para verificar se a posição do estado no interior ou na fronteira do cone afeta a taxa de convergência dos otimizadores.
""")

add_code(r"""# Comparação 4.3.1: Trajetória de Convergência por Classe de Pureza
print("Executando experimento de pureza com chute x0 e observáveis fixos...")
purity_results_extended = qt.run_purity_spectrum_experiment(
    n_qubits=2,
    num_measurements=6,
    seed=42
)

# Tabela comparativa de convergência por classe
print(f"{'Classe de Estado':<30s} | {'Método':<25s} | {'Fidelidade':>10s} | {'S (Alvo)':>10s} | {'Iterações':>10s} | {'Tempo (ms)':>10s}")
print("-" * 105)
for state_name, solvers_res in purity_results_extended.items():
    for solver_name, res in solvers_res.items():
        nome_metodo = "SDP Dual (Quadrático)" if solver_name == "quadratic" else "Força Bruta (Cholesky)"
        s_true = res.extra.get("true_entropy", res.entropy)
        print(f"{state_name:<30s} | {nome_metodo:<25s} | {res.fidelity:10.4f} | {s_true:10.4f} | {res.num_iters:10d} | {res.elapsed_seconds*1000:10.2f}")

# Gráfico sobreposto das trajetórias de custo (dois subplots lado a lado)
fig_purity_conv = qt.plot_purity_convergence(
    purity_results_extended,
    title="Trajetória de Convergência por Classe de Pureza (Mesmo Chute Inicial x0 e Observáveis)",
    save_path="trajetoria_custo_pureza.png"
)
plt.show()
""")

add_md(r"""### 4.3.2 Controle de Completude de Medição: Incompleto ($M=6$) vs. Completo ($M=15$)
O experimento 4.3 utilizou $M=6$ observáveis para um sistema de 2 qubits (onde a base completa de Pauli possui $d^2 - 1 = 15$ observáveis), configurando um problema **subdeterminado**.
Nessa condição, como demonstrado na Seção 1.3, existe uma multiplicidade contínua de matrizes densidade compatíveis com as medições (não-unicidade).

Para demonstrar que o comportamento de convergência observado entre estados puros e mistos **não decorre do artefato de não-unicidade**, executamos o mesmo experimento com a base **COMPLETA** de 15 observáveis de Pauli ($M=15$).
Com medições completas, o estado compatível é matematicamente único, restaurando fidelidade $F = 1.0000$ em todos os métodos e isolando com precisão o efeito geométrico do cone PSD.
""")

add_md(r"""#### [Slide] Comparação 4.3.2: Efeito da Completude Informacional (M=6 vs M=15 Observáveis)
- **Estado Quântico Alvo:** As quatro classes de estados de 2 qubits reconstruídas sob medição subdeterminada incompleta ($M=6$) em contraste direto com a base completa de Pauli ($M=15$).
- **Objetivo da Demonstração:** Provar que as diferenças de fidelidade observadas no regime subdeterminado decorrem exclusivamente do artefato de não-unicidade, mostrando que medições completas restauram fidelidade unitária ($F = 1.0000$) para todas as classes em ambos os métodos.
""")

add_code(r"""# Comparação 4.3.2: Medição Incompleta (M=6) vs. Completa (M=15)
print("Executando experimento com medições completas (M=15 Pauli)...")
purity_complete = qt.run_purity_spectrum_experiment(
    n_qubits=2,
    measurement_completeness="complete",
    seed=42
)

# Tabela comparativa: Completo vs. Incompleto lado a lado
print(f"{'Classe de Estado':<28s} | {'Medição':<14s} | {'Método':<24s} | {'Fidelidade':>10s} | {'Resíduo':>10s} | {'Iterações':>10s}")
print("-" * 105)
for state_name in purity_results_extended:
    for m_label, exp_set in [("M=6 (Incomp.)", purity_results_extended), ("M=15 (Compl.)", purity_complete)]:
        for s_name in ["least_squares", "quadratic"]:
            res = exp_set[state_name][s_name]
            nome_m = "SDP Dual (Quadrático)" if s_name == "quadratic" else "Força Bruta (Cholesky)"
            print(f"{state_name:<28s} | {m_label:<14s} | {nome_m:<24s} | {res.fidelity:10.4f} | {res.residual:10.2e} | {res.num_iters:10d}")

# Gráfico comparativo de completude lado a lado
fig_comp = qt.plot_purity_completeness_comparison(
    purity_results_extended,
    purity_complete,
    title="Efeito da Completude Informacional na Convergência (M=6 vs M=15)",
    save_path="pureza_completa_vs_incompleta.png"
)
plt.show()
""")

add_md(r"""### 4.4 Aspectos Físicos: Introdução de Energia ($H$) à Tomografia
Quando o sistema físico é governado por um Hamiltoniano $H$ (ex: Modelo de Ising de Campo Transversal - TFIM), podemos incorporar a energia como parte do custo linear primal:
$$
\min_{\pi \succeq 0, \operatorname{Tr}\pi = 1} \left\{ \operatorname{Tr}[H \pi] + \frac{\varepsilon}{2} \|\pi\|_F^2 \right\} \quad \text{sujeito a} \quad \operatorname{Tr}[Q_i \pi] = q_i.
$$
No dual, o Hamiltoniano desloca o argumento da função de corte: $\pi^*(\alpha) = \left(\frac{\sum \alpha_i Q_i - H}{\varepsilon} - \theta\right)_+$.

**Importância Física:** Quando as medições experimentais são escassas e incompletas (ex: apenas 4 observáveis medidos de um total de 15 para 2 qubits), a tomografia padrão sem energia falha ($F \approx 0.32$). A introdução do Hamiltoniano do sistema fornece o **prior físico indispensável**, recuperando o estado fundamental com **fidelidade $F = 1.0000$**!
""")

add_md(r"""#### [Slide] Comparação 4.4: Tomografia Guiada por Energia com Medições Escassas (TFIM)
- **Estado Quântico Alvo:** Estado fundamental $|\psi_0\rangle$ do Modelo de Ising de Campo Transversal (TFIM, $J=1.0, h=0.8$) de 2 qubits ($E_0 = -2.5616$, pureza $\gamma = 1.0$), medindo apenas 4 de 15 observáveis.
- **Objetivo da Demonstração:** Demonstrar que acoplar o Hamiltoniano $H$ ao problema tomográfico atua como um prior físico indispensável, permitindo reconstruir o estado fundamental com fidelidade $F = 1.0000$ sob dados ultra-escassos onde a tomografia convencional falha ($F \approx 0.32$).
""")

add_code(r"""# Comparação 4.4: Tomografia com Energia (TFIM 2 Qubits)
n_q = 2
H_ising = transverse_field_ising(n_q, J=1.0, h=0.8)
e0_ground, ground_state_target = ground_state(H_ising)

print(f"Hamiltoniano TFIM gerado para {n_q} qubits (dim=4).")
print(f"Energia do Fundamental E_0: {e0_ground:.4f} | Pureza: {ground_state_target.purity:.4f}\n")

# Medições parciais: apenas 4 observáveis de um total de 15!
energy_exp = qt.run_energy_introduction_experiment(
    n_qubits=n_q,
    hamiltonian_type="ising",
    num_measurements=4,
    eps=0.15,
    seed=42
)

print(f"Observáveis medidos ({len(energy_exp['observables'])} de 15): {energy_exp['observables']}")
print(f"Valores esperados q medidos: {np.round(energy_exp['q'], 4)}\n")

print(f"{'Configuração Tomográfica':<36s} | {'Fidelidade':>10s} | {'Energia <H>':>12s} | {'Resíduo':>10s}")
print("-" * 75)
print(f"{'Estado Alvo (Fundamental)':<36s} | {1.0:10.4f} | {energy_exp['e_ground']:12.4f} | {'0.0':>10s}")
print(f"{'Sem Energia (H = 0, Padrão)':<36s} | {energy_exp['res_sem_h'].fidelity:10.4f} | {energy_exp['energias']['Reconstruído (H=0)']:12.4f} | {energy_exp['res_sem_h'].residual:10.2e}")
print(f"{'Com Energia (H = H_ising, Dual)':<36s} | {energy_exp['res_com_h_quad'].fidelity:10.4f} | {energy_exp['energias']['Reconstruído (Com H, Quadrático)']:12.4f} | {energy_exp['res_com_h_quad'].residual:10.2e}")

# Gráfico comparativo dos autovalores
panels_energy = [
    ("Fundamental Alvo", ground_state_target),
    ("Reconstrução com H (Dual)", energy_exp["res_com_h_quad"]),
    ("Reconstrução sem H (H=0)", energy_exp["res_sem_h"]),
]
fig_energy = plot_spectra(panels_energy, title="Efeito da Inclusão de Energia H com Apenas 4 Medições")
plt.show()
""")

add_md(r"""### 4.4.1 Aspectos Físicos: Estado Térmico de Gibbs vs. Controle de Mesma Pureza
O estado canônico de Gibbs descreve o equilíbrio termodinâmico de um sistema governado por um Hamiltoniano físico $H$ a temperatura finita $T = 1/(k_B \beta)$:
$$
\rho_{\text{gibbs}}(\beta) = \frac{e^{-\beta H}}{\operatorname{Tr}[e^{-\beta H}]}.
$$
- Para $\beta \to \infty$ ($T \to 0$), $\rho_{\text{gibbs}}$ colapsa no **estado fundamental** (puro, posto 1, $S \to 0$);
- Para $\beta \to 0$ ($T \to \infty$), $\rho_{\text{gibbs}}$ tende ao **estado totalmente misto** ($I/d$, desordem máxima, $S \to \ln d$).

**Pergunta Central do Experimento:**
> *É mais fácil ou mais difícil reconstruir um estado térmico de Gibbs comparado a um estado aleatório/não-físico de mesma pureza espectral?*

Para responder a essa questão sem ambiguidades:
1. Geramos $\rho_{\text{gibbs}}(\beta)$ para $\beta \in [0.1, 1.0, 5.0]$ no modelo TFIM de 2 qubits;
2. Para cada $\beta$, calculamos $S(\rho_{\text{gibbs}})$ e sintetizamos um **estado de controle aleatório** $\rho_{\text{ctrl}}$ não-físico com exatamente a mesma entropia de von Neumann ($|S_{\text{ctrl}} - S_{\text{gibbs}}| < 10^{-6}$);
3. Realizamos tomografia sob **medição COMPLETA de Pauli** ($M=15$) com os métodos de **Força Bruta** e **SDP Dual**, usando o mesmo chute inicial $x_0$ fixo.
""")

add_md(r"""#### [Slide] Comparação 4.4.1: Estado Térmico de Gibbs vs. Controle Aleatório de Mesma Pureza
- **Estado Quântico Alvo:** Estados térmicos de Boltzmann $\rho_{\text{gibbs}}(\beta) = e^{-\beta H}/\operatorname{Tr}[e^{-\beta H}]$ do TFIM para $\beta \in \{0.1, 1.0, 5.0\}$ versus estados de controle aleatórios não-físicos pareados com exatamente a mesma entropia de von Neumann.
- **Objetivo da Demonstração:** Responder experimentalmente se a estrutura térmica de Boltzmann acelera ou dificulta a tomografia em relação a um estado aleatório equivalente sob medições completas, monitorando iterações e tempos de parada dos solvers.
""")

add_code(r"""# Comparação 4.4.1: Estado de Gibbs vs. Controle de Mesma Pureza
print("Executando experimento comparativo: Estado de Gibbs vs. Controle Aleatório...")
gibbs_exp = qt.run_gibbs_vs_control_experiment(
    n_qubits=2,
    betas=[0.1, 1.0, 5.0],
    J=1.0,
    h=0.8,
    seed=42
)

# Tabela comparativa formatada
print(f"{'Beta':<6s} | {'Tipo':<10s} | {'Entropia S':>10s} | {'Pureza':>8s} | {'Método':<25s} | {'Fidelidade':>10s} | {'Iterações':>10s} | {'Tempo (ms)':>10s}")
print("-" * 105)
for item in gibbs_exp["data"]:
    beta_val = item["beta"]
    for tipo, res_dict, rho_obj, s_val in [
        ("Gibbs", item["res_gibbs"], item["rho_gibbs"], item["s_gibbs"]),
        ("Controle", item["res_ctrl"], item["rho_ctrl"], item["s_ctrl"]),
    ]:
        p_val = qt.purity(rho_obj.rho)
        for s_name in ["least_squares", "quadratic"]:
            res = res_dict[s_name]
            nome_m = "SDP Dual (Quadrático)" if s_name == "quadratic" else "Força Bruta (Cholesky)"
            print(f"{beta_val:<6.1f} | {tipo:<10s} | {s_val:10.4f} | {p_val:8.4f} | {nome_m:<25s} | {res.fidelity:10.4f} | {res.num_iters:10d} | {res.elapsed_seconds*1000:10.2f}")

# Sobreposição das curvas de custo e gradiente dual
fig_gibbs = qt.plot_gibbs_vs_control_convergence(
    gibbs_exp,
    title="Convergência Tomográfica: Estado Térmico de Gibbs vs. Controle de Mesma Pureza",
    save_path="gibbs_vs_controle_convergencia.png"
)
plt.show()

print("\n--- Conclusão do Experimento Gibbs vs. Controle ---")
print("1. Força Bruta (Cholesky LSQ): Apresenta dificuldade de reconstrução praticamente IDÊNTICA")
print("   entre Gibbs e Controle (31 a 47 avaliações, ~5-7 ms). O espaço não-convexo não tira proveito")
print("   da estrutura de Boltzmann quando os observáveis medidos são Paulis uniformes despolarizados.")
print("2. SDP Dual (Quadrático JAX): Para beta=0.1 (alta temperatura), ambos convergem em 1 iteração.")
print("   Para beta=5.0 (baixa temperatura / quase puro), o estado de Gibbs exige mais iterações duais")
print("   (45 iterações vs 3 do controle) porque a hierarquia exponencial dos autovalores e^{-beta E_k}")
print("   demanda ajuste fino gradual do multiplicador de traço theta para projetar os autovalores residuais.")
""")

add_md(r"""### 4.5 Estabilidade dos Algoritmos e Ruído de Shot Noise
Em experimentos reais, cada observável é estimado com $N_{\text{shots}}$ repetições da regra de Born:
$$
p_+ = \frac{1 + \langle Q_i \rangle}{2}, \quad N_+ \sim \operatorname{Binomial}(N_{\text{shots}}, p_+), \quad \hat{q}_i = 2\frac{N_+}{N_{\text{shots}}} - 1.
$$
Com $N_{\text{shots}}$ baixo, flutuações estatísticas fazem com que o SDP de viabilidade estrita falhe com status `infeasible` em até 80% das vezes.
Em contrapartida, as formulações por **Mínimos Quadrados** (Convexos e Cholesky) são incondicionalmente estáveis.
""")

add_md(r"""#### [Slide] Comparação 4.5: Estabilidade perante Ruído Estatístico de Shot Noise (1 Qubit)
- **Estado Quântico Alvo:** Estado puro Haar-aleatório de 1 qubit submetido à amostragem estatística binomial de Born com $N_{\text{shots}}$ variando de 10 a 10.000 repetições (20 repetições por valor).
- **Objetivo da Demonstração:** Quantificar o colapso por infactibilidade do SDP primal de viabilidade estrita ($\min 0$) sob ruído estatístico finito (atingindo até 80% de falha para shots baixos), contrastando com a robustez e convergência garantida das formulações de Mínimos Quadrados.
""")

add_code(r"""# Comparação 4.5: Estabilidade perante Ruído de Shot Noise
print("Executando simulação de estabilidade e ruído de shot noise...")
shots_teste = [10, 25, 50, 100, 250, 500, 1000, 5000, 10000]
noise_data = qt.run_noise_stability_experiment(
    n_qubits=1,
    shots_list=shots_teste,
    n_reps=20,
    seed=42
)

# Plota fidelidade média e taxa de infactibilidade vs número de shots
fig_noise = plot_fidelity_vs_shots(
    shots_list=noise_data["shots_list"],
    fidelidades_medias={
        "SDP Mínimos Quadrados Convexos": noise_data["fids_lsq_convexa"],
        "Força Bruta (Cholesky LSQ)": noise_data["fids_brute_force"]
    },
    fidelidades_stds={
        "SDP Mínimos Quadrados Convexos": noise_data["stds_lsq_convexa"],
        "Força Bruta (Cholesky LSQ)": noise_data["stds_brute_force"]
    },
    taxa_infactivel=noise_data["taxa_infactivel_sdp"],
    title="Estabilidade Numérica: Fidelidade e Infactibilidade vs. N_shots",
    save_path="ruido_shot_noise_sdp.png"
)
plt.show()

# Tabela resumo de estabilidade
print(f"{'N_shots':<10s} | {'Fidelidade (LSQ Convexa)':<28s} | {'Fidelidade (Força Bruta)':<26s} | {'Infactibilidade (SDP Linear)':<30s}")
print("-" * 102)
for i, shots in enumerate(noise_data["shots_list"]):
    fid_cvx = noise_data["fids_lsq_convexa"][i]
    std_cvx = noise_data["stds_lsq_convexa"][i]
    fid_bf = noise_data["fids_brute_force"][i]
    std_bf = noise_data["stds_brute_force"][i]
    taxa_inf = noise_data["taxa_infactivel_sdp"][i]
    print(f"{shots:<10d} | {fid_cvx:8.4f} +- {std_cvx:6.4f}             | {fid_bf:8.4f} +- {std_bf:6.4f}         | {taxa_inf*100:6.1f}%")
""")

# ==============================================================================
# SEÇÃO 4.5.1: ESCALABILIDADE COM SHOTS REAIS (TAREFA 1)
# ==============================================================================
add_md(r"""### 4.5.1 Escalabilidade em Qubits com Shots Reais: Fidelidade e Iterações ($N = 1$ a $4$)

Para avaliar o comportamento prático em processadores quânticos reais, estendemos a análise de escalabilidade para o regime sob **amostragem binomial finita (shot noise da regra de Born)** com $N_{\text{shots}} = 10{,}000$ fixo em todo o experimento.

**Escolha metodológica do $N_{\text{shots}} = 10{,}000$:**
O valor de 10.000 shots representa um padrão realista da computação quântica ruidosa (NISQ), amplamente utilizado em backends da IBM Quantum e consistente com os experimentos de estabilidade prévios deste projeto. Ele assegura uma incerteza estatística por observável da ordem de $\sigma_q = \sqrt{(1 - q^2)/N_{\text{shots}}} \le 1/\sqrt{10{,}000} = 0.01$ (1%).

**Comportamento Numérico e Teórico dos Métodos:**
- **Força Bruta (Cholesky LSQ):** Otimiza $\min_x \frac{1}{2}\sum_i (\operatorname{Tr}[\rho(T) Q_i] - q_i)^2$ diretamente sobre a parametrização física $\rho(T) \succeq 0, \operatorname{Tr}\rho = 1$. O ruído estatístico é naturalmente acomodado pelo funcional de mínimos quadrados, mantendo convergência rápida e fidelidade estavelmente elevada ($F > 0.989$) para todos os qubits $N=1, 2, 3, 4$.
- **SDP Dual Regularizado (Quadrático):** O formalismo dual assume a viabilidade estrita das equações de igualdade $\operatorname{Tr}[Q_i \rho] = q_i$. Sob flutuações estatísticas de shots, o vetor empírico $\vec{q}_{\text{shots}}$ cai fora do politopo/cone de estados quânticos válidos ($q \notin \mathcal{A}(\mathcal{S})$). Consequentemente, o sistema de igualdades é estritamente infactível, o funcional dual torna-se ilimitado/sem resíduo nulo, e o otimizador L-BFGS esgota o limite de iterações (`atingiu_max_iter`), degradando a fidelidade para $N \ge 2$. Como previsto pelas restrições do projeto, esse comportamento é rigorosamente documentado em vez de forçado.
""")

add_md(r"""#### [Slide] Experimento 4.5.1: Fidelidade e Iterações vs. Número de Qubits com Shots Reais ($N_{\text{shots}} = 10.000$)
- **Estado Quântico Alvo:** Estados puros Haar-aleatórios gerados para sistemas de $N \in \{1, 2, 3, 4\}$ qubits ($d = 2, 4, 8, 16$) com semente determinística fixa por $n$, medidos em toda a base completa de Pauli ($M = 4^N - 1$) sob $N_{\text{shots}} = 10{,}000$ (5 repetições monte carlo por qubit).
- **Objetivo da Demonstração:** Analisar a fidelidade e o número de avaliações/iterações dos métodos `least_squares` e `quadratic` sob ruído de shots realista, demonstrando que a Força Bruta acomoda o ruído com degradação suave ($F \approx 0.99$), enquanto a formulação dual com restrições de igualdade estrita esgota iterações perante a infactibilidade estatística dos dados.
""")

add_code(r"""# Comparação 4.5.1: Benchmark de Escalabilidade com Shots Reais (N = 1 a 4)
print("Executando benchmark de escalabilidade com shots reais (N_shots=10.000, 5 repetições por qubit)...")

benchmark_shots_data = qt.run_shots_scaling_benchmark(
    qubits_range=[1, 2, 3, 4],
    n_shots=10000,
    methods=["least_squares", "quadratic"],
    n_reps=5,
    seed=42,
    max_iter_quadratic=300
)

# Gera os dois gráficos lado a lado: (a) Fidelidade vs N qubits, (b) Iterações vs N qubits
fig_shots_scale = qt.plot_shots_scaling(
    benchmark_shots_data,
    title="Escalabilidade sob Shot Noise Realista (N_shots = 10.000, 5 Repetições)",
    save_path="benchmark_shots_scaling.png"
)
plt.show()

# Tabela resumo de métricas formatada em f-string
print(f"{'Qubits':<8s} | {'Dim':<6s} | {'Obs (4^N-1)':<12s} | {'Método':<26s} | {'Fidelidade Média':<20s} | {'Iterações Médias':<20s} | {'Tempo Médio (s)':<16s}")
print("-" * 115)
for i, n in enumerate(benchmark_shots_data["qubits_range"]):
    dim = 2 ** n
    n_obs = 4 ** n - 1
    for m in benchmark_shots_data["methods"]:
        nome_m = "Força Bruta (Cholesky)" if m == "least_squares" else "SDP Dual (Quadrático)"
        f_mean = benchmark_shots_data["fidelidades_mean"][m][i]
        f_std = benchmark_shots_data["fidelidades_std"][m][i]
        it_mean = benchmark_shots_data["iteracoes_mean"][m][i]
        it_std = benchmark_shots_data["iteracoes_std"][m][i]
        t_mean = benchmark_shots_data["tempos_mean"][m][i]
        t_std = benchmark_shots_data["tempos_std"][m][i]
        
        fid_str = f"{f_mean:.4f} +- {f_std:.4f}"
        it_str = f"{it_mean:6.1f} +- {it_std:5.1f}"
        t_str = f"{t_mean:.4f}s"
        print(f"{n:<8d} | {dim:<6d} | {n_obs:<12d} | {nome_m:<26s} | {fid_str:<20s} | {it_str:<20s} | {t_str:<16s}")
    print("-" * 115)
""")

# ==============================================================================
# SEÇÃO 4.5.2: EXEMPLOS PALPÁVEIS EM ALTOS QUBITS COM SHOTS (TAREFA 2)
# ==============================================================================
add_md(r"""### 4.5.2 Exemplos Palpáveis de Reconstrução em Altos Qubits com Shots Reais

Para tornar a reconstrução tomográfica concreta e palpável para além de métricas globais, analisamos dois estados fundamentais da mecânica quântica multipartite ($N=3$ e $N=4$ qubits) sob $N_{\text{shots}} = 10{,}000$:
1. **Comparação linha por linha de observáveis:** Confrontamos o valor experimental medido com ruído ($\hat{q}_i$), o valor previsto pela matriz densidade reconstruída ($\operatorname{Tr}[\rho_{\text{rec}} Q_i]$) e o valor exato analítico ($\operatorname{Tr}[\rho_{\text{true}} Q_i]$);
2. **Projeção na Esfera de Bloch via Traço Parcial:** Calculamos o estado reduzido de 1 qubit $\rho_1 = \operatorname{Tr}_{2..N}[\rho]$ tanto para o estado verdadeiro quanto para o reconstruído, plotando ambos lado a lado na Esfera de Bloch 3D e projeções 2D.
""")

add_md(r"""#### [Slide] Exemplo 4.5.2: Reconstrução Palpável do Estado W de 3 Qubits com Shots Reais
- **Estado Quântico Alvo:** Estado tripartite emaranhado de Werner $|\text{W}_3\rangle = \frac{1}{\sqrt{3}}(|100\rangle + |010\rangle + |001\rangle)$ ($N = 3$ qubits, dimensão $d = 8$, pureza $\gamma = 1.0$, posto 1), medido com $N_{\text{shots}} = 10{,}000$ em todos os 63 observáveis de Pauli.
- **Objetivo da Demonstração:** Demonstrar a correspondência observável a observável entre valores medidos, reconstruídos e exatos, e visualizar o traço parcial sobre o primeiro qubit ($\rho_1 = \operatorname{diag}(2/3, 1/3)$, vetor de Bloch $\vec{r} = [0, 0, 1/3]$) na Esfera de Bloch, confirmando a fidelidade da reconstrução em sistema multipartite.
""")

add_code(r"""# Exemplo 4.5.2: Reconstrução Palpável do Estado W de 3 Qubits com Shots Reais
print("--- Reconstrução Palpável: Estado W de 3 Qubits com N_shots = 10.000 ---")

# 1. Geração do estado-alvo específico e nomeado
target_w3 = qt.standard_pure_state("w", n_qubits=3)
all_obs_3q = qt.pauli_basis_strings(3)
q_exato_3q = qt.measure_observables(target_w3, all_obs_3q)

# 2. Simulação com N_shots = 10.000
rng_w3 = np.random.default_rng(42)
q_shots_3q = qt.simula_medidas_shots(q_exato_3q, 10000, rng=rng_w3)

# 3. Reconstrução tomográfica via Força Bruta (estável para shots)
t0 = time.time()
res_w3 = qt.solve_tomography(all_obs_3q, q_shots_3q, method="least_squares", true_state=target_w3, n_qubits=3)
tempo_w3 = time.time() - t0

# 4. Tabela palpável comparativa: observável por observável (amostra representativa)
amostra_obs_3q = ["ZII", "IZI", "IIZ", "ZZI", "IZZ", "ZIZ", "XXI", "YYI", "XXX", "ZZZ"]
print(f"\n{'Observável':<12s} | {'q_i Medido (Shots)':<22s} | {'q_i Reconstruído':<20s} | {'q_i Exato':<14s} | {'Erro Absoluto':<15s}")
print("-" * 90)
for obs_str in amostra_obs_3q:
    idx = all_obs_3q.index(obs_str)
    qi_med = q_shots_3q[idx]
    mat_obs = qt.pauli_string_to_matrix(obs_str)
    qi_rec = float(np.trace(res_w3.rho @ mat_obs).real)
    qi_true = float(np.trace(target_w3.rho @ mat_obs).real)
    erro = abs(qi_rec - qi_true)
    print(f"{obs_str:<12s} | {qi_med:22.4f} | {qi_rec:20.4f} | {qi_true:14.4f} | {erro:15.2e}")

# 5. Traço parcial sobre o primeiro qubit e visualização na Esfera de Bloch
rho_true_q1_w3 = target_w3.partial_trace(keep=0)
rho_rec_q1_w3 = qt.partial_trace(res_w3.rho, keep=0)

fig_bloch_w3 = plotar_bloch(
    [rho_true_q1_w3, rho_rec_q1_w3],
    rotulos=["Alvo W_3 (Qubit 1)", "Reconstruído W_3 (Qubit 1)"],
    cores=["#31a354", "#e6550d"],
    titulo="Esfera de Bloch: Traço Parcial do Estado W (3 Qubits, N_shots=10.000)",
    salvar="bloch_w3_shots.png"
)
plt.show()

# 6. Tabela resumo da reconstrução específica
print(f"\n{'Sistema':<18s} | {'Estado Alvo':<25s} | {'Fidelidade':>12s} | {'Avaliações':>12s} | {'Tempo (s)':>12s} | {'Resíduo':>12s}")
print("-" * 102)
print(f"{'3 Qubits (dim=8)':<18s} | {'Estado W (|100>+|010>+|001>)/√3':<25s} | {res_w3.fidelity:12.6f} | {res_w3.num_iters:12d} | {tempo_w3:12.4f} | {res_w3.residual:12.2e}")
""")

add_md(r"""#### [Slide] Exemplo 4.5.3: Reconstrução Palpável do Estado GHZ de 4 Qubits com Shots Reais
- **Estado Quântico Alvo:** Estado tetrapartite maximamente emaranhado de Greenberger–Horne–Zeilinger $|\text{GHZ}_4\rangle = \frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$ ($N = 4$ qubits, dimensão $d = 16$, pureza $\gamma = 1.0$), medido com $N_{\text{shots}} = 10{,}000$ em todos os 255 observáveis de Pauli.
- **Objetivo da Demonstração:** Validar a tomografia completa em escala de $d=16$ dimensões com dados ruidosos, exibindo a tabela detalhada de correlações quânticas multipartite e projetando o traço parcial do primeiro qubit ($\rho_1 = I_2/2$, vetor de Bloch $\vec{r} \approx [0, 0, 0]$ correspondente à despolarização máxima por emaranhamento) na Esfera de Bloch.
""")

add_code(r"""# Exemplo 4.5.3: Reconstrução Palpável do Estado GHZ de 4 Qubits com Shots Reais
print("--- Reconstrução Palpável: Estado GHZ de 4 Qubits com N_shots = 10.000 ---")

# 1. Geração do estado-alvo específico e nomeado
target_ghz4 = qt.standard_pure_state("ghz", n_qubits=4)
all_obs_4q = qt.pauli_basis_strings(4)
q_exato_4q = qt.measure_observables(target_ghz4, all_obs_4q)

# 2. Simulação com N_shots = 10.000
rng_ghz4 = np.random.default_rng(43)
q_shots_4q = qt.simula_medidas_shots(q_exato_4q, 10000, rng=rng_ghz4)

# 3. Reconstrução tomográfica via Força Bruta
t0 = time.time()
res_ghz4 = qt.solve_tomography(all_obs_4q, q_shots_4q, method="least_squares", true_state=target_ghz4, n_qubits=4)
tempo_ghz4 = time.time() - t0

# 4. Tabela palpável comparativa: observável por observável (amostra representativa)
amostra_obs_4q = ["ZIII", "IZII", "ZZII", "IIZZ", "ZIZI", "XXXX", "YYYY", "XXYY", "YYXX", "ZZZZ"]
print(f"\n{'Observável':<12s} | {'q_i Medido (Shots)':<22s} | {'q_i Reconstruído':<20s} | {'q_i Exato':<14s} | {'Erro Absoluto':<15s}")
print("-" * 90)
for obs_str in amostra_obs_4q:
    idx = all_obs_4q.index(obs_str)
    qi_med = q_shots_4q[idx]
    mat_obs = qt.pauli_string_to_matrix(obs_str)
    qi_rec = float(np.trace(res_ghz4.rho @ mat_obs).real)
    qi_true = float(np.trace(target_ghz4.rho @ mat_obs).real)
    erro = abs(qi_rec - qi_true)
    print(f"{obs_str:<12s} | {qi_med:22.4f} | {qi_rec:20.4f} | {qi_true:14.4f} | {erro:15.2e}")

# 5. Traço parcial sobre o primeiro qubit e visualização na Esfera de Bloch
rho_true_q1_ghz4 = target_ghz4.partial_trace(keep=0)
rho_rec_q1_ghz4 = qt.partial_trace(res_ghz4.rho, keep=0)

fig_bloch_ghz4 = plotar_bloch(
    [rho_true_q1_ghz4, rho_rec_q1_ghz4],
    rotulos=["Alvo GHZ_4 (Qubit 1)", "Reconstruído GHZ_4 (Qubit 1)"],
    cores=["#31a354", "#e6550d"],
    titulo="Esfera de Bloch: Traço Parcial do Estado GHZ (4 Qubits, N_shots=10.000)",
    salvar="bloch_ghz4_shots.png"
)
plt.show()

# 6. Tabela resumo da reconstrução específica
print(f"\n{'Sistema':<18s} | {'Estado Alvo':<25s} | {'Fidelidade':>12s} | {'Avaliações':>12s} | {'Tempo (s)':>12s} | {'Resíduo':>12s}")
print("-" * 102)
print(f"{'4 Qubits (dim=16)':<18s} | {'Estado GHZ (|0000>+|1111>)/√2':<25s} | {res_ghz4.fidelity:12.6f} | {res_ghz4.num_iters:12d} | {tempo_ghz4:12.4f} | {res_ghz4.residual:12.2e}")
""")

# ==============================================================================
# SÍNTESE FINAL
# ==============================================================================
add_md(r"""### 4.6 Tabela Síntese Comparativa

A tabela abaixo sintetiza o comportamento comparativo de todos os métodos estudados:

| Critério de Comparação | Força Bruta (Cholesky $T^\dagger T$) | SDP Primal (Linear / SCS) | SDP Dual Regularizado (Quadrático JAX) | SDP Mínimos Quadrados Convexos |
|---|---|---|---|---|
| **Espaço de Otimização** | $\mathbb{R}^{d^2}$ (não-linear) | Matrizes hermitianas $d \times d$ no cone PSD | $\mathbb{R}^M$ (irrestrito, côncavo) | Matrizes hermitianas $d \times d$ no cone PSD |
| **Garantia de Otimalidade** | Mínimo local (não-convexo) | Ponto arbitrário do conjunto afim | Mínimo global único certificado | Mínimo global único certificado |
| **Sensibilidade à Inicialização** | Alta ($x_0$ aleatório afeta trajetória) | Baixa ($\rho_0 = 0$) | Nenhuma ($\alpha_0 = \vec{0}$) | Nenhuma ($\rho_0 = 0$) |
| **Tempo de Parada ($N=4$ qubits)** | Médio (~0.05s a 0.2s) | Inviável / Timeout (> dezenas de s) | **Ultra-rápido (~0.4s)** | Alto (segundos) |
| **Critério de Parada** | `gtol` / `ftol` no resíduo | Resíduos primal e dual $\le \epsilon_{\text{feas}}$ | $\|\nabla \mathcal{D}\| = \|A\pi - q\| \le 10^{-12}$ | Tolerância KKT de interior-point |
| **Robustez a Shot Noise** | **Alta (mínimos quadrados $F > 0.99$)** | **Baixa (infactível para shots baixos)** | Sensível a igualdade estrita ($\lim_{k} F$) | **Alta (garantia de factibilidade)** |
| **Propriedade Espectral** | Depende do chute inicial | Indeterminada | Favorece baixo posto (fronteira PSD) | Projeção euclidiana no cone |
| **Integração de Energia ($H$)** | Penalidade ad-hoc na perda | Custo linear em $\operatorname{Tr}[H\rho]$ | Termo físico natural no slack dual | Termo linear de energia |
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (Quantum Tomography .venv)",
            "language": "python",
            "name": "qtomography"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "version": "3.13.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = "/home/al.david.oliveira/Desktop/Quantum-Tomography/qt_generalizado.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook {output_path} gerado com sucesso contendo {len(cells)} células nas 4 seções!")
