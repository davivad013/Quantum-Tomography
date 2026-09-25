# Quantum State Tomography (QST)

Repositório desenvolvido para o projeto de **Tomografia de Estados Quânticos** na disciplina de *Aspectos Matemáticos e Computacionais da Mecânica Quântica*.

O projeto implementa, compara e estende as principais abordagens modernas para reconstrução de matrizes densidade quânticas ($\rho \succeq 0, \operatorname{Tr}\rho = 1$):
1. **Programação Semidefinida Regularizada (SDP)** utilizando a biblioteca `sdplab` e `spacecore` com aceleração compilada em JAX e Optax (L-BFGS).
2. **Programação Semidefinida Linear e Mínimos Quadrados Convexos** via `CVXPY` (solvers CLARABEL e SCS).
3. **Mínimos Quadrados Não-Lineares de Força Bruta** com parametrização de Cholesky $\rho(T) = \frac{T^\dagger T}{\operatorname{Tr}(T^\dagger T)}$ e Jacobiano Analítico exato via `scipy.optimize.least_squares`.

---

## Estrutura do Repositório

```
Quantum-Tomography/
├── quantum_tomography/             # Pacote Python modular e extensível
│   ├── states.py                   # Classes de estado (QuantState), geradores (puros, misturas, térmicos) e métricas (fidelidade, entropia, pureza)
│   ├── observables.py              # Álgebra de Pauli, geração de bases completas/incompletas e simulação de Born/Shot Noise
│   ├── hamiltonians.py             # Modelos físicos de energia (Ising TFIM, Heisenberg XXZ, Zeeman)
│   ├── visualization.py            # Visualização na Esfera de Bloch (3D + 2D), espectros de autovalores e benchmarks
│   ├── experiments.py              # Suíte de experimentos reprodutíveis em 1 linha
│   └── solvers/
│       ├── base.py                 # TomographyResult (dataclass unificada com compatibilidade dict)
│       ├── sdp_solver.py           # Regularização Entrópica (MaxEnt), Quadrática (L2) e CVXPY
│       └── brute_force.py          # Parametrização T†T com Jacobiano analítico fechado
├── qt_generalizado.ipynb           # Notebook Mestre interativo com todos os experimentos executados
├── tomography.ipynb                # Notebook de referência dos autores do sdplab
├── tomografia_pipeline.py          # Módulo de compatibilidade com pipelines legados
├── bloch_reconstrucao.py           # Módulo de compatibilidade para plot da esfera de Bloch
├── benchmark_tempo_sdp.png         # Gráfico de escalabilidade em qubits (tempo)
├── benchmark_shots_scaling.png     # Gráfico de escalabilidade em qubits com shots reais (Tarefa 1)
├── bloch_reconstrucoes.png         # Gráfico da não-unicidade na esfera de Bloch
├── bloch_w3_shots.png              # Gráfico do traço parcial na esfera de Bloch para estado W de 3 qubits (Tarefa 2)
├── bloch_ghz4_shots.png            # Gráfico do traço parcial na esfera de Bloch para estado GHZ de 4 qubits (Tarefa 2)
├── ruido_shot_noise_sdp.png        # Gráfico de fidelidade e infactibilidade vs. shot noise
└── objetivos                       # Lista de metas do projeto (todas concluídas)
```

---

## Principais Resultados e Tópicos Desenvolvidos

### 1. Reconstrução e Sanity Check
- Todos os solvers recuperam estados puros exatos com fidelidade $F = 1.000000$ e resíduos da ordem de $10^{-14}$ a $10^{-16}$.

### 2. O Problema da Não-Unicidade com Medições Incompletas ($M < 4^N - 1$)
- Quando observáveis são omitidos (ex: medindo apenas $\sigma_x$ e $\sigma_y$), infinitos estados físicos reproduzem as medidas.
- A **Regularização Entrópica (MaxEnt)** seleciona o estado com $r_z = 0$, que maximiza a entropia de von Neumann no interior da esfera de Bloch.
- A **Força Bruta** cai em diferentes pontos da reta permitida dependendo do chute inicial, evidenciando a multiplicidade de soluções na ausência de regularizador estrito.

### 3. Escalabilidade em Múltiplos Qubits ($N = 1, 2, 3, 4, 5$)
- O número de parâmetros reais cresce como $4^N - 1$.
- Enquanto solvers convencionais como CVXPY SCS e métodos de parametrização direta começam a falhar ou demorar dezenas de segundos para $N \ge 4$, a formulação dual do `sdplab` com otimizador L-BFGS em JAX resolve instâncias de $N=4$ qubits ($d=16$) em frações de segundo (~0.5s), convergindo em poucas dezenas de iterações.

### 4. Espectro de Pureza (Puros vs. Mistura de Puros vs. Muito Mistos)
- **Regularização Entrópica:** o estado recuperado é da forma de Gibbs $\pi^* \propto \exp\left(\frac{1}{\varepsilon}\sum \alpha_i Q_i\right)$, residindo estritamente no **interior** do cone semidefinido positivo (todos os autovalores são estritamente positivos).
- **Regularização Quadrática ($L^2$ / Frobenius):** o estado recuperado é uma projeção positiva $\pi^* = \left(\frac{1}{\varepsilon}\sum \alpha_i Q_i - \theta\right)_+$, truncando autovalores negativos em zero. Reside na **fronteira** do cone, favorecendo estados de baixo posto (ideal para estados puros ou com baixo ruído).

### 5. Introdução de Energia ao Problema (Hamiltoniano Físico $H$)
- Permite acoplar um Hamiltoniano de sistema (ex: Modelo de Ising de Campo Transversal - TFIM):
  $$\min_\pi \operatorname{Tr}[H \pi] + \varepsilon \operatorname{Tr}[\varphi(\pi)] \quad \text{s.t.} \quad \operatorname{Tr}[Q_i \pi] = q_i.$$
- **Tomografia Guiada por Energia com Poucas Medições:** Quando apenas uma pequena fração dos observáveis é conhecida (ex: 4 de 15 observáveis para 2 qubits), a tomografia padrão falha ($F \approx 0.32$). A introdução do Hamiltoniano atua como um prior físico, recuperando o estado fundamental e de baixa energia com fidelidade $F > 0.99$!

### 6. Estabilidade dos Algoritmos e Shot Noise
- Sob ruído binomial da regra de Born ($N_{\text{shots}}$ finito), as flutuações estatísticas podem violar as desigualdades matriciais de positividade.
- O problema de viabilidade linear SDP ($\min 0$) torna-se **infactível** em até 80% das realizações para $N_{\text{shots}} \le 100$.
- Em contraste, a formulação por **Mínimos Quadrados Convexos** e a parametrização de **Cholesky** são incondicionalmente estáveis e convergentes, alcançando $F \to 1$ conforme $N_{\text{shots}} \to \infty$.

### 7. Escalabilidade com Shots Reais ($N_{\text{shots}} = 10{,}000$)
- Avaliação para $N \in \{1, 2, 3, 4\}$ qubits sob amostragem binomial realista com $N_{\text{shots}} = 10{,}000$ (5 repetições monte carlo por qubit).
- A **Força Bruta (Cholesky)** acomoda as flutuações amostrais no funcional de mínimos quadrados, mantendo fidelidade média $F \ge 0.99$ em todas as dimensões testadas.
- O **SDP Dual Regularizado**, operando com restrições de igualdade estrita em dados empíricos fora do politopo quântico, esgota as iterações do L-BFGS demonstrando a sensibilidade cônica esperada.

### 8. Exemplos Palpáveis de Reconstrução em Altos Qubits
- Reconstruções completas dos estados fundamentais $|\text{W}_3\rangle$ (3 qubits, 63 observáveis) e $|\text{GHZ}_4\rangle$ (4 qubits, 255 observáveis) sob $N_{\text{shots}} = 10{,}000$.
- Tabelas comparativas linha por linha ($q_i$ medido vs. reconstruído vs. exato) confirmando precisão dentro de $1\% \sim 1/\sqrt{N_{\text{shots}}}$.
- Projeção do traço parcial do primeiro qubit na **Esfera de Bloch**, fornecendo confirmação geométrica direta mesmo em sistemas de dimensão 8 e 16.

---

## Como Usar o Toolkit em Código Python

```python
import quantum_tomography as qt

# 1. Definir ou gerar um estado quântico alvo
estado = qt.random_pure_state(dim=4, seed=42)  # 2 qubits
observaveis = qt.pauli_basis_strings(n_qubits=2)

# 2. Medir valores esperados (exatos ou com ruído)
q = qt.measure_observables(estado, observaveis, N_shots=50000)

# 3. Resolver com qualquer um dos métodos disponíveis
res_ent  = qt.solve_tomography(observaveis, q, method="entropy", true_state=estado)
res_quad = qt.solve_tomography(observaveis, q, method="quadratic", true_state=estado)
res_lsq  = qt.solve_tomography(observaveis, q, method="least_squares", true_state=estado)
res_cvx  = qt.solve_tomography(observaveis, q, method="cvxpy_lsq", true_state=estado)

# 4. Imprimir relatório detalhado
print(res_ent.summary())
```
