# Backup de Dados e Resultados - apresentacao.ipynb

Este diretório contém todos os dados, tabelas, gráficos gerados e saídas de texto extraídos diretamente do notebook `apresentacao.ipynb`.

> **Objetivo:** Garantir a preservação total de todos os dados numéricos e plots gerados (que demandaram horas de processamento computacional, como os cálculos de escalabilidade de Gibbs e as 138 execuções de tomografia quântica) antes de qualquer re-execução do notebook via *Run All*.

---

## Estrutura do Diretório

- `apresentacao_backup.ipynb`: Cópia integral do notebook original com outputs e metadados intactos.
- `all_data_consolidated.json`: Dados consolidados em formato JSON para fácil carregamento programático.
- `figures/`: Todas as figuras e gráficos em alta resolução (formato PNG).
  - `cell_07_gd_vs_sgd_convergence.png`: Comparação da convergência entre Gradient Descent completo e SGD.
  - `cell_10_sgd_tuning_learning_rate.png`: Ajuste de taxa de aprendizado (lr = 0.01 a 0.75, eps = 1.0).
  - `cell_10_sgd_tuning_epsilon.png`: Ajuste de epsilon de regularização (eps = 0.75 a 2.0, lr = 0.5).
  - `cell_12_sgd_convergence_by_state_type.png`: Convergência por tipo de estado (Puro, Mistura, Gibbs, Totalmente Misto).
  - `cell_16_gibbs_scaling_n3_to_n8_convergence.png`: Curvas de convergência do SGD de memória reduzida (n = 3 a 8 qubits).
  - `cell_17_gibbs_scaling_iterations_comparison.png`: Comparação de iterações e escalabilidade para n = 3 a 8 qubits.
  - `cell_20_trace_distance_vs_num_observables_5panels.png`: Painéis com n = 1 a 5 qubits mostrando distância de traço vs número de observáveis medidos (m).
- `tables/`: Tabelas numéricas extraídas em formato CSV.
  - `cell_12_state_types_comparison.csv`: Comparação dos 4 tipos de estados quânticos testados.
  - `cell_14_gibbs_scaling_maxiters_100k.csv`: Benchmark de escalabilidade do estado de Gibbs (n=3..7 qubits, 100k iterações, até ~56 minutos de execução para n=7).
  - `cell_17_gibbs_scaling_maxiters_10k.csv`: Benchmark de escalabilidade do estado de Gibbs (n=3..8 qubits, 10k iterações, até 64 GB de matrizes economizadas).
  - `cell_19_observables_scaling_full_dataset.csv`: Conjunto completo de dados com todas as 138 execuções experimentais de tomografia (n, m, seed, iters, gradiente, distância de traço, fidelidade).
- `logs_and_text/`: Saídas textuais completas de cada célula do notebook.
  - `cell_03_sanity_check_ga_vs_sga.txt`: Verificação analítica dos métodos GA e SGA.
  - `cell_05_target_state_3qubits.txt`: Propriedades do estado-alvo de 3 qubits (posto, autovalores, traço).
  - `cell_06_gd_vs_sgd_metrics.txt`: Tempos de execução e fidelidade comparativa entre GD e SGD.
  - `cell_12_sgd_by_state_type_output.txt`: Saída impressa da execução dos tipos de estado.
  - `cell_14_gibbs_scaling_100k_output.txt`: Saída impressa do benchmark de 100k iterações.
  - `cell_15_interruption_status.txt`: Ponto de interrupção com contagem de iterações.
  - `cell_17_gibbs_scaling_10k_output.txt`: Saída impressa do benchmark de 10k iterações.
  - `cell_19_observables_scaling_raw.txt`: Registro textual integral das 139 linhas de resultados de observáveis.

---

## Como Carregar os Dados em Python

```python
import pandas as pd
import json

# Carregar qualquer uma das tabelas via pandas
df_observaveis = pd.read_csv('apresentacao_saved_outputs/tables/cell_19_observables_scaling_full_dataset.csv')
df_estados = pd.read_csv('apresentacao_saved_outputs/tables/cell_12_state_types_comparison.csv')
df_gibbs_100k = pd.read_csv('apresentacao_saved_outputs/tables/cell_14_gibbs_scaling_maxiters_100k.csv')
df_gibbs_10k = pd.read_csv('apresentacao_saved_outputs/tables/cell_17_gibbs_scaling_maxiters_10k.csv')

# Ou carregar todos os dados de uma só vez pelo arquivo consolidado
with open('apresentacao_saved_outputs/all_data_consolidated.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)
```
