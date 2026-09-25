"""
quantum_tomography.visualization
================================
Visualizações gráficas para tomografia quântica:
- Esfera de Bloch 3D e projeções 2D (xy, xz, yz)
- Espectros de autovalores (fronteira vs. interior do cone PSD)
- Benchmarks de escalabilidade com N qubits
- Fidelidade vs. Shot noise
- Análise de reconstrução de energia
"""

from __future__ import annotations
from typing import Sequence, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

from .states import QuantState, _symmetrize_hermitian
from .solvers.base import TomographyResult

# Operadores de Pauli para vetor de Bloch
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def vetor_bloch(rho: np.ndarray | QuantState | TomographyResult) -> np.ndarray:
    """Extrai o vetor de Bloch (rx, ry, rz) = Tr(rho * sigma_i) para 1 qubit."""
    if isinstance(rho, TomographyResult):
        mat = rho.rho
    elif isinstance(rho, QuantState):
        mat = rho.rho
    else:
        mat = np.asarray(rho, dtype=complex)
    return np.array([float(np.trace(mat @ s).real) for s in (sigma_x, sigma_y, sigma_z)])


# ---------------------------------------------------------------------------
# Visualização na Esfera de Bloch (3D + 2D)
# ---------------------------------------------------------------------------

def desenhar_esfera(ax):
    """Desenha a esfera de Bloch unitária em 3D com linhas de grade e eixos."""
    u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="#9ecae1", alpha=0.10, linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color="#6baed6", alpha=0.10, linewidth=0.4, rstride=4, cstride=4)

    # Círculos máximos: equador (xy) e meridianos (xz, yz)
    t = np.linspace(0, 2 * np.pi, 300)
    c, s, o = np.cos(t), np.sin(t), np.zeros_like(t)
    ax.plot(c, s, o, color="gray", alpha=0.5, lw=1)
    ax.plot(c, o, s, color="gray", alpha=0.5, lw=1)
    ax.plot(o, c, s, color="gray", alpha=0.5, lw=1)

    L = 1.25
    for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        d_arr = np.array(d) * L
        ax.plot([-d_arr[0], d_arr[0]], [-d_arr[1], d_arr[1]], [-d_arr[2], d_arr[2]], color="k", alpha=0.6, lw=0.9)

    k = 1.42
    ax.text(0, 0, k, r"$|0\rangle$", ha="center", fontsize=11)
    ax.text(0, 0, -k, r"$|1\rangle$", ha="center", fontsize=11)
    ax.text(k, 0, 0, r"$|+\rangle$", ha="center", fontsize=11)
    ax.text(-k, 0, 0, r"$|-\rangle$", ha="center", fontsize=11)
    ax.text(0, k, 0, r"$|{+i}\rangle$", ha="center", fontsize=11)
    ax.text(0, -k, 0, r"$|{-i}\rangle$", ha="center", fontsize=11)


def desenhar_plano(ax, vetores, cores, rotulos, marcadores, chave, medido=False):
    """Desenha a projeção em um plano 2D."""
    planos = {
        "xz": (0, 2, "Plano x–z", (r"$|+\rangle$", r"$|-\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
        "xy": (0, 1, "Plano x–y", (r"$|+\rangle$", r"$|-\rangle$", r"$|{+i}\rangle$", r"$|{-i}\rangle$")),
        "yz": (1, 2, "Plano y–z", (r"$|{+i}\rangle$", r"$|{-i}\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
    }
    ih, iv, nome, (dir_, esq, cima, baixo) = planos[chave]

    ax.add_patch(Circle((0, 0), 1, fc="#9ecae1", ec="gray", alpha=0.15, lw=0))
    ax.add_patch(Circle((0, 0), 1, fc="none", ec="gray", lw=1.2))
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)

    for r, cor, rot, m in zip(vetores, cores, rotulos, marcadores):
        h, v = r[ih], r[iv]
        ax.annotate("", xy=(h, v), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=cor, lw=2.2))
        ax.scatter(h, v, color=cor, s=100 if m == "o" else 65, marker=m, edgecolor="k", zorder=10)

    if len(vetores) == 2:
        a, b = vetores
        ax.plot([a[ih], b[ih]], [a[iv], b[iv]], color="k", ls=":", lw=1.3)

    k = 1.18
    ax.text(k, 0, dir_, ha="left", va="center", fontsize=11)
    ax.text(-k, 0, esq, ha="right", va="center", fontsize=11)
    ax.text(0, k, cima, ha="center", va="bottom", fontsize=11)
    ax.text(0, -k, baixo, ha="center", va="top", fontsize=11)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    titulo = nome + ("  (observáveis medidos)" if medido else "")
    ax.set_title(titulo, fontsize=11, fontweight="bold" if medido else "normal")


def plotar_bloch(
    rhos: Sequence[Any],
    rotulos: Sequence[str] | None = None,
    cores: Sequence[str] | None = None,
    titulo: str = "Esfera de Bloch–Poincaré (Reconstrução Tomográfica)",
    elev: float = 20,
    azim: float = 35,
    medidos: Sequence[str] = ("x", "y"),
    salvar: str | None = None,
) -> plt.Figure:
    """
    Renderiza a visualização canônica da esfera de Bloch (3D + projeções xz, xy, yz).
    """
    rotulos = list(rotulos) if rotulos else [f"ρ{i + 1}" for i in range(len(rhos))]
    cores = list(cores) if cores else ["#e6550d", "#31a354", "#756bb1", "#de2d26", "#3182bd"]
    marcadores = ["o", "D", "s", "^", "v"]
    vetores = [vetor_bloch(rho) for rho in rhos]

    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08, left=0.04, right=0.96, top=0.92, bottom=0.12)

    # 3D
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    desenhar_esfera(ax3d)
    for r, cor, rot, m in zip(vetores, cores, rotulos, marcadores):
        ax3d.quiver(0, 0, 0, *r, color=cor, lw=2.4, arrow_length_ratio=0.10)
        ax3d.scatter(*r, color=cor, s=90, marker=m, edgecolor="k", zorder=10, label=rot)
    if len(vetores) == 2:
        ax3d.plot(*zip(*vetores), color="k", ls=":", lw=1.5)

    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-1, 1)
    ax3d.set_axis_off()
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("Esfera de Bloch (3D)", fontsize=12)
    ax3d.legend(loc="upper left", fontsize=10)

    # Projeções 2D
    conj_medidos = {m.lower() for m in medidos}
    for chave, pos in (("xz", (0, 1)), ("xy", (1, 0)), ("yz", (1, 1))):
        ax = fig.add_subplot(gs[pos])
        desenhar_plano(ax, vetores, cores, rotulos, marcadores, chave, medido=(set(chave) == conj_medidos))

    # Caixa informativa
    linhas = [f"{rot}: r = ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})  |r| = {np.linalg.norm(r):.3f}" for rot, r in zip(rotulos, vetores)]
    if len(vetores) == 2:
        linhas.append(f"Distância entre eles: {np.linalg.norm(vetores[0] - vetores[1]):.4f}")
    fig.text(0.5, 0.02, "\n".join(linhas), ha="center", va="bottom", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    fig.suptitle(titulo, fontsize=15, y=0.97)
    if salvar:
        fig.savefig(salvar, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Espectros de Autovalores (Comparações de Posto e Interior/Fronteira)
# ---------------------------------------------------------------------------

def plot_spectra(
    panels: Sequence[tuple[str, Any]],
    title: str = "Espectro de Autovalores dos Estados Reconstruídos",
    floor: float = 1e-18,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plota autovalores ordenados em escala logarítmica.
    Distingue estados no interior do cone (autovalores positivos finitos) de estados na fronteira (autovalores nulos).
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (name, obj) in enumerate(panels):
        mat = obj.rho if hasattr(obj, "rho") else np.asarray(obj, dtype=complex)
        ev = np.sort(np.linalg.eigvalsh(mat))[::-1]
        style = dict(lw=4.5, alpha=0.35) if k == 0 else dict(lw=1.8, alpha=0.9)
        ax.semilogy(np.arange(1, len(ev) + 1), np.clip(ev, floor, None), "o-", label=name, **style)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Índice do autovalor (ordem decrescente)", fontsize=11)
    ax.set_ylabel(r"Autovalor $\lambda_i(\rho)$ (escala log)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Gráficos de Escalabilidade e Benchmarks
# ---------------------------------------------------------------------------

def plot_qubit_scaling(
    n_qubits_list: Sequence[int],
    tempos_por_metodo: dict[str, Sequence[float]],
    title: str = "Tempo de Execução vs. Número de Qubits",
    save_path: str | None = None,
) -> plt.Figure:
    """Plota tempo de computação em função do número de qubits em escala semi-log."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    markers = ["o", "s", "^", "D", "v"]
    for i, (metodo, tempos) in enumerate(tempos_por_metodo.items()):
        m = markers[i % len(markers)]
        ax.plot(n_qubits_list, tempos, marker=m, lw=2, markersize=7, label=metodo)

    ax.set_yscale("log")
    ax.set_xlabel("Número de Qubits (n)", fontsize=11)
    ax.set_ylabel("Tempo de Execução (s, escala log)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_fidelity_vs_shots(
    shots_list: Sequence[int],
    fidelidades_medias: dict[str, Sequence[float]],
    fidelidades_stds: dict[str, Sequence[float]] | None = None,
    taxa_infactivel: Sequence[float] | None = None,
    title: str = "Fidelidade de Reconstrução vs. Número de Shots",
    save_path: str | None = None,
) -> plt.Figure:
    """Plota fidelidade média e desvio padrão em função de N_shots com barra de erros."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    markers = ["o", "s", "^", "D"]

    for i, (metodo, fids) in enumerate(fidelidades_medias.items()):
        m = markers[i % len(markers)]
        if fidelidades_stds and metodo in fidelidades_stds:
            ax1.errorbar(shots_list, fids, yerr=fidelidades_stds[metodo], marker=m, lw=1.8, capsize=3, label=metodo)
        else:
            ax1.plot(shots_list, fids, marker=m, lw=1.8, label=metodo)

    ax1.set_xscale("log")
    ax1.set_xlabel("Número de Shots por Observável ($N_{\\mathrm{shots}}$)", fontsize=11)
    ax1.set_ylabel("Fidelidade com o Estado Alvo", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)

    if taxa_infactivel is not None:
        ax2 = ax1.twinx()
        ax2.plot(shots_list, taxa_infactivel, color="crimson", ls="--", marker="x", label="Taxa Infactibilidade (SDP Linear)")
        ax2.set_ylabel("Fração de Instâncias Infactíveis", color="crimson", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Gráficos de Trajetória de Convergência por Classe de Pureza e Estado Térmico
# ---------------------------------------------------------------------------

def plot_purity_convergence(
    purity_results: dict[str, dict[str, TomographyResult]],
    title: str = "Trajetória de Convergência por Classe de Pureza",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plota as curvas de custo e norma de gradiente dual (escala log) sobrepostas por classe de estado:
    - Subplot 1 (esquerda): Força Bruta (Cholesky LSQ) - Custo vs. Avaliações de Função.
    - Subplot 2 (direita): SDP Dual (Quadrático) - Norma do Gradiente Dual vs. Iterações L-BFGS.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (state_name, solvers_res) in enumerate(purity_results.items()):
        cor = cores[idx % len(cores)]
        res_lsq = solvers_res.get("least_squares")
        res_quad = solvers_res.get("quadratic")

        # Entropia de von Neumann para legenda
        s_val = None
        if res_lsq is not None and "true_entropy" in res_lsq.extra:
            s_val = res_lsq.extra["true_entropy"]
        elif res_quad is not None and "true_entropy" in res_quad.extra:
            s_val = res_quad.extra["true_entropy"]
        elif res_lsq is not None:
            s_val = res_lsq.entropy
        elif res_quad is not None:
            s_val = res_quad.entropy
        s_str = f"S = {s_val:.4f}" if s_val is not None else ""

        # Subplot 1: Força Bruta
        if res_lsq is not None:
            custos = res_lsq.cost_history or res_lsq.extra.get("cost_history", [])
            if len(custos) > 0:
                ax1.semilogy(
                    np.arange(len(custos)),
                    np.clip(custos, 1e-16, None),
                    "o-",
                    color=cor,
                    lw=1.8,
                    markersize=3.5,
                    label=f"{state_name} ({s_str})",
                )

        # Subplot 2: SDP Dual Quadrático
        if res_quad is not None:
            grad_norms = res_quad.grad_norm_history or res_quad.extra.get("grad_norm_history", [])
            if len(grad_norms) > 0:
                ax2.semilogy(
                    np.arange(1, len(grad_norms) + 1),
                    np.clip(grad_norms, 1e-16, None),
                    "s-",
                    color=cor,
                    lw=1.8,
                    markersize=4.5,
                    label=f"{state_name} ({s_str})",
                )

    ax1.set_title("Força Bruta (Cholesky LSQ)\nCusto vs. Avaliações de Função", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Avaliações de Função ($k$)", fontsize=10)
    ax1.set_ylabel(r"Função de Custo $F(x) = \frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=9, loc="upper right")

    ax2.set_title("SDP Dual (Regularização Quadrática)\nNorma do Gradiente Dual vs. Iterações", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iterações L-BFGS", fontsize=10)
    ax2.set_ylabel(r"Norma do Gradiente Dual $\|\nabla \mathcal{D}\|_2 = \|A\pi^* - q\|_2$", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_gibbs_vs_control_convergence(
    gibbs_exp_results: dict[str, Any],
    title: str = "Convergência Tomográfica: Estado Térmico de Gibbs vs. Controle de Mesma Pureza",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Sobrepõe as curvas de convergência de estados térmicos de Gibbs e estados de controle equivalentes:
    - Subplot 1 (esquerda): Força Bruta (Cholesky LSQ).
    - Subplot 2 (direita): SDP Dual (Quadrático JAX).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, item in enumerate(gibbs_exp_results["data"]):
        beta = item["beta"]
        s_g = item["s_gibbs"]
        cor = cores[idx % len(cores)]

        res_g_lsq = item["res_gibbs"].get("least_squares")
        res_c_lsq = item["res_ctrl"].get("least_squares")

        # Subplot 1: Força Bruta
        if res_g_lsq is not None:
            c_g = res_g_lsq.cost_history or res_g_lsq.extra.get("cost_history", [])
            ax1.semilogy(np.arange(len(c_g)), np.clip(c_g, 1e-16, None), "-", color=cor, lw=2.0, label=f"Gibbs beta={beta} (S={s_g:.3f})")
        if res_c_lsq is not None:
            c_c = res_c_lsq.cost_history or res_c_lsq.extra.get("cost_history", [])
            ax1.semilogy(np.arange(len(c_c)), np.clip(c_c, 1e-16, None), "--", color=cor, lw=1.8, alpha=0.85, label=f"Controle beta={beta} (S={s_g:.3f})")

        # Subplot 2: SDP Dual
        res_g_quad = item["res_gibbs"].get("quadratic")
        res_c_quad = item["res_ctrl"].get("quadratic")
        if res_g_quad is not None:
            gn_g = res_g_quad.grad_norm_history or res_g_quad.extra.get("grad_norm_history", [])
            ax2.semilogy(np.arange(1, len(gn_g) + 1), np.clip(gn_g, 1e-16, None), "o-", color=cor, lw=2.0, markersize=4, label=f"Gibbs beta={beta} (S={s_g:.3f})")
        if res_c_quad is not None:
            gn_c = res_c_quad.grad_norm_history or res_c_quad.extra.get("grad_norm_history", [])
            ax2.semilogy(np.arange(1, len(gn_c) + 1), np.clip(gn_c, 1e-16, None), "s--", color=cor, lw=1.8, markersize=4, alpha=0.85, label=f"Controle beta={beta} (S={s_g:.3f})")

    ax1.set_title("Força Bruta (Cholesky LSQ)\nCusto vs. Avaliações (Medição Completa)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Avaliações de Função", fontsize=10)
    ax1.set_ylabel(r"Custo $\frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=8.5, loc="upper right")

    ax2.set_title("SDP Dual (Regularização Quadrática)\nGradiente Dual vs. Iterações (Medição Completa)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iterações L-BFGS", fontsize=10)
    ax2.set_ylabel(r"Norma Gradiente Dual $\|\nabla \mathcal{D}\|_2$", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_purity_completeness_comparison(
    incomplete_results: dict[str, dict[str, TomographyResult]],
    complete_results: dict[str, dict[str, TomographyResult]],
    title: str = "Comparação de Completude: Medição Incompleta (M=6) vs. Completa (M=15)",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Compara o comportamento da convergência em dados incompletos (M=6) vs dados completos (M=15).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5))
    cores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (state_name, solvers_res) in enumerate(incomplete_results.items()):
        cor = cores[idx % len(cores)]
        res_lsq = solvers_res.get("least_squares")
        if res_lsq is not None:
            c = res_lsq.cost_history or res_lsq.extra.get("cost_history", [])
            s_val = res_lsq.extra.get("true_entropy", res_lsq.entropy)
            ax1.semilogy(np.arange(len(c)), np.clip(c, 1e-16, None), "o-", color=cor, lw=1.8, markersize=3.5, label=f"{state_name} (S={s_val:.3f})")

    for idx, (state_name, solvers_res) in enumerate(complete_results.items()):
        cor = cores[idx % len(cores)]
        res_lsq = solvers_res.get("least_squares")
        if res_lsq is not None:
            c = res_lsq.cost_history or res_lsq.extra.get("cost_history", [])
            s_val = res_lsq.extra.get("true_entropy", res_lsq.entropy)
            ax2.semilogy(np.arange(len(c)), np.clip(c, 1e-16, None), "o-", color=cor, lw=1.8, markersize=3.5, label=f"{state_name} (S={s_val:.3f})")

    ax1.set_title("Medição Incompleta ($M = 6$ de 15 observáveis)\nSubdeterminado (Não-Unicidade)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Avaliações de Função", fontsize=10)
    ax1.set_ylabel(r"Custo $\frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=9, loc="upper right")

    ax2.set_title("Medição Completa ($M = 15$ observáveis de Pauli)\nDeterminismo Total (Estado Único)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Avaliações de Função", fontsize=10)
    ax2.set_ylabel(r"Custo $\frac{1}{2} \|f(x)\|^2$", fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_shots_scaling(
    benchmark_data: dict[str, Any],
    title: str = "Escalabilidade Tomográfica com Shots Reais ($N_{\\text{shots}} = 10.000$)",
    save_path: str | None = None,
) -> plt.Figure:
    r"""
    Gera dois gráficos lado a lado:
      (a) Fidelidade média vs. n_qubits com barras de erro (desvio padrão);
      (b) Número médio de iterações/avaliações vs. n_qubits com barras de erro.
    """
    qubits = benchmark_data["qubits_range"]
    methods = benchmark_data["methods"]
    fids_mean = benchmark_data["fidelidades_mean"]
    fids_std = benchmark_data["fidelidades_std"]
    iters_mean = benchmark_data["iteracoes_mean"]
    iters_std = benchmark_data["iteracoes_std"]

    rotulos_metodos = {
        "least_squares": "Força Bruta (Cholesky LSQ)",
        "quadratic": "SDP Dual (Quadrático JAX)",
        "cvxpy_lsq": "SDP Mínimos Quadrados Convexos",
        "cvxpy_scs": "SDP Primal Linear (SCS)",
    }
    cores_metodos = {
        "least_squares": "#31a354",
        "quadratic": "#e6550d",
        "cvxpy_lsq": "#3182bd",
        "cvxpy_scs": "#756bb1",
    }
    marcadores_metodos = {
        "least_squares": "o",
        "quadratic": "s",
        "cvxpy_lsq": "^",
        "cvxpy_scs": "D",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for m in methods:
        nome = rotulos_metodos.get(m, m)
        cor = cores_metodos.get(m, "#333333")
        marker = marcadores_metodos.get(m, "o")

        # Subplot (a): Fidelidade vs Qubits
        ax1.errorbar(
            qubits,
            fids_mean[m],
            yerr=fids_std[m],
            fmt=f"{marker}-",
            color=cor,
            capsize=4,
            capthick=1.2,
            lw=1.8,
            markersize=6,
            label=nome,
        )

        # Subplot (b): Iterações vs Qubits
        ax2.errorbar(
            qubits,
            iters_mean[m],
            yerr=iters_std[m],
            fmt=f"{marker}-",
            color=cor,
            capsize=4,
            capthick=1.2,
            lw=1.8,
            markersize=6,
            label=nome,
        )

    ax1.set_title("(a) Fidelidade vs. Número de Qubits", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Número de Qubits ($N$)", fontsize=10)
    ax1.set_ylabel("Fidelidade Média ($F$)", fontsize=10)
    ax1.set_xticks(qubits)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="lower left")

    ax2.set_title("(b) Iterações / Avaliações vs. Número de Qubits", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Número de Qubits ($N$)", fontsize=10)
    ax2.set_ylabel("Iterações Médias (escala log)", fontsize=10)
    ax2.set_yscale("log")
    ax2.set_xticks(qubits)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


