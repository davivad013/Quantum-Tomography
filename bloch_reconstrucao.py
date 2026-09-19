import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------
# Matrizes de Pauli e vetor de Bloch
# ---------------------------------------------------------------
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def vetor_bloch(rho):
    """Retorna (<sx>, <sy>, <sz>) = Tr(rho * sigma_i) para um estado de 1 qubit."""
    return np.array([np.real(np.trace(rho @ s)) for s in (sigma_x, sigma_y, sigma_z)])


# ---------------------------------------------------------------
# Esfera 3D
# ---------------------------------------------------------------
def desenhar_esfera(ax):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_surface(xs, ys, zs, color="#9ecae1", alpha=0.10, linewidth=0,
                    antialiased=True, shade=False)
    ax.plot_wireframe(xs, ys, zs, color="#6baed6", alpha=0.10,
                      linewidth=0.4, rstride=4, cstride=4)

    # Círculos máximos: equador (xy) e planos xz e yz
    t = np.linspace(0, 2 * np.pi, 300)
    c, s, o = np.cos(t), np.sin(t), np.zeros_like(t)
    ax.plot(c, s, o, color="gray", alpha=0.5, lw=1)
    ax.plot(c, o, s, color="gray", alpha=0.5, lw=1)
    ax.plot(o, c, s, color="gray", alpha=0.5, lw=1)

    L = 1.25
    for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        d = np.array(d) * L
        ax.plot([-d[0], d[0]], [-d[1], d[1]], [-d[2], d[2]],
                color="k", alpha=0.6, lw=0.9)

    k = 1.42
    ax.text(0, 0, k, r"$|0\rangle$", ha="center", fontsize=12)
    ax.text(0, 0, -k, r"$|1\rangle$", ha="center", fontsize=12)
    ax.text(k, 0, 0, r"$|+\rangle$", ha="center", fontsize=12)
    ax.text(-k, 0, 0, r"$|-\rangle$", ha="center", fontsize=12)
    ax.text(0, k, 0, r"$|{+i}\rangle$", ha="center", fontsize=12)
    ax.text(0, -k, 0, r"$|{-i}\rangle$", ha="center", fontsize=12)


def desenhar_estado_3d(ax, r, cor, rotulo, marcador):
    ax.quiver(0, 0, 0, *r, color=cor, lw=2.5, arrow_length_ratio=0.10)
    ax.scatter(*r, color=cor, s=90, marker=marcador, edgecolor="k",
               zorder=10, label=rotulo)


# ---------------------------------------------------------------
# Projeções 2D nos três planos
# ---------------------------------------------------------------
# (índice do eixo horizontal, índice do eixo vertical, nome, rótulos dos 4 lados)
# ordem dos rótulos: direita, esquerda, cima, baixo
PLANOS = {
    "xz": (0, 2, "Plano x–z", (r"$|+\rangle$", r"$|-\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
    "xy": (0, 1, "Plano x–y", (r"$|+\rangle$", r"$|-\rangle$", r"$|{+i}\rangle$", r"$|{-i}\rangle$")),
    "yz": (1, 2, "Plano y–z", (r"$|{+i}\rangle$", r"$|{-i}\rangle$", r"$|0\rangle$", r"$|1\rangle$")),
}


def desenhar_plano(ax, vetores, cores, rotulos, marcadores, chave, medido=False):
    ih, iv, nome, (dir_, esq, cima, baixo) = PLANOS[chave]

    ax.add_patch(Circle((0, 0), 1, fc="#9ecae1", ec="gray", alpha=0.15, lw=0))
    ax.add_patch(Circle((0, 0), 1, fc="none", ec="gray", lw=1.2))
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)

    for r, cor, rot, m in zip(vetores, cores, rotulos, marcadores):
        h, v = r[ih], r[iv]
        ax.annotate("", xy=(h, v), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=cor, lw=2.2))
        ax.scatter(h, v, color=cor, s=110 if m == "o" else 60, marker=m,
                   edgecolor="k", zorder=10)

    if len(vetores) == 2:
        a, b = vetores
        ax.plot([a[ih], b[ih]], [a[iv], b[iv]], color="k", ls=":", lw=1.3)

    k = 1.17
    ax.text(k, 0, dir_, ha="left", va="center", fontsize=12)
    ax.text(-k, 0, esq, ha="right", va="center", fontsize=12)
    ax.text(0, k, cima, ha="center", va="bottom", fontsize=12)
    ax.text(0, -k, baixo, ha="center", va="top", fontsize=12)

    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal"); ax.axis("off")
    titulo = nome + ("  (observáveis medidos)" if medido else "")
    ax.set_title(titulo, fontsize=12, fontweight="bold" if medido else "normal")


# ---------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------
def plotar_bloch(rhos, rotulos=None, cores=None, titulo="Esfera de Bloch–Poincaré",
                 elev=20, azim=35, medidos=("x", "z"), salvar=None):
    """
    rhos    : lista de matrizes densidade 2x2
    medidos : eixos dos observáveis realmente medidos (destaca o plano correspondente)
    Layout  : esfera 3D + projeções nos planos x–z, x–y e y–z.
    """
    rotulos = rotulos or [f"ρ{i + 1}" for i in range(len(rhos))]
    cores = cores or ["#e6550d", "#31a354", "#756bb1", "#de2d26"]
    marcadores = ["o", "D", "s", "^"]
    vetores = [vetor_bloch(rho) for rho in rhos]

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05,
                          left=0.03, right=0.97, top=0.92, bottom=0.14)

    # --- esfera 3D ---
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    desenhar_esfera(ax3d)
    for r, cor, rot, m in zip(vetores, cores, rotulos, marcadores):
        desenhar_estado_3d(ax3d, r, cor, rot, m)
    if len(vetores) == 2:
        ax3d.plot(*zip(*vetores), color="k", ls=":", lw=1.5)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
    ax3d.set_axis_off()
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("Esfera de Bloch (3D)", fontsize=12)
    ax3d.legend(loc="upper left", fontsize=11)

    # --- três planos ---
    conj_medidos = set(medidos)
    for chave, pos in (("xz", (0, 1)), ("xy", (1, 0)), ("yz", (1, 1))):
        ax = fig.add_subplot(gs[pos])
        desenhar_plano(ax, vetores, cores, rotulos, marcadores, chave,
                       medido=(set(chave) == conj_medidos))

    # --- caixa de texto ---
    linhas = [f"{rot}:  r = ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})   |r| = {np.linalg.norm(r):.3f}"
              for rot, r in zip(rotulos, vetores)]
    if len(vetores) == 2:
        linhas.append(f"Distância entre eles = {np.linalg.norm(vetores[0] - vetores[1]):.3f}")
    fig.text(0.5, 0.02, "\n".join(linhas), ha="center", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    fig.suptitle(titulo, fontsize=16, y=0.97)
    if salvar:
        fig.savefig(salvar, dpi=200, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------
# Uso com os seus resultados (rode depois do seu código)
# ---------------------------------------------------------------
# plotar_bloch(
#     [rho_reconstruido1, rho_reconstruido2],
#     rotulos=["Reconstrução 1", "Reconstrução 2"],
#     medidos=("x", "z"),
#     salvar="bloch_reconstrucoes.png",
# )
