"""
quantum_tomography.solvers.base
===============================
Estruturas de dados unificadas e classes base para resultados de tomografia quântica.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ..states import (
    von_neumann_entropy,
    purity,
    numerical_rank,
    fidelidade,
    frobenius_distance,
    trace_distance,
)


@dataclass
class TomographyResult:
    """
    Resultado padronizado de uma reconstrução tomográfica.
    """
    rho: np.ndarray
    method: str
    success: bool
    status: str
    num_iters: int
    elapsed_seconds: float
    residual: float
    fidelity: float | None = None
    entropy: float = 0.0
    purity: float = 0.0
    rank: int = 0
    frobenius_norm: float = 0.0
    energy: float | None = None
    cost_or_loss: float | None = None
    trace_dist: float | None = None
    frobenius_dist: float | None = None
    dual_variables: np.ndarray | None = None
    cost_history: list[float] | None = None
    grad_norm_history: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Calcula métricas automáticas do estado reconstruído
        self.rho = np.asarray(self.rho, dtype=complex)
        self.entropy = von_neumann_entropy(self.rho)
        self.purity = purity(self.rho)
        self.rank = numerical_rank(self.rho)
        self.frobenius_norm = float(np.linalg.norm(self.rho))
        if self.cost_history is None and "cost_history" in self.extra:
            self.cost_history = self.extra["cost_history"]
        if self.grad_norm_history is None and "grad_norm_history" in self.extra:
            self.grad_norm_history = self.extra["grad_norm_history"]

    def __getitem__(self, key: str) -> Any:
        """Compatibilidade retroativa com código que acessa o resultado como dicionário."""
        alias_map = {
            "rho_reconstruido": self.rho,
            "rho": self.rho,
            "fidelidade": self.fidelity,
            "fidelity": self.fidelity,
            "tempo_s": self.elapsed_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "nfev": self.num_iters,
            "num_iters": self.num_iters,
            "status": self.status,
            "mensagem": self.status,
            "custo_final": self.cost_or_loss,
            "cost_or_loss": self.cost_or_loss,
            "residual": self.residual,
            "cost_history": self.cost_history if self.cost_history is not None else self.extra.get("cost_history", None),
            "grad_norm_history": self.grad_norm_history if self.grad_norm_history is not None else self.extra.get("grad_norm_history", None),
            "resultado": self.extra.get("resultado_scipy", None),
            "chute_inicial": self.extra.get("chute_inicial", None),
            "q_medidos": self.extra.get("q_medidos", None),
        }
        if key in alias_map:
            return alias_map[key]
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.extra:
            return self.extra[key]
        raise KeyError(f"Chave '{key}' não encontrada em TomographyResult.")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in self.extra or key in (
            "rho_reconstruido", "fidelidade", "tempo_s", "custo_final", "nfev", "mensagem"
        )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def summary(self) -> str:
        """Retorna uma visão geral limpa e legível dos resultados."""
        lines = [
            f"=== Resultado Tomografia: {self.method} ===",
            f"  Status        : {self.status} (Sucesso: {self.success})",
            f"  Tempo         : {self.elapsed_seconds * 1000.0:.2f} ms ({self.elapsed_seconds:.4f} s)",
            f"  Iterações     : {self.num_iters}",
            f"  Resíduo ||Aq-q||: {self.residual:.3e}",
            f"  Entropia S(pi): {self.entropy:.6f}",
            f"  Pureza        : {self.purity:.6f}",
            f"  Posto numérico: {self.rank}",
            f"  Norma Frobenius: {self.frobenius_norm:.6f}",
        ]
        if self.fidelity is not None:
            lines.append(f"  Fidelidade    : {self.fidelity:.6f}")
        if self.trace_dist is not None:
            lines.append(f"  Distância Traço: {self.trace_dist:.6f}")
        if self.frobenius_dist is not None:
            lines.append(f"  Dist. Frobenius: {self.frobenius_dist:.6f}")
        if self.energy is not None:
            lines.append(f"  Energia <H>   : {self.energy:.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        fid_str = f", Fid={self.fidelity:.4f}" if self.fidelity is not None else ""
        return (
            f"<TomographyResult ({self.method}, sucesso={self.success}, "
            f"tempo={self.elapsed_seconds:.3f}s{fid_str}, res={self.residual:.2e})>"
        )
