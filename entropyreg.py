"""Módulo de compatibilidade que reexporta as funções organizadas em `entropy_reg`."""

from entropy_reg import (
    dual_cost,
    dual_gradient,
    gradient_descent,
    reconstruct_pi,
    reconstruct_spi,
    sgd,
    stochastic_gradient,
)

__all__ = [
    "dual_cost",
    "dual_gradient",
    "gradient_descent",
    "reconstruct_pi",
    "reconstruct_spi",
    "sgd",
    "stochastic_gradient",
]
