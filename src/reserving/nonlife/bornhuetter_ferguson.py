"""Bornhuetter-Ferguson.

Chain-ladder apalanca la experiencia observada: para un anio de origen reciente
con muy poco desarrollo, multiplica una cifra pequenia por un factor acumulado
grande, y el resultado es inestable. Bornhuetter-Ferguson sustituye ese
apalancamiento por una estimacion *a priori* del siniestro ultimo, normalmente
prima por siniestralidad esperada, y solo usa el triangulo para repartirla en el
tiempo.

Esto hace de BF el metodo adecuado para los anios de la cola superior del
triangulo, y explica por que en la practica se usa una combinacion: chain-ladder
para los anios maduros y BF para los recientes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..triangle import Triangle
from .chainladder import chain_ladder

__all__ = ["BornhuetterFergusonResult", "bornhuetter_ferguson"]


@dataclass(frozen=True)
class BornhuetterFergusonResult:
    """Resultado de Bornhuetter-Ferguson."""

    ultimate: np.ndarray
    """Siniestro ultimo estimado por anio de origen."""

    ibnr: np.ndarray
    """Reserva por anio de origen."""

    developed_fraction: np.ndarray
    """Fraccion del siniestro ultimo ya desarrollada, ``1 / CDF``."""

    a_priori_ultimate: np.ndarray
    """Siniestro ultimo a priori utilizado."""

    @property
    def total_ibnr(self) -> float:
        """Reserva total."""
        return float(np.sum(self.ibnr))


def bornhuetter_ferguson(
    triangle: Triangle,
    premium: np.ndarray,
    expected_loss_ratio: float | np.ndarray,
) -> BornhuetterFergusonResult:
    """Estima la reserva por Bornhuetter-Ferguson.

    La reserva es la parte **no desarrollada** del ultimo a priori:

    .. math::

        R_i^{BF} = \\left(1 - \\frac{1}{\\mathrm{CDF}_i}\\right) \\cdot
                   P_i \\cdot \\mathrm{ELR}_i

    donde ``CDF_i`` es el factor acumulado desde el desarrollo corriente hasta el
    ultimo. El patron de desarrollo se toma del triangulo via chain-ladder; el
    nivel, del a priori. Observese que el siniestro pagado a la fecha **no**
    entra en la reserva, solo en el ultimo.

    Parameters
    ----------
    triangle:
        Triangulo de siniestros.
    premium:
        Prima devengada por anio de origen.
    expected_loss_ratio:
        Siniestralidad esperada, escalar o un valor por anio de origen.

    Returns
    -------
    BornhuetterFergusonResult
    """
    tri = triangle.to_cumulative()
    n = tri.n
    premium = np.asarray(premium, dtype=float)
    if premium.shape != (n,):
        raise ValueError(f"premium debe tener {n} elementos, tiene {premium.shape}")
    elr = np.broadcast_to(np.asarray(expected_loss_ratio, dtype=float), (n,)).astype(float)

    cl = chain_ladder(tri)
    latest = cl.latest

    cdf = np.empty(n)
    for i in range(n):
        k = tri.last_observed_dev(i)
        cdf[i] = float(np.prod(cl.factors[k:])) if k < n - 1 else 1.0

    developed = 1.0 / cdf
    a_priori = premium * elr
    ibnr = (1.0 - developed) * a_priori
    return BornhuetterFergusonResult(
        ultimate=latest + ibnr,
        ibnr=ibnr,
        developed_fraction=developed,
        a_priori_ultimate=a_priori,
    )
