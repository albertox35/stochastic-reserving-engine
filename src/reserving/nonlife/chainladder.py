"""Chain-ladder determinista.

Estima los factores de desarrollo por el promedio ponderado por volumen, que es
el estimador estandar y el unico para el que Mack (1993) deriva el error de
prediccion analitico. Los promedios simples y los de los ultimos ``k`` anios se
ofrecen aparte porque cambian las propiedades distribucionales: usarlos invalida
las formulas de :mod:`reserving.nonlife.mack`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..triangle import Triangle

__all__ = ["ChainLadderResult", "development_factors", "chain_ladder"]


@dataclass(frozen=True)
class ChainLadderResult:
    """Resultado del chain-ladder."""

    full_triangle: np.ndarray
    """Triangulo acumulado completado (observado + proyectado), ``n x n``."""

    factors: np.ndarray
    """Factores de desarrollo ``f[0..n-2]``, ponderados por volumen."""

    column_sums: np.ndarray
    """``S[j]``: suma de los acumulados usados como denominador de ``f[j]``.

    Interviene directamente en el error de parametro de Mack, por eso se expone.
    """

    ultimate: np.ndarray
    """Siniestro ultimo estimado por anio de origen."""

    latest: np.ndarray
    """Diagonal corriente observada."""

    @property
    def ibnr(self) -> np.ndarray:
        """Reserva por anio de origen: ultimo menos pagado a la fecha."""
        return self.ultimate - self.latest

    @property
    def total_ibnr(self) -> float:
        """Reserva total."""
        return float(np.sum(self.ibnr))


def development_factors(triangle: Triangle) -> tuple[np.ndarray, np.ndarray]:
    """Factores de desarrollo ponderados por volumen y sus denominadores.

    El factor ``f[j]`` usa las ``n - 1 - j`` parejas ``(C[i, j], C[i, j+1])``
    completamente observadas:

    .. math::

        \\hat f_j = \\frac{\\sum_{i=0}^{n-2-j} C_{i,j+1}}{\\sum_{i=0}^{n-2-j} C_{i,j}}

    Returns
    -------
    factors, column_sums
        ``factors[j]`` es el factor de ``j`` a ``j+1``; ``column_sums[j]`` es el
        denominador, necesario para el error de parametro.
    """
    cum = triangle.to_cumulative().values
    n = triangle.n
    factors = np.empty(n - 1)
    col_sums = np.empty(n - 1)
    for j in range(n - 1):
        rows = slice(0, n - 1 - j)
        denom = float(np.sum(cum[rows, j]))
        numer = float(np.sum(cum[rows, j + 1]))
        if denom <= 0.0:
            raise ValueError(
                f"el denominador del factor de desarrollo {j} no es positivo "
                f"({denom}); chain-ladder no esta definido para esta columna"
            )
        factors[j] = numer / denom
        col_sums[j] = denom
    return factors, col_sums


def chain_ladder(triangle: Triangle) -> ChainLadderResult:
    """Completa el triangulo y estima la reserva por chain-ladder.

    Parameters
    ----------
    triangle:
        Triangulo de siniestros, acumulado o incremental.

    Returns
    -------
    ChainLadderResult

    Examples
    --------
    >>> from reserving.data import taylor_ashe
    >>> res = chain_ladder(taylor_ashe())
    >>> round(res.total_ibnr)
    18680856
    """
    tri = triangle.to_cumulative()
    cum = tri.values
    n = tri.n
    factors, col_sums = development_factors(tri)

    full = cum.copy()
    for i in range(1, n):
        for j in range(tri.last_observed_dev(i), n - 1):
            full[i, j + 1] = full[i, j] * factors[j]

    return ChainLadderResult(
        full_triangle=full,
        factors=factors,
        column_sums=col_sums,
        ultimate=full[:, -1].copy(),
        latest=tri.latest_diagonal,
    )
