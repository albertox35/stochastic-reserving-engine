"""Error de prediccion analitico de Mack (1993).

Mack deriva el error cuadratico medio de prediccion (MSEP) del chain-ladder sin
suponer una distribucion para los siniestros: solo requiere los dos primeros
momentos condicionales

.. math::

    E[C_{i,j+1} \\mid C_{i,0..j}] = f_j C_{i,j}, \\qquad
    \\mathrm{Var}[C_{i,j+1} \\mid C_{i,0..j}] = \\sigma_j^2 C_{i,j}

El MSEP se descompone en **error de proceso** (la aleatoriedad futura, que
existiria incluso conociendo los factores) y **error de parametro** (la
incertidumbre de haber estimado los factores con un triangulo finito). La
distincion importa porque solo el segundo se reduce al acumular experiencia.

El total agregado no es la suma de los MSEP por anio: los anios comparten los
mismos factores estimados, asi que sus errores de parametro estan correlacionados
y aparece un termino cruzado positivo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..triangle import Triangle
from .chainladder import ChainLadderResult, chain_ladder

__all__ = ["MackResult", "mack_sigma_squared", "mack"]


@dataclass(frozen=True)
class MackResult:
    """Resultado del analisis de Mack."""

    chain_ladder: ChainLadderResult
    sigma_squared: np.ndarray
    """``sigma^2[j]`` para ``j = 0..n-2``; el ultimo se extrapola."""

    process_variance: np.ndarray
    """Varianza de proceso de la reserva, por anio de origen."""

    parameter_variance: np.ndarray
    """Varianza de parametro de la reserva, por anio de origen."""

    total_msep: float
    """MSEP de la reserva agregada, incluido el termino cruzado."""

    @property
    def msep(self) -> np.ndarray:
        """MSEP por anio de origen."""
        return self.process_variance + self.parameter_variance

    @property
    def standard_error(self) -> np.ndarray:
        """Error estandar de prediccion por anio de origen."""
        return np.sqrt(self.msep)

    @property
    def total_standard_error(self) -> float:
        """Error estandar de prediccion de la reserva agregada."""
        return float(np.sqrt(self.total_msep))

    @property
    def coefficient_of_variation(self) -> float:
        """Coeficiente de variacion de la reserva total."""
        total = self.chain_ladder.total_ibnr
        if total == 0.0:
            return float("nan")
        return self.total_standard_error / total


def mack_sigma_squared(triangle: Triangle) -> np.ndarray:
    """Estima ``sigma^2[j]`` para cada periodo de desarrollo.

    Para ``j <= n - 3`` se usa el estimador no sesgado

    .. math::

        \\hat\\sigma_j^2 = \\frac{1}{n-j-2}
            \\sum_{i=0}^{n-2-j} C_{i,j}
            \\left(\\frac{C_{i,j+1}}{C_{i,j}} - \\hat f_j\\right)^2

    El ultimo periodo (``j = n - 2``) tiene una sola observacion, asi que no
    tiene grados de libertad. Se aplica la extrapolacion de Mack, que es
    deliberadamente conservadora y mantiene la sucesion decreciente:

    .. math::

        \\hat\\sigma_{n-2}^2 = \\min\\left(
            \\frac{\\hat\\sigma_{n-3}^4}{\\hat\\sigma_{n-4}^2},\\;
            \\min(\\hat\\sigma_{n-4}^2, \\hat\\sigma_{n-3}^2)\\right)
    """
    tri = triangle.to_cumulative()
    cum = tri.values
    n = tri.n
    if n < 4:
        raise ValueError("se requieren al menos 4 anios de origen para extrapolar sigma^2")

    factors, _ = tri_factors(tri)
    sigma2 = np.full(n - 1, np.nan)
    for j in range(n - 2):
        dof = n - j - 2
        rows = slice(0, n - 1 - j)
        ratios = cum[rows, j + 1] / cum[rows, j]
        sigma2[j] = float(np.sum(cum[rows, j] * (ratios - factors[j]) ** 2) / dof)

    s_last, s_prev = sigma2[n - 3], sigma2[n - 4]
    if s_prev > 0.0:
        sigma2[n - 2] = min(s_last**2 / s_prev, min(s_prev, s_last))
    else:
        sigma2[n - 2] = s_last
    return sigma2


def tri_factors(triangle: Triangle) -> tuple[np.ndarray, np.ndarray]:
    """Alias interno de :func:`~reserving.nonlife.chainladder.development_factors`."""
    from .chainladder import development_factors

    return development_factors(triangle)


def mack(triangle: Triangle) -> MackResult:
    """Calcula el MSEP de Mack del chain-ladder.

    Parameters
    ----------
    triangle:
        Triangulo de siniestros.

    Returns
    -------
    MackResult

    Notes
    -----
    Para el anio de origen ``i`` con ultimo desarrollo observado
    ``k = n - 1 - i``, el MSEP del siniestro ultimo es

    .. math::

        \\widehat{\\mathrm{msep}}(\\hat C_{i,n-1}) = \\hat C_{i,n-1}^2
            \\sum_{j=k}^{n-2} \\frac{\\hat\\sigma_j^2}{\\hat f_j^2}
            \\left(\\frac{1}{\\hat C_{i,j}} + \\frac{1}{S_j}\\right)

    El termino ``1 / C_{i,j}`` es el error de proceso y ``1 / S_j`` el de
    parametro. El MSEP de la reserva coincide con el del ultimo, porque lo ya
    pagado es conocido.

    Examples
    --------
    >>> from reserving.data import taylor_ashe
    >>> res = mack(taylor_ashe())
    >>> round(res.total_standard_error)
    2447095
    """
    tri = triangle.to_cumulative()
    n = tri.n
    cl = chain_ladder(tri)
    sigma2 = mack_sigma_squared(tri)
    full = cl.full_triangle
    f = cl.factors
    s = cl.column_sums

    process = np.zeros(n)
    parameter = np.zeros(n)
    for i in range(1, n):
        k = tri.last_observed_dev(i)
        ult2 = cl.ultimate[i] ** 2
        process[i] = ult2 * float(
            np.sum([(sigma2[j] / f[j] ** 2) / full[i, j] for j in range(k, n - 1)])
        )
        parameter[i] = ult2 * float(
            np.sum([(sigma2[j] / f[j] ** 2) / s[j] for j in range(k, n - 1)])
        )

    # Termino cruzado: los anios comparten los factores estimados, asi que sus
    # errores de parametro covarian. Ignorarlo subestima el riesgo agregado.
    cross = 0.0
    for i in range(1, n):
        k = tri.last_observed_dev(i)
        tail_sum = float(np.sum(cl.ultimate[i + 1 :]))
        if tail_sum == 0.0:
            continue
        shared = float(np.sum([(sigma2[j] / f[j] ** 2) / s[j] for j in range(k, n - 1)]))
        cross += 2.0 * cl.ultimate[i] * tail_sum * shared

    total = float(np.sum(process) + np.sum(parameter) + cross)
    return MackResult(
        chain_ladder=cl,
        sigma_squared=sigma2,
        process_variance=process,
        parameter_variance=parameter,
        total_msep=total,
    )
