"""Bootstrap ODP de England y Verrall (2002).

Mack entrega el segundo momento de la reserva; el bootstrap sobredisperso de
Poisson entrega la **distribucion predictiva completa**, que es lo que se necesita
para un percentil (VaR de reserva) o un ajuste por riesgo al 75 % como pide la
CUSF.

El procedimiento separa las dos fuentes de incertidumbre:

1. **Error de estimacion**: se remuestrean los residuos de Pearson del ajuste
   ODP, se construye un pseudo-triangulo y se vuelve a correr chain-ladder. Esto
   reproduce la variabilidad de haber estimado los factores.
2. **Error de proceso**: cada celda futura se simula de una distribucion con
   media igual al valor ajustado y varianza ``phi`` veces la media, que es la
   relacion media-varianza del modelo ODP.

Omitir el paso 2 produce un intervalo demasiado estrecho; es el error habitual
al implementar este bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..triangle import Triangle
from .chainladder import chain_ladder

__all__ = ["BootstrapResult", "fitted_incremental", "pearson_residuals", "odp_bootstrap"]

_EPS = 1e-8


@dataclass(frozen=True)
class BootstrapResult:
    """Distribucion predictiva simulada de la reserva."""

    reserves: np.ndarray
    """Reserva total simulada, una por iteracion."""

    reserves_by_origin: np.ndarray
    """Reserva simulada por anio de origen, ``(n_sims, n)``."""

    scale: float
    """Parametro de sobredispersion ``phi`` estimado."""

    point_estimate: float
    """Reserva del chain-ladder determinista, como referencia."""

    @property
    def mean(self) -> float:
        """Media de la distribucion predictiva."""
        return float(np.mean(self.reserves))

    @property
    def standard_error(self) -> float:
        """Error de prediccion: desviacion estandar de la distribucion."""
        return float(np.std(self.reserves, ddof=1))

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        """Percentil de la reserva total.

        Parameters
        ----------
        q:
            Nivel o niveles en ``[0, 1]``. El 0.75 es el que usa la CUSF para el
            margen de riesgo; el 0.995 el que corresponde al VaR de Solvencia II.
        """
        return np.quantile(self.reserves, q)

    @property
    def coefficient_of_variation(self) -> float:
        """Coeficiente de variacion de la reserva total."""
        return self.standard_error / self.mean if self.mean else float("nan")


def fitted_incremental(triangle: Triangle) -> np.ndarray:
    """Valores incrementales ajustados por el modelo ODP.

    Se obtienen por recursion hacia atras desde el siniestro ultimo del
    chain-ladder. Esa recursion coincide exactamente con el ajuste de maxima
    verosimilitud del GLM Poisson sobredisperso con enlace logaritmico y efectos
    de fila y columna, que es la equivalencia en la que se apoya el metodo.

    Returns
    -------
    numpy.ndarray
        Matriz ``n x n`` con el ajuste en **todas** las celdas, observadas y
        futuras.
    """
    tri = triangle.to_cumulative()
    n = tri.n
    cl = chain_ladder(tri)

    fitted_cum = np.empty((n, n))
    fitted_cum[:, -1] = cl.ultimate
    for j in range(n - 2, -1, -1):
        fitted_cum[:, j] = fitted_cum[:, j + 1] / cl.factors[j]

    fitted_inc = np.empty((n, n))
    fitted_inc[:, 0] = fitted_cum[:, 0]
    fitted_inc[:, 1:] = np.diff(fitted_cum, axis=1)
    return fitted_inc


def pearson_residuals(triangle: Triangle) -> tuple[np.ndarray, float, np.ndarray]:
    """Residuos de Pearson ajustados por grados de libertad.

    El residuo crudo ``(y - m) / sqrt(m)`` subestima la variabilidad porque el
    modelo tiene ``2n - 1`` parametros ajustados sobre ``n(n+1)/2``
    observaciones. Se corrige por ``sqrt(N / (N - p))``, que es la correccion
    estandar de England y Verrall.

    Returns
    -------
    residuals, scale, mask
        Residuos ajustados (solo celdas observadas), parametro de
        sobredispersion ``phi``, y la mascara de celdas observadas.
    """
    tri = triangle.to_incremental()
    n = tri.n
    mask = tri.observed_mask
    obs = tri.values[mask]
    fit = fitted_incremental(tri)[mask]

    if np.any(fit <= 0.0):
        # Un ajuste no positivo aparece cuando el triangulo tiene incrementales
        # negativos grandes. El residuo de Pearson no esta definido ahi.
        n_bad = int(np.sum(fit <= 0.0))
        raise ValueError(
            f"{n_bad} celdas tienen ajuste ODP no positivo; el bootstrap de Pearson "
            "no es aplicable. Considerar un modelo que admita incrementales negativos."
        )

    n_obs = n * (n + 1) // 2
    n_par = 2 * n - 1
    dof = n_obs - n_par
    if dof <= 0:
        raise ValueError("triangulo demasiado pequenio: no quedan grados de libertad")

    raw = (obs - fit) / np.sqrt(fit)
    scale = float(np.sum(raw**2) / dof)
    adjusted = raw * np.sqrt(n_obs / dof)
    return adjusted, scale, mask


def odp_bootstrap(
    triangle: Triangle,
    n_sims: int = 10_000,
    seed: int = 20260101,
    center_residuals: bool = True,
    process_error: bool = True,
) -> BootstrapResult:
    """Distribucion predictiva de la reserva por bootstrap ODP.

    Parameters
    ----------
    triangle:
        Triangulo de siniestros.
    n_sims:
        Numero de iteraciones. Con 10 000 el error de simulacion del percentil
        99.5 ya es pequenio frente al error de modelo.
    seed:
        Semilla del generador. Fijarla hace el resultado reproducible bit a bit,
        que es requisito de auditoria.
    center_residuals:
        Centrar los residuos a media cero antes de remuestrear. Evita introducir
        un sesgo sistematico en la reserva simulada.
    process_error:
        Si ``False`` solo se mide el error de estimacion. Util para
        descomponer el riesgo, pero el resultado **no** es la distribucion
        predictiva.

    Returns
    -------
    BootstrapResult
    """
    tri = triangle.to_incremental()
    n = tri.n
    mask = tri.observed_mask
    residuals, scale, _ = pearson_residuals(tri)
    if center_residuals:
        residuals = residuals - residuals.mean()

    fit_inc = fitted_incremental(tri)
    fit_obs = fit_inc[mask]
    rng = np.random.default_rng(seed)
    cl_point = chain_ladder(tri).total_ibnr

    future = ~mask
    reserves = np.empty(n_sims)
    by_origin = np.zeros((n_sims, n))

    for s in range(n_sims):
        # --- error de estimacion: pseudo-triangulo y re-ajuste -----------------
        drawn = rng.choice(residuals, size=fit_obs.size, replace=True)
        pseudo = fit_obs + drawn * np.sqrt(fit_obs)
        pseudo_tri = np.full((n, n), np.nan)
        pseudo_tri[mask] = pseudo
        pseudo_cum = np.full((n, n), np.nan)
        for i in range(n):
            row = pseudo_tri[i, mask[i]]
            pseudo_cum[i, mask[i]] = np.cumsum(row)

        try:
            pseudo_fit = fitted_incremental(Triangle(pseudo_cum, cumulative=True))
        except ValueError:
            # Un pseudo-triangulo degenerado se descarta reutilizando el ajuste
            # original; con triangulos reales es un evento raro.
            pseudo_fit = fit_inc

        means = pseudo_fit[future]
        means = np.maximum(means, _EPS)

        # --- error de proceso: media-varianza del modelo ODP ------------------
        if process_error:
            shape = means / scale
            sim = rng.gamma(shape=shape, scale=scale)
        else:
            sim = means

        cell = np.zeros((n, n))
        cell[future] = sim
        by_origin[s] = cell.sum(axis=1)
        reserves[s] = float(sim.sum())

    return BootstrapResult(
        reserves=reserves,
        reserves_by_origin=by_origin,
        scale=scale,
        point_estimate=cl_point,
    )
