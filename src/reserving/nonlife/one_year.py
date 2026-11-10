"""Riesgo de reserva a un anio (claims development result).

Solvencia II y la CUSF no piden la incertidumbre hasta el siniestro ultimo, sino
la del **resultado del desarrollo de siniestros a un anio**: cuanto puede moverse
la estimacion de la reserva cuando llegue la informacion del proximo ejercicio.
Esa es una cantidad distinta y menor que el riesgo a ultimo, y confundirlas
sobrestima el requerimiento de capital.

Se implementan dos estimadores:

``simulated_cdr``
    Re-reserva por simulacion, el metodo que la practica llama *actuary in the
    box*: se simula la siguiente diagonal, se vuelve a correr chain-ladder con la
    informacion ampliada y se mide cuanto cambio la estimacion. Mide **solo
    riesgo de proceso**: al simular con los factores estimados como si fueran los
    verdaderos, la incertidumbre de parametro no entra por construccion. Es una
    cota inferior del CDR, y se documenta como tal porque es un malentendido
    frecuente de este metodo.

``merz_wuthrich``
    La formula cerrada de Merz y Wuthrich (2008), que incluye proceso **y**
    parametro. Es la que corresponde usar para capital.

La formula cerrada no se da por buena: se valida en ``tests/test_one_year.py``
contra cuatro condiciones verificables, entre ellas una frontera exacta. Para el
anio de origen mas antiguo incompleto solo resta un periodo de desarrollo, asi
que un anio de informacion resuelve todo el siniestro ultimo y el MSEP del CDR
tiene que **coincidir exactamente** con el MSEP a ultimo de Mack. Esa condicion
es lo que fija el indexado de los denominadores de la formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..triangle import Triangle
from .chainladder import chain_ladder
from .mack import mack_sigma_squared

__all__ = ["CDRResult", "simulated_cdr", "merz_wuthrich"]


@dataclass(frozen=True)
class CDRResult:
    """Riesgo de reserva a un anio."""

    msep_by_origin: np.ndarray
    """MSEP del CDR por anio de origen."""

    total_msep: float
    """MSEP del CDR agregado."""

    mean_cdr: np.ndarray | None = None
    """Media simulada del CDR por anio; debe ser cercana a cero."""

    @property
    def standard_error(self) -> np.ndarray:
        """Error estandar del CDR por anio de origen."""
        return np.sqrt(self.msep_by_origin)

    @property
    def total_standard_error(self) -> float:
        """Error estandar del CDR agregado."""
        return float(np.sqrt(self.total_msep))


def _trapezoid_ultimates(data: np.ndarray, extra_diagonals: int) -> np.ndarray:
    """Chain-ladder sobre un trapecio con diagonales adicionales.

    ``data`` es ``n x n`` con celdas observadas donde
    ``i + j <= n - 1 + extra_diagonals``. Con ``extra_diagonals = 1`` representa
    la informacion disponible al cierre del proximo ejercicio.
    """
    n = data.shape[0]
    limit = n - 1 + extra_diagonals
    factors = np.empty(n - 1)
    for j in range(n - 1):
        # Filas con la pareja (C[i, j], C[i, j+1]) completamente observada.
        n_rows = min(n, limit - j)
        rows = slice(0, n_rows)
        denom = float(np.sum(data[rows, j]))
        numer = float(np.sum(data[rows, j + 1]))
        factors[j] = numer / denom if denom > 0.0 else 1.0

    ultimates = np.empty(n)
    for i in range(n):
        k = min(limit - i, n - 1)
        value = data[i, k]
        for j in range(k, n - 1):
            value *= factors[j]
        ultimates[i] = value
    return ultimates


def simulated_cdr(
    triangle: Triangle,
    n_sims: int = 20_000,
    seed: int = 20260101,
) -> CDRResult:
    """Riesgo a un anio por re-reserva simulada.

    La siguiente diagonal se simula bajo los dos primeros momentos de Mack,

    .. math::

        E[C_{i,k+1}] = \\hat f_k C_{i,k}, \\qquad
        \\mathrm{Var}[C_{i,k+1}] = \\hat\\sigma_k^2 C_{i,k}

    usando una gamma con esa media y varianza para mantener el soporte positivo.
    Despues se vuelve a correr chain-ladder sobre el trapecio ampliado y se mide

    .. math::

        \\mathrm{CDR}_i = \\hat C_{i,J}^{(I)} - \\hat C_{i,J}^{(I+1)}

    Parameters
    ----------
    triangle:
        Triangulo de siniestros.
    n_sims:
        Numero de simulaciones.
    seed:
        Semilla, para reproducibilidad.
    """
    tri = triangle.to_cumulative()
    n = tri.n
    cl = chain_ladder(tri)
    sigma2 = mack_sigma_squared(tri)
    base = cl.full_triangle
    rng = np.random.default_rng(seed)

    # Celdas de la diagonal siguiente: una por anio de origen i = 1..n-1.
    rows = np.arange(1, n)
    k = n - 1 - rows
    c_now = base[rows, k]
    mean_next = c_now * cl.factors[k]
    var_next = sigma2[k] * c_now
    shape = mean_next**2 / var_next
    scale = var_next / mean_next

    cdr = np.empty((n_sims, n))
    for s in range(n_sims):
        data = base.copy()
        data[rows, k + 1] = rng.gamma(shape=shape, scale=scale)
        # Reproyectar lo que sigue siendo futuro tras la nueva diagonal, para
        # que el trapecio quede completo en las columnas que lo requieren.
        new_ult = _trapezoid_ultimates(_reproject(data, n), extra_diagonals=1)
        cdr[s] = cl.ultimate - new_ult

    msep = np.var(cdr, axis=0, ddof=1)
    total = float(np.var(cdr.sum(axis=1), ddof=1))
    return CDRResult(
        msep_by_origin=msep,
        total_msep=total,
        mean_cdr=np.mean(cdr, axis=0),
    )


def _reproject(data: np.ndarray, n: int) -> np.ndarray:
    """Rellena las celdas aun futuras tras anadir una diagonal.

    Solo sirve para que ``_trapezoid_ultimates`` pueda leer ``data[i, k]`` en el
    borde; los valores reproyectados no entran en la estimacion de factores,
    que se limita a las parejas observadas.
    """
    out = data.copy()
    limit = n  # i + j <= n tras anadir una diagonal
    for i in range(n):
        for j in range(n - 1):
            if i + j + 1 > limit and np.isnan(out[i, j + 1]):
                out[i, j + 1] = out[i, j]
    return out


def merz_wuthrich(triangle: Triangle) -> CDRResult:
    """Riesgo a un anio por la formula cerrada de Merz y Wuthrich (2008).

    El MSEP condicional del CDR para el anio de origen ``i`` con ultimo
    desarrollo observado ``k = n - 1 - i`` es

    .. math::

        \\widehat{\\mathrm{msep}}(\\mathrm{CDR}_i)
          = \\hat C_{i,J}^2 \\left(\\hat\\Phi_i + \\hat\\Delta_i\\right)

    con un termino de proceso que solo involucra el siguiente paso de desarrollo
    y un termino de parametro que recoge el efecto de reestimar los factores con
    una diagonal mas.

    El indexado de los denominadores es el punto delicado de la formula. Los
    pesos de la suma usan el denominador **ampliado** ``S_j^{I+1}``, porque la
    reestimacion del proximo anio ya dispondra de esa observacion, mientras que
    el termino de parametro usa el denominador **corriente** ``S_j^{I}``, que es
    el que respalda la estimacion de hoy. La eleccion no es cosmetica: con
    ``S_j^{I+1}`` en el termino de parametro la formula subestima el MSEP un 19 %
    en el triangulo de Taylor-Ashe y deja de cumplir la frontera exacta del anio
    mas antiguo incompleto.

    Notes
    -----
    Validada en ``tests/test_one_year.py`` contra:

    1. la frontera exacta frente a Mack para el anio mas antiguo incompleto;
    2. ``MSEP(CDR_i) <= MSEP(ultimo_i)`` en todos los anios;
    3. dominancia sobre la re-reserva simulada, que solo mide proceso;
    4. el valor analitico del riesgo de proceso del anio mas antiguo.
    """
    tri = triangle.to_cumulative()
    n = tri.n
    cl = chain_ladder(tri)
    sigma2 = mack_sigma_squared(tri)
    full = cl.full_triangle
    f = cl.factors
    s_now = cl.column_sums

    # Denominador ampliado: el proximo cierre anade la fila n-1-j en el
    # desarrollo j, que hoy es la ultima celda observada de esa fila.
    s_next = np.array([s_now[j] + full[n - 1 - j, j] for j in range(n - 1)])

    msep = np.zeros(n)
    terms = np.zeros((n, n - 1))
    for i in range(1, n):
        k = n - 1 - i
        ult2 = cl.ultimate[i] ** 2

        phi = (sigma2[k] / f[k] ** 2) / full[i, k]
        delta = (sigma2[k] / f[k] ** 2) / s_now[k]
        terms[i, k] = (sigma2[k] / f[k] ** 2) / s_now[k]

        for j in range(k + 1, n - 1):
            w = full[n - 1 - j, j] / s_next[j]
            phi += w**2 * (sigma2[j] / f[j] ** 2) / full[n - 1 - j, j]
            delta += w**2 * (sigma2[j] / f[j] ** 2) / s_now[j]
            terms[i, j] = w**2 * (sigma2[j] / f[j] ** 2) / s_now[j]

        msep[i] = ult2 * (phi + delta)

    # Termino cruzado: los CDR de distintos anios comparten los factores
    # reestimados, igual que en Mack.
    cross = 0.0
    for i in range(1, n):
        tail = float(np.sum(cl.ultimate[i + 1 :]))
        if tail == 0.0:
            continue
        cross += 2.0 * cl.ultimate[i] * tail * float(np.sum(terms[i]))

    return CDRResult(msep_by_origin=msep, total_msep=float(np.sum(msep) + cross))
