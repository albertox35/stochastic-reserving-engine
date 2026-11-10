"""Modelos multiestado de Markov para seguros de Vida y Salud.

Aqui es donde las cadenas de Markov pertenecen de verdad. Un contrato de
invalidez, de dependencia o de rentas con reversion no se describe con un
triangulo de desarrollo: se describe con un conjunto de estados (activo,
invalido, fallecido) y con intensidades de transicion entre ellos.

El objeto central es el **generador** ``Q(t)``, una matriz cuyas entradas fuera
de la diagonal son las intensidades de transicion ``mu_{gh}(t) >= 0`` y cuya
diagonal es ``-sum_h mu_{gh}(t)``, de modo que cada fila suma cero. Las
probabilidades de transicion resuelven la ecuacion de Kolmogorov progresiva

.. math::

    \\frac{\\partial}{\\partial t} P(s, t) = P(s, t)\\, Q(t), \\qquad P(s,s) = I

Con generador constante la solucion es ``P(s,t) = exp(Q (t-s))``, lo que da una
prueba exacta de la integracion numerica frente a la exponencial matricial.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["MultiStateModel", "constant_generator", "illness_death_generator"]


def constant_generator(matrix: np.ndarray) -> Callable[[float], np.ndarray]:
    """Generador independiente del tiempo a partir de una matriz de intensidades."""
    arr = np.asarray(matrix, dtype=float)
    return lambda t: arr


def illness_death_generator(
    mu_ai: Callable[[float], float],
    mu_ad: Callable[[float], float],
    mu_id: Callable[[float], float],
    mu_ia: Callable[[float], float] | None = None,
) -> Callable[[float], np.ndarray]:
    """Generador del modelo clasico de invalidez con tres estados.

    Estados: ``0`` activo, ``1`` invalido, ``2`` fallecido. El estado 2 es
    absorbente.

    Parameters
    ----------
    mu_ai, mu_ad, mu_id:
        Intensidades activo->invalido, activo->fallecido e invalido->fallecido.
    mu_ia:
        Intensidad de recuperacion invalido->activo. Si es ``None`` el modelo es
        de invalidez permanente.
    """
    recovery = mu_ia if mu_ia is not None else (lambda t: 0.0)

    def generator(t: float) -> np.ndarray:
        ai, ad, idd, ia = mu_ai(t), mu_ad(t), mu_id(t), recovery(t)
        return np.array(
            [
                [-(ai + ad), ai, ad],
                [ia, -(ia + idd), idd],
                [0.0, 0.0, 0.0],
            ]
        )

    return generator


@dataclass(frozen=True)
class MultiStateModel:
    """Cadena de Markov en tiempo continuo con generador dependiente del tiempo.

    Parameters
    ----------
    n_states:
        Numero de estados.
    generator:
        Funcion ``t -> Q(t)``, con ``Q`` de forma ``(n_states, n_states)`` y filas
        que suman cero.
    state_names:
        Nombres opcionales, solo para presentacion.
    """

    n_states: int
    generator: Callable[[float], np.ndarray]
    state_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.generator(0.0), dtype=float)
        if q.shape != (self.n_states, self.n_states):
            raise ValueError(f"el generador debe ser {self.n_states}x{self.n_states}, es {q.shape}")
        if not np.allclose(q.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError(
                "las filas del generador deben sumar cero; revisar que la diagonal "
                "sea el negativo de la suma de las intensidades de salida"
            )
        if np.any(q[~np.eye(self.n_states, dtype=bool)] < -1e-12):
            raise ValueError("las intensidades fuera de la diagonal no pueden ser negativas")

    # --------------------------------------------------------------- transiciones

    def transition_matrix(self, s: float, t: float, n_steps: int = 2000) -> np.ndarray:
        """Matriz de probabilidades de transicion ``P(s, t)``.

        Integra la ecuacion de Kolmogorov progresiva con Runge-Kutta de cuarto
        orden. El error es ``O(h^4)``, asi que con unos pocos miles de pasos el
        resultado es exacto a precision de maquina para intensidades suaves.

        Parameters
        ----------
        s, t:
            Tiempos inicial y final, con ``t >= s``.
        n_steps:
            Pasos de integracion.

        Returns
        -------
        numpy.ndarray
            Matriz ``(n_states, n_states)`` cuyas filas suman uno.
        """
        if t < s:
            raise ValueError("se requiere t >= s")
        if t == s:
            return np.eye(self.n_states)

        h = (t - s) / n_steps
        p = np.eye(self.n_states)
        for step in range(n_steps):
            u = s + step * h
            k1 = p @ self.generator(u)
            k2 = (p + 0.5 * h * k1) @ self.generator(u + 0.5 * h)
            k3 = (p + 0.5 * h * k2) @ self.generator(u + 0.5 * h)
            k4 = (p + h * k3) @ self.generator(u + h)
            p = p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return p

    def occupancy_probabilities(
        self, initial_state: int, s: float, t: float, n_steps: int = 2000
    ) -> np.ndarray:
        """Distribucion sobre estados en ``t`` partiendo de ``initial_state`` en ``s``."""
        return self.transition_matrix(s, t, n_steps)[initial_state]

    # -------------------------------------------------------------------- reserva

    def prospective_reserve(
        self,
        initial_state: int,
        horizon: float,
        discount_rate: float,
        annuity_rates: np.ndarray | None = None,
        transition_benefits: np.ndarray | None = None,
        s: float = 0.0,
        n_steps: int = 2000,
    ) -> float:
        """Reserva prospectiva: valor presente esperado de las prestaciones.

        Se integra numericamente

        .. math::

            V(s) = \\int_s^{s+T} e^{-\\delta (u-s)} \\left[
                       \\sum_g {}_{u-s}p^{\\,g}\\, b_g(u)
                     + \\sum_{g \\neq h} {}_{u-s}p^{\\,g}\\, \\mu_{gh}(u)\\, B_{gh}(u)
                   \\right] du

        El primer sumando son las prestaciones en forma de renta mientras se
        permanece en un estado; el segundo, los capitales pagaderos al momento de
        la transicion, que es la forma en que entra un capital por fallecimiento
        o por reconocimiento de invalidez.

        Parameters
        ----------
        initial_state:
            Estado en el que se encuentra el asegurado en ``s``.
        horizon:
            Duracion ``T`` del periodo de cobertura.
        discount_rate:
            Fuerza de interes ``delta``, continua.
        annuity_rates:
            Tasa de renta continua por estado, forma ``(n_states,)``.
        transition_benefits:
            Capital pagadero en cada transicion, forma
            ``(n_states, n_states)``; la diagonal se ignora.
        s:
            Tiempo inicial.
        n_steps:
            Pasos de la cuadratura; se usa Simpson, asi que debe ser par.

        Returns
        -------
        float
            Valor presente esperado de las prestaciones.
        """
        if annuity_rates is None and transition_benefits is None:
            raise ValueError("se requiere al menos annuity_rates o transition_benefits")
        if n_steps % 2 != 0:
            n_steps += 1

        rates = (
            np.zeros(self.n_states)
            if annuity_rates is None
            else np.asarray(annuity_rates, dtype=float)
        )
        lumps = (
            np.zeros((self.n_states, self.n_states))
            if transition_benefits is None
            else np.asarray(transition_benefits, dtype=float)
        )
        off_diag = ~np.eye(self.n_states, dtype=bool)

        grid = np.linspace(0.0, horizon, n_steps + 1)
        integrand = np.empty(grid.size)
        # Integrar la ecuacion de Kolmogorov una sola vez y leer la ocupacion en
        # cada nodo de la cuadratura, en lugar de reintegrar por nodo.
        p = np.eye(self.n_states)
        h = horizon / n_steps
        for idx, tau in enumerate(grid):
            u = s + tau
            probs = p[initial_state]
            q = self.generator(u)
            annuity_part = float(np.dot(probs, rates))
            lump_part = float(np.sum(probs[:, None] * q * lumps * off_diag))
            integrand[idx] = np.exp(-discount_rate * tau) * (annuity_part + lump_part)
            if idx < n_steps:
                k1 = p @ self.generator(u)
                k2 = (p + 0.5 * h * k1) @ self.generator(u + 0.5 * h)
                k3 = (p + 0.5 * h * k2) @ self.generator(u + 0.5 * h)
                k4 = (p + h * k3) @ self.generator(u + h)
                p = p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Simpson compuesta
        weights = np.ones(n_steps + 1)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        return float(h / 3.0 * np.dot(weights, integrand))
