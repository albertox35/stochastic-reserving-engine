"""Estructura de datos de triangulo de siniestros.

Un triangulo de desarrollo es una matriz ``C[i, j]`` donde ``i`` indexa el anio de
origen (accident year) y ``j`` el periodo de desarrollo. La celda esta observada
cuando ``i + j <= n - 1``; el resto es el futuro a estimar.

Toda la libreria trabaja con indices base cero y con la diagonal corriente en
``i + j == n - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Triangle"]


@dataclass(frozen=True)
class Triangle:
    """Triangulo de desarrollo cuadrado.

    Parameters
    ----------
    values:
        Matriz ``n x n``. Las celdas no observadas (``i + j > n - 1``) deben ser
        ``nan``. Se valida en la construccion.
    cumulative:
        ``True`` si ``values`` son importes acumulados, ``False`` si son
        incrementales.
    origin_labels:
        Etiquetas opcionales de los anios de origen, solo para presentacion.

    Notes
    -----
    La clase es inmutable: ``to_cumulative`` y ``to_incremental`` devuelven
    instancias nuevas. Eso hace que los metodos de reserva sean funciones puras
    del triangulo y que los resultados sean reproducibles.
    """

    values: np.ndarray
    cumulative: bool = True
    origin_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"se espera un triangulo cuadrado, se recibio {arr.shape}")
        n = arr.shape[0]
        upper = ~np.isnan(arr)
        expected = np.add.outer(np.arange(n), np.arange(n)) <= n - 1
        if not np.array_equal(upper, expected):
            raise ValueError(
                "el patron de celdas observadas no corresponde a un triangulo: "
                "se esperan valores finitos exactamente donde i + j <= n - 1"
            )
        object.__setattr__(self, "values", arr)
        if self.origin_labels is not None and len(self.origin_labels) != n:
            raise ValueError("origin_labels debe tener una etiqueta por anio de origen")

    # ------------------------------------------------------------------ geometria

    @property
    def n(self) -> int:
        """Numero de anios de origen (y de periodos de desarrollo)."""
        return int(self.values.shape[0])

    @property
    def observed_mask(self) -> np.ndarray:
        """Mascara booleana de celdas observadas."""
        n = self.n
        return np.add.outer(np.arange(n), np.arange(n)) <= n - 1

    def last_observed_dev(self, i: int) -> int:
        """Indice de desarrollo de la ultima celda observada del anio ``i``."""
        return self.n - 1 - i

    @property
    def latest_diagonal(self) -> np.ndarray:
        """Diagonal corriente: ultimo importe acumulado observado por anio."""
        cum = self.to_cumulative().values
        return np.array([cum[i, self.last_observed_dev(i)] for i in range(self.n)])

    # ------------------------------------------------------------- transformaciones

    def to_cumulative(self) -> Triangle:
        """Devuelve el triangulo en importes acumulados."""
        if self.cumulative:
            return self
        out = np.full_like(self.values, np.nan)
        mask = self.observed_mask
        for i in range(self.n):
            row = self.values[i, mask[i]]
            out[i, mask[i]] = np.cumsum(row)
        return Triangle(out, cumulative=True, origin_labels=self.origin_labels)

    def to_incremental(self) -> Triangle:
        """Devuelve el triangulo en importes incrementales."""
        if not self.cumulative:
            return self
        out = np.full_like(self.values, np.nan)
        mask = self.observed_mask
        for i in range(self.n):
            row = self.values[i, mask[i]]
            out[i, mask[i]] = np.concatenate(([row[0]], np.diff(row)))
        return Triangle(out, cumulative=False, origin_labels=self.origin_labels)

    # ------------------------------------------------------------------- utilidades

    @classmethod
    def from_incremental(
        cls, values, origin_labels: tuple[str, ...] | None = None
    ) -> Triangle:
        """Construye un triangulo a partir de importes incrementales."""
        return cls(np.asarray(values, dtype=float), cumulative=False, origin_labels=origin_labels)

    @classmethod
    def from_rows(cls, rows: list[list[float]], cumulative: bool = False) -> Triangle:
        """Construye un triangulo desde filas de longitud decreciente.

        Es la forma en la que normalmente llegan los datos de un sistema de
        siniestros: la fila ``i`` tiene ``n - i`` observaciones.
        """
        n = len(rows)
        if any(len(r) != n - i for i, r in enumerate(rows)):
            raise ValueError("la fila i debe tener exactamente n - i observaciones")
        arr = np.full((n, n), np.nan)
        for i, row in enumerate(rows):
            arr[i, : n - i] = row
        return cls(arr, cumulative=cumulative)
