"""Motor de reservas estocasticas.

Dos submodulos deliberadamente separados, porque son dos disciplinas distintas:

:mod:`reserving.nonlife`
    Reservas de siniestros No-Vida sobre triangulos de desarrollo:
    chain-ladder, error de prediccion de Mack, bootstrap ODP,
    Bornhuetter-Ferguson y riesgo de reserva a un anio.

:mod:`reserving.life`
    Modelos multiestado de Markov para Vida y Salud: generador de
    intensidades, ecuacion de Kolmogorov y reserva prospectiva.
"""

from __future__ import annotations

from .data import taylor_ashe
from .triangle import Triangle

__version__ = "0.1.0"

__all__ = ["Triangle", "taylor_ashe", "__version__"]
