"""Conjuntos de datos de referencia.

Se incluyen triangulos publicados para que la validacion sea verificable contra
cifras externas y no solo contra la propia implementacion.
"""

from __future__ import annotations

from .triangle import Triangle

__all__ = ["TAYLOR_ASHE_INCREMENTAL", "taylor_ashe"]


TAYLOR_ASHE_INCREMENTAL: list[list[float]] = [
    [357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
    [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046],
    [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405],
    [310608, 1108250, 776189, 1562400, 272482, 352053, 206286],
    [443160, 693190, 991983, 769488, 504851, 470639],
    [396132, 937085, 847498, 805037, 705960],
    [440832, 847631, 1131398, 1063269],
    [359480, 1061648, 1443370],
    [376686, 986608],
    [344014],
]
"""Triangulo incremental de Taylor y Ashe (1983), 10 x 10.

Es el triangulo de referencia de la literatura de reservas: aparece en Mack
(1993) y en England y Verrall (2002), lo que permite contrastar tanto la reserva
puntual como el error de prediccion contra valores publicados.
"""


def taylor_ashe() -> Triangle:
    """Triangulo de Taylor-Ashe como :class:`~reserving.triangle.Triangle`."""
    return Triangle.from_rows(TAYLOR_ASHE_INCREMENTAL, cumulative=False)
