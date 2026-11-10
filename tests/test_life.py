"""Validacion de los modelos multiestado de Vida.

Todas las pruebas de esta bateria contrastan contra **soluciones cerradas**: con
generador constante la ecuacion de Kolmogorov tiene solucion exponencial, y las
reservas de un seguro temporal y de una renta tienen forma cerrada.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from reserving.life import MultiStateModel, constant_generator, illness_death_generator

MU = 0.02
DELTA = 0.04
T = 10.0


def dos_estados(mu: float = MU) -> MultiStateModel:
    """Modelo vivo/fallecido con fuerza de mortalidad constante."""
    q = np.array([[-mu, mu], [0.0, 0.0]])
    return MultiStateModel(2, constant_generator(q), state_names=("vivo", "fallecido"))


# ------------------------------------------------------------------- validacion


def test_generador_debe_tener_filas_que_sumen_cero():
    malo = np.array([[-0.1, 0.2], [0.0, 0.0]])
    with pytest.raises(ValueError, match="sumar cero"):
        MultiStateModel(2, constant_generator(malo))


def test_intensidades_no_pueden_ser_negativas():
    malo = np.array([[0.1, -0.1], [0.0, 0.0]])
    with pytest.raises(ValueError, match="no pueden ser negativas"):
        MultiStateModel(2, constant_generator(malo))


def test_generador_de_dimension_incorrecta():
    with pytest.raises(ValueError, match="debe ser 3x3"):
        MultiStateModel(3, constant_generator(np.zeros((2, 2))))


# ---------------------------------------------------- probabilidades exactas


def test_supervivencia_coincide_con_la_exponencial_analitica():
    """``p_00(t) = exp(-mu t)`` para fuerza constante."""
    modelo = dos_estados()
    p = modelo.transition_matrix(0.0, T, n_steps=500)
    assert p[0, 0] == pytest.approx(np.exp(-MU * T), rel=1e-11)
    assert p[0, 1] == pytest.approx(1.0 - np.exp(-MU * T), rel=1e-11)


def test_generador_constante_coincide_con_la_exponencial_matricial():
    """Con ``Q`` constante, ``P(0,t) = exp(Qt)``: prueba exacta de la integracion."""
    q = np.array(
        [
            [-0.07, 0.05, 0.02],
            [0.01, -0.06, 0.05],
            [0.0, 0.0, 0.0],
        ]
    )
    modelo = MultiStateModel(3, constant_generator(q))
    numerico = modelo.transition_matrix(0.0, 7.0, n_steps=1500)
    np.testing.assert_allclose(numerico, expm(q * 7.0), rtol=1e-10, atol=1e-12)


def test_filas_de_la_matriz_de_transicion_suman_uno():
    modelo = MultiStateModel(
        3,
        illness_death_generator(
            mu_ai=lambda t: 0.03 + 0.001 * t,
            mu_ad=lambda t: 0.01,
            mu_id=lambda t: 0.08,
            mu_ia=lambda t: 0.02,
        ),
    )
    p = modelo.transition_matrix(0.0, 12.0, n_steps=1200)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(3), rtol=1e-11)
    assert np.all(p >= -1e-12)


def test_chapman_kolmogorov():
    """``P(0,t2) = P(0,t1) P(t1,t2)``, incluso con intensidades no constantes."""
    modelo = MultiStateModel(
        3,
        illness_death_generator(
            mu_ai=lambda t: 0.02 * np.exp(0.05 * t),
            mu_ad=lambda t: 0.005 * np.exp(0.07 * t),
            mu_id=lambda t: 0.06,
        ),
    )
    directo = modelo.transition_matrix(0.0, 9.0, n_steps=2400)
    compuesto = modelo.transition_matrix(0.0, 4.0, n_steps=1200) @ modelo.transition_matrix(
        4.0, 9.0, n_steps=1200
    )
    np.testing.assert_allclose(directo, compuesto, rtol=1e-8, atol=1e-11)


def test_estado_absorbente_no_se_abandona():
    modelo = dos_estados()
    p = modelo.transition_matrix(0.0, 30.0, n_steps=400)
    np.testing.assert_allclose(p[1], np.array([0.0, 1.0]))


# ------------------------------------------------------------ reservas exactas


def test_seguro_temporal_coincide_con_su_valor_cerrado():
    """EPV de un capital 1 al fallecimiento, fuerza y descuento constantes.

    .. math:: \\bar A = \\frac{\\mu}{\\mu + \\delta}\\left(1 - e^{-(\\mu+\\delta)T}\\right)
    """
    modelo = dos_estados()
    capital = np.zeros((2, 2))
    capital[0, 1] = 1.0
    numerico = modelo.prospective_reserve(
        initial_state=0,
        horizon=T,
        discount_rate=DELTA,
        transition_benefits=capital,
        n_steps=2000,
    )
    exacto = MU / (MU + DELTA) * (1.0 - np.exp(-(MU + DELTA) * T))
    assert numerico == pytest.approx(exacto, rel=1e-9)


def test_renta_continua_coincide_con_su_valor_cerrado():
    """EPV de una renta continua de 1 mientras se vive.

    .. math:: \\bar a = \\frac{1 - e^{-(\\mu+\\delta)T}}{\\mu + \\delta}
    """
    modelo = dos_estados()
    numerico = modelo.prospective_reserve(
        initial_state=0,
        horizon=T,
        discount_rate=DELTA,
        annuity_rates=np.array([1.0, 0.0]),
        n_steps=2000,
    )
    exacto = (1.0 - np.exp(-(MU + DELTA) * T)) / (MU + DELTA)
    assert numerico == pytest.approx(exacto, rel=1e-9)


def test_relacion_entre_seguro_y_renta():
    """Relacion clasica ``1 = delta * a + A`` en el limite de cobertura infinita."""
    modelo = dos_estados()
    horizonte = 400.0
    capital = np.zeros((2, 2))
    capital[0, 1] = 1.0
    seguro = modelo.prospective_reserve(
        0, horizonte, DELTA, transition_benefits=capital, n_steps=4000
    )
    renta = modelo.prospective_reserve(
        0, horizonte, DELTA, annuity_rates=np.array([1.0, 0.0]), n_steps=4000
    )
    assert DELTA * renta + seguro == pytest.approx(1.0, rel=1e-8)


def test_reserva_de_invalidez_es_positiva_y_crece_con_la_prestacion():
    modelo = MultiStateModel(
        3,
        illness_death_generator(
            mu_ai=lambda t: 0.015,
            mu_ad=lambda t: 0.008,
            mu_id=lambda t: 0.05,
        ),
    )
    base = modelo.prospective_reserve(
        0, 20.0, 0.03, annuity_rates=np.array([0.0, 1.0, 0.0]), n_steps=1000
    )
    doble = modelo.prospective_reserve(
        0, 20.0, 0.03, annuity_rates=np.array([0.0, 2.0, 0.0]), n_steps=1000
    )
    assert base > 0.0
    assert doble == pytest.approx(2.0 * base, rel=1e-10)


def test_la_recuperacion_reduce_la_reserva_de_invalidez():
    def modelo_con(mu_ia):
        return MultiStateModel(
            3,
            illness_death_generator(
                mu_ai=lambda t: 0.015,
                mu_ad=lambda t: 0.008,
                mu_id=lambda t: 0.05,
                mu_ia=mu_ia,
            ),
        )

    renta = np.array([0.0, 1.0, 0.0])
    permanente = modelo_con(None).prospective_reserve(0, 20.0, 0.03, annuity_rates=renta)
    con_recuperacion = modelo_con(lambda t: 0.1).prospective_reserve(
        0, 20.0, 0.03, annuity_rates=renta
    )
    assert con_recuperacion < permanente


def test_reserva_requiere_al_menos_una_prestacion():
    with pytest.raises(ValueError, match="al menos"):
        dos_estados().prospective_reserve(0, T, DELTA)
