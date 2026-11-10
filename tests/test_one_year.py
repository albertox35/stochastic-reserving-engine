"""Validacion del riesgo de reserva a un anio.

La formula cerrada de Merz-Wuthrich no se da por buena: se fija contra una
condicion de frontera exacta y se contrasta con la re-reserva simulada.
"""

from __future__ import annotations

import numpy as np
import pytest

from reserving import taylor_ashe
from reserving.nonlife import mack, mack_sigma_squared, merz_wuthrich, simulated_cdr
from reserving.nonlife.chainladder import chain_ladder


def test_frontera_exacta_del_anio_mas_antiguo_incompleto():
    """Condicion que fija el indexado de la formula.

    Para el anio de origen 1 solo resta un periodo de desarrollo. Un anio de
    informacion nueva resuelve por completo el siniestro ultimo, asi que el MSEP
    del CDR tiene que coincidir **exactamente** con el MSEP a ultimo de Mack.
    """
    tri = taylor_ashe()
    cdr = merz_wuthrich(tri)
    ultimo = mack(tri)
    assert cdr.standard_error[1] == pytest.approx(ultimo.standard_error[1], rel=1e-10)


def test_riesgo_a_un_anio_no_excede_el_riesgo_a_ultimo_por_anio():
    tri = taylor_ashe()
    cdr = merz_wuthrich(tri)
    ultimo = mack(tri)
    assert np.all(cdr.standard_error <= ultimo.standard_error + 1e-9)


def test_riesgo_a_un_anio_total_es_menor_que_a_ultimo():
    tri = taylor_ashe()
    assert merz_wuthrich(tri).total_standard_error < mack(tri).total_standard_error


def test_la_simulacion_solo_captura_proceso_y_queda_por_debajo():
    """La re-reserva con factores fijos es una cota inferior del CDR.

    Simular con los factores estimados como si fueran los verdaderos elimina la
    incertidumbre de parametro, asi que la formula cerrada debe dominar a la
    simulacion en todos los anios.
    """
    tri = taylor_ashe()
    sim = simulated_cdr(tri, n_sims=3000, seed=11)
    cerrada = merz_wuthrich(tri)
    assert np.all(cerrada.standard_error[1:] >= sim.standard_error[1:])
    # Y no deben estar lejos: la diferencia es solo el error de parametro.
    ratio = cerrada.total_standard_error / sim.total_standard_error
    assert 1.0 < ratio < 1.5


def test_riesgo_de_proceso_simulado_coincide_con_su_valor_analitico():
    """Para el anio 1 el riesgo de proceso tiene forma cerrada.

    Al quedar un solo paso de desarrollo, ``CDR_1 = U_1 - C_{1,J}`` y por tanto
    ``Var(CDR_1) = sigma^2_{J-1} C_{1,J-1}``.
    """
    tri = taylor_ashe()
    n = tri.n
    cum = tri.to_cumulative().values
    s2 = mack_sigma_squared(tri)
    esperado = float(np.sqrt(s2[n - 2] * cum[1, n - 2]))

    sim = simulated_cdr(tri, n_sims=20_000, seed=2024)
    assert sim.standard_error[1] == pytest.approx(esperado, rel=0.03)


def test_media_del_cdr_simulado_es_cercana_a_cero():
    """El CDR es una diferencia de martingala: su media debe anularse."""
    tri = taylor_ashe()
    sim = simulated_cdr(tri, n_sims=8000, seed=77)
    total = chain_ladder(tri).total_ibnr
    assert abs(float(np.sum(sim.mean_cdr))) < 0.02 * total


def test_el_primer_anio_no_tiene_riesgo_a_un_anio():
    cdr = merz_wuthrich(taylor_ashe())
    assert cdr.msep_by_origin[0] == pytest.approx(0.0)


def test_simulacion_reproducible():
    a = simulated_cdr(taylor_ashe(), n_sims=300, seed=1234)
    b = simulated_cdr(taylor_ashe(), n_sims=300, seed=1234)
    np.testing.assert_allclose(a.msep_by_origin, b.msep_by_origin)
