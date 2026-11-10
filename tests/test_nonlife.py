"""Validacion de los metodos No-Vida.

La prueba central contrasta contra **cifras publicadas**, no contra la propia
implementacion: el triangulo de Taylor-Ashe con los resultados de Mack (1993).
"""

from __future__ import annotations

import numpy as np
import pytest

from reserving import Triangle, taylor_ashe
from reserving.nonlife import (
    bornhuetter_ferguson,
    chain_ladder,
    development_factors,
    mack,
    mack_sigma_squared,
    odp_bootstrap,
    pearson_residuals,
)

# Cifras publicadas por Mack (1993) para el triangulo de Taylor-Ashe.
MACK_TOTAL_IBNR = 18_680_856
MACK_TOTAL_SE = 2_447_095
MACK_SE_BY_ORIGIN = [
    0,
    75_535,
    121_699,
    133_549,
    261_406,
    411_010,
    558_317,
    875_328,
    971_258,
    1_363_155,
]


# --------------------------------------------------------------------- triangulo


def test_triangle_rechaza_patron_invalido():
    bad = np.ones((3, 3))
    with pytest.raises(ValueError, match="patron de celdas observadas"):
        Triangle(bad)


def test_triangle_rechaza_no_cuadrado():
    with pytest.raises(ValueError, match="cuadrado"):
        Triangle(np.full((3, 4), np.nan))


def test_conversion_acumulado_incremental_es_involutiva():
    tri = taylor_ashe()
    ida_vuelta = tri.to_cumulative().to_incremental()
    np.testing.assert_allclose(ida_vuelta.values, tri.values, equal_nan=True)


def test_diagonal_corriente_coincide_con_suma_de_fila():
    tri = taylor_ashe()
    inc = tri.to_incremental().values
    for i in range(tri.n):
        fila = inc[i, : tri.n - i]
        assert tri.latest_diagonal[i] == pytest.approx(float(np.sum(fila)))


# ------------------------------------------------------------------ chain-ladder


def test_chain_ladder_reproduce_ibnr_publicado():
    res = chain_ladder(taylor_ashe())
    assert round(res.total_ibnr) == MACK_TOTAL_IBNR


def test_factores_de_desarrollo_son_mayores_que_uno_y_decrecen_en_tendencia():
    factors, _ = development_factors(taylor_ashe())
    assert np.all(factors > 1.0)
    # El patron global decrece, pero no de forma estrictamente monotona: en este
    # triangulo el factor 7->8 repunta sobre el 6->7. Exigir monotonia estricta
    # seria imponer al dato una suavidad que no tiene.
    assert factors[0] == factors.max()
    assert factors[0] > factors[-1]
    assert np.all(np.diff(factors[:6]) < 0.0)


def test_el_primer_anio_no_tiene_reserva():
    res = chain_ladder(taylor_ashe())
    assert res.ibnr[0] == pytest.approx(0.0)


def test_triangulo_completado_es_monotono_en_desarrollo():
    res = chain_ladder(taylor_ashe())
    assert np.all(np.diff(res.full_triangle, axis=1) > 0.0)


# -------------------------------------------------------------------------- Mack


def test_mack_reproduce_error_estandar_total_publicado():
    res = mack(taylor_ashe())
    assert round(res.total_standard_error) == MACK_TOTAL_SE


def test_mack_reproduce_error_estandar_por_anio_publicado():
    res = mack(taylor_ashe())
    obtenido = [round(x) for x in res.standard_error]
    assert obtenido == MACK_SE_BY_ORIGIN


def test_mack_total_excede_la_raiz_de_la_suma_de_cuadrados():
    # Los anios comparten los factores estimados: el termino cruzado del error de
    # parametro es positivo, asi que el total supera la agregacion independiente.
    res = mack(taylor_ashe())
    independiente = float(np.sqrt(np.sum(res.msep)))
    assert res.total_standard_error > independiente


def test_sigma_cuadrado_extrapolado_no_excede_a_los_anteriores():
    s2 = mack_sigma_squared(taylor_ashe())
    assert s2[-1] <= max(s2[-3], s2[-2]) + 1e-9


def test_mack_requiere_cuatro_anios():
    tri = Triangle.from_rows([[1.0, 2.0, 3.0], [1.0, 2.0], [1.0]], cumulative=False)
    with pytest.raises(ValueError, match="al menos 4 anios"):
        mack(tri)


# -------------------------------------------------------------- bootstrap ODP


def test_residuos_de_pearson_tienen_media_casi_nula():
    # Los residuos de Pearson no estan estandarizados: su escala es sqrt(phi),
    # del orden de 230 en este triangulo. La media debe ser pequenia *en relacion
    # con esa escala*, no en valor absoluto.
    resid, _, _ = pearson_residuals(taylor_ashe())
    error_estandar_de_la_media = float(np.std(resid, ddof=1)) / np.sqrt(resid.size)
    assert abs(float(np.mean(resid))) < 0.5 * error_estandar_de_la_media


def test_escala_de_sobredispersion_es_positiva():
    _, phi, _ = pearson_residuals(taylor_ashe())
    assert phi > 0.0


def test_bootstrap_es_reproducible_con_la_misma_semilla():
    a = odp_bootstrap(taylor_ashe(), n_sims=250, seed=4242)
    b = odp_bootstrap(taylor_ashe(), n_sims=250, seed=4242)
    np.testing.assert_array_equal(a.reserves, b.reserves)


def test_bootstrap_media_cercana_al_estimador_puntual():
    res = odp_bootstrap(taylor_ashe(), n_sims=2000, seed=99)
    assert res.mean == pytest.approx(res.point_estimate, rel=0.03)


def test_bootstrap_percentiles_ordenados():
    res = odp_bootstrap(taylor_ashe(), n_sims=1500, seed=5)
    q = res.quantile([0.5, 0.75, 0.995])
    assert q[0] < q[1] < q[2]


def test_el_error_de_proceso_aumenta_la_dispersion():
    # Desactivar el error de proceso deja solo el de estimacion: la distribucion
    # resultante tiene que ser estrictamente mas estrecha.
    con = odp_bootstrap(taylor_ashe(), n_sims=1500, seed=3, process_error=True)
    sin = odp_bootstrap(taylor_ashe(), n_sims=1500, seed=3, process_error=False)
    assert sin.standard_error < con.standard_error


def test_bootstrap_reservas_por_anio_suman_el_total():
    res = odp_bootstrap(taylor_ashe(), n_sims=200, seed=8)
    np.testing.assert_allclose(res.reserves_by_origin.sum(axis=1), res.reserves, rtol=1e-9)


def test_cv_del_bootstrap_en_rango_publicado():
    # El bootstrap ODP sobre este triangulo da un coeficiente de variacion en
    # torno al 16 %, consistente con BootChainLadder del paquete R ChainLadder
    # sobre el mismo conjunto (GenIns).
    res = odp_bootstrap(taylor_ashe(), n_sims=4000, seed=20260101)
    assert 0.14 < res.coefficient_of_variation < 0.18


# ------------------------------------------------------- Bornhuetter-Ferguson


def test_bf_con_a_priori_igual_al_chain_ladder_reproduce_su_reserva():
    # Si el ultimo a priori coincide con el del chain-ladder, BF devuelve la
    # misma reserva: es la propiedad que fija la escala del metodo.
    tri = taylor_ashe()
    cl = chain_ladder(tri)
    res = bornhuetter_ferguson(tri, premium=cl.ultimate, expected_loss_ratio=1.0)
    np.testing.assert_allclose(res.ibnr, cl.ibnr, rtol=1e-9)
    assert res.total_ibnr == pytest.approx(cl.total_ibnr, rel=1e-9)


def test_bf_reserva_crece_con_el_a_priori():
    tri = taylor_ashe()
    primas = np.full(tri.n, 10_000_000.0)
    bajo = bornhuetter_ferguson(tri, primas, 0.6)
    alto = bornhuetter_ferguson(tri, primas, 0.9)
    assert alto.total_ibnr > bajo.total_ibnr


def test_bf_fraccion_desarrollada_entre_cero_y_uno_y_creciente():
    res = bornhuetter_ferguson(taylor_ashe(), np.full(10, 1e7), 0.7)
    assert np.all(res.developed_fraction > 0.0)
    assert np.all(res.developed_fraction <= 1.0 + 1e-12)
    # El anio mas antiguo esta completamente desarrollado; los recientes menos.
    assert res.developed_fraction[0] == pytest.approx(1.0)
    assert np.all(np.diff(res.developed_fraction) < 0.0)


def test_bf_valida_longitud_de_primas():
    with pytest.raises(ValueError, match="premium debe tener"):
        bornhuetter_ferguson(taylor_ashe(), np.ones(3), 0.7)
