"""Analisis completo del triangulo de Taylor-Ashe.

Ejecutar con:

    python examples/taylor_ashe.py
"""

from __future__ import annotations

import numpy as np

from reserving import taylor_ashe
from reserving.nonlife import (
    bornhuetter_ferguson,
    chain_ladder,
    mack,
    merz_wuthrich,
    odp_bootstrap,
    simulated_cdr,
)

SEED = 20260101


def main() -> None:
    tri = taylor_ashe()

    cl = chain_ladder(tri)
    print("=== Chain-ladder ===")
    print("factores:", np.round(cl.factors, 4))
    print(f"reserva total: {cl.total_ibnr:,.0f}")

    mk = mack(tri)
    print("\n=== Mack: error de prediccion a ultimo ===")
    print(f"error estandar total: {mk.total_standard_error:,.0f}")
    print(f"coeficiente de variacion: {mk.coefficient_of_variation:.1%}")
    print("\npor anio de origen:")
    print(f"{'anio':>5} {'reserva':>14} {'error est.':>13} {'CV':>7}")
    for i in range(tri.n):
        reserva = cl.ibnr[i]
        cv = mk.standard_error[i] / reserva if reserva > 0 else float("nan")
        print(f"{i:>5} {reserva:>14,.0f} {mk.standard_error[i]:>13,.0f} {cv:>7.1%}")

    boot = odp_bootstrap(tri, n_sims=10_000, seed=SEED)
    print("\n=== Bootstrap ODP: distribucion predictiva ===")
    print(f"sobredispersion phi: {boot.scale:,.1f}")
    print(f"media:               {boot.mean:,.0f}")
    print(f"error de prediccion: {boot.standard_error:,.0f}")
    print(f"CV:                  {boot.coefficient_of_variation:.1%}")
    for q in (0.50, 0.75, 0.95, 0.995):
        print(f"percentil {q:>6.1%}:    {float(boot.quantile(q)):,.0f}")

    mw = merz_wuthrich(tri)
    sim = simulated_cdr(tri, n_sims=10_000, seed=SEED)
    print("\n=== Riesgo de reserva a un anio ===")
    print(f"Merz-Wuthrich (proceso + parametro): {mw.total_standard_error:,.0f}")
    print(f"re-reserva simulada (solo proceso):  {sim.total_standard_error:,.0f}")
    print(f"riesgo a ultimo (Mack):              {mk.total_standard_error:,.0f}")
    pct = 100.0 * mw.total_standard_error / mk.total_standard_error
    print(f"el riesgo a un anio es el {pct:.1f}% del riesgo a ultimo")

    print("\n=== Bornhuetter-Ferguson ===")
    primas = np.full(tri.n, 12_000_000.0)
    bf = bornhuetter_ferguson(tri, premium=primas, expected_loss_ratio=0.85)
    print(f"reserva BF total: {bf.total_ibnr:,.0f}")
    print(f"reserva CL total: {cl.total_ibnr:,.0f}")
    print("\nBF frente a CL en los anios recientes, donde CL esta mas apalancado:")
    for i in range(tri.n - 3, tri.n):
        print(
            f"  anio {i}: BF {bf.ibnr[i]:>12,.0f}   CL {cl.ibnr[i]:>12,.0f}"
            f"   desarrollado {bf.developed_fraction[i]:.1%}"
        )


if __name__ == "__main__":
    main()
