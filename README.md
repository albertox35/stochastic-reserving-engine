# stochastic-reserving-engine

Reservas estocásticas de siniestros: métodos No-Vida sobre triángulos de
desarrollo y modelos multiestado de Markov para Vida y Salud.

La validación es el eje del repositorio. El chain-ladder y el error de predicción
de Mack **reproducen exactamente** las cifras publicadas para el triángulo de
Taylor-Ashe, y la fórmula cerrada de riesgo a un año se fija contra una condición
de frontera exacta en lugar de darse por buena.

```python
from reserving import taylor_ashe
from reserving.nonlife import chain_ladder, mack, odp_bootstrap, merz_wuthrich

triangulo = taylor_ashe()

print(chain_ladder(triangulo).total_ibnr)          # 18 680 856
print(mack(triangulo).total_standard_error)        #  2 447 095
print(merz_wuthrich(triangulo).total_standard_error)  # 1 724 910

dist = odp_bootstrap(triangulo, n_sims=10_000, seed=20260101)
print(dist.quantile(0.995))                        # 28 048 129
```

---

## Qué hace

**No-Vida** (`reserving.nonlife`), sobre triángulos de desarrollo:

| Método | Qué entrega |
|---|---|
| `chain_ladder` | Reserva puntual, factores ponderados por volumen, triángulo completado |
| `mack` | MSEP analítico descompuesto en error de proceso y de parámetro, con término cruzado |
| `odp_bootstrap` | Distribución predictiva completa, para percentiles y margen de riesgo |
| `bornhuetter_ferguson` | Reserva basada en un *a priori*, para los años recientes poco desarrollados |
| `merz_wuthrich` | Riesgo de reserva a un año (CDR), fórmula cerrada |
| `simulated_cdr` | Riesgo a un año por re-reserva simulada, solo proceso |

**Vida y Salud** (`reserving.life`), modelos multiestado:

- Cadena de Markov en tiempo continuo con generador dependiente del tiempo.
- Probabilidades de transición por integración de la ecuación de Kolmogorov
  progresiva con Runge-Kutta de cuarto orden.
- Reserva prospectiva con prestaciones en forma de renta por estado y capitales
  pagaderos al momento de la transición.
- Generador prearmado del modelo clásico de invalidez (activo / inválido /
  fallecido), con recuperación opcional.

---

## Resultados de validación

Triángulo de Taylor-Ashe (1983), 10 × 10. Semilla `20260101`, 10 000 simulaciones.

| Cantidad | Obtenido | Referencia | Coincide |
|---|---|---|---|
| Reserva chain-ladder | 18 680 856 | 18 680 856 (Mack 1993) | exacto |
| Error estándar de Mack, total | 2 447 095 | 2 447 095 (Mack 1993) | exacto |
| Error estándar de Mack, por año | 9 valores | tabla publicada de Mack | exacto, los 9 |
| CV del bootstrap ODP | 15.8 % | ≈ 16 % (`BootChainLadder`, triángulo `GenIns`) | sí |
| MSEP del CDR, año más antiguo | = MSEP a último de Mack | identidad exacta | exacto |

Riesgo a un año frente a riesgo a último: **1 724 910 contra 2 447 095**, es decir
el 70.5 %. Que el riesgo a un año sea sustancialmente menor es lo esperado y es la
razón por la que confundirlos sobrestima el capital.

El bootstrap ODP da un error de predicción de 2 986 829 (CV 15.8 %) frente a los
2 447 095 de Mack. **No es una discrepancia**: los dos modelos suponen relaciones
media-varianza distintas —`Var ∝ C` en Mack, `Var ∝ media` en ODP— y por tanto no
tienen por qué coincidir. Esperar que lo hagan es un malentendido habitual.

---

## Qué no hace

Declarado explícitamente, porque un alcance inflado es más dañino que uno
limitado:

- **No hay ejecución distribuida.** El bootstrap de 10 000 iteraciones sobre un
  triángulo 10 × 10 corre en menos de tres segundos en un solo núcleo. Añadir MPI
  o Spark a un problema de este tamaño sería una decisión arquitectónica
  equivocada: el problema es *embarrassingly parallel* y no está limitado por
  cómputo a esta escala. Para triángulos grandes o muchas líneas de negocio, el
  bucle de `odp_bootstrap` es paralelizable con `joblib` sin cambios de diseño.
- **No calcula el RCS ni el SCR completos.** Entrega el riesgo de reserva, que es
  un componente del módulo de suscripción. Agregarlo al requerimiento de capital
  exige la matriz de correlación del régimen correspondiente, que está fuera del
  alcance.
- **El bootstrap de Pearson no admite incrementales negativos grandes.** Si el
  ajuste ODP produce celdas no positivas, `pearson_residuals` lanza una excepción
  explícita en lugar de devolver un número silenciosamente inválido. Para
  triángulos con salvamentos y recobros fuertes hace falta otro modelo.
- **`simulated_cdr` mide solo riesgo de proceso.** Al simular con los factores
  estimados como si fueran los verdaderos, la incertidumbre de parámetro no entra
  por construcción. Es una cota inferior, y está documentado como tal porque es el
  malentendido más frecuente del método *actuary in the box*. Para capital hay que
  usar `merz_wuthrich`.
- **Sin tablas de mortalidad precargadas.** El módulo de Vida toma intensidades
  como funciones de `t`; no incluye tablas de experiencia mexicanas ni de otros
  mercados.

---

## Alcance y decisiones de diseño

**Markov está en Vida, no en No-Vida.** Es la corrección más importante respecto
del diseño inicial. Las cadenas de Markov no son el modelo de reservas No-Vida:
ahí la cadena correcta es chain-ladder → Mack → bootstrap ODP → Merz-Wüthrich.
Los modelos multiestado describen contratos con estados (activo, inválido,
fallecido) e intensidades de transición, que es otra rama. Mezclarlos produce un
repositorio que no hace bien ninguna de las dos cosas, así que están en submódulos
separados y sin dependencias cruzadas.

**Chain-ladder solo con promedio ponderado por volumen.** Es el único estimador
para el que Mack deriva el error de predicción analítico. Ofrecer promedios simples
o de los últimos *k* años en la misma función invitaría a combinarlos con `mack()`,
lo que invalidaría las fórmulas sin aviso.

**El repositorio es autocontenido.** No depende de una librería núcleo compartida
con los demás repositorios del conjunto. La contrapartida es que hay duplicación
numérica entre repositorios; la ventaja es que este se clona, se audita y se
evalúa por sí solo. Es una decisión consciente, no un descuido.

**Determinismo.** Toda función estocástica recibe una semilla y es reproducible bit
a bit. No es cosmética: es requisito para que un cálculo de reservas sea auditable.

---

## Marco regulatorio

Las cantidades que calcula este repositorio corresponden a:

- **Solvencia II**, Directiva 2009/138/CE, con las enmiendas del *Solvency II
  Review* introducidas por la **Directiva (UE) 2025/2** (transposición prevista
  para enero de 2027). El riesgo de reserva a un año es la base del submódulo de
  riesgo de primas y reservas.
- **LISF y Circular Única de Seguros y Fianzas (CUSF)** de la CNSF para México,
  donde la mejor estimación más el margen de riesgo conforma la reserva de
  siniestros pendientes y el RCS se calcula por fórmula general.

No existe, ni está anunciada, una «Solvencia III». Si alguna documentación
anterior de este proyecto la mencionaba, era un error.

---

## Instalación

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

Requiere Python 3.11 o superior, NumPy, SciPy y pandas.

---

## Referencias

- Mack, T. (1993). *Distribution-free calculation of the standard error of chain
  ladder reserve estimates*. ASTIN Bulletin 23(2), 213-225.
- England, P. & Verrall, R. (2002). *Stochastic claims reserving in general
  insurance*. British Actuarial Journal 8(3), 443-518.
- Merz, M. & Wüthrich, M. (2008). *Modelling the claims development result for
  solvency purposes*. CAS E-Forum, Fall 2008, 542-568.
- Taylor, G. & Ashe, F. (1983). *Second moments of estimates of outstanding
  claims*. Journal of Econometrics 23(1), 37-61.
- Hoem, J. (1969). *Markov chain models in life insurance*. Blätter der DGVFM 9(2).
- Wüthrich, M. & Merz, M. (2008). *Stochastic Claims Reserving Methods in
  Insurance*. Wiley.

## Licencia

MIT. Ver [LICENSE](LICENSE).
