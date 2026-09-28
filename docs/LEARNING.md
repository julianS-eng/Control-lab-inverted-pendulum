# Guía de aprendizaje — Control Lab: Inverted Pendulum

Este documento acompaña al código. Explica la teoría paso a paso, **por qué** se tomó
cada decisión de diseño, qué alternativas se probaron y se descartaron, y cierra con
10 preguntas de entrevista técnica con sus respuestas.

> Todas las cifras que aparecen aquí se refieren a las tablas del
> [README](../README.md), que se generan automáticamente a partir de
> [`docs/results/results.json`](results/results.json). Si cambias el código, vuelve a
> ejecutar `pendulum-lab all` y las tablas se actualizan solas.

---

## 1. El sistema físico

Un carro de masa `M` se desplaza sobre un riel horizontal empujado por una fuerza `u`.
Sobre el carro gira libremente una varilla uniforme de masa `m`, con su centro de masa
a una distancia `l` del pivote y momento de inercia `J = m(2l)²/12` respecto a su
centro. Hay fricción viscosa en el carro (`b_c`) y en el pivote (`b_p`).

Convenciones (importantes para no equivocarse de signo):

- Estado `s = [x, ẋ, θ, θ̇]`.
- `θ` se mide **desde la vertical hacia arriba**; `θ > 0` significa que la punta del
  péndulo se inclina hacia `+x`.
- `θ = 0` es el equilibrio invertido (inestable); `θ = π` es el colgante (estable).

## 2. Modelo no lineal con Lagrange

1. **Posición del centro de masa del péndulo**: `(x + l sen θ, l cos θ)`.
2. **Energía cinética**:
   `T = ½(M+m)ẋ² + m l cos θ ẋ θ̇ + ½(J + m l²) θ̇²`.
   El término cruzado `m l cos θ ẋ θ̇` es el acoplamiento carro–péndulo.
3. **Energía potencial**: `V = m g l cos θ`.
4. **Ecuaciones de Euler–Lagrange** `d/dt(∂L/∂q̇) − ∂L/∂q = Q` con fuerzas
   generalizadas `Q_x = u − b_c ẋ` y `Q_θ = −b_p θ̇`:

```
(M+m) ẍ + m l cos θ θ̈ − m l sen θ θ̇² = u − b_c ẋ
m l cos θ ẍ + (J + m l²) θ̈ − m g l sen θ = −b_p θ̇
```

5. Se escribe como `M(θ) q̈ = f(q, q̇, u)` y se resuelve el sistema 2×2 en cada
   evaluación. La matriz de masa es siempre invertible porque su determinante
   `(M+m)(J+ml²) − (ml cos θ)² ≥ M m l² + (M+m)J > 0`.

**Verificación**: el test `test_frictionless_energy_is_conserved` comprueba que, sin
fricción, la energía total se conserva con el integrador RK4, y
`test_partial_feedback_linearization_achieves_requested_acceleration` comprueba la
inversión del modelo usada por el swing-up.

## 3. Linealización

Alrededor de `θ = 0` (`sen θ ≈ θ`, `cos θ ≈ 1`, términos cuadráticos ≈ 0), con
`I = J + ml²` y `D = (M+m)I − (ml)²`:

```
A = [0      1          0              0      ]     B = [ 0    ]
    [0  −I b_c/D   −(ml)² g/D     ml b_p/D  ]         [ I/D  ]
    [0      0          0              1      ]         [ 0    ]
    [0  ml b_c/D   (M+m) m g l/D  −(M+m)b_p/D]         [−ml/D ]
```

Datos clave que salen del modelo (ver tabla *Structural analysis* del README):

- Un polo real positivo (`≈ +5.5 rad/s`): el péndulo se cae con una constante de
  tiempo de unos 180 ms. Cualquier controlador necesita un ancho de banda mayor.
- Un polo en `0`: la posición del carro es un integrador puro (el sistema no "sabe"
  dónde está el centro del riel).
- La función de transferencia `x/u` tiene un **cero en el semiplano derecho**
  (`≈ +4.95 rad/s`). Consecuencia práctica: para mover el carro hacia delante, primero
  hay que moverlo un poco hacia atrás para inclinar el péndulo (se ve como
  *undershoot* en la respuesta al escalón). Este cero limita el ancho de banda del
  lazo de posición.

La linealización se valida numéricamente contra diferencias finitas (tests) y
dinámicamente contra el modelo no lineal (figura `open_loop.png`: el periodo de
oscilación colgante coincide para 0.1 rad pero difiere claramente para 1 rad).

## 4. Controlabilidad y observabilidad

- **Controlabilidad**: `rank [B, AB, A²B, A³B] = 4` → con una sola fuerza se pueden
  llevar los cuatro estados a donde se quiera. Se verifica también con el test PBH
  (Popov–Belevitch–Hautus), que identifica *qué* modo falla si falla alguno.
- **Observabilidad**, dependiendo del sensor:
  - `x` y `θ`: rango 4 (lo que usamos).
  - solo `x`: rango 4 — sorprendente pero cierto: el movimiento del carro "delata" el
    ángulo a través del acoplamiento. En la práctica está peor condicionado.
  - solo `θ`: rango 3. El modo no observable es el polo en `0` (la posición
    absoluta del carro). **Lección**: un encoder en el péndulo no basta; sin un
    sensor de posición del carro, el carro deriva sin que nadie lo note.
- El número de condición de la matriz de observabilidad y el Gramiano de
  controlabilidad a 1 s (autovalores mínimo/máximo) dicen *cuánto* cuesta controlar u
  observar cada dirección, no solo *si* se puede.

## 5. Implementación discreta

Un controlador real corre en un microcontrolador a periodo fijo `Ts = 10 ms`:

1. **Discretización ZOH exacta**: `Ad = e^{A Ts}`, `Bd = ∫ e^{As} ds B`, calculadas con
   una única exponencial de la matriz bloque `[[A, B], [0, 0]]`.
2. **Diseño directo en discreto** (DARE, `place_poles` sobre `(Ad, Bd)`), en vez de
   diseñar en continuo y "emular". El estudio *Sampling* del README muestra que la
   emulación se vuelve inestable antes que el diseño discreto al aumentar `Ts`.
3. **Retardo de cómputo** de 1 muestra: el comando calculado en `k` se aplica en
   `[t_{k+1}, t_{k+2})`. Se modela con una línea de retardo y el filtro de Kalman se
   propaga con la fuerza *realmente aplicada* (que el controlador conoce porque la
   emitió).
4. **Compensación de retardo**: se aumenta el estado con los comandos "en vuelo"
   `z = [x; u_{k−d}; …; u_{k−1}]` y se diseña un LQR sobre ese sistema. El resultado es
   equivalente a un predictor de Smith discreto / asignación de espectro finito: los
   polos de lazo cerrado son los del LQR nominal más `d` polos en el origen (lo
   verifica `test_delay_compensated_lqr_places_design_poles_plus_zeros`).
5. **Saturación** de ±10 N y **anti-windup** por integración condicional en el PID.
6. **Planta integrada con RK4 a 4 sub-pasos por periodo**: la física continua se
   resuelve más fino que la tasa del controlador (el error de energía en 10 s sin
   fricción es del orden de 1e-11 J).

## 6. Controladores

### 6.1 PID en cascada

- **Lazo interno** (rápido, ángulo): `u = Kp(θ − θ_ref) + Ki∫(θ − θ_ref) + Kd θ̇_f`.
  El signo es *positivo* en `θ`: si el péndulo se inclina hacia delante, hay que
  empujar el carro hacia delante.
- **Lazo externo** (lento, posición): `θ_ref = Kp_x (r − x) − Kd_x ẋ_f`, recortado a
  ±0.2 rad. Con el lazo interno cerrado, `ẍ ≈ g θ_ref`, así que el lazo externo ve un
  doble integrador.
- Derivada **sobre la medición** (sin "patada derivativa" al cambiar la referencia) y
  filtrada con `s/(τs+1)`.
- **Decisión**: el lazo externo no tiene integral. Se probó con `Ki_x > 0` y el
  resultado fue peor (sobreimpulso por windup durante el escalón) sin ningún
  beneficio: la integral del lazo interno ya garantiza error nulo en posición ante
  fuerzas constantes, porque en equilibrio el péndulo debe estar vertical
  (`θ = θ_ref = 0` ⇒ `x = r`).
- **Sintonía**: búsqueda aleatoria de ganancias sobre el modelo LTI exacto del lazo
  (con retardo), filtrando por márgenes mínimos (GM ↓ < −6 dB, GM ↑ > 6 dB,
  PM > 35°), y luego refinamiento del lazo externo en simulación no lineal
  minimizando tiempo de asentamiento y sobreimpulso.
- El PID expone `to_lti()`, que construye el modelo de estado **sondeando la propia
  función de actualización** con vectores unitarios. Así el modelo de análisis y el
  código que se simula son, por construcción, las mismas ecuaciones (test
  `test_pid_lti_matches_implementation`).

### 6.2 Asignación de polos

Polos continuos elegidos: un par dominante `ζ = 0.8, ωn = 2.5 rad/s` para el carro,
el **reflejo del polo inestable** (`−5.5 rad/s`, que es la elección de mínima energía
para estabilizar ese modo) y un polo rápido en `−8 rad/s`. Se mapean con
`z = e^{sTs}` y se colocan con el algoritmo robusto de Kautsky–Nichols–Van Dooren.

### 6.3 LQR y la justificación de Q y R

Se usa la **regla de Bryson**: cada variable se pondera con el inverso del cuadrado
de su excursión aceptable, de forma que el coste vale ≈1 cuando cualquier variable
llega a su límite:

| Variable | Límite | Razón |
|---|---|---|
| `x` | 0.25 m | un cuarto del semirriel (1 m) |
| `ẋ` | 1 m/s | velocidad cómoda para un riel de laboratorio |
| `θ` | 0.1 rad (5.7°) | muy dentro de la zona lineal (sen θ ≈ θ con error < 0.2 %) |
| `θ̇` | 1.5 rad/s | limita la agresividad del lazo de ángulo |
| `u` | 3 N | menos de un tercio de la saturación de 10 N, deja margen para rechazar perturbaciones |

La justificación no es solo "de libro": el barrido de `R → ρR` (figura
`lqr_tradeoff.png`) muestra que con `ρ < 1` el controlador satura en el escalón, el
ruido de sensor se amplifica en la fuerza (RMS en reposo mucho mayor) y el margen de
fase cae; con `ρ > 1` el sistema se vuelve lento. `ρ = 1` está en el codo de la curva.

### 6.4 Swing-up por energía + LQR

1. Energía del péndulo con cero arriba: `E = ½ I θ̇² + m g l (cos θ − 1)`.
2. De la ecuación del péndulo: `Ė = −m l cos θ θ̇ ẍ − b_p θ̇²`.
3. Ley de Åström–Furuta: `ẍ = sat(k_E (E − E_ref) · sign(θ̇ cos θ)) − k_x x − k_v ẋ`.
   Con esto `Ė ∝ −(E − E_ref)|θ̇ cos θ|`: la energía converge monótonamente.
4. La aceleración deseada se convierte en fuerza **invirtiendo el modelo nominal**
   (linealización por realimentación parcial colocada, Spong 1995).
5. **Conmutación con histéresis**: se pasa al LQR cuando `|θ| < 0.35 rad` y
   `|θ̇| < 2.5 rad/s`; se vuelve al swing-up si `|θ| > 0.9 rad`.
6. **Robustez**: la ley regula la energía *del modelo nominal*. Si el péndulo real es
   más corto, la energía nominal no alcanza para llegar arriba. Se añadió una
   **adaptación por oscilación**: si el péndulo da la vuelta en la mitad superior sin
   entrar en la ventana de captura, se sube el objetivo de energía en el déficit
   observado `m g l (1 − cos θ)`; si pasa por arriba demasiado rápido, se baja.
   La tabla *Swing-up ablation* del README cuantifica la mejora.
7. **Signo con histéresis**: `sign(θ̇ cos θ)` se mantiene mientras el argumento esté
   dentro de la banda de ruido, lo que evita que la fuerza "castañetee" en reposo y
   a la vez garantiza el empujón inicial.

## 7. Estimación: filtro de Kalman y LQG

- **Modelo de ruido de proceso**: fuerza en el carro y par en el pivote como ruido
  blanco continuo, discretizado con el **método de Van Loan** (exacto, no la
  aproximación `Q·Ts`).
- **Forma "current estimator"**: predicción con la fuerza aplicada y corrección con
  la medida del mismo instante antes de calcular el control (no se pierde una
  muestra).
- **Actualización de Joseph** `P = (I−KC)P(I−KC)ᵀ + KRKᵀ`: numéricamente robusta,
  mantiene `P` simétrica y definida positiva.
- **Consistencia**: se reporta el NIS medio (debería ser 2 con dos medidas). Un valor
  menor indica que el filtro es conservador (supone más ruido de proceso del real),
  lo cual es deliberado para tolerar errores de modelo.
- **EKF para el swing-up**: el KF lineal se diseña alrededor de la vertical y es
  inútil cuando el péndulo da vueltas completas. La tabla *Estimator in the swing-up
  loop* muestra que el swing-up falla con KF lineal y funciona con EKF. El jacobiano
  del EKF se calcula por diferencias centrales **del mapa discreto RK4**, vectorizado
  para todo el lote.
- **Principio de separación**: los polos del lazo LQG son los del regulador más los
  del estimador (test `test_lqg_separation_principle`). Pero **no hay garantía de
  márgenes** (Doyle, 1978, *"Guaranteed margins for LQG regulators: there are
  none"*). La tabla de márgenes muestra cómo LQG pierde margen de ganancia superior
  frente al LQR con estado ideal.

## 8. Robustez

- **Márgenes clásicos** (lazo abierto en el actuador) calculados exactamente:
  intervalo de ganancia por barrido de autovalores + bisección, margen de fase en los
  cruces `|L| = 1`, y margen de retardo tanto estimado (`PM/ωc`) como exacto en
  muestras enteras (aumentando el retardo hasta perder estabilidad). Para una planta
  inestable en lazo abierto existe un **margen de ganancia inferior**: con muy poca
  autoridad de actuador el péndulo se cae.
- **Monte Carlo** con números aleatorios comunes (misma planta, mismas condiciones
  iniciales y mismo ruido para todos los controladores): la comparación es pareada y
  mucho más sensible que con muestras independientes. Intervalos de confianza de
  Wilson (se comportan bien cerca del 100 %, donde el intervalo normal falla).
- **Región de atracción** empírica en el plano `(θ₀, θ̇₀)` con saturación, retardo y
  ruido. Se inicializa el estimador con el estado verdadero para aislar la cuenca del
  lazo cerrado del transitorio del estimador.

## 9. Decisiones de ingeniería de software

- **Todo vectorizado por lotes**: la dinámica, los controladores, el KF y el EKF
  aceptan `B` lazos a la vez. Así un Monte Carlo de cientos de plantas o un mapa de
  3721 condiciones iniciales tarda segundos sin dependencias extra (sin numba).
- **Factories en `Strategy`**: cada simulación recibe objetos nuevos; no hay estado
  compartido accidental entre corridas.
- **Semillas fijas** (`DEFAULT_SEED = 20260928`) y resultados en JSON.
- **README generado**: las tablas del README se reemplazan entre marcadores
  `<!-- BEGIN GENERATED -->` desde el JSON; CI falla si el README no coincide con los
  resultados comprometidos.

## 10. Alternativas probadas y descartadas

| Alternativa | Por qué se descartó |
|---|---|
| Integral en el lazo externo del PID | Windup en escalones de referencia (sobreimpulsos del orden del 30 % en las pruebas de sintonía); la integral interna ya elimina el error estacionario. |
| Ruido de sensor σθ = 5 mrad | Con un LQR razonable el ruido dominaba la fuerza (RMS de varios N en reposo); se eligió σθ = 2 mrad (aún 4× peor que un encoder típico de 4096 cuentas). |
| Rampa temporal del objetivo de energía en el swing-up | Sobrecargaba de energía a las plantas nominales (vueltas completas y más fallos). Reemplazada por la adaptación por oscilación. |
| Captura del péndulo con velocidad angular de hasta 5 rad/s | El LQR saturaba y el carro se salía del riel tras la captura. Se bajó a 2.5 rad/s y la adaptación reduce la energía en los pasos demasiado rápidos. |
| Transferencia "sin salto" de la referencia del carro al capturar | Probado: sin efecto en la tasa de éxito (el problema era la velocidad angular en la captura, no la posición). Eliminado para no añadir complejidad. |
| Compuerta de convergencia de energía en la adaptación | Acelera la captura nominal pero reduce la robustez; se deja como opción (`adapt_tolerance`) y se cuantifica en la ablación. |
| KF lineal para el swing-up | Falla (ver tabla); se usa EKF. |
| Diseñar en continuo y emular | Pierde estabilidad antes al crecer `Ts` (tabla *Sampling*). |
| numba/JAX para acelerar | Innecesario: la vectorización por lotes en NumPy basta y evita dependencias. |

---

## 11. Diez preguntas de entrevista técnica (con respuestas)

**1. ¿Por qué el péndulo invertido es "difícil" de controlar, más allá de ser inestable?**
Tiene un polo inestable (`+5.5 rad/s`) *y* un cero de fase no mínima en `x/u`
(`+4.95 rad/s`) muy cercano. El polo exige un ancho de banda mínimo; el cero impone un
máximo al lazo de posición. Con ambos tan próximos, el margen de diseño es estrecho;
además es subactuado (una entrada, dos grados de libertad).

**2. ¿Qué pasa si solo mides el ángulo del péndulo?**
El sistema pierde observabilidad: rango 3 de 4. El modo no observable es la posición
del carro (polo en 0). Se puede estabilizar el péndulo, pero el carro deriva sin que
el controlador lo sepa y acaba chocando contra el final del riel.

**3. ¿Cómo eliges Q y R en un LQR sin que sea "a ojo"?**
Regla de Bryson como punto de partida (inverso del cuadrado de la excursión aceptable
de cada variable), y luego un barrido de `ρ` en `R → ρR` evaluando tiempo de
asentamiento, esfuerzo, saturación, ruido en la fuerza y margen de fase. Se elige el
codo de la curva de Pareto, no el mínimo de un solo criterio.

**4. El LQR continuo garantiza 60° de margen de fase. ¿Por qué tus márgenes son menores?**
La garantía es para realimentación de estado *continua* y *sin retardo*. En discreto
con retardo de una muestra se pierde fase (`ωTs` rad por muestra), y con un
observador (LQG) no hay ninguna garantía (Doyle 1978). Por eso se calculan márgenes
del lazo real, no se suponen.

**5. ¿Qué es un "current estimator" y por qué usarlo?**
Es la forma del KF en la que la medida `y_k` corrige la predicción *antes* de calcular
`u_k`. La forma "predictor" usa `x̂_{k|k−1}` y añade efectivamente una muestra de
retardo al lazo. Con retardo de actuador ya presente, no conviene perder otra muestra.

**6. ¿Cómo compensas un retardo de actuador conocido?**
Aumentando el estado con los comandos en vuelo y diseñando el LQR sobre el sistema
aumentado. La ganancia "predice" dónde estará el sistema cuando llegue el comando. Con
el modelo exacto, los polos de lazo cerrado son los nominales más `d` polos en cero.
En el estudio de retardo, esta variante es la única que sigue estable a retardos
grandes en el lazo lineal nominal; con incertidumbre de parámetros también degrada,
pero mucho más tarde.

**7. ¿Por qué un EKF para el swing-up y no un KF lineal?**
El KF lineal usa `A` linealizada en `θ = 0`; cuando `θ ≈ π` el signo de la gravedad en
el modelo es el opuesto al real, así que la predicción es sistemáticamente errónea y
las velocidades estimadas son basura. El EKF re-linealiza en cada muestra alrededor
de la estimación actual.

**8. ¿Cómo sabes si tu filtro de Kalman está bien sintonizado?**
Con pruebas de consistencia: el NIS (innovación normalizada al cuadrado) debe tener
media igual al número de medidas (2). NIS ≫ 2 indica que el filtro es demasiado
confiado (Q o R demasiado pequeños); NIS < 2, que es conservador. También el NEES
cuando se conoce el estado real (en simulación).

**9. ¿Qué es el anti-windup y cuál implementaste?**
Cuando el actuador satura, el integrador sigue acumulando error y, al salir de la
saturación, produce un sobreimpulso enorme. Se implementó integración condicional:
se congela el integrador si la salida está saturada *y* el error empujaría más hacia
la saturación. Alternativas: back-calculation (realimentar `u_sat − u` al integrador)
o limitar el integrador.

**10. ¿Por qué usar números aleatorios comunes en el Monte Carlo?**
Porque comparamos controladores, no estimamos una tasa absoluta. Si cada controlador
recibe las mismas plantas, condiciones iniciales y ruido, la diferencia entre tasas
de éxito se debe al controlador y no a la suerte del muestreo; la varianza de la
diferencia se reduce drásticamente. Los intervalos se reportan con Wilson, que es
válido cerca de 0 % y 100 %.
