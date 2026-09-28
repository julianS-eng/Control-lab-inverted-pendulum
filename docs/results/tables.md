### structure

| Quantity | Value |
| :--- | ---: |
| Open-loop poles [rad/s] | -5.660, -0.143, 0.000, 5.515 |
| Unstable pole (upright) | +5.515 rad/s |
| RHP zero of x/u | +4.952 rad/s |
| Controllability rank (continuous / discrete) | 4 / 4 (of 4) |
| Controllability matrix condition number | 134 |
| 1 s Gramian eigenvalues (min / max) | 3.31e-05 / 279 |

| Measured output | Observability rank | Condition number | Unobservable modes |
| :--- | ---: | ---: | ---: |
| x and theta | 4 / 4 | 33.8 | none |
| x only | 4 / 4 | 3.11 | none |
| theta only | 3 / 4 | inf | 0 |

### linearization

| Check | Nonlinear | Linear |
| :--- | ---: | ---: |
| θ after 0.8 s from θ₀ = 0.02 rad (upright) [rad] | 0.7934 | 0.8352 |
| Hanging period, 0.1 rad swing [s] | 1.126 | 1.125 |
| Hanging period, 1.0 rad swing [s] | 1.221 | 1.125 |
| RK4 energy drift, 10 s frictionless [J] | 4.26e-11 | — |

### lqr

| Item | Value |
| :--- | ---: |
| Bryson limits (x, ẋ, θ, θ̇, u) | 0.25, 1.00, 0.10, 1.50, 3.00 |
| Q = diag(...) | 16, 1, 100, 0.444 |
| R | 0.1111 |
| K (u = −K x) | -10.63, -10.53, -53.80, -9.19 |
| Equivalent s-plane poles | -10.77+5.43j, -10.77-5.43j, -1.52+1.17j, -1.52-1.17j |

### lqr_sweep

| ρ (R → ρR) | Settling [s] | ∫u² [N²s] | Peak \|u\| [N] | Noise RMS u [N] | PM [deg] | Lower GM [dB] |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 8.52 | 293.6 | 10.0 | 5.938 | 17.0 | -21.4 |
| 0.0316228 | 7.70 | 131.6 | 10.0 | 3.846 | 21.1 | -19.1 |
| 0.1 | 7.70 | 47.0 | 10.0 | 2.205 | 28.0 | -16.4 |
| 0.316228 | 2.72 | 19.3 | 10.0 | 1.350 | 36.3 | -13.6 |
| 1 | 2.74 | 8.3 | 6.0 | 0.875 | 41.7 | -11.0 |
| 3.16228 | 3.30 | 4.0 | 3.4 | 0.611 | 43.7 | -8.8 |
| 10 | 3.56 | 2.3 | 1.9 | 0.463 | 44.4 | -7.3 |
| 31.6228 | 4.29 | 1.4 | 1.3 | 0.380 | 44.6 | -6.3 |
| 100 | 5.30 | 1.0 | 1.1 | 0.333 | 44.9 | -5.8 |

### balance

| Controller | Step settling [s] | Overshoot [%] | Undershoot [%] | ∫u² step [N²s] | Peak \|u\| [N] | Recovery settling [s] | Peak cart [m] | RMS θ at rest [mrad] | RMS u at rest [N] |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PID cascade | 2.47 | 2.6 | 7.0 | 2.6 | 2.7 | 4.96 | 0.22 | 1.85 | 0.429 |
| Pole placement + KF | 2.17 | 1.8 | 6.8 | 3.4 | 3.1 | 2.04 | 0.22 | 1.90 | 0.543 |
| LQR (ideal state) | 2.87 | 2.1 | 7.4 | 1.7 | 5.3 | 1.90 | 0.17 | 0.24 | 0.022 |
| LQG (LQR + KF) | 2.74 | 2.3 | 7.6 | 8.3 | 6.0 | 2.00 | 0.17 | 2.26 | 0.898 |
| LQG + delay comp. | 2.75 | 2.3 | 7.2 | 7.7 | 4.9 | 2.00 | 0.18 | 2.07 | 0.879 |

### margins

| Loop (broken at the actuator) | Gain margin ↓ [dB] | Gain margin ↑ [dB] | Phase margin [deg] | Crossover [rad/s] | Delay margin [ms] | Extra samples tolerated |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PID cascade | -5.9 | 10.7 | 39.4 | 14.8 | 46.5 | 4 |
| Pole placement + KF | -7.1 | 10.6 | 45.1 | 14.8 | 53.3 | 5 |
| LQR (ideal state) | -11.6 | 12.5 | 47.0 | 22.5 | 36.4 | 3 |
| LQG (LQR + KF) | -11.0 | 8.0 | 41.7 | 21.9 | 33.2 | 3 |
| LQG + delay comp. | -10.0 | 9.3 | 44.5 | 19.3 | 40.2 | 4 |

### kalman

| State | Raw measurement RMS error | Finite-difference RMS error | Kalman RMS error |
| :--- | ---: | ---: | ---: |
| x [mm] | 1.01 | — | 0.69 |
| ẋ [mm/s] | — | 140.3 | 25.4 |
| θ [mrad] | 1.96 | — | 1.44 |
| θ̇ [mrad/s] | — | 287.6 | 82.8 |

Mean normalised innovation squared (NIS): **1.52** (expected 2.00 for a perfectly tuned filter with two measurements).

### kalman_swingup

| Estimator in the swing-up loop | Swing-up succeeded | RMS θ error [rad] | RMS θ̇ error [rad/s] |
| :--- | ---: | ---: | ---: |
| linear KF | no | 0.030 | 3.988 |
| EKF | yes | 0.001 | 0.086 |

### swingup

| Metric | Value |
| :--- | ---: |
| Hand-over to LQR (catch time) | 4.07 s |
| Settled upright (\|θ\| < 0.05 rad, \|x\| < 5 cm) | 5.19 s |
| Peak cart excursion | 0.46 m |
| Peak force | 7.9 N |
| Control energy ∫u² dt | 45.7 N²s |
| Mode switches | 1 |
| GIF size | 0.51 MB |

### roa

| Controller | Recovered fraction of grid | Area [rad²/s] | Largest recoverable θ₀ at rest [deg] |
| :--- | ---: | ---: | ---: |
| PID cascade | 54.7 % | 8.15 | 30.9 |
| Pole placement + KF | 61.7 % | 9.18 | 34.4 |
| LQR (ideal state) | 63.6 % | 9.46 | 36.1 |
| LQG (LQR + KF) | 63.5 % | 9.45 | 36.1 |
| LQG + delay comp. | 63.3 % | 9.43 | 34.4 |

### monte_carlo

| Controller | Success, moderate | Success, severe | Success, extreme | Median settling, moderate [s] | Median ∫u², moderate [N²s] |
| :--- | ---: | ---: | ---: | ---: | ---: |
| PID cascade | 100.0 % [99.2, 100.0] | 99.0 % [97.7, 99.6] | 87.4 % [84.2, 90.0] | 9.17 | 14.9 |
| Pole placement + KF | 100.0 % [99.2, 100.0] | 100.0 % [99.2, 100.0] | 92.0 % [89.3, 94.1] | 7.82 | 14.2 |
| LQR (ideal state) | 100.0 % [99.2, 100.0] | 100.0 % [99.2, 100.0] | 99.8 % [98.9, 100.0] | 7.58 | 12.6 |
| LQG (LQR + KF) | 100.0 % [99.2, 100.0] | 100.0 % [99.2, 100.0] | 93.2 % [90.6, 95.1] | 7.69 | 21.5 |
| LQG + delay comp. | 100.0 % [99.2, 100.0] | 100.0 % [99.2, 100.0] | 95.8 % [93.7, 97.2] | 7.78 | 19.9 |
| Energy swing-up + LQR | 99.6 % [98.6, 99.9] | 89.8 % [86.8, 92.2] | 67.0 % [62.8, 71.0] | 5.61 | 53.3 |

*n = 500 sampled plants per level (common random numbers across controllers); brackets are Wilson 95 % confidence intervals.*

### uncertainty

| Level | Cart mass M | Pole mass m | Length l | Friction b_c, b_p |
| :--- | ---: | ---: | ---: | ---: |
| moderate | ±20 % | ±20 % | ±20 % | ×/÷ 2 |
| severe | ±40 % | ±50 % | ±40 % | ×/÷ 4 |
| extreme | ±60 % | ±70 % | ±60 % | ×/÷ 8 |

### swingup_ablation

| Swing-up variant | Success, moderate | Success, severe | Success, extreme |
| :--- | ---: | ---: | ---: |
| no adaptation | 65.4 % [61.1, 69.4] | 39.4 % [35.2, 43.7] | 36.8 % [32.7, 41.1] |
| adaptation (default) | 99.6 % [98.6, 99.9] | 89.8 % [86.8, 92.2] | 67.0 % [62.8, 71.0] |
| adaptation + convergence gate | 88.8 % [85.7, 91.3] | 60.8 % [56.5, 65.0] | 49.0 % [44.6, 53.4] |

### delay

| Controller | Largest stable delay, nominal linear loop [ms] | MC success at 30 ms (severe) | MC success at 50 ms (severe) |
| :--- | ---: | ---: | ---: |
| PID cascade | 50 | 80 % | 56 % |
| Pole placement + KF | 60 | 89 % | 65 % |
| LQR (ideal state) | 40 | 82 % | 18 % |
| LQG (LQR + KF) | 40 | 73 % | 30 % |
| LQG + delay comp. | ≥ 150 (max tested) | 95 % | 84 % |

### sampling

| Design (with one sample of computation delay) | Largest stable Ts [ms] |
| :--- | ---: |
| Continuous LQR, gains emulated | 30 |
| Discrete LQR (DARE on the ZOH model) | 40 |
| Discrete LQR + delay compensation | ≥ 150 (max tested) |

*Tested periods: 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120, 150 ms.*

### config

| Parameter | Value |
| :--- | ---: |
| Cart mass M | 0.5 kg |
| Pole mass m | 0.2 kg |
| Pivot to centre of mass l | 0.3 m (rod length 0.6 m) |
| Pole inertia about CoM J | 0.0060 kg·m² |
| Cart / pivot viscous friction | 0.1 N·s/m / 0.002 N·m·s/rad |
| Sampling period Ts | 10 ms (4 RK4 substeps) |
| Actuator | ±10.0 N, 1 sample(s) of delay |
| Sensor noise (1σ) | x: 1.0 mm, θ: 2.0 mrad |
| Random force disturbance (1σ) | 0.05 N |
| Rail half-length | 1.0 m |
