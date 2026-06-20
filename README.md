  # Non-Linear Covariant Scalar Field & Self-Interacting Dark Matter (SIDM) Solver

[![License: MIT](https://shields.io)](https://opensource.org)
[![Physics: Astro](https://shields.io)]()

## Theoretical Overview
This repository contains a high-performance computational astrophysics pipeline designed to simulate the non-linear cosmic evolution of non-baryonic dark matter candidates. 

The core engine solves the coupled **Schrödinger-Poisson** and **Vlasov-Poisson equations** within an expanding Friedmann-Lemaître-Robertson-Walker (FLRW) metric. By utilizing pseudospectral methods and a 4th-order Runge-Kutta-Fehlberg (RKF45) time-integration scheme, the simulation models dark matter dynamics, including Bohmian quantum pressure tensors and self-interacting dark matter (SIDM) damping mechanisms.

## Architecture
The project is built using a highly scalable, modular software architecture following modern HPC (High-Performance Computing) standards:

1. `simulation.py`: The core physics engine. Handles spectral Laplace operators via Fast Fourier Transforms (FFT), gravitational potential inversion, and differential cosmic time-stepping.
2. `main.py`: The executive pipeline controller. Initializes the cosmological grid configuration, executes the numerical integration loop, and handles server-side diagnostics validation.

## Prerequisites
To execute this computational pipeline on your server or cluster, you require Python 3.8+ along with the following scientific computing libraries:

```bash
pip install numpy scipy
```

## Execution & Server Validation
To initialize the cosmological grid (128³ resolution) and execute the non-linear integration loop, run the executive core script:

```bash
python main.py
```

### Expected Server Diagnostics Output
Upon successful validation, the HPC logging framework will output real-time metric scaling and mass conservation integrity logs:

```text
[MAIN-INTERFACE] Systeemcontrole voltooid. Starten van de Quantum Dark Matter Pipeline...
[COSMO-ENGINE] Physics Engine geïnitialiseerd. Grid cellen: 2097152
[COSMO-ENGINE] Iniciële kosmologische condities genereren via Gaußiaans wit-ruis spectrum...
[COSMO-ENGINE] Stap 0001/0050 | Scale Factor a: 0.01100 | Massa-Behoud: 2.1482e+06 | Systeementropie: -5.1242
...
[MAIN-INTERFACE] Simulatie-fase succesvol afgerond op de clusterserver.
PIPELINE EXECUTION STATUS: SUCCESSFUL VALIDATION
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

