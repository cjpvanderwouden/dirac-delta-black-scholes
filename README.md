# Dirac-BS: Event-Driven Option Pricing via PDE Forcing

Code repository for **"Black-Scholes with Impulse Forcing: A PDE Framework with Dirac Delta Forcing"** (van der Wouden, 2025).

## Overview

This framework treats scheduled market events (earnings, FOMC decisions, etc.) as Dirac delta forcing terms in the Black-Scholes PDE rather than modifying the underlying stochastic process:

$$\mathcal{L}_{BS} V(S,t) = \sum_{j=1}^{N} \alpha_j(S) \delta(t - t_j)$$

Key results:
- **Equivalence theorem**: Dirac forcing ≡ jump conditions at event times
- **Greek discontinuities**: $\Delta\Delta = \alpha'(S)$, $\Delta\Gamma = \alpha''(S)$ (Theorems 5.2, 5.4)
- **Explicit propagation**: Closed-form shock contribution via Green's functions

## Requirements

- Python 3.8+
- NumPy ≥ 2.0
- SciPy ≥ 1.7
- Matplotlib ≥ 3.5

## Repository Structure

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `core.py` | BS analytics, shock classes, propagation formulas | Sections 3–5 |
| `solver.py` | Crank-Nicolson PDE solver with shock injection | Section 7 |
| `run_convergence.py` | Spatial convergence + Greek identity validation | Figures 2, 4 |
| `run_calibration.py` | Tikhonov-regularized shock recovery | Figure 1 |
| `run_hedging.py` | MTM hedging comparison (BS vs shock-aware) | Figure 5 |

## Shock Types

| Type | Specification | Use Case |
|------|--------------|----------|
| I | $\alpha(S) = A$ | Uniform repricing (systemic events) |
| II | $\alpha(S) = \beta S$ | Proportional gaps (earnings) |
| III | $\alpha(S) = \gamma \exp\left(-\frac{(\log S/K)^2}{2\xi^2}\right)$ | ATM-localized (vol events) |
| IV | $\alpha(S) = \eta \cdot \nu_{BS}(S)$ | Vega-proportional (IV crush) |

## License

MIT
