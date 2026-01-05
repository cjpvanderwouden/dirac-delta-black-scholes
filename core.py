"""
Core analytics for Dirac-BS framework.
Reference: van der Wouden (2025), Sections 3-5.
"""
import numpy as np
from scipy.stats import norm


# --- Black-Scholes analytics ---

def bs_price(S, K, T, r, sigma):
    """Call price. Handles T→0."""
    S = np.atleast_1d(S).astype(float)
    T = float(T)
    
    if T <= 1e-12:
        out = np.maximum(S - K, 0.0)
    else:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        out = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    return out.item() if out.size == 1 else out


def bs_delta(S, K, T, r, sigma):
    """N(d1). Handles T→0."""
    S = np.atleast_1d(S).astype(float)
    T = float(T)
    
    if T <= 1e-12:
        out = np.where(S > K, 1.0, 0.0)
    else:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        out = norm.cdf(d1)
    
    return out.item() if out.size == 1 else out


def bs_gamma(S, K, T, r, sigma):
    """φ(d1)/(Sσ√T). Handles T→0."""
    S = np.atleast_1d(S).astype(float)
    T = float(T)
    
    if T <= 1e-12:
        out = np.zeros_like(S)
    else:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        out = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return out.item() if out.size == 1 else out


def bs_vega(S, K, T, r, sigma):
    """S√T φ(d1). Handles T→0."""
    S = np.atleast_1d(S).astype(float)
    T = float(T)
    
    if T <= 1e-12:
        out = np.zeros_like(S)
    else:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        out = S * np.sqrt(T) * norm.pdf(d1)
    
    return out.item() if out.size == 1 else out


# --- Shock propagation (Corollary 3.13) ---

def propagate_gaussian(S, K, r, sigma, tau, gamma, xi):
    """
    Gaussian shock propagated backward from event time.
    Returns (value, delta) tuple. Eq. (64) in paper.
    """
    S = np.atleast_1d(S).astype(float)
    tau = max(tau, 0.0)
    
    var_eff = xi**2 + sigma**2 * tau
    d = np.log(S/K) + (r - 0.5*sigma**2) * tau
    amp = (xi / np.sqrt(var_eff)) * np.exp(-r * tau)
    
    val = gamma * amp * np.exp(-d**2 / (2*var_eff))
    delta = val * (-d / (var_eff * S))
    
    if val.size == 1:
        return val.item(), delta.item()
    return val, delta


# --- Shock classes ---

class Shock:
    """Base class for shock amplitude α(S)."""
    
    def __init__(self, tau):
        self.tau = tau
    
    def alpha(self, S):
        raise NotImplementedError
    
    def alpha_prime(self, S, h=1e-6):
        """Δ jump: α'(S). Theorem 5.2."""
        S = np.atleast_1d(S).astype(float)
        dS = h * np.maximum(S, 1.0)
        out = (self.alpha(S + dS) - self.alpha(S - dS)) / (2*dS)
        out = np.atleast_1d(out)
        return out.item() if out.size == 1 else out
    
    def alpha_pp(self, S, h=1e-5):
        """Γ jump: α''(S). Theorem 5.4."""
        S = np.atleast_1d(S).astype(float)
        dS = h * np.maximum(S, 1.0)
        out = (self.alpha(S + dS) - 2*self.alpha(S) + self.alpha(S - dS)) / dS**2
        out = np.atleast_1d(out)
        return out.item() if out.size == 1 else out
    
    def as_dict(self):
        """For compatibility with solver.py interface."""
        return {'tau': self.tau, 'func': self.alpha}
    
    def __call__(self, S):
        return self.alpha(S)


class UniformShock(Shock):
    """Type I: α(S) = A. Section 4.2."""
    
    def __init__(self, tau, A):
        super().__init__(tau)
        self.A = A
    
    def alpha(self, S):
        S = np.atleast_1d(S)
        out = np.full_like(S, self.A, dtype=float)
        return out.item() if out.size == 1 else out
    
    def alpha_prime(self, S, h=None):
        S = np.atleast_1d(S)
        out = np.zeros_like(S, dtype=float)
        return out.item() if out.size == 1 else out
    
    def alpha_pp(self, S, h=None):
        S = np.atleast_1d(S)
        out = np.zeros_like(S, dtype=float)
        return out.item() if out.size == 1 else out


class ProportionalShock(Shock):
    """Type II: α(S) = βS. Section 4.3."""
    
    def __init__(self, tau, beta):
        super().__init__(tau)
        self.beta = beta
    
    def alpha(self, S):
        out = self.beta * np.atleast_1d(S).astype(float)
        return out.item() if out.size == 1 else out
    
    def alpha_prime(self, S, h=None):
        S = np.atleast_1d(S)
        out = np.full_like(S, self.beta, dtype=float)
        return out.item() if out.size == 1 else out
    
    def alpha_pp(self, S, h=None):
        S = np.atleast_1d(S)
        out = np.zeros_like(S, dtype=float)
        return out.item() if out.size == 1 else out


class GaussianShock(Shock):
    """Type III: α(S) = γ exp(-(log S/K)²/2ξ²). Section 4.4."""
    
    def __init__(self, tau, K, gamma, xi):
        super().__init__(tau)
        self.K = K
        self.gamma = gamma
        self.xi = xi
    
    def alpha(self, S):
        S = np.atleast_1d(S).astype(float)
        ell = np.log(S / self.K)
        out = self.gamma * np.exp(-ell**2 / (2*self.xi**2))
        return out.item() if out.size == 1 else out
    
    def alpha_prime(self, S, h=None):
        """Analytical: -γℓ/(Sξ²) exp(-ℓ²/2ξ²). Eq. (94)."""
        S = np.atleast_1d(S).astype(float)
        ell = np.log(S / self.K)
        out = -self.gamma * ell / (S * self.xi**2) * np.exp(-ell**2 / (2*self.xi**2))
        return out.item() if out.size == 1 else out
    
    def alpha_pp(self, S, h=None):
        """Analytical: γ(ℓ²/ξ² + ℓ - 1)/(S²ξ²) exp(-ℓ²/2ξ²). Eq. (95)."""
        S = np.atleast_1d(S).astype(float)
        ell = np.log(S / self.K)
        coef = (ell**2/self.xi**2 + ell - 1) / (S**2 * self.xi**2)
        out = self.gamma * coef * np.exp(-ell**2 / (2*self.xi**2))
        return out.item() if out.size == 1 else out


class VegaShock(Shock):
    """Type IV: α(S) = η·ν_BS(S). Section 4.5."""
    
    def __init__(self, tau, K, T, r, sigma, eta):
        super().__init__(tau)
        self.K, self.T, self.r, self.sigma, self.eta = K, T, r, sigma, eta
    
    def alpha(self, S):
        tau_mat = self.T - (self.T - self.tau)  # time to maturity at event
        out = self.eta * bs_vega(S, self.K, tau_mat, self.r, self.sigma)
        out = np.atleast_1d(out)
        return out.item() if out.size == 1 else out


# --- Legacy function interface (for solver.py compatibility) ---

def shock_uniform(A):
    """Returns callable α(S) = A."""
    return lambda S: np.full_like(np.atleast_1d(S), A, dtype=float)


def shock_proportional(beta):
    """Returns callable α(S) = βS."""
    return lambda S: beta * np.atleast_1d(S).astype(float)


def shock_gaussian(K, gamma, xi):
    """Returns callable α(S) = γ exp(-(log S/K)²/2ξ²)."""
    def alpha(S):
        S = np.atleast_1d(S).astype(float)
        return gamma * np.exp(-np.log(S/K)**2 / (2*xi**2))
    return alpha


# --- Quick validation ---

if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    # BS sanity checks
    print("BS analytics:")
    print(f"  Price: {bs_price(S0, K, T, r, sigma):.4f}")
    print(f"  Delta: {bs_delta(S0, K, T, r, sigma):.4f}")
    print(f"  Gamma: {bs_gamma(S0, K, T, r, sigma):.6f}")
    
    # Shock classes
    print("\nShock classes:")
    shocks = [
        UniformShock(0.5, 2.0),
        ProportionalShock(0.5, 0.05),
        GaussianShock(0.5, K, 3.0, 0.15),
    ]
    for s in shocks:
        print(f"  {s.__class__.__name__}: α(100)={s.alpha(100):.4f}, "
              f"α'(100)={s.alpha_prime(100):.4f}, α''(100)={s.alpha_pp(100):.6f}")
    
    # Propagation
    print("\nGaussian propagation (τ=0.5):")
    val, delta = propagate_gaussian(S0, K, r, sigma, 0.5, 3.0, 0.15)
    print(f"  Value: {val:.4f}, Delta: {delta:.6f}")
