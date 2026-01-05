"""
Crank-Nicolson PDE solver for shock-forced Black-Scholes.
Reference: van der Wouden (2025), Section 7.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import interp1d

from core import shock_uniform, shock_proportional, shock_gaussian


class ShockPDE:
    """
    Solves V_τ = (σ²/2)V_xx + (r - σ²/2)V_x - rV with shock injection.
    Uses log-price coordinates x = log(S) on bounded domain.
    """
    
    def __init__(self, r, sigma, T, S_ref, K, Nx=500, Nt=2000):
        self.r, self.sigma, self.T, self.K = r, sigma, T, K
        self.Nx, self.Nt = Nx, Nt
        
        # Log-price grid centered at S_ref
        x_width = 5.0
        self.x_min = np.log(S_ref) - x_width
        self.x_max = np.log(S_ref) + x_width
        self.dx = (self.x_max - self.x_min) / (self.Nx - 1)
        self.grid_x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.grid_S = np.exp(self.grid_x)
        
        # PDE coefficients: a = σ²/2, b = r - σ²/2
        self.a = 0.5 * sigma**2
        self.b = r - 0.5 * sigma**2

    def _build_matrices(self, dt):
        """Crank-Nicolson tridiagonal matrices."""
        alpha = (self.a * dt) / (2 * self.dx**2)
        beta = (self.b * dt) / (4 * self.dx)
        gamma = (self.r * dt) / 2

        # LHS: (I - dt/2 · L)
        d_main = np.full(self.Nx, 1 + 2*alpha + gamma)
        d_upper = np.full(self.Nx-1, -alpha - beta)
        d_lower = np.full(self.Nx-1, -alpha + beta)
        d_main[0], d_main[-1] = 1.0, 1.0
        d_upper[0], d_lower[-1] = 0.0, 0.0
        A = sp.diags([d_lower, d_main, d_upper], [-1, 0, 1], format='csc')
        
        # RHS: (I + dt/2 · L)
        d_main = np.full(self.Nx, 1 - 2*alpha - gamma)
        d_upper = np.full(self.Nx-1, alpha + beta)
        d_lower = np.full(self.Nx-1, alpha - beta)
        d_main[0], d_main[-1] = 0.0, 0.0
        d_upper[0], d_lower[-1] = 0.0, 0.0
        B = sp.diags([d_lower, d_main, d_upper], [-1, 0, 1], format='csc')
        
        return A, B

    def solve(self, payoff, shocks=None):
        """
        Solve PDE from τ=0 to τ=T, injecting shocks at specified times.
        
        Args:
            payoff: callable S -> payoff values
            shocks: list of {'tau': float, 'func': callable} or Shock objects
        
        Returns:
            Interpolated solution V(S) at τ=T
        """
        if shocks is None:
            shocks = []
        
        # Normalize shock format (support both dict and Shock class)
        shock_list = []
        for s in shocks:
            if hasattr(s, 'as_dict'):
                shock_list.append(s.as_dict())
            else:
                shock_list.append(s)
        shock_list = sorted(shock_list, key=lambda x: x['tau'])
        
        U = payoff(self.grid_S).astype(float)
        current_tau = 0.0
        checkpoints = [s['tau'] for s in shock_list] + [self.T]
        
        for i, target_tau in enumerate(checkpoints):
            duration = target_tau - current_tau
            if duration <= 1e-9:
                continue
            
            # Time-stepping within segment
            steps = max(int(np.ceil(duration / (self.T / self.Nt))), 5)
            dt = duration / steps
            A, B = self._build_matrices(dt)
            solve_A = spla.factorized(A)
            
            for _ in range(steps):
                tau_new = current_tau + dt
                rhs = B @ U
                rhs[0] = 0.0
                rhs[-1] = self.grid_S[-1] - self.K * np.exp(-self.r * tau_new)
                U = solve_A(rhs)
                current_tau = tau_new
            
            # Inject shock (Algorithm 2, line 9)
            if i < len(shock_list):
                U = U + shock_list[i]['func'](self.grid_S)
                U = np.maximum(U, 0)
        
        return interp1d(self.grid_S, U, kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    solver = ShockPDE(r, sigma, T, S0, K)
    payoff = lambda S: np.maximum(S - K, 0)
    
    # Baseline
    V_base = solver.solve(payoff)
    print(f"Baseline: {V_base(S0):.4f}")
    
    # Type I: ΔV = A·e^{-rτ}
    V_t1 = solver.solve(payoff, [{'tau': 0.5, 'func': shock_uniform(2.0)}])
    impact = V_t1(S0) - V_base(S0)
    theory = 2.0 * np.exp(-r * 0.5)
    print(f"Type I:   {impact:.6f} (theory: {theory:.6f}, err: {abs(impact-theory):.2e})")
    
    # Type II: ΔV = βS
    V_t2 = solver.solve(payoff, [{'tau': 0.5, 'func': shock_proportional(0.05)}])
    impact = V_t2(S0) - V_base(S0)
    theory = 0.05 * S0
    print(f"Type II:  {impact:.6f} (theory: {theory:.6f}, err: {abs(impact-theory):.2e})")
    
    # Type III: no closed form
    V_t3 = solver.solve(payoff, [{'tau': 0.5, 'func': shock_gaussian(K, 3.0, 0.15)}])
    print(f"Type III: {V_t3(S0) - V_base(S0):.4f}")
    
    # Figure 3
    S = np.linspace(80, 120, 200)
    plt.figure(figsize=(9, 5))
    plt.plot(S, V_base(S), 'k--', lw=2, label='Black-Scholes')
    plt.plot(S, V_t1(S), 'b-', label='Type I (+$2)')
    plt.plot(S, V_t2(S), 'r-', label='Type II (+5%S)')
    plt.plot(S, V_t3(S), 'g-', label='Type III (Gaussian)')
    plt.xlabel('Spot Price S')
    plt.ylabel('Option Value')
    plt.title('Option Value Surfaces with Shock at τ=0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_shock_surfaces.png', dpi=150)
    plt.show()
