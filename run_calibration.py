"""
Calibration: recover α(S) from pre/post-event prices via Tikhonov regularization.
Generates Figure 1 from van der Wouden (2025).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from solver import ShockPDE


class Calibrator:
    """Tikhonov-regularized shock amplitude recovery (Section 6)."""
    
    def __init__(self, solver, strikes, pre_prices, post_prices, event_tau):
        self.solver = solver
        self.strikes = np.array(strikes)
        self.pre_prices = np.array(pre_prices)
        self.post_prices = np.array(post_prices)
        self.event_tau = event_tau
        self.payoff = lambda S: np.maximum(S - solver.K, 0)
    
    def objective(self, node_values, node_positions, reg_lambda):
        """J_λ(α) = SSE + λ·∫(α'')² ds. Definition 6.2."""
        alpha = CubicSpline(node_positions, node_values, bc_type='natural')
        shock = {'tau': self.event_tau, 'func': alpha}
        V = self.solver.solve(self.payoff, [shock])
        
        # Pricing error
        sse = np.sum((V(self.strikes) - self.post_prices)**2)
        
        # Roughness penalty
        alpha_pp = alpha.derivative(nu=2)
        s = np.linspace(node_positions[0], node_positions[-1], 100)
        roughness = np.trapezoid(alpha_pp(s)**2, s)
        
        return sse + reg_lambda * roughness
    
    def calibrate(self, num_nodes=12, reg_lambda=1e-4):
        """Algorithm 1: Regularized shock calibration."""
        K_min, K_max = self.strikes.min(), self.strikes.max()
        pad = 0.15 * (K_max - K_min)
        node_positions = np.linspace(K_min - pad, K_max + pad, num_nodes)
        
        # Initial guess from raw price differences
        initial = np.interp(node_positions, self.strikes, 
                           self.post_prices - self.pre_prices)
        initial = np.maximum(initial, 0)
        
        bounds = [(0, None) for _ in range(num_nodes)]
        res = minimize(self.objective, initial,
                      args=(node_positions, reg_lambda),
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 50})
        
        alpha = CubicSpline(node_positions, res.x, bc_type='natural')
        return alpha, node_positions, res.x
    
    def compute_l_curve(self, num_nodes, lambdas):
        """L-curve data for regularization parameter selection."""
        K_min, K_max = self.strikes.min(), self.strikes.max()
        pad = 0.15 * (K_max - K_min)
        node_positions = np.linspace(K_min - pad, K_max + pad, num_nodes)
        initial = np.interp(node_positions, self.strikes, 
                           self.post_prices - self.pre_prices)
        initial = np.maximum(initial, 0)
        bounds = [(0, None) for _ in range(num_nodes)]
        
        errors, roughnesses = [], []
        for lam in lambdas:
            res = minimize(self.objective, initial, args=(node_positions, lam),
                          method='L-BFGS-B', bounds=bounds, options={'maxiter': 30})
            alpha = CubicSpline(node_positions, res.x, bc_type='natural')
            
            shock = {'tau': self.event_tau, 'func': alpha}
            V = self.solver.solve(self.payoff, [shock])
            errors.append(np.sum((V(self.strikes) - self.post_prices)**2))
            
            alpha_pp = alpha.derivative(nu=2)
            s = np.linspace(node_positions[0], node_positions[-1], 100)
            roughnesses.append(np.trapezoid(alpha_pp(s)**2, s))
        
        return np.array(errors), np.array(roughnesses)


def true_alpha(S):
    """Ground truth: Gaussian centered at K=100, width=12 in price space."""
    return 3.0 * np.exp(-((S - 100)**2) / (2 * 12**2))


if __name__ == "__main__":
    # Setup
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    event_tau = 0.5
    strikes = np.linspace(70, 130, 15)
    S = np.linspace(60, 140, 200)
    
    solver = ShockPDE(r, sigma, T, S0, K, Nx=300, Nt=800)
    payoff = lambda S: np.maximum(S - K, 0)
    
    # Generate synthetic data
    V_pre = solver.solve(payoff)
    V_post = solver.solve(payoff, [{'tau': event_tau, 'func': true_alpha}])
    pre_prices = V_pre(strikes)
    post_prices_clean = V_post(strikes)
    
    # Noiseless calibration
    print("Noiseless calibration...")
    cal_clean = Calibrator(solver, strikes, pre_prices, post_prices_clean, event_tau)
    alpha_clean, nodes_x, nodes_y = cal_clean.calibrate(num_nodes=12, reg_lambda=1e-5)
    
    # Noisy calibration
    print("Noisy calibration (±5¢)...")
    np.random.seed(42)
    post_prices_noisy = post_prices_clean + np.random.normal(0, 0.05, len(strikes))
    cal_noisy = Calibrator(solver, strikes, pre_prices, post_prices_noisy, event_tau)
    alpha_noisy, _, _ = cal_noisy.calibrate(num_nodes=10, reg_lambda=1.0)
    
    # L-curve
    print("Computing L-curve...")
    lambdas = np.logspace(-8, -4, 8)
    errors, roughnesses = cal_clean.compute_l_curve(10, lambdas)
    
    # --- Figure 1: 4-panel calibration results ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Panel 1: Noiseless reconstruction
    ax = axes[0, 0]
    ax.plot(S, true_alpha(S), 'k--', lw=2, label=r'True $\alpha(S)$')
    ax.plot(S, alpha_clean(S), 'r-', lw=2, label='Calibrated')
    ax.scatter(strikes, post_prices_clean - pre_prices, c='blue', alpha=0.5, 
               s=40, label='Raw diff')
    ax.scatter(nodes_x, nodes_y, c='red', marker='s', s=50, zorder=5)
    ax.set_xlabel('Spot Price S')
    ax.set_ylabel(r'$\alpha(S)$')
    ax.set_title('Noiseless Reconstruction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Price fit
    ax = axes[0, 1]
    V_fit = solver.solve(payoff, [{'tau': event_tau, 'func': alpha_clean}])
    ax.scatter(strikes, post_prices_clean, c='black', s=50, label='Target')
    ax.plot(S, V_fit(S), 'r-', lw=2, label='Model')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Option Price')
    ax.set_title('Price Fit (Noiseless)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Noisy reconstruction
    ax = axes[1, 0]
    ax.plot(S, true_alpha(S), 'k--', lw=2, label=r'True $\alpha(S)$')
    ax.plot(S, alpha_noisy(S), 'r-', lw=2, label=r'Calibrated ($\lambda$=1)')
    ax.fill_between(S, alpha_noisy(S) - 0.5, alpha_noisy(S) + 0.5, 
                    color='red', alpha=0.2)
    ax.set_xlabel('Spot Price S')
    ax.set_ylabel(r'$\alpha(S)$')
    ax.set_title(r'Noisy Reconstruction ($\pm$5¢)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: L-curve
    ax = axes[1, 1]
    ax.loglog(errors, roughnesses, 'b.-', lw=1.5, ms=8)
    for i in [0, 2, -1]:
        ax.annotate(f'λ={lambdas[i]:.0e}', (errors[i], roughnesses[i]),
                   textcoords='offset points', xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Fitting Error (SSE)')
    ax.set_ylabel(r"Roughness $\int(\alpha'')^2\,ds$")
    ax.set_title('L-Curve')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('fig_calibration_results.png', dpi=150)
    plt.show()
    
    print("Done.")
