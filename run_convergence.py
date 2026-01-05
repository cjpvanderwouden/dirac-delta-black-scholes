"""
Convergence study and Greek discontinuity validation.
Generates Figures 2 and 4 from van der Wouden (2025).
"""
import numpy as np
import matplotlib.pyplot as plt
from solver import ShockPDE
from core import shock_uniform, shock_gaussian, GaussianShock, bs_delta, bs_gamma


def convergence_study():
    """
    Verify O(Δx²) spatial convergence using Type I shock.
    Theoretical: ΔV = A·e^{-rτ} (Corollary 3.10).
    """
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    tau_event = 0.5
    A = 2.0
    
    theory = A * np.exp(-r * tau_event)
    payoff = lambda S: np.maximum(S - K, 0)
    
    Nx_vals = [125, 250, 500, 1000, 2000]
    errors = []
    
    for Nx in Nx_vals:
        solver = ShockPDE(r, sigma, T, S0, K, Nx=Nx, Nt=4*Nx)
        V_base = solver.solve(payoff)
        V_shock = solver.solve(payoff, [{'tau': tau_event, 'func': shock_uniform(A)}])
        errors.append(abs(V_shock(S0) - V_base(S0) - theory))
    
    return Nx_vals, errors, theory


def greek_identity_validation():
    """
    Verify Theorems 5.2 and 5.4: ΔΔ = α'(S), ΔΓ = α''(S).
    Computes Greeks via finite difference on PDE solution.
    """
    S0, K, r, sigma = 100.0, 100.0, 0.05, 0.2
    tau_event = 0.5
    gamma_amp, xi = 3.0, 0.15
    
    # Solve up to event time (set T = tau_event)
    solver = ShockPDE(r, sigma, tau_event, S0, K, Nx=800, Nt=3200)
    payoff = lambda S: np.maximum(S - K, 0)
    shock = GaussianShock(tau_event, K, gamma_amp, xi)
    
    V_pre = solver.solve(payoff)
    V_post = solver.solve(payoff, [shock])
    
    S = np.linspace(70, 130, 600)
    
    # Numerical Greeks via central difference
    def fd_greeks(V_func, S, h=1e-5):
        S = np.asarray(S)
        dS = h * np.maximum(S, 1.0)
        Vp, Vm, V0 = V_func(S + dS), V_func(S - dS), V_func(S)
        delta = (Vp - Vm) / (2 * dS)
        gamma = (Vp - 2*V0 + Vm) / dS**2
        return delta, gamma
    
    delta_pre, gamma_pre = fd_greeks(V_pre, S)
    delta_post, gamma_post = fd_greeks(V_post, S)
    
    # Numerical jumps
    delta_jump_num = delta_post - delta_pre
    gamma_jump_num = gamma_post - gamma_pre
    
    # Theoretical jumps
    delta_jump_theory = shock.alpha_prime(S)
    gamma_jump_theory = shock.alpha_pp(S)
    
    # Errors
    err_delta = np.max(np.abs(delta_jump_num - delta_jump_theory))
    err_gamma = np.max(np.abs(gamma_jump_num - gamma_jump_theory))
    
    return {
        'S': S,
        'alpha': shock.alpha(S),
        'delta_jump_num': delta_jump_num,
        'delta_jump_theory': delta_jump_theory,
        'gamma_jump_num': gamma_jump_num,
        'gamma_jump_theory': gamma_jump_theory,
        'err_delta': err_delta,
        'err_gamma': err_gamma,
        'K': K
    }


if __name__ == "__main__":
    # --- Convergence Study (Figure 2 / Table 4) ---
    print("Convergence study...")
    Nx_vals, errors, theory = convergence_study()
    
    print(f"Theoretical impact: {theory:.6f}")
    print(f"{'Nx':>6} | {'Error':>12} | {'Order':>6}")
    print("-" * 30)
    for i, (nx, err) in enumerate(zip(Nx_vals, errors)):
        if i > 0 and errors[i-1] > 1e-14 and err > 1e-14:
            order = np.log(errors[i-1]/err) / np.log(Nx_vals[i]/Nx_vals[i-1])
            print(f"{nx:>6} | {err:>12.2e} | {order:>6.2f}")
        else:
            print(f"{nx:>6} | {err:>12.2e} |      -")
    
    # --- Greek Identity Validation (Figure 4) ---
    print("\nGreek identity validation...")
    g = greek_identity_validation()
    print(f"Max error ΔΔ: {g['err_delta']:.2e}")
    print(f"Max error ΔΓ: {g['err_gamma']:.2e}")
    
    # --- Figure 2: Convergence ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.loglog(Nx_vals, errors, 'bo-', lw=2, ms=8, label='Error')
    ref = errors[0] * (Nx_vals[0] / np.array(Nx_vals))**2
    ax1.loglog(Nx_vals, ref, 'k--', lw=1.5, label=r'$O(N_x^{-2})$')
    ax1.set_xlabel('Grid Points $N_x$')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Spatial Convergence (Type I Shock)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    fig1.tight_layout()
    fig1.savefig('fig_convergence.png', dpi=150)
    
    # --- Figure 4: Greek Discontinuities ---
    fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
    S, K = g['S'], g['K']
    
    # Panel 1: α(S)
    axes[0].plot(S, g['alpha'], 'b-', lw=2)
    axes[0].axvline(K, color='k', ls='--', alpha=0.5)
    axes[0].set_xlabel('Spot S')
    axes[0].set_ylabel(r'$\alpha(S)$')
    axes[0].set_title('Shock Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: ΔΔ = α'(S)
    axes[1].plot(S, g['delta_jump_theory'], 'r-', lw=2, label=r"Theory: $\alpha'(S)$")
    axes[1].plot(S, g['delta_jump_num'], 'k--', lw=1.5, 
                 label=f"Numeric (err={g['err_delta']:.2e})")
    axes[1].axhline(0, color='k', lw=0.5, alpha=0.3)
    axes[1].axvline(K, color='k', ls='--', alpha=0.5)
    axes[1].set_xlabel('Spot S')
    axes[1].set_ylabel(r'$\Delta\Delta(S)$')
    axes[1].set_title(r"Theorem 5.2: $\Delta\Delta = \alpha'(S)$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: ΔΓ = α''(S)
    axes[2].plot(S, g['gamma_jump_theory'], 'g-', lw=2, label=r"Theory: $\alpha''(S)$")
    axes[2].plot(S, g['gamma_jump_num'], 'k--', lw=1.5,
                 label=f"Numeric (err={g['err_gamma']:.2e})")
    axes[2].axhline(0, color='k', lw=0.5, alpha=0.3)
    axes[2].axvline(K, color='k', ls='--', alpha=0.5)
    axes[2].set_xlabel('Spot S')
    axes[2].set_ylabel(r'$\Delta\Gamma(S)$')
    axes[2].set_title(r"Theorem 5.4: $\Delta\Gamma = \alpha''(S)$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig2.tight_layout()
    fig2.savefig('fig_greek_identity_validation.png', dpi=150)
    
    plt.show()
