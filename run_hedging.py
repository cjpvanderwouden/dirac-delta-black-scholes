"""
Hedging validation: compare BS vs shock-aware delta hedging.
Generates Figure 5 from van der Wouden (2025).
"""
import numpy as np
import matplotlib.pyplot as plt
from core import bs_price, bs_delta, propagate_gaussian


def market_price(S, K, T, r, sigma, t, event_time, gamma, xi):
    """
    Market option price including event premium (pre-event) or pure BS (post-event).
    """
    S = np.atleast_1d(S).astype(float)
    tau_mat = max(T - t, 0.0)
    base = bs_price(S, K, tau_mat, r, sigma)
    
    if t < event_time:
        tau_to_event = event_time - t
        prem, _ = propagate_gaussian(S, K, r, sigma, tau_to_event, gamma, xi)
        out = base + prem
    else:
        out = base
    
    out = np.atleast_1d(out)
    return out.item() if out.size == 1 else out


class HedgingValidator:
    """MTM tracking error comparison: BS vs shock-aware hedging."""
    
    def __init__(self, S0, K, T, r, sigma, event_time, gamma=5.0, xi=0.30):
        self.S0, self.K, self.T = float(S0), float(K), float(T)
        self.r, self.sigma = float(r), float(sigma)
        self.event_time = float(event_time)
        self.gamma, self.xi = gamma, xi
    
    def simulate_paths(self, n_paths, n_steps, seed=42):
        np.random.seed(seed)
        dt = self.T / n_steps
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        
        for i in range(n_steps):
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z[:, i]
            )
        return S, dt
    
    def run_mtm_validation(self, n_paths=5000, n_steps=2500, seed=42):
        """
        Track Π(t) - V_mkt(S,t) for both strategies up to event time.
        Returns dict with tracking errors, IMSE, and event jump P&L.
        """
        S, dt = self.simulate_paths(n_paths, n_steps, seed)
        
        # Align event to grid
        idx_event = min(max(int(round(self.event_time / dt)), 2), n_steps - 1)
        t_event = idx_event * dt
        
        # Initial market price
        V0 = market_price(self.S0, self.K, self.T, self.r, self.sigma,
                         0.0, self.event_time, self.gamma, self.xi)
        
        cash_bs = np.full(n_paths, V0, dtype=float)
        cash_pde = np.full(n_paths, V0, dtype=float)
        units_bs = np.zeros(n_paths)
        units_pde = np.zeros(n_paths)
        
        times_pre, std_bs, std_pde = [], [], []
        imse_bs, imse_pde = 0.0, 0.0
        
        # Pre-event loop
        for t_idx in range(idx_event):
            t = t_idx * dt
            S_t = S[:, t_idx]
            tau_mat = self.T - t
            tau_event = self.event_time - t
            
            # BS delta (naive)
            delta_A = bs_delta(S_t, self.K, tau_mat, self.r, self.sigma)
            
            # Shock-aware delta
            _, delta_prem = propagate_gaussian(S_t, self.K, self.r, self.sigma,
                                               tau_event, self.gamma, self.xi)
            delta_B = delta_A + delta_prem
            
            # Rebalance
            cash_bs -= (delta_A - units_bs) * S_t
            units_bs = delta_A
            cash_pde -= (delta_B - units_pde) * S_t
            units_pde = delta_B
            
            # Portfolio values
            port_bs = cash_bs + units_bs * S_t
            port_pde = cash_pde + units_pde * S_t
            
            # Market value
            prem_val, _ = propagate_gaussian(S_t, self.K, self.r, self.sigma,
                                             tau_event, self.gamma, self.xi)
            V_mkt = bs_price(S_t, self.K, tau_mat, self.r, self.sigma) + prem_val
            
            # Tracking errors
            err_bs = port_bs - V_mkt
            err_pde = port_pde - V_mkt
            
            times_pre.append(t)
            std_bs.append(np.std(err_bs))
            std_pde.append(np.std(err_pde))
            imse_bs += np.mean(err_bs**2) * dt
            imse_pde += np.mean(err_pde**2) * dt
            
            # Accrue cash
            cash_bs *= np.exp(self.r * dt)
            cash_pde *= np.exp(self.r * dt)
        
        # Snapshot at t_j^-
        S_event = S[:, idx_event]
        port_bs_event = cash_bs + units_bs * S_event
        port_pde_event = cash_pde + units_pde * S_event
        
        eps = 1e-10
        V_pre = market_price(S_event, self.K, self.T, self.r, self.sigma,
                            t_event - eps, self.event_time, self.gamma, self.xi)
        V_post = market_price(S_event, self.K, self.T, self.r, self.sigma,
                             t_event + eps, self.event_time, self.gamma, self.xi)
        
        err_snap_bs = port_bs_event - V_pre
        err_snap_pde = port_pde_event - V_pre
        jump_pnl = V_pre - V_post  # Short option P&L at event
        
        return {
            'times': np.array(times_pre),
            'std_bs': np.array(std_bs),
            'std_pde': np.array(std_pde),
            'err_snap_bs': err_snap_bs,
            'err_snap_pde': err_snap_pde,
            'jump_pnl': jump_pnl,
            'imse_bs': imse_bs,
            'imse_pde': imse_pde,
            't_event': t_event,
            'V0': V0
        }


if __name__ == "__main__":
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.10
    event_time = 0.5
    gamma, xi = 5.0, 0.30
    
    validator = HedgingValidator(S0, K, T, r, sigma, event_time, gamma, xi)
    
    # Main validation
    print("Running MTM validation (seed=42 for reproducibility)...")
    out = validator.run_mtm_validation(n_paths=5000, n_steps=2500)
    
    red = 100 * (1 - out['imse_pde'] / (out['imse_bs'] + 1e-18))
    print(f"IMSE: BS={out['imse_bs']:.2e}, PDE={out['imse_pde']:.2e}, Reduction={red:.1f}%")
    print(f"Std at t_j^-: BS={np.std(out['err_snap_bs']):.4f}, PDE={np.std(out['err_snap_pde']):.4f}")
    
    # Stress test (σ=5%)
    print("\nStress test (σ=5%)...")
    val_stress = HedgingValidator(S0, K, T, r, 0.05, event_time, gamma, xi)
    out_s = val_stress.run_mtm_validation(n_paths=5000, n_steps=2500, seed=43)
    red_s = 100 * (1 - out_s['imse_pde'] / (out_s['imse_bs'] + 1e-18))
    print(f"IMSE: BS={out_s['imse_bs']:.2e}, PDE={out_s['imse_pde']:.2e}, Reduction={red_s:.1f}%")
    
    # Sensitivity analysis
    print("\nSensitivity analysis...")
    gamma_vals = [1, 2, 3, 5, 7, 10]
    red_gamma = []
    for g in gamma_vals:
        v = HedgingValidator(S0, K, T, r, sigma, event_time, g, xi)
        tmp = v.run_mtm_validation(n_paths=2000, n_steps=1000, seed=11)
        rg = 100 * (1 - tmp['imse_pde'] / (tmp['imse_bs'] + 1e-18))
        red_gamma.append(rg)
        print(f"  γ={g}: {rg:.1f}%")
    
    event_times = [0.1, 0.25, 0.5, 0.75, 0.9]
    red_time = []
    for et in event_times:
        v = HedgingValidator(S0, K, T, r, sigma, et, gamma, xi)
        tmp = v.run_mtm_validation(n_paths=2000, n_steps=1000, seed=12)
        rt = 100 * (1 - tmp['imse_pde'] / (tmp['imse_bs'] + 1e-18))
        red_time.append(rt)
        print(f"  t={et}: {rt:.1f}%")
    
    # --- Figure 5: 4-panel hedging results ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Panel 1: Std over time
    ax = axes[0, 0]
    ax.plot(out['times'], out['std_bs'], lw=2, label='BS')
    ax.plot(out['times'], out['std_pde'], lw=2, label='Shock-aware')
    ax.set_xlabel('Time t (pre-event)')
    ax.set_ylabel(r'Std$[\Pi(t) - V_{mkt}(S,t)]$')
    ax.set_title('Pre-Event MTM Tracking Risk')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Tracking error at t_j^-
    ax = axes[0, 1]
    bins = 70
    ax.hist(out['err_snap_bs'], bins=bins, alpha=0.5, 
            label=f"BS (Std={np.std(out['err_snap_bs']):.4f})", color='gray')
    ax.hist(out['err_snap_pde'], bins=bins, alpha=0.8,
            label=f"PDE (Std={np.std(out['err_snap_pde']):.4f})", color='blue')
    ax.set_yscale('log')
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel(r'Tracking error at $t_j^-$')
    ax.set_ylabel('Frequency (log)')
    ax.set_title(r'Tracking Error at $t_j^-$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Event jump P&L
    ax = axes[1, 0]
    ax.hist(out['jump_pnl'], bins=70, alpha=0.85, density=True, color='purple')
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel(r'Event jump P&L: $V_{pre} - V_{post}$')
    ax.set_ylabel('Density')
    ax.set_title('Event-Time Jump P&L (Unspanned)')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Sensitivity
    ax = axes[1, 1]
    ax.plot(gamma_vals, red_gamma, 'bo-', lw=2, ms=7, label='vs γ')
    ax.set_xlabel('Shock amplitude γ')
    ax.set_ylabel('IMSE Reduction (%)')
    ax.set_title('Sensitivity: IMSE Reduction')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twiny()
    ax2.plot(event_times, red_time, 'rs--', lw=2, ms=7)
    ax2.set_xlabel('Event time (t/T)', color='red')
    ax2.tick_params(axis='x', labelcolor='red')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('fig_mtm_hedging_results.png', dpi=150)
    plt.show()
    
    print("\nDone.")
