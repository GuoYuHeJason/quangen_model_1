import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

def response_to_selection(V0, A0, P):
    """Calculate equilibrium response to selection R* for given P."""
    h0_2 = A0 / V0
    
    # Calculate selection parameters
    x = stats.norm.ppf(1 - P)
    f_x = stats.norm.pdf(x)
    I = f_x / P
    k = -I * (I - x)
    
    # Calculate d*
    Delta = 1 - 4 * k * h0_2 * (1 - h0_2)
    numerator = 2 * k * h0_2 - 1 + np.sqrt(Delta)
    denominator = 2 * (1 - k)
    d_star = V0 * numerator / denominator
    
    # Calculate equilibrium response
    V_star = V0 + d_star
    A_star = A0 + d_star
    h_star2 = A_star / V_star
    R_star = h_star2 * I * np.sqrt(V_star)
    
    return R_star, d_star, I, h_star2

def find_optimal_P(V0, A0, P_bounds=(0.001, 0.999)):
    """Find the P that maximizes R*."""
    def objective(P):
        R_star, _, _, _ = response_to_selection(V0, A0, P)
        return -R_star  # Minimize negative to find maximum
    
    result = minimize_scalar(objective, bounds=P_bounds, method='bounded')
    return result.x, -result.fun

if __name__ == "__main__":
    output_file = "/Users/heguoyu/Desktop/eeb314Final/optimal_P_analysis.png"
    # Bulmer's parameters
    V0, A0 = 100, 50

    # Find optimal P
    P_opt, R_opt = find_optimal_P(V0, A0)

    # Analyze around optimal P
    P_test = np.linspace(0.001, 0.999, 2000)
    R_values = []
    dR_dP_values = []

    for P in P_test:
        R_star, _, _, _ = response_to_selection(V0, A0, P)
        R_values.append(R_star)

    # Calculate derivative numerically
    dR_dP = np.gradient(R_values, P_test)

    # Find where derivative crosses zero
    zero_crossings = np.where(np.diff(np.sign(dR_dP)))[0]

    print("OPTIMAL SELECTION PROPORTION ANALYSIS")
    print("=" * 50)
    print(f"Parameters: V0 = {V0}, A0 = {A0}, h0² = {A0/V0:.3f}")
    print(f"\nOptimal selection proportion: P = {P_opt:.4f}")
    print(f"Maximum response to selection: R* = {R_opt:.4f}")

    # Detailed analysis at optimal point
    R_opt_calc, d_star_opt, I_opt, h_star2_opt = response_to_selection(V0, A0, P_opt)
    V_star_opt = V0 + d_star_opt
    A_star_opt = A0 + d_star_opt

    print(f"\nAt optimal P = {P_opt:.4f}:")
    print(f"  Selection intensity: I = {I_opt:.4f}")
    print(f"  Disequilibrium: d* = {d_star_opt:.4f}")
    print(f"  Phenotypic variance: V* = {V_star_opt:.4f}")
    print(f"  Additive genetic variance: A* = {A_star_opt:.4f}")
    print(f"  Heritability: h*² = {h_star2_opt:.4f}")

    # Compare with Bulmer's P = 0.2
    R_02, d_star_02, I_02, h_star2_02 = response_to_selection(V0, A0, 0.2)
    print(f"\nAt P = 0.2 (Bulmer's example):")
    print(f"  Response to selection: R* = {R_02:.4f}")
    print(f"  This is {100*R_02/R_opt:.2f}% of maximum")

    # Create detailed visualization
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot R* vs P
    ax1.plot(P_test, R_values, 'b-', linewidth=2, label='R*')
    ax1.axvline(x=0.2, color='green', linestyle='--', 
                label='Bulmer P = 0.2')
    ax1.set_xlabel('Proportion Selected (P)')
    ax1.set_ylabel('Response to Selection (R*)')
    ax1.set_title('Response to Selection vs Selection Proportion')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot derivative dR*/dP
    ax2.plot(P_test, dR_dP, 'r-', linewidth=2, label='dR*/dP')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # Mark zero crossings
    for cross in zero_crossings:
        ax2.plot(P_test[cross], dR_dP[cross], 'ro', markersize=6)
    ax2.set_xlabel('Proportion Selected (P)')
    ax2.set_ylabel('dR*/dP')
    ax2.set_title('Derivative of Response to Selection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)