import numpy as np
from scipy import stats

def calculate_equilibrium_disequilibrium(V0, A0, P):
    """
    Calculate the equilibrium disequilibrium component d* using Bulmer's formula.
    
    Parameters:
    V0 (float): Initial phenotypic variance
    A0 (float): Initial additive genetic variance
    P (float): Proportion selected (between 0 and 1)
    
    Returns:
    tuple: (d_star, k, I, x) - equilibrium disequilibrium and parameters used
    """
    # Calculate initial heritability
    h0_2 = A0 / V0
    
    # Calculate selection parameters
    x = stats.norm.ppf(1 - P)  # Standard normal deviate
    f_x = stats.norm.pdf(x)     # Standard normal density at x
    I = f_x / P                # Selection intensity
    
    # Calculate k = -I(I - x)
    k = -I * (I - x)
    
    # Calculate the expression inside the square root
    sqrt_term = 1 - 4 * k * h0_2 * (1 - h0_2)
    
    # Check if the square root term is non-negative
    if sqrt_term < 0:
        raise ValueError(f"Square root term is negative: {sqrt_term}. "
                        f"This can happen with extreme parameter values.")
    
    # Calculate d* using Bulmer's formula
    numerator = 2 * k * h0_2 - 1 + np.sqrt(sqrt_term)
    denominator = 2 * (1 - k)
    
    d_star = V0 * numerator / denominator
    
    return d_star, k, I, x

def calculate_equilibrium_vars(V0, A0, P):
    """
    Calculate all equilibrium values: d*, V*, A*, h*^2
    
    Parameters:
    V0 (float): Initial phenotypic variance
    A0 (float): Initial additive genetic variance
    P (float): Proportion selected
    
    Returns:
    dict: Dictionary containing all equilibrium values
    """
    d_star, k, I, x = calculate_equilibrium_disequilibrium(V0, A0, P)
    
    # Calculate equilibrium variances
    V_star = V0 + d_star
    A_star = A0 + d_star
    h_star_2 = A_star / V_star
    
    return {
        'd_star': d_star,
        'V_star': V_star,
        'A_star': A_star,
        'h_star_2': h_star_2,
        'k': k,
        'I': I,
        'x': x
    }

# Example: Reproduce the results from Bulmer's paper (Table 1)
if __name__ == "__main__":
    # Parameters from Bulmer's paper
    V0 = 100.0
    A0 = 50.0
    P = 0.2
    
    print("Bulmer's Paper Example (P = 0.2):")
    print(f"Initial parameters: V0 = {V0}, A0 = {A0}, h0² = {A0/V0:.3f}")

    results = calculate_equilibrium_vars(V0, A0, P)
    
    print(f"\nSelection parameters:")
    print(f"  Proportion selected (P): {P}")
    print(f"  Selection intensity (I): {results['I']:.4f}")
    print(f"  Standard normal deviate (x): {results['x']:.4f}")
    print(f"  k = -I(I-x): {results['k']:.4f}")
    
    print(f"\nEquilibrium values:")
    print(f"  Disequilibrium component (d*): {results['d_star']:.4f}")
    print(f"  Phenotypic variance (V*): {results['V_star']:.4f}")
    print(f"  Additive genetic variance (A*): {results['A_star']:.4f}")
    print(f"  Heritability (h*²): {results['h_star_2']:.4f}")
    
    # Calculate equilibrium response to selection
    equilibrium_response = results['h_star_2'] * results['I'] * np.sqrt(results['V_star'])
    print(f"  Equilibrium response to selection: {equilibrium_response:.4f}")