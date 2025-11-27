from scipy import stats
def calculate_selection_parameters(P):
    """
    Calculate selection parameters from proportion selected.
    Assumes a standard normal distribution.
    
    Parameters:
    P (float): Proportion selected (between 0 and 1)
    
    Returns:
    dict: Dictionary containing x, f(x), and I
    """
    # Calculate the standard normal deviate x
    # For upper tail selection: x = Φ^(-1)(1 - P)
    x = stats.norm.ppf(1 - P)
    
    # Calculate the standard normal density at x
    f_x = stats.norm.pdf(x)
    
    # Calculate selection intensity I
    I = f_x / P
    
    return x, f_x, I

def calculate_dV(I, x, V):
    """
    Calculate the change in variance due to selection.
    Assumes truncation selection on a normally distributed trait.

    Parameters:
    I (float): Intensity of selection
    x (float): Standard normal deviate
    V (float): Phenotypic variance

    Returns:
    float: Change in variance
    """
    return V * (0 - (I * (I - x)))

def phenotypic_variance(V0, d):
    """
    Calculate the phenotypic variance in the current generation.

    Parameters:
    V0 (float): Initial phenotypic variance
    d (float): Disequilibrium variance component in the current generation

    Returns:
    float: Phenotypic variance in the current generation
    """
    return V0 + d

def additive_genetic_variance(A0, d):
    """
    Calculate the additive genetic variance in the current generation.

    Parameters:
    A0 (float): Initial additive genetic variance
    d (float): Disequilibrium variance component in the current generation

    Returns:
    float: Additive genetic variance in the current generation
    """
    return A0 + d

def heritability(A, V):
    """
    Calculate heritability.

    Parameters:
    A (float): Additive genetic variance
    V (float): Phenotypic variance

    Returns:
    float: Heritability
    """
    return A / V