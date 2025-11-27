import numpy as np

def mean_phenotype_recursion(M, h2, I, V):
    """
    Calculate the mean phenotype at the next generation.

    mean phenotype of the selected parents = Mt + I * sqrt(V)
    and the response to selection R = h2 * (I * sqrt(V))

    Parameters:
    M (float): Mean phenotype in the current generation
    h2 (float): Heritability
    I (float): Intensity of selection
    V (float): Phenotypic variance

    Returns:
    float: Mean phenotype in the next generation
    """
    return M + h2 * I * np.sqrt(V)

def disequilibrium_contribution_recursion(d, h2, dV):
    """
    Calculate the contribution of disequilibrium to variance in the next generation.

    Parameters:
    d (float): Disequilibrium variance component in the current generation
    h2 (float): Heritability
    dV (float): Change in variance due to selection

    Returns:
    float: Disequilibrium variance component in the next generation
    """
    return 0.5 * d + 0.5 * (h2 ** 2) * dV