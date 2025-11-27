from recursion_equations import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def simulation(M0, V0, A0, P, generations, d0 = 0.0):
    """
    Simulate the evolution of mean phenotype and variances over a number of generations.

    Parameters:
    M0 (float): Initial mean phenotype
    V0 (float): Initial phenotypic variance
    A0 (float): Initial additive genetic variance
    P (float): Proportion selected
    generations (int): Number of generations to simulate

    Returns:
    dict: A dictionary containing the history of mean phenotype, phenotypic variance,
          additive genetic variance, and disequilibrium contribution, heritability, response to selection
          as numpy arrays.
    """
    # Store history, add initial values
    history = {
        'mean_phenotype': [M0],
        'phenotypic_variance': [V0],
        'additive_genetic_variance': [A0],
        'disequilibrium_contribution': [d0],
        'heritability': [],
    }

    # get selection parameters
    # capital letters for constants across generations
    X, _, I = calculate_selection_parameters(P)

    # Initialize variables
    m = M0
    v = V0
    a = A0
    d = d0  # Initial disequilibrium contribution

    # simulate over generations
    for gen in range(generations):
        # calculate current and changes
        # Calculate heritability
        h2 = heritability(a, v)
        dv = calculate_dV(I, X, v)

        # run recursions
        m = mean_phenotype_recursion(m, h2, I, v)
        d = disequilibrium_contribution_recursion(d, h2, dv)

        # Update variances
        v = phenotypic_variance(V0, d)
        a = additive_genetic_variance(A0, d)
        
        # Store history
        history['mean_phenotype'].append(m)
        history['phenotypic_variance'].append(v)
        history['additive_genetic_variance'].append(a)
        history['disequilibrium_contribution'].append(d)
        history['heritability'].append(h2)

    # Final heritability
    history['heritability'].append(heritability(a, v))

    # convert to numpy arrays for easier handling
    for key in history:
        history[key] = np.array(history[key])

    # calculate response to selection history
    history['response_to_selection'] = np.diff(history['mean_phenotype'])
    return history

def plot_results(history, output_file=None):
    """
    Plot the results of the simulation.

    Parameters:
    history (dict): The history of the simulation results.
    output_file (str, optional): If specified, save the plot to this file.
    """

    generations = len(history['mean_phenotype'])

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(range(generations), history['mean_phenotype'], label='Mean Phenotype')
    plt.xlabel('Generation')
    plt.ylabel('Mean Phenotype')
    plt.title('Mean Phenotype over Generations')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(range(generations), history['phenotypic_variance'], label='Phenotypic Variance', color='orange')
    plt.xlabel('Generation')
    plt.ylabel('Phenotypic Variance')
    plt.title('Phenotypic Variance over Generations')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(range(generations), history['additive_genetic_variance'], label='Additive Genetic Variance', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Additive Genetic Variance')
    plt.title('Additive Genetic Variance over Generations')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(range(generations - 1), history['response_to_selection'], label='Response to Selection', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Response to Selection')
    plt.title('Response to Selection over Generations')
    plt.legend()

    # do disequilibrium and heritability as well
    plt.subplot(3, 2, 5)
    plt.plot(range(generations), history['disequilibrium_contribution'], label='Disequilibrium Contribution', color='purple')
    plt.xlabel('Generation')
    plt.ylabel('Disequilibrium Contribution')
    plt.title('Disequilibrium Contribution over Generations')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(range(generations), history['heritability'], label='Heritability', color='brown')
    plt.xlabel('Generation')
    plt.ylabel('Heritability')
    plt.title('Heritability over Generations')
    plt.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)

    # plt.show()

def sim_with_unstable_d(M0, V0, A0, P, generations, d0 = 0.0):
    # Store history, add initial values
    history = {
        'mean_phenotype': [M0],
        'phenotypic_variance': [V0],
        'additive_genetic_variance': [A0],
        'disequilibrium_contribution': [d0],
        'heritability': [],
    }

    # get selection parameters
    # capital letters for constants across generations
    X, _, I = calculate_selection_parameters(P)

    # Initialize variables
    m = M0
    v = V0
    a = A0
    d = d0  # Initial disequilibrium contribution

    # simulate over generations
    for gen in range(generations):
        # calculate current and changes
        # Calculate heritability
        h2 = heritability(a, v)
        dv = calculate_dV(I, X, v)

        # run recursions
        m = mean_phenotype_recursion(m, h2, I, v)
        d = disequilibrium_contribution_recursion(d, h2, dv)
        # change d randomly by 1%
        d *= np.random.uniform(0.99, 1.01)

        # Update variances
        v = phenotypic_variance(V0, d)
        a = additive_genetic_variance(A0, d)
        
        # Store history
        history['mean_phenotype'].append(m)
        history['phenotypic_variance'].append(v)
        history['additive_genetic_variance'].append(a)
        history['disequilibrium_contribution'].append(d)
        history['heritability'].append(h2)

    # Final heritability
    history['heritability'].append(heritability(a, v))

    # convert to numpy arrays for easier handling
    for key in history:
        history[key] = np.array(history[key])

    # calculate response to selection history
    history['response_to_selection'] = np.diff(history['mean_phenotype'])
    return history

def sim_with_periodic_selection(M0, V0, A0, P, generations = 50, d0 = 0.0, select_for = 10):
    # Store history, add initial values
    history = {
        'mean_phenotype': [M0],
        'phenotypic_variance': [V0],
        'additive_genetic_variance': [A0],
        'disequilibrium_contribution': [d0],
        'heritability': [],
    }

    # get selection parameters
    # capital letters for constants across generations
    X, _, I = calculate_selection_parameters(P)

    # Initialize variables
    m = M0
    v = V0
    a = A0
    d = d0  # Initial disequilibrium contribution

    # simulate over generations
    for gen in range(generations):
        # calculate current and changes
        # Calculate heritability
        h2 = heritability(a, v)
        dv = calculate_dV(I, X, v)

        # run recursions
        m = mean_phenotype_recursion(m, h2, I, v)
        # select for 10 generations, then relax for 10 generations, then repeat
        if (gen // select_for) % 2 == 0:
            d = disequilibrium_contribution_recursion(d, h2, dv)
        else:
            d = 0.5 * d  # decay disequilibrium when not selecting

        # Update variances
        v = phenotypic_variance(V0, d)
        a = additive_genetic_variance(A0, d)
        
        # Store history
        history['mean_phenotype'].append(m)
        history['phenotypic_variance'].append(v)
        history['additive_genetic_variance'].append(a)
        history['disequilibrium_contribution'].append(d)
        history['heritability'].append(h2)

    # Final heritability
    history['heritability'].append(heritability(a, v))

    # convert to numpy arrays for easier handling
    for key in history:
        history[key] = np.array(history[key])

    # calculate response to selection history
    history['response_to_selection'] = np.diff(history['mean_phenotype'])
    return history
if __name__ == "__main__":
    # Example parameters
    M0 = 0.0  # Initial mean phenotype
    V0 = 100.0  # Initial phenotypic variance
    A0 = 50.0  # Initial additive genetic variance
    P = 0.2   # Proportion selected
    generations = 10  # Number of generations to simulate

    # Run simulation
    results = sim_with_periodic_selection(M0, V0, A0, P)

    # Print results
    for key, value in results.items():
        print(f"{key}: {value}")

    # Plot results
    plot_results(results, "/Users/heguoyu/Desktop/eeb314Final/figure_return")
    

    # calc eq, general solution? for some
    # stability
    # non-linear, multivariate model
    # Jacobian, lambda.

