"""Evalutes the experiment design used in the Vision Architecture Comparison"""

import matplotlib.pyplot as plt

import raxpy
from raxpy.does.measure import assess_design, compute_opt_coverage
from raxpy.does.plots import plot_scatterplot_matrix
from raxpy.does.random import generate_design as gd_random
from raxpy.adapters.torch_sobol import design_experiment as gd_flat_sobol

import run_experiment

n_points = 20
seed = 42

input_space = raxpy.function_spec.extract_input_space(
    run_experiment.evaluate_trial
)

# Lets (re)create the design that will be used in the experiment that considers
# hierarchical conditions
print("-" * 80)
print("Space-filling optimized experiment design")
print("-" * 80)

sf_doe = raxpy.design_experiment(
    input_space,
    n_points=n_points,
    seed=seed,
)

print(f"Concept Coverage Percentage: {compute_opt_coverage(sf_doe)}")
assess_design(sf_doe)
fig = plot_scatterplot_matrix(sf_doe)
fig.savefig("sf_doe_scatterplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n\n")


# Lets create a uMaxPro design for comparison
print("-" * 80)
print("Space-filling (uMaxPro) optimized experiment design")
print("-" * 80)

sf_umaxpro_doe = raxpy.design_experiment(
    input_space, n_points=n_points, seed=seed, optimize_projections="uMaxPro"
)

print(f"Concept Coverage Percentage: {compute_opt_coverage(sf_umaxpro_doe)}")
assess_design(sf_umaxpro_doe)
fig = plot_scatterplot_matrix(sf_umaxpro_doe)
fig.savefig("umaxpro_doe_scatterplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n\n")


# Lets compare these results to a random expertiment design that ignores
# hierarchical conditions
print("-" * 80)
print("Non-optimized (random) experiment design")
print("-" * 80)

random_doe = raxpy.design_experiment(
    input_space,
    design_algorithm=gd_random,
    n_points=n_points,
    optimize_projections=False,
    seed=seed,
)

print(f"Concept Coverage Percentage: {compute_opt_coverage(random_doe)}")
assess_design(random_doe)
fig = plot_scatterplot_matrix(random_doe)
fig.savefig("random_doe_scatterplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\n\n")


# Lets compare these results to a sobol design that ignores hierarchical
# conditions
print("-" * 80)
print("Non-optimized (flat-sobol) experiment design")
print("-" * 80)

flat_sobol_doe = raxpy.design_experiment(
    input_space,
    design_algorithm=gd_flat_sobol,
    n_points=n_points,
    optimize_projections=False,
    seed=seed,
)

print(f"Concept Coverage Percentage: {compute_opt_coverage(flat_sobol_doe)}")
assess_design(flat_sobol_doe)
fig = plot_scatterplot_matrix(flat_sobol_doe)
fig.savefig("flat_sobol_doe_scatterplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
