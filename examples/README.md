# `raxpy` Examples and Benchmarks

This directory contains demonstrations and tutorials showing the capabilities of `raxpy` across several problem domains.

---

## Directory Overview

| Example / Directory | Domain | Key Features Demonstrated |
| :---- | :---- | :---- |
| **`Simple_Experiment_Demonstration.ipynb`** | Numerical Function | Minimal quickstart: annotated scalar functions, optional parameters, space-filling batch generation, and result extraction. |
| **`api_walkthrough.ipynb`** | Core API Tour | Progressive type annotation, `@validate_at_runtime`, hierarchical dataclasses, branching `Union` types, space-filling diagnostic metrics, and pairplot visualizations. |
| **`distributed_mpi_example.py`** | High-Performance Computing | Distributed parallel batch execution across multiple worker ranks using MPI (`mpi4py`). |
| **`California_Housing_NAS_HPO.ipynb`** | Neural Architecture Search | ANN architecture search and hyperparameter tuning on California Housing regression using TensorFlow/Keras. |
| **`California_Housing_NAS_Optuna_HPO.ipynb`** | HPO Framework Adapter | Integrating `raxpy` space definitions with Optuna's study optimization engine (`raxpy.adapters.optuna`). |
| **`Study_Experiment_Design_Generation.ipynb`** | Methodological Study | Systematic benchmark comparing space-filling algorithms (LHS, MaxPro SA, random) across concept-subspaces. |
| **`vision_benchmark/`** | Deep Learning / Vision | ResNet-34 vs. ViT-Small architecture search, branching factor spaces, conditional data augmentation, automated tabular results export. |

---

## 1\. Interactive Notebooks

To run the interactive Jupyter notebooks, install JupyterLab or Notebook:

pip install jupyterlab

jupyter lab examples/

### `Simple_Experiment_Demonstration.ipynb`

A self-contained, 5-minute introduction:

1. Defines an annotated mathematical function with a continuous input \$x\_1 \\in \[3.0, 4.0\]\$ and an optional continuous input \$x\_2 \\in \[0.0, 3.0\]\$ defaulting to `1.5`.  
2. Generates an optimized 100-point space-filling design using `raxpy.perform_experiment`.  
3. Demonstrates how null values (`None`) are automatically handled during function execution and outputs are gathered.

### `api_walkthrough.ipynb`

A comprehensive tour of the `raxpy` API:

- **Progressive typing**: Evolving unannotated Python functions into annotated design spaces.  
- **Runtime validation**: Applying `@raxpy.validate_at_runtime()` to enforce domain bounds during standard Python execution.  
- **Hierarchical & branching types**: Structuring complex input spaces using nested dataclasses and `typing.Union`.  
- **Diagnostic evaluation**: Measuring geometric properties with `raxpy.measure` (`compute_opt_coverage`, `compute_min_interpoint_dist`, `compute_star_discrepancy`, `compute_max_pro`).  
- **Visualization**: Generating scatterplot matrices with `raxpy.does.plots.plot_scatterplot_matrix`.

### `California_Housing_NAS_HPO.ipynb` & `California_Housing_NAS_Optuna_HPO.ipynb`

Explore deep neural network architectures and hyperparameter optimization for tabular regression:

- Modular layer stacks with variable depth, neuron counts, and dropout rates.  
- Dynamic optimizer choice (SGD with momentum vs. Adam with decoupled schedules).  
- Interoperability with Optuna via `raxpy.adapters.optuna`.  
- *Additional requirements*: `pip install tensorflow scikit-learn optuna`

### `Study_Experiment_Design_Generation.ipynb`

An empirical methodology notebook that benchmarks different design-synthesis strategies:

- Evaluates tree-traversal merging, subspace allocation, and Latin Hypercube sampling with value pooling.  
- Quantifies discrepancy and minimum interpoint distance across multi-level concept-subspaces.  
- Reproduces diagnostic figures from published research.

---

## 2\. Distributed Execution with MPI (`distributed_mpi_example.py`)

`raxpy` includes parallel execution runners for high-performance computing clusters via MPI. Rank 0 acts as the experiment coordinator (introspecting the function, designing the experiment, and dispatching tasks), while non-zero ranks act as workers executing trials.

### Prerequisites

Ensure an MPI implementation (e.g., OpenMPI, MPICH) and `mpi4py` are installed:

pip install mpi4py

### Running the Example

mpirun \-n 4 python examples/distributed\_mpi\_example.py

---

## 3\. Computer Vision Architecture Exploration (`vision_benchmark/`)

This benchmark compares convolutional (ResNet-34) and vision transformer (ViT-Small) backbones at matched parameter scales (\~21.3M parameters each) on MNIST digit classification under varying optimization and data augmentation regimes.

### Features

- Branching factors: `Union[ResNetConfig, ViTConfig]` selecting mutually exclusive architecture sub-spaces.

- Nullable factors: `Optional[AugmentationConfig]` conditionally enabling affine rotation and random crop padding.

- Hierarchical configurations: Modular dataclasses encapsulating component-specific hyperparameters.

- Global hyperparameters: Shared learning rate, weight decay, and mini-batch size.



More details are located in a separate readme.md file within that folder



---

Recommended Environment Setup

To run all examples and adapters in an isolated environment:

python \-m venv .venv

source .venv/bin/activate

pip install ".\[dev\]"

pip install ".\[mpi\]"

pip install torch torchvision optuna tensorflow scikit-learn matplotlib