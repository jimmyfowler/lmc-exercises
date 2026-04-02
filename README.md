# lmc-exercises

This repository contains the exercises and Jupyter notebook demos that accompany the [LMC book](https://uw-ctrl.github.io/lmc-book/).

As the course progresses, the notebooks (exercises for homework and any demo presented in class) will be posted here.

## Contents

- `exercises`: This folder will contain notebooks where you need to fill in some code and run cells.
- `demos`: This folder will contain notebooks of some demo code used in class, or notebooks where you can just run and analyze the outputs more closely.




## Requirements

- Python 3.10 or newer (see `pyproject.toml` for the declared minimum).

## Setup

Create a virtual environment and install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs the packages listed in `pyproject.toml` (NumPy, SciPy, Matplotlib, JAX, Equinox, CVXPy, JupyterLab, etc.).

Alternatively, you can use Google Colab to run the notebooks. In that case, you may need to install any required packages yourself. To install a package in a Colab notebook, run the following command in a code cell:

```python
!pip install <package_name>
```
