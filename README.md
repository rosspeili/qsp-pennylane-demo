# Learning QSP Phase Angles via Gradient Descent

Training Quantum Signal Processing (QSP) phase angles from scratch via JAX gradient descent.

A PennyLane community demo by Ross Peili.

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-efcefa?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-bae6fd?style=flat-square)
![PennyLane](https://img.shields.io/badge/PennyLane-0.44%2B-e9d5ff?style=flat-square)
![JAX](https://img.shields.io/badge/JAX-0.4.25%2B-a5f3fc?style=flat-square)
![Optax](https://img.shields.io/badge/Optax-0.2.2%2B-fce7f3?style=flat-square)
[![nbviewer](https://img.shields.io/badge/View-nbviewer-fef9c3?style=flat-square&logo=jupyter&logoColor=555)](https://nbviewer.org/github/rosspeili/qsp-pennylane-demo/blob/main/demo.ipynb)
[![Colab](https://img.shields.io/badge/Open_in-Colab-fed7aa?style=flat-square&logo=google-colab&logoColor=555)](https://colab.research.google.com/drive/1AQ3ZNxwo4ed0DtVDlsz_jLeMAgf_UJgQ?usp=sharing)
[![PennyLane Community](https://img.shields.io/badge/PennyLane-Community_Demo-bbf7d0?style=flat-square)](https://pennylane.ai/qml/demos_community)

</div>

---

## What This Is

Quantum Signal Processing encodes polynomial transformations of a signal into a quantum circuit
via a sequence of phase-shifted oracle calls. The standard approach computes the required phase
angles analytically given a target polynomial. This demo takes the opposite route: it starts
from random phase angles and trains them using gradient-based optimization (JAX + Optax) to
recover a target polynomial transformation.

The result is a working recipe for practitioners who need to optimize QSP angles for novel
polynomials where no analytic solver exists or where the polynomial is implicitly defined by
a loss function.

## Setup

```bash
git clone https://github.com/rosspeili/qsp-pennylane-demo
cd qsp-pennylane-demo
pip install -r requirements.txt
```

## Running the Demo

```bash
jupyter notebook demo.ipynb
```

Or run the tests:

```bash
pytest tests/ -v
```

## Structure

```
qsp-pennylane-demo/
├── demo.ipynb          # Main community demo notebook
├── qsp_jax/
│   ├── __init__.py
│   └── circuit.py      # Circuit construction and loss function
├── tests/
│   └── test_circuit.py # Unit tests
├── requirements.txt
├── LICENSE             # Apache 2.0
└── README.md
```

## Key Concepts

- **Signal oracle**: `W(x) = H @ RZ(-2*arccos(x)) @ H`, encoding signal `x ∈ (-1, 1)` in the top-left matrix element
- **QSP sequence**: Flat alternating circuit — one phase rotation `RZ(-2*phi_k)` per signal query `W(x)`
- **Polynomial encoding**: The expectation value `<X>` encodes a degree-d polynomial in `x` determined by the phase angles
- **Training**: Adam optimizer (Optax) minimizes MSE between circuit output and target polynomial via `jax.grad`
- **Note**: The circuit is implemented as inline `qp.RZ` + `qp.Hadamard` gates, not `qp.QSVT`, to preserve JAX traceability

## Target Polynomial

Default target: degree-5 Chebyshev approximation of `sin(x)` on `[-1, 1]`.

This is an odd polynomial, consistent with QSP conventions for odd-degree transformations.

## Results

After ~500 Adam steps, the trained phase angles reproduce the target polynomial to MSE < 1e-3
on a uniform grid of 64 signal values.

## References

- Martyn et al., [A Grand Unification of Quantum Algorithms](https://arxiv.org/abs/2105.02859), PRX Quantum 2021
- Gilyen et al., [Quantum singular value transformation](https://arxiv.org/abs/1806.01838), STOC 2019

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Press Release

[📄 View the Press Release](https://docs.google.com/document/d/1pamiIksg56KKkIIrCPj18YvCLocLVchkUFW3TM5Xs_o/edit?usp=sharing)
