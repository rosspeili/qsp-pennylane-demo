# qsp-pennylane-demo

Training Quantum Signal Processing (QSP) phase angles from scratch via JAX gradient descent.

A PennyLane community demo by Ross Peili.

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
- **Polynomial encoding**: The expectation value `<Z>` encodes a degree-d polynomial in `x` determined by the phase angles
- **Training**: Adam optimizer (Optax) minimizes MSE between circuit output and target polynomial via `jax.grad`
- **Note**: The circuit is implemented as inline `qml.RZ` + `qml.Hadamard` gates, not `qml.QSVT`, to preserve JAX traceability

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
