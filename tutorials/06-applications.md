# Chapter 6: Real-World Applications of Quantum Computing

> *"The question is no longer whether quantum computers will be useful, but when and for which problems they will first deliver advantage."* — McKinsey Quantum Technology Monitor 2025

---

## 🎯 Learning Goals

By the end of this chapter, you will understand:
- How quantum computing is applied in drug discovery and chemistry
- Financial applications: portfolio optimization, risk analysis, fraud detection
- Cryptography implications and post-quantum security
- Logistics and supply chain optimization
- AI and machine learning quantum enhancements
- Materials science and energy applications
- The concept of quantum-centric supercomputing

---

## 6.1 Drug Discovery and Molecular Simulation

### The Problem

Designing new drugs requires understanding how molecules interact at the atomic level. Classical computers struggle because:

- A molecule with N electrons has a quantum state space that grows as 2^N
- A caffeine molecule (24 atoms) would need more classical bits than atoms in the universe to simulate exactly
- Current methods use approximations that miss critical details

### How Quantum Helps

Quantum computers simulate molecules **natively** — they use quantum mechanics to simulate quantum mechanics.

| Task | Classical Approach | Quantum Approach |
|---|---|---|
| Molecular energy | Approximate (DFT, coupled cluster) | Exact via VQE / QPE |
| Protein folding | Statistical sampling | Quantum optimization |
| Drug-target binding | Classical MD simulation | Quantum chemistry simulation |

### Qiskit Example: Simulating H2 Molecule

```python
"""
Simulating the Hydrogen molecule (H2) energy curve
This shows how bond distance affects molecular energy
"""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import numpy as np
from scipy.optimize import minimize

def run_vqe(hamiltonian_coefficients):
    """Run VQE for a given Hamiltonian"""
    hamiltonian = SparsePauliOp.from_list(hamiltonian_coefficients)
    
    theta = Parameter('theta')
    ansatz = QuantumCircuit(2)
    ansatz.ry(theta, 0)
    ansatz.cx(0, 1)
    
    estimator = StatevectorEstimator()
    
    def cost(params):
        bound = ansatz.assign_parameters({theta: params[0]})
        job = estimator.run([(bound, hamiltonian)])
        return float(job.result()[0].data.evs)
    
    result = minimize(cost, x0=[0.0], method='COBYLA')
    return result.fun

# H2 Hamiltonian at equilibrium bond distance (0.735 Angstroms)
h2_hamiltonian = [
    ("II", -1.052),
    ("IZ",  0.398),
    ("ZI", -0.398),
    ("ZZ", -0.011),
    ("XX",  0.181)
]

energy = run_vqe(h2_hamiltonian)
print(f"H2 ground state energy: {energy:.6f} Hartree")
print(f"Expected (exact):       -1.137284 Hartree")
```

### Real-World Impact

| Company/Lab | Application | Status |
|---|---|---|
| IBM + pharmaceutical partners | Molecular simulation for drug candidates | Active research |
| Google Quantum AI | Simulating chemical reactions | Published results |
| Roche + Cambridge Quantum | Drug discovery pipelines | Pilot programs |
| Cleveland Clinic + IBM | Healthcare and genomics research | Partnership active |

---

## 6.2 Finance

### Portfolio Optimization

**Problem**: Given N assets, find the allocation that maximizes return while minimizing risk. With many assets and constraints, this becomes a combinatorial explosion.

**Quantum approach**: QAOA and VQE can explore the solution space more efficiently.

```python
"""
Simplified Quantum Portfolio Optimization
Using QAOA-style approach for a 3-asset portfolio
"""
import numpy as np

# Asset data (simplified)
expected_returns = [0.05, 0.08, 0.12]  # 5%, 8%, 12%
risk = [0.02, 0.05, 0.10]

n_assets = 3

def evaluate_portfolio(bitstring, returns, risks, risk_aversion=0.5):
    """Score a portfolio selection (1 = include asset, 0 = exclude)"""
    total_return = sum(int(b) * r for b, r in zip(bitstring, returns))
    total_risk = sum(int(b) * v for b, v in zip(bitstring, risks))
    return total_return - risk_aversion * total_risk

print("=== All Portfolio Combinations ===")
best_score = -999
best_portfolio = ""
for i in range(2 ** n_assets):
    bits = format(i, f'0{n_assets}b')
    if bits == '000':
        continue
    assets = [f"{'ABC'[j]}" for j in range(n_assets) if bits[j] == '1']
    score = evaluate_portfolio(bits, expected_returns, risk, 0.5)
    ret = sum(int(b) * r for b, r in zip(bits, expected_returns))
    rsk = sum(int(b) * v for b, v in zip(bits, risk))
    print(f"  {bits} ({','.join(assets):>5}): Return={ret:.2%}, Risk={rsk:.2%}, Score={score:.4f}")
    if score > best_score:
        best_score = score
        best_portfolio = bits

print(f"\nBest portfolio: {best_portfolio} (Score: {best_score:.4f})")
print("A quantum optimizer (QAOA/VQE) finds this efficiently even for 1000s of assets!")
```

### Industry Adoption

| Institution | Application | Details |
|---|---|---|
| Goldman Sachs + QC Ware | Portfolio optimization | Quantum algorithms for allocation |
| JPMorgan Chase | Option pricing, risk | Quantum Monte Carlo research |
| HSBC + IBM | Portfolio optimization | Qiskit-based pilot |
| Vanguard + IBM | Index tracking | Quantum optimization |

---

## 6.3 Cryptography and Cybersecurity

### The Threat: Shor's Algorithm

Shor's algorithm can factor large numbers in polynomial time, breaking:

| Encryption | Status | Quantum Threat |
|---|---|---|
| **RSA** (2048-bit) | Widely used | BROKEN by Shor's (needs ~4,000 logical qubits) |
| **ECC** (Elliptic Curve) | Widely used | BROKEN by Shor's variant |
| **AES-256** | Symmetric encryption | WEAKENED (Grover's: effectively 128-bit security) |
| **SHA-256** | Hash function | WEAKENED (Grover's: effectively 128-bit security) |

### The Solution: Post-Quantum Cryptography (PQC)

NIST has standardized new cryptographic algorithms resistant to quantum attacks:

| PQC Algorithm | Type | Use Case |
|---|---|---|
| **ML-KEM (CRYSTALS-Kyber)** | Lattice-based | Key encapsulation |
| **ML-DSA (CRYSTALS-Dilithium)** | Lattice-based | Digital signatures |
| **SLH-DSA (SPHINCS+)** | Hash-based | Digital signatures |
| **FN-DSA (FALCON)** | Lattice-based | Digital signatures |

> **"Harvest now, decrypt later"** attacks are happening TODAY — adversaries store encrypted data to decrypt it later when quantum computers are powerful enough.

---

## 6.4 Logistics and Supply Chain Optimization

### The Problem

Real-world optimization problems are often NP-hard:
- Vehicle routing (deliver to 1,000 locations — how?)
- Warehouse layout optimization
- Supply chain scheduling
- Network flow optimization

### Quantum Approach

QAOA and quantum annealing can explore solution spaces more efficiently.

```python
"""
Vehicle Routing Conceptual Example
"""
from itertools import permutations

cities = ['Warehouse', 'City A', 'City B', 'City C']
distances = {
    (0,1): 10, (0,2): 15, (0,3): 20,
    (1,2): 12, (1,3): 18, (2,3): 8
}
for (i,j), d in list(distances.items()):
    distances[(j,i)] = d

best_route = None
best_distance = float('inf')
for perm in permutations(range(1, len(cities))):
    route = [0] + list(perm) + [0]
    total = sum(distances[(route[i], route[i+1])] for i in range(len(route)-1))
    if total < best_distance:
        best_distance = total
        best_route = route

print(f"Optimal route: {' -> '.join(cities[i] for i in best_route)}")
print(f"Total distance: {best_distance} km")
print(f"\nWith 20 cities: 20! = 2.4 x 10^18 routes (classical nightmare)")
print(f"Quantum optimizer (QAOA) finds near-optimal solutions much faster")
```

---

## 6.5 Artificial Intelligence and Machine Learning

### Quantum Machine Learning (QML)

| Technique | Description | Potential Advantage |
|---|---|---|
| **Quantum Kernels** | Compute similarity in quantum feature space | Access to richer feature maps |
| **Quantum Neural Networks** | Parameterized quantum circuits as neural nets | Expressiveness for certain data |
| **Quantum Data Encoding** | Encode classical data into quantum states | Novel data representations |
| **Quantum Sampling** | Generate samples from complex distributions | Faster generative models |

### Qiskit Example: Quantum Feature Map

```python
"""
Quantum Kernel for Classification
"""
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

def quantum_feature_map(data_point, n_qubits=2):
    """Encode a classical data point into a quantum state"""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        qc.rz(data_point[i % len(data_point)], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(data_point[i % len(data_point)] ** 2, i)
    return qc

def quantum_kernel(x1, x2, n_qubits=2):
    """Compute quantum kernel (similarity) between two data points"""
    sv1 = Statevector.from_instruction(quantum_feature_map(x1, n_qubits))
    sv2 = Statevector.from_instruction(quantum_feature_map(x2, n_qubits))
    return float(np.abs(sv1.inner(sv2)) ** 2)

# Sample data
data = [[0.5, 1.2], [0.6, 1.1], [2.1, 0.3], [2.3, 0.4]]
labels = ['A', 'A', 'B', 'B']

print("=== Quantum Kernel Matrix ===")
n = len(data)
for i in range(n):
    row = [f"{quantum_kernel(data[i], data[j]):.3f}" for j in range(n)]
    print(f"  x{i}({labels[i]}): {' '.join(row)}")

print("\nSame-class points have HIGHER kernel values (more similar)")
```

---

## 6.6 Materials Science and Energy

### Applications

| Domain | Quantum Application | Impact |
|---|---|---|
| **Batteries** | Simulate lithium-ion interactions | Better energy storage |
| **Solar Cells** | Model photovoltaic materials | Higher efficiency |
| **Catalysts** | Design better industrial catalysts | Cleaner chemical processes |
| **Superconductors** | Simulate high-Tc materials | Room-temperature superconductors |
| **Carbon Capture** | Model CO2 absorption | Climate change mitigation |

---

## 6.7 Quantum-Centric Supercomputing

### The Vision

IBM's roadmap envisions **quantum processors embedded alongside classical HPC systems**:

```
+--------------------------------------------------------------+
|                  Quantum-Centric Supercomputer                |
|  +------------------+     +------------------+               |
|  | Classical HPC    |<--->| Quantum          |               |
|  | (CPU + GPU)      |     | Processors       |               |
|  +------------------+     +------------------+               |
|         Workload Orchestration (Qiskit Runtime)              |
+--------------------------------------------------------------+
```

Key enablers (2025-2026):
- **Qiskit C API**: Call quantum circuits from C/C++/Fortran HPC codes
- **Circuit knitting**: Split large quantum circuits across multiple processors
- **Error mitigation**: Software techniques to improve noisy results

---

## 6.8 Timeline: When Will Quantum Impact Each Industry?

| Timeline | Industries | Type of Impact |
|---|---|---|
| **Now (2025-2026)** | Research, pharma, finance | Quantum utility demonstrated; pilot programs |
| **Near-term (2027-2029)** | Chemistry, materials, optimization | First quantum advantage for specific tasks |
| **Medium-term (2030-2035)** | Crypto, AI, logistics, energy | Broad quantum advantage; PQC mandatory |
| **Long-term (2035+)** | All industries | Fault-tolerant QC; transformative impact |

---

## Summary

1. **Drug Discovery**: Quantum simulates molecules natively; VQE finds ground state energies
2. **Finance**: Portfolio optimization, risk analysis with quadratic speedups
3. **Cryptography**: Shor's threatens RSA; post-quantum cryptography is the solution
4. **Logistics**: QAOA tackles combinatorial optimization
5. **AI/ML**: Quantum kernels, quantum neural networks
6. **Materials**: Better batteries, catalysts, superconductors via quantum simulation
7. **The future is hybrid**: Quantum-centric supercomputing combines quantum + classical HPC

---

## Next Chapter

**[Chapter 7: Hardware and Landscape -->](07-hardware-landscape.md)**

---

*[<-- Previous: Chapter 5 - Quantum Algorithms](05-quantum-algorithms.md) | [Back to Main README](../README.md)*