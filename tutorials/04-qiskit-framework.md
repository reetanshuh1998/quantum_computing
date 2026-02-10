# Chapter 4: IBM Qiskit Framework

> *"Qiskit is the most popular open-source quantum SDK — with 13M+ downloads, 600K+ users, and 700+ universities teaching with it."* — IBM Quantum

---

## 🎯 Learning Goals

By the end of this chapter, you will:
- Install and set up Qiskit
- Understand the Qiskit architecture and components
- Build quantum circuits programmatically
- Run circuits on simulators
- Run circuits on real IBM quantum hardware
- Understand transpilation, primitives, and shots
- Use Qiskit Runtime for cloud execution

---

## 4.1 What Is Qiskit?

[Qiskit](https://www.ibm.com/quantum/qiskit) (pronounced "kiss-kit") is IBM's open-source quantum computing SDK. It provides everything you need to:

1. **Build** quantum circuits
2. **Optimize** them for specific hardware (transpilation)
3. **Execute** them on simulators or real quantum processors
4. **Analyze** the results

### Qiskit Architecture

```
+-------------------------------------------------------------+
|                    YOUR APPLICATION                           |
+-------------------------------------------------------------+
|            Qiskit Application Functions                       |
|   (Chemistry, Finance, ML, Optimization -- ready-made)       |
+-------------------------------------------------------------+
|                  Qiskit SDK (v2.x)                            |
|  +--------------+ +---------------+ +--------------------+   |
|  | Quantum      | | Transpiler    | | Primitives         |   |
|  | Circuit      | | (83x faster   | | (Sampler &         |   |
|  | Library      | |  than others) | |  Estimator)        |   |
|  +--------------+ +---------------+ +--------------------+   |
+-------------------------------------------------------------+
|                  Qiskit Runtime                               |
|         (Cloud execution, session management)                 |
+-------------------------------------------------------------+
|              IBM Quantum Hardware                             |
|   (Eagle 127q, Heron 156q, up to 1,121+ qubits)             |
+-------------------------------------------------------------+
```

---

## 4.2 Installation

### Basic Installation

```bash
# Install Qiskit SDK
pip install qiskit

# Install IBM Quantum Runtime (for real hardware access)
pip install qiskit-ibm-runtime

# Install visualization tools (optional but recommended)
pip install qiskit[visualization]
```

### Verify Installation

```python
import qiskit
print(f"Qiskit version: {qiskit.__version__}")

# You should see something like: Qiskit version: 2.x.x
```

### Set Up IBM Quantum Account (Free!)

1. Go to [https://quantum.ibm.com/](https://quantum.ibm.com/)
2. Create a free account
3. Copy your API token from the dashboard
4. Save it in your environment:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your account (only need to do this once)
# IMPORTANT: Use "ibm_quantum_platform" as the channel (not "ibm_quantum")
# The older "ibm_quantum" channel was deprecated in recent versions.
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_API_TOKEN_HERE",
    overwrite=True
)

# Verify it works
service = QiskitRuntimeService()
print("Available backends:")
for backend in service.backends():
    print(f"  {backend.name} - {backend.num_qubits} qubits")
```

> **Note on channel values (version-dependent):**
>
> | qiskit-ibm-runtime version | Valid channel values |
> |---|---|
> | Older (< 0.30) | "ibm_quantum", "ibm_cloud" |
> | Latest (0.30+) | "ibm_quantum_platform", "ibm_cloud" |
> |
> If you see `InvalidAccountError: Invalid channel value`, switch to "ibm_quantum_platform".

---

## 4.3 Building Your First Circuit

### Hello Quantum World!

```python
from qiskit import QuantumCircuit

# Create a circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Step 1: Apply Hadamard to qubit 0 (create superposition)
qc.h(0)

# Step 2: Apply CNOT (entangle qubit 0 and qubit 1)
qc.cx(0, 1)

# Step 3: Measure both qubits
qc.measure([0, 1], [0, 1])

# Draw the circuit
print(qc.draw())
```

This creates a **Bell State** -- the simplest entangled state!

### Understanding the Circuit

```python
# Circuit properties
print(f"Number of qubits: {qc.num_qubits}")       # 2
print(f"Number of classical bits: {qc.num_clbits}") # 2
print(f"Circuit depth: {qc.depth()}")               # 3
print(f"Gate counts: {qc.count_ops()}")             # {'h': 1, 'cx': 1, 'measure': 2}
```

---

## 4.4 All the Ways to Build Circuits

### Method 1: Gate by Gate (Most Common)

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(3, 3)

# Single-qubit gates
qc.x(0)           # Pauli-X (NOT) on qubit 0
qc.y(1)           # Pauli-Y on qubit 1
qc.z(2)           # Pauli-Z on qubit 2
qc.h(0)           # Hadamard on qubit 0
qc.s(1)           # S gate on qubit 1
qc.t(2)           # T gate on qubit 2

# Rotation gates
qc.rx(3.14, 0)    # Rotate around X by pi
qc.ry(1.57, 1)    # Rotate around Y by pi/2
qc.rz(0.78, 2)    # Rotate around Z

# Multi-qubit gates
qc.cx(0, 1)       # CNOT: control=0, target=1
qc.cz(1, 2)       # CZ: control=1, target=2
qc.ccx(0, 1, 2)   # Toffoli: controls=0,1, target=2
qc.swap(0, 1)     # SWAP qubits 0 and 1

# Barriers (visual separators)
qc.barrier()

# Measurement
qc.measure([0, 1, 2], [0, 1, 2])

print(qc.draw())
```

### Method 2: Apply to Multiple Qubits

```python
qc = QuantumCircuit(4)

# Apply H to all qubits at once
qc.h([0, 1, 2, 3])

# Apply CNOT in a chain
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

print(qc.draw())
```

### Method 3: Compose Circuits

```python
# Build sub-circuits and combine them
bell = QuantumCircuit(2, name="Bell")
bell.h(0)
bell.cx(0, 1)

# Main circuit
main = QuantumCircuit(4)
main.compose(bell, qubits=[0, 1], inplace=True)
main.compose(bell, qubits=[2, 3], inplace=True)

print(main.draw())
```

### Method 4: Parameterized Circuits (for Variational Algorithms)

```python
from qiskit.circuit import Parameter

theta = Parameter('theta')
phi = Parameter('phi')

qc = QuantumCircuit(2)
qc.ry(theta, 0)
qc.ry(phi, 1)
qc.cx(0, 1)

print(qc.draw())

# Bind values later
bound_qc = qc.assign_parameters({theta: 1.57, phi: 3.14})
print(bound_qc.draw())
```

---

## 4.5 Circuit Visualization

```python
print(qc.draw())          # ASCII text (default)
print(qc.draw('text'))    # Same as above
qc.draw('mpl')            # Matplotlib figure (pretty)
qc.draw('latex')           # LaTeX source (publication quality)
```

---

## 4.6 Running on Simulators

### Using StatevectorSampler (Exact Simulation)

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

# Build circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Run on simulator
sampler = StatevectorSampler()
job = sampler.run([qc], shots=1024)
result = job.result()

# Get measurement counts
counts = result[0].data.c.get_counts()
print(counts)
# Output: {'00': 512, '11': 512}  (approximately)
```

### Visualizing Results

```python
from qiskit.visualization import plot_histogram
fig = plot_histogram(counts)
fig.savefig('results.png')
```

---

## 4.7 Running on Real IBM Quantum Hardware

### Step 1: Connect to IBM Quantum

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print(f"Selected backend: {backend.name}")
```

### Step 2: Transpile Your Circuit

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
transpiled_qc = pm.run(qc)
```

**Optimization levels:**

| Level | Speed | Quality | Use When |
|-------|-------|---------|----------|
| 0 | Fastest | Minimal optimization | Quick tests |
| 1 | Fast | Light optimization | Development |
| 2 | Medium | Heavy optimization | Good results |
| 3 | Slowest | Maximum optimization | Production / best results |

### Step 3: Execute with Sampler

```python
from qiskit_ibm_runtime import SamplerV2

sampler = SamplerV2(backend)
job = sampler.run([transpiled_qc], shots=4096)
result = job.result()
counts = result[0].data.c.get_counts()
print(counts)
```

### Step 3b: Execute with Estimator

```python
from qiskit_ibm_runtime import EstimatorV2
from qiskit.quantum_info import SparsePauliOp

observable = SparsePauliOp("ZZ")
qc_est = QuantumCircuit(2)
qc_est.h(0)
qc_est.cx(0, 1)
transpiled_est = pm.run(qc_est)

estimator = EstimatorV2(backend)
job = estimator.run([(transpiled_est, observable)])
result = job.result()
print(f"<ZZ> = {result[0].data.evs}")
```

| Primitive | Returns | Use When |
|-----------|---------|----------|
| **Sampler** | Measurement counts/probabilities | You want the distribution of outcomes |
| **Estimator** | Expectation values of observables | You want a physical quantity (energy, correlation) |

---

## 4.8 Troubleshooting Common Issues

### Issue: `InvalidAccountError: Invalid channel value`

```python
# OLD (deprecated in qiskit-ibm-runtime >= 0.30):
QiskitRuntimeService.save_account(channel="ibm_quantum", ...)

# NEW (correct for latest versions):
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", ...)
```

### Issue: `ModuleNotFoundError`

```bash
pip install qiskit-ibm-runtime
```

### Issue: Results are very noisy on real hardware

- Use more shots (4096+)
- Use error mitigation techniques (covered in Chapter 8)
- Choose a less busy backend with better error rates

---

## 4.9 Complete Working Example

```python
"""
Complete Qiskit Example: Creating and Analyzing a GHZ State
"""
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.barrier()
qc.measure([0, 1, 2], [0, 1, 2])

print("=== GHZ State Circuit ===")
print(qc.draw())

sampler = StatevectorSampler()
job = sampler.run([qc], shots=4096)
result = job.result()
counts = result[0].data.c.get_counts()

print("\n=== Measurement Results (4096 shots) ===")
total = sum(counts.values())
for state in sorted(counts.keys()):
    count = counts[state]
    prob = count / total
    bar = '#' * int(prob * 50)
    print(f"|{state}>: {count:4d} ({prob:6.2%}) {bar}")
```

---

## Summary

1. **Qiskit** is the world's most popular quantum SDK
2. **Installation**: `pip install qiskit qiskit-ibm-runtime`
3. **QuantumCircuit** is the core object
4. **Transpiler** optimizes circuits for real hardware
5. **Two primitives**: Sampler (counts) and Estimator (expectation values)
6. **Channel**: Use "ibm_quantum_platform" (not deprecated "ibm_quantum")
7. **Real hardware** introduces noise -- use error mitigation

---

## Next Chapter

**[Chapter 5: Quantum Algorithms -->](05-quantum-algorithms.md)**

---

*[<-- Previous: Chapter 3 - Gates and Circuits](03-gates-and-circuits.md) | [Back to Main README](../README.md)*