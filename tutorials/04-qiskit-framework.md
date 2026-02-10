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
┌─────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                       │
├─────────────────────────────────────────────────────────┤
│            Qiskit Application Functions                   │
│   (Chemistry, Finance, ML, Optimization — ready-made)    │
├─────────────────────────────────────────────────────────┤
│                  Qiskit SDK (v2.x)                        │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────┐ │
│  │ Quantum      │ │ Transpiler   │ │ Primitives        │ │
│  │ Circuit      │ │ (83x faster  │ │ (Sampler &        │ │
│  │ Library      │ │  than others)│ │  Estimator)       │ │
│  └─────────────┘ └──────────────┘ └───────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Qiskit Runtime                           │
│         (Cloud execution, session management)             │
├─────────────────────────────────────────────────────────┤
│              IBM Quantum Hardware                         │
│   (Eagle 127q, Heron 156q, up to 1,121+ qubits)         │
└─────────────────────────────────────────────────────────┘
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
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_API_TOKEN_HERE",
    overwrite=True
)

# Verify it works
service = QiskitRuntimeService()
print("Available backends:")
for backend in service.backends():
    print(f"  {backend.name} - {backend.num_qubits} qubits")
```

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

**Output:**
```
     ┌───┐     ┌─┐
q_0: ┤ H ├──■──┤M├───
     └───┘┌─┴─┐└╥┘┌─┐
q_1: ────┤ X ├─╫─┤M├
          └───┘ ║ └╥┘
c: 2/═══════════╩══╩═
                0  1
```

This creates a **Bell State** — the simplest entangled state!

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
main.compose(bell, qubits=[0, 1], inplace=True)  # Apply bell to qubits 0,1
main.compose(bell, qubits=[2, 3], inplace=True)  # Apply bell to qubits 2,3

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
# Output shows theta and phi as symbolic parameters

# Bind values later
bound_qc = qc.assign_parameters({theta: 1.57, phi: 3.14})
print(bound_qc.draw())
```

---

## 4.5 Circuit Visualization

### Text Drawing (Default)

```python
print(qc.draw())          # ASCII text
print(qc.draw('text'))    # Same as above
```

### Matplotlib Drawing (Pretty)

```python
qc.draw('mpl')            # Returns a matplotlib figure
qc.draw('mpl', filename='my_circuit.png')  # Save to file
```

### LaTeX Drawing (Publication Quality)

```python
qc.draw('latex')           # Returns LaTeX source
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

### Understanding Results

```python
counts = {'00': 523, '11': 501}

# What this means:
# - We ran the circuit 1024 times (shots)
# - 523 times we measured both qubits as 0 (state |00>)
# - 501 times we measured both qubits as 1 (state |11>)
# - We NEVER got |01> or |10> (because they're entangled!)
# - This confirms the Bell State (|00> + |11>)/sqrt(2)

# Calculate probabilities
total_shots = sum(counts.values())
for state, count in counts.items():
    prob = count / total_shots
    print(f"|{state}>: {count}/{total_shots} = {prob:.3f} ({prob*100:.1f}%)")
```

### Visualizing Results

```python
from qiskit.visualization import plot_histogram

# Plot a histogram of results
fig = plot_histogram(counts)
fig.savefig('results.png')
```

---

## 4.7 Running on Real IBM Quantum Hardware

### Step 1: Connect to IBM Quantum

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Load saved account
service = QiskitRuntimeService()

# Choose a backend (real quantum computer)
backend = service.least_busy(operational=True, simulator=False)
print(f"Selected backend: {backend.name}")
print(f"Number of qubits: {backend.num_qubits}")
```

### Step 2: Transpile Your Circuit

Your ideal circuit uses abstract qubits and gates. Real hardware has:
- **Limited connectivity** (not all qubits are connected)
- **Native gate set** (only certain gates are physically implemented)
- **Noise characteristics** (some qubits/gates are better than others)

The **transpiler** converts your circuit to run on specific hardware:

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Build your ideal circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Transpile for the specific backend
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
transpiled_qc = pm.run(qc)

print(f"Original depth: {qc.depth()}")
print(f"Transpiled depth: {transpiled_qc.depth()}")
print(f"Original gates: {qc.count_ops()}")
print(f"Transpiled gates: {transpiled_qc.count_ops()}")
```

**Optimization levels:**

| Level | Speed | Quality | Use When |
|-------|-------|---------|----------|
| 0 | Fastest | Minimal optimization | Quick tests |
| 1 | Fast | Light optimization | Development |
| 2 | Medium | Heavy optimization | Good results |
| 3 | Slowest | Maximum optimization | Production / best results |

### Step 3: Execute with Primitives

Qiskit v2.x uses two **primitives** as the main execution interfaces:

#### Sampler (Get Measurement Counts)

```python
from qiskit_ibm_runtime import SamplerV2

sampler = SamplerV2(backend)
job = sampler.run([transpiled_qc], shots=4096)

# Wait for results (real hardware takes seconds to minutes)
result = job.result()
counts = result[0].data.c.get_counts()
print(counts)
# Output: {'00': 1985, '11': 1943, '01': 87, '10': 81}
#          Notice small counts for 01/10 -- this is HARDWARE NOISE!
```

#### Estimator (Get Expectation Values)

```python
from qiskit_ibm_runtime import EstimatorV2
from qiskit.quantum_info import SparsePauliOp

# Define an observable to measure
observable = SparsePauliOp("ZZ")  # Measure Z tensor Z correlation

# Build circuit (no measurement gates needed for Estimator!)
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# Transpile
transpiled_qc = pm.run(qc)

# Run estimator
estimator = EstimatorV2(backend)
job = estimator.run([(transpiled_qc, observable)])
result = job.result()

expectation_value = result[0].data.evs
print(f"<ZZ> = {expectation_value}")
# For a perfect Bell state: <ZZ> = 1.0
# On real hardware: <ZZ> ~ 0.95 (due to noise)
```

### Sampler vs Estimator: When to Use Which?

| Primitive | Returns | Use When |
|-----------|---------|----------|
| **Sampler** | Measurement counts/probabilities | You want to see the distribution of outcomes |
| **Estimator** | Expectation values of observables | You want a physical quantity (energy, correlation) |

---

## 4.8 Key Qiskit Concepts Deep Dive

### Shots: Statistical Sampling

Quantum computing is probabilistic — you need multiple runs to build statistics:

```python
# Few shots = noisy statistics
job_100 = sampler.run([qc], shots=100)      # Rough estimate
job_1000 = sampler.run([qc], shots=1000)     # Good estimate
job_10000 = sampler.run([qc], shots=10000)   # High precision

# More shots = better precision but more time/cost
# Rule of thumb: 1024-4096 shots for most experiments
```

### Barriers: Visual Organization

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.barrier()       # Visual separator -- does NOT affect computation
qc.cx(0, 1)
qc.barrier()
qc.measure_all()
```

### Classical Registers

```python
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Named registers for clarity
qr = QuantumRegister(2, name='q')
cr = ClassicalRegister(2, name='result')
qc = QuantumCircuit(qr, cr)

qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.measure(qr, cr)
```

---

## 4.9 Qiskit Application Functions (2025-2026)

IBM Qiskit now provides **ready-made application functions** for common quantum tasks:

| Function | Domain | What It Does |
|----------|--------|-------------|
| **Quantum Portfolio Optimizer** | Finance | Optimizes investment portfolios |
| **QUICK-PDE** | Science | Solves partial differential equations |
| **Chemistry Templates** | Chemistry | Simulates molecular ground states |
| **Implicit Solvent Models** | Pharma | Models molecular environments |
| **Hamiltonian Simulation** | Physics | Simulates quantum systems |

These abstract away the quantum circuit design so domain experts can use quantum computing without deep quantum knowledge.

---

## 4.10 Complete Working Example

Here's a full end-to-end program that creates, simulates, and visualizes a quantum experiment:

```python
"""
Complete Qiskit Example: Creating and Analyzing a GHZ State
A GHZ state is a 3-qubit entangled state: (|000> + |111>)/sqrt(2)
"""

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

# --- BUILD THE CIRCUIT ---
qc = QuantumCircuit(3, 3)

# Create GHZ state
qc.h(0)          # Superposition on qubit 0
qc.cx(0, 1)      # Entangle qubit 0 -> 1
qc.cx(0, 2)      # Entangle qubit 0 -> 2
qc.barrier()
qc.measure([0, 1, 2], [0, 1, 2])

print("=== GHZ State Circuit ===")
print(qc.draw())
print(f"\nCircuit depth: {qc.depth()}")
print(f"Gate counts: {qc.count_ops()}")

# --- ANALYZE THE STATE (before measurement) ---
qc_no_measure = QuantumCircuit(3)
qc_no_measure.h(0)
qc_no_measure.cx(0, 1)
qc_no_measure.cx(0, 2)

statevector = Statevector.from_instruction(qc_no_measure)
print(f"\n=== Statevector ===")
print(statevector)
# Shows: (|000> + |111>)/sqrt(2)

# --- SIMULATE ---
sampler = StatevectorSampler()
job = sampler.run([qc], shots=4096)
result = job.result()
counts = result[0].data.c.get_counts()

print(f"\n=== Measurement Results (4096 shots) ===")
total = sum(counts.values())
for state in sorted(counts.keys()):
    count = counts[state]
    prob = count / total
    bar = '#' * int(prob * 50)
    print(f"|{state}>: {count:4d} ({prob:6.2%}) {bar}")

# Expected output:
# |000>: ~2048 (50%)  #########################
# |111>: ~2048 (50%)  #########################
# All other states: 0 (0%)
```

---

## Summary

1. **Qiskit** is the world's most popular quantum SDK — open source, free, with cloud hardware access
2. **Installation**: `pip install qiskit qiskit-ibm-runtime`
3. **QuantumCircuit** is the core object — add gates, measurements, and visualize
4. **Transpiler** optimizes your ideal circuit for specific real hardware (83x faster than competitors)
5. **Two primitives**: Sampler (measurement counts) and Estimator (expectation values)
6. **Shots** = repeated runs to build probability statistics
7. **Real hardware** introduces noise — results won't be perfect
8. **Application functions** provide ready-made quantum solutions for chemistry, finance, etc.

---

## Next Chapter

**[Chapter 5: Quantum Algorithms →](05-quantum-algorithms.md)**

We'll implement the major quantum algorithms — Deutsch-Jozsa, Grover's Search, Shor's Factoring, VQE, and QAOA — with full Qiskit code!

---

*[← Previous: Chapter 3 - Gates and Circuits](03-gates-and-circuits.md) | [Back to Main README](../README.md)*