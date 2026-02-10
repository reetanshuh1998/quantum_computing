# Chapter 5: Quantum Algorithms

> *"Quantum algorithms are the reason quantum computers exist — they are the recipes that unlock exponential speedups."*

---

## 🎯 Learning Goals

By the end of this chapter, you will:
- Understand what makes a quantum algorithm "quantum"
- Implement the Deutsch-Jozsa algorithm
- Implement Grover's Search algorithm with full Qiskit code
- Understand Shor's Factoring algorithm conceptually
- Build a Variational Quantum Eigensolver (VQE)
- Understand QAOA for optimization problems

---

## 5.1 What Makes Quantum Algorithms Special?

Classical algorithms process data step by step. Quantum algorithms exploit three quantum phenomena to achieve speedups:

| Phenomenon | Role in Algorithms |
|---|---|
| **Superposition** | Explore many solutions simultaneously |
| **Entanglement** | Correlate qubits to share information |
| **Interference** | Amplify correct answers, cancel wrong ones |

### Types of Quantum Speedup

| Speedup Type | Example | Classical | Quantum |
|---|---|---|---|
| **Exponential** | Shor's (factoring) | O(2^n) | O(n^3) |
| **Quadratic** | Grover's (search) | O(N) | O(sqrt(N)) |
| **Polynomial** | Quantum simulation | O(2^n) | O(n^k) |

---

## 5.2 The Deutsch-Jozsa Algorithm

### The Problem

Given a black-box function f(x) that takes n-bit input and returns 0 or 1:
- **Constant**: f(x) returns the same value for ALL inputs (all 0s or all 1s)
- **Balanced**: f(x) returns 0 for exactly half the inputs and 1 for the other half

**Promise**: The function is guaranteed to be either constant or balanced. Determine which.

### Classical Approach
- Worst case: Need to check 2^(n-1) + 1 inputs
- For n=10: up to 513 function evaluations

### Quantum Approach
- Needs exactly **1** function evaluation — exponential speedup!

### How It Works

```
1. Start with n qubits in |0> and 1 auxiliary qubit in |1>
2. Apply Hadamard to all qubits (create superposition)
3. Apply the oracle (black-box function) once
4. Apply Hadamard to the n input qubits again
5. Measure the input qubits
   - All zeros -> CONSTANT
   - Any non-zero -> BALANCED
```

### Qiskit Implementation

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

def deutsch_jozsa(oracle_type, n_qubits=3):
    """
    Deutsch-Jozsa Algorithm
    oracle_type: 'constant' or 'balanced'
    """
    # Total qubits = n input + 1 auxiliary
    qc = QuantumCircuit(n_qubits + 1, n_qubits)
    
    # Step 1: Initialize auxiliary qubit to |1>
    qc.x(n_qubits)
    
    # Step 2: Apply Hadamard to ALL qubits
    qc.h(range(n_qubits + 1))
    qc.barrier()
    
    # Step 3: Apply Oracle
    if oracle_type == 'constant':
        # Constant oracle: do nothing (f(x) = 0 for all x)
        pass
    elif oracle_type == 'balanced':
        # Balanced oracle: CNOT from each input to auxiliary
        for i in range(n_qubits):
            qc.cx(i, n_qubits)
    
    qc.barrier()
    
    # Step 4: Apply Hadamard to input qubits
    qc.h(range(n_qubits))
    
    # Step 5: Measure input qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc

# Test with CONSTANT oracle
print("=== CONSTANT Oracle ===")
qc_const = deutsch_jozsa('constant', n_qubits=3)
print(qc_const.draw())

sampler = StatevectorSampler()
result = sampler.run([qc_const], shots=1024).result()
counts = result[0].data.c.get_counts()
print(f"Results: {counts}")
print(f"Verdict: {'CONSTANT' if '000' in counts else 'BALANCED'}")

# Test with BALANCED oracle
print("\n=== BALANCED Oracle ===")
qc_bal = deutsch_jozsa('balanced', n_qubits=3)
print(qc_bal.draw())

result = sampler.run([qc_bal], shots=1024).result()
counts = result[0].data.c.get_counts()
print(f"Results: {counts}")
print(f"Verdict: {'CONSTANT' if '000' in counts else 'BALANCED'}")

# Expected output:
# CONSTANT -> always measures '000' -> CONSTANT
# BALANCED -> never measures '000' -> BALANCED
```

---

## 5.3 Grover's Search Algorithm

### The Problem

Find a specific item in an **unsorted** database of N items.

| Approach | Evaluations Needed |
|---|---|
| Classical | O(N) — check each item |
| Grover's | O(sqrt(N)) — quadratic speedup |

**Example**: Search 1,000,000 items:
- Classical: up to 1,000,000 checks
- Grover's: about 1,000 checks

### How It Works

```
1. Start with all qubits in equal superposition (Hadamard)
2. Repeat sqrt(N) times:
   a. ORACLE: Flip the phase of the target state (mark it)
   b. DIFFUSION: Amplify the marked state's amplitude
3. Measure — high probability of getting the target!
```

### The Key Insight: Amplitude Amplification

Each Grover iteration:
- The oracle makes the target state "negative"
- The diffusion operator "reflects" all amplitudes around the mean
- This gradually increases the target's amplitude while decreasing others

### Qiskit Implementation

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import math

def grovers_search(target_state='11', n_qubits=2):
    """
    Grover's Algorithm to search for target_state
    """
    # Calculate optimal number of iterations
    N = 2 ** n_qubits
    num_iterations = max(1, round(math.pi / 4 * math.sqrt(N)))
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Step 1: Create equal superposition
    qc.h(range(n_qubits))
    
    for iteration in range(num_iterations):
        qc.barrier()
        
        # Step 2a: Oracle — flip phase of target state
        # For target '11': apply CZ
        if target_state == '11':
            qc.cz(0, 1)
        elif target_state == '00':
            qc.x([0, 1])
            qc.cz(0, 1)
            qc.x([0, 1])
        elif target_state == '01':
            qc.x(0)
            qc.cz(0, 1)
            qc.x(0)
        elif target_state == '10':
            qc.x(1)
            qc.cz(0, 1)
            qc.x(1)
        
        qc.barrier()
        
        # Step 2b: Diffusion operator (reflect around mean)
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.cz(0, 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))
    
    # Step 3: Measure
    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc, num_iterations

# Search for state |11>
target = '11'
qc, iters = grovers_search(target, n_qubits=2)

print(f"Searching for: |{target}>")
print(f"Grover iterations: {iters}")
print(qc.draw())

# Run
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
counts = result[0].data.c.get_counts()

print(f"\nResults:")
total = sum(counts.values())
for state in sorted(counts.keys()):
    count = counts[state]
    prob = count / total
    marker = " <-- TARGET" if state == target else ""
    print(f"  |{state}>: {count} ({prob:.1%}){marker}")

# Expected: |11> has ~100% probability!
```

### Grover's with Qiskit Circuit Library

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.primitives import StatevectorSampler

# Define the oracle (marks |11|)
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# Build Grover operator automatically
grover_op = GroverOperator(oracle)

# Full circuit
qc = QuantumCircuit(2, 2)
qc.h([0, 1])                            # Initial superposition
qc.compose(grover_op, inplace=True)     # Apply Grover
qc.measure([0, 1], [0, 1])

# Run
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
counts = result[0].data.c.get_counts()
print(f"Results: {counts}")
# Expected: {'11': ~1024}
```

---

## 5.4 Shor's Algorithm (Factoring)

### The Problem

Given a large composite number N, find its prime factors.

| Approach | Time Complexity |
|---|---|
| Classical (best known) | O(exp(n^(1/3))) — sub-exponential |
| Shor's Algorithm | O(n^3) — polynomial! |

### Why It Matters

- **RSA encryption** relies on the difficulty of factoring large numbers
- A quantum computer running Shor's algorithm could break RSA
- This drives the urgent shift to **post-quantum cryptography**

### How Shor's Works (Conceptual)

```
1. Pick a random number 'a' less than N
2. Use a QUANTUM SUBROUTINE to find the period 'r' of f(x) = a^x mod N
   (This is where the quantum speedup happens — uses Quantum Fourier Transform)
3. Use the period 'r' to compute factors classically:
   - If r is even: factors = gcd(a^(r/2) +/- 1, N)
```

### The Quantum Part: Period Finding

The quantum subroutine uses:
- **Quantum Fourier Transform (QFT)** — the quantum analog of the classical FFT
- **Modular exponentiation** — computed in superposition
- **Interference** — amplifies the correct period

### Quantum Fourier Transform in Qiskit

```python
from qiskit import QuantumCircuit
import math

def qft_circuit(n_qubits):
    """Build a Quantum Fourier Transform circuit"""
    qc = QuantumCircuit(n_qubits, name="QFT")
    
    for target in range(n_qubits):
        # Hadamard on target qubit
        qc.h(target)
        
        # Controlled phase rotations
        for control in range(target + 1, n_qubits):
            angle = math.pi / (2 ** (control - target))
            qc.cp(angle, control, target)
    
    # Swap qubits to get correct order
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)
    
    return qc

# Build and display 4-qubit QFT
qft = qft_circuit(4)
print("Quantum Fourier Transform (4 qubits):")
print(qft.draw())
```

### Shor's Factoring Example (Small Number)

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import math

# Factor N = 15 (= 3 x 5)
# Using a = 7, we need to find the period of 7^x mod 15

N = 15
a = 7

# For this small example, the period r = 4
# Because: 7^1=7, 7^2=4, 7^3=13, 7^4=1 (mod 15) -> period = 4

# Verify classically:
for x in range(8):
    print(f"  {a}^{x} mod {N} = {pow(a, x, N)}")

# With r = 4 (even), compute factors:
r = 4
factor1 = math.gcd(a ** (r // 2) - 1, N)
factor2 = math.gcd(a ** (r // 2) + 1, N)
print(f"\nFactors of {N}: {factor1} and {factor2}")
# Output: Factors of 15: 3 and 5
```

> **Note**: Full Shor's implementation requires many qubits. For N=15, you need ~8+ qubits. For RSA-2048, you'd need thousands of error-corrected qubits — still years away.

---

## 5.5 Variational Quantum Eigensolver (VQE)

### The Problem

Find the **ground state energy** of a molecule or quantum system. This is crucial for:
- Drug discovery (molecular stability)
- Materials science (new materials design)
- Chemistry (reaction predictions)

### Why Quantum?

The Hilbert space of a quantum system grows exponentially. Simulating a molecule with 50 electrons classically is essentially impossible, but a quantum computer can represent it naturally.

### How VQE Works

VQE is a **hybrid quantum-classical** algorithm:

```
Classical Computer                    Quantum Computer
+------------------+                 +------------------+
|                  |  parameters     |                  |
| Classical        | ------------->  | Parameterized    |
| Optimizer        |                 | Quantum Circuit  |
| (minimize energy)|  energy         | (ansatz)         |
|                  | <-------------  |                  |
+------------------+                 +------------------+
        |                                    |
        |  Repeat until energy converges     |
        +------------------------------------+
```

### Qiskit Implementation

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import numpy as np
from scipy.optimize import minimize

# Step 1: Define the Hamiltonian (H2 molecule simplified)
hamiltonian = SparsePauliOp.from_list([
    ("II", -1.052),
    ("IZ",  0.398),
    ("ZI", -0.398),
    ("ZZ", -0.011),
    ("XX",  0.181)
])

# Step 2: Build parameterized ansatz circuit
theta = Parameter('theta')
ansatz = QuantumCircuit(2)
ansatz.ry(theta, 0)
ansatz.cx(0, 1)

print("Ansatz circuit:")
print(ansatz.draw())

# Step 3: Define the cost function
estimator = StatevectorEstimator()

def cost_function(params):
    """Evaluate energy for given parameters"""
    bound_circuit = ansatz.assign_parameters({theta: params[0]})
    job = estimator.run([(bound_circuit, hamiltonian)])
    result = job.result()
    energy = result[0].data.evs
    return float(energy)

# Step 4: Classical optimization
print("\nOptimizing...")
result = minimize(cost_function, x0=[0.0], method='COBYLA',
                  options={'maxiter': 100})

print(f"Optimal angle: {result.x[0]:.4f} radians")
print(f"Ground state energy: {result.fun:.6f} Hartree")
print(f"Optimization converged: {result.success}")

# The exact ground state energy of H2 at equilibrium is ~ -1.137 Hartree
```

### Using Qiskit's EfficientSU2 Ansatz

```python
from qiskit.circuit.library import EfficientSU2

# More expressive ansatz with multiple parameters
ansatz = EfficientSU2(num_qubits=2, reps=1, entanglement='linear')
print(f"Number of parameters: {ansatz.num_parameters}")
print(ansatz.draw())
```

---

## 5.6 Quantum Approximate Optimization Algorithm (QAOA)

### The Problem

Solve **combinatorial optimization** problems:
- Traveling salesman
- Portfolio optimization
- Job scheduling
- Graph coloring

### How QAOA Works

QAOA is another hybrid quantum-classical algorithm, but for optimization:

```
1. Encode the optimization problem as a "cost Hamiltonian"
2. Build a parameterized circuit alternating between:
   - Cost layer: encodes the problem
   - Mixer layer: explores the solution space
3. Classical optimizer tunes parameters to minimize the cost
4. Measure to get the approximate optimal solution
```

### Qiskit Implementation: MaxCut Problem

```python
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler
import numpy as np
from scipy.optimize import minimize

def build_qaoa_circuit(gamma, beta, edges, n_qubits):
    """Build a 1-layer QAOA circuit for MaxCut"""
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initial superposition
    qc.h(range(n_qubits))
    
    # Cost layer: ZZ interaction for each edge
    for (i, j) in edges:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    
    # Mixer layer: X rotation on each qubit
    for i in range(n_qubits):
        qc.rx(2 * beta, i)
    
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

# Define a simple graph: triangle (3 nodes, 3 edges)
#   0 --- 1
#    \   /
#      2
n_qubits = 3
edges = [(0, 1), (1, 2), (0, 2)]

def maxcut_cost(bitstring, edges):
    """Calculate the MaxCut value for a bitstring"""
    cost = 0
    for (i, j) in edges:
        if bitstring[i] != bitstring[j]:
            cost += 1
    return cost

def qaoa_objective(params):
    """Evaluate QAOA with given parameters"""
    gamma, beta = params
    qc = build_qaoa_circuit(gamma, beta, edges, n_qubits)
    
    sampler = StatevectorSampler()
    result = sampler.run([qc], shots=1024).result()
    counts = result[0].data.c.get_counts()
    
    # Calculate expected cost
    total_cost = 0
    total_shots = sum(counts.values())
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring]
        cost = maxcut_cost(bits, edges)
        total_cost += cost * count
    
    return -total_cost / total_shots  # Negative because we minimize

# Optimize
result = minimize(qaoa_objective, x0=[0.5, 0.5], method='COBYLA')
gamma_opt, beta_opt = result.x

print(f"Optimal gamma: {gamma_opt:.4f}")
print(f"Optimal beta: {beta_opt:.4f}")

# Run with optimal parameters
qc_opt = build_qaoa_circuit(gamma_opt, beta_opt, edges, n_qubits)
sampler = StatevectorSampler()
final_result = sampler.run([qc_opt], shots=4096).result()
counts = final_result[0].data.c.get_counts()

print("\nResults (sorted by frequency):")
for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
    bits = [int(b) for b in bitstring]
    cost = maxcut_cost(bits, edges)
    print(f"  |{bitstring}>: {count} times, MaxCut value = {cost}")

# Optimal MaxCut for triangle = 2 (e.g., |011>, |101>, |110>)
```

---

## 5.7 Algorithm Comparison Summary

| Algorithm | Problem | Speedup | Type | Hardware Needed |
|---|---|---|---|---|
| **Deutsch-Jozsa** | Function classification | Exponential | Pure quantum | Small |
| **Bernstein-Vazirani** | Hidden string finding | Exponential | Pure quantum | Small |
| **Grover's** | Unstructured search | Quadratic | Pure quantum | Medium |
| **Shor's** | Integer factoring | Exponential | Pure quantum | Large (fault-tolerant) |
| **QFT** | Fourier transform | Exponential | Subroutine | Medium |
| **VQE** | Ground state energy | Polynomial (expected) | Hybrid | Small-Medium (NISQ) |
| **QAOA** | Combinatorial optimization | Polynomial (expected) | Hybrid | Small-Medium (NISQ) |

### NISQ-Era vs Fault-Tolerant Algorithms

| Category | Algorithms | When |
|---|---|---|
| **NISQ (now)** | VQE, QAOA, variational methods | Works on today's noisy hardware |
| **Fault-Tolerant (future)** | Shor's (large), quantum simulation, QPE | Needs error-corrected qubits (2028+) |

---

## Summary

1. **Deutsch-Jozsa**: Determines if a function is constant or balanced with 1 query (exponential speedup)
2. **Grover's**: Searches unsorted databases in O(sqrt(N)) time (quadratic speedup)
3. **Shor's**: Factors integers in polynomial time (threatens RSA encryption)
4. **VQE**: Hybrid algorithm to find molecular ground state energies (NISQ-friendly)
5. **QAOA**: Hybrid algorithm for combinatorial optimization (NISQ-friendly)
6. All quantum algorithms use superposition + entanglement + interference

---

## Next Chapter

**[Chapter 6: Real-World Applications -->](06-applications.md)**

---

*[<-- Previous: Chapter 4 - Qiskit Framework](04-qiskit-framework.md) | [Back to Main README](../README.md)*