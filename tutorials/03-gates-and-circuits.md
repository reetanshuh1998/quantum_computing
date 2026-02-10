# Chapter 3: Quantum Gates & Circuits

> *"A quantum gate is to a qubit what a logic gate is to a bit — the fundamental operation that transforms information."*

---

## 🎯 Learning Goals

By the end of this chapter, you will understand:
- What quantum gates are and how they differ from classical gates
- All major single-qubit gates (X, Y, Z, H, S, T, rotations)
- Multi-qubit gates (CNOT, CZ, Toffoli, SWAP)
- The Bloch sphere visualization
- How to read and build quantum circuits
- Matrix representations of gates

---

## 3.1 What Are Quantum Gates?

In classical computing, **logic gates** (AND, OR, NOT) transform bits. In quantum computing, **quantum gates** transform qubits.

### Key Differences from Classical Gates

| Property | Classical Gates | Quantum Gates |
|----------|----------------|---------------|
| **Reversibility** | Most are irreversible (AND: can't recover input from output) | ALL are reversible (you can always undo) |
| **Representation** | Truth tables | Unitary matrices |
| **Operation** | Flips bits | Rotates qubit state on the Bloch sphere |
| **Information loss** | Can lose information | Never loses information (until measurement) |
| **Deterministic?** | Yes | Yes (the state evolution is deterministic; only measurement is probabilistic) |

### The Math: Unitary Matrices

Every quantum gate is represented by a **unitary matrix** U, meaning:

```
U†U = UU† = I    (where U† is the conjugate transpose)
```

This guarantees:
- The operation is **reversible** (U⁻¹ = U†)
- **Probabilities are preserved** (|α|² + |β|² = 1 before and after)

Applying a gate to a qubit:

```
|ψ_new⟩ = U |ψ_old⟩

Example: Apply X gate to |0⟩
[0 1] [1]   [0]
[1 0] [0] = [1]  → |1⟩  ✓ (bit flip!)
```

---

## 3.2 Single-Qubit Gates

### 3.2.1 Pauli-X Gate (Quantum NOT)

The X gate **flips** the qubit — it's the quantum equivalent of a classical NOT gate.

```
Matrix:         Action:
X = [0  1]      |0⟩ → |1⟩
    [1  0]      |1⟩ → |0⟩
```

**On the Bloch sphere:** 180° rotation around the X-axis.

### 3.2.2 Pauli-Y Gate

The Y gate rotates around the Y-axis, combining a bit flip with a phase flip.

```
Matrix:           Action:
Y = [0  -i]      |0⟩ → i|1⟩
    [i   0]      |1⟩ → -i|0⟩
```

### 3.2.3 Pauli-Z Gate (Phase Flip)

The Z gate **flips the phase** of |1⟩ but leaves |0⟩ unchanged.

```
Matrix:         Action:
Z = [1   0]     |0⟩ → |0⟩
    [0  -1]     |1⟩ → -|1⟩
```

**On the Bloch sphere:** 180° rotation around the Z-axis.

> 💡 Note: |1⟩ and -|1⟩ give the **same measurement probabilities** (both 100% chance of measuring 1). The phase difference only matters when combined with other operations (interference!).

### 3.2.4 Hadamard Gate (H) — The Most Important Gate!

The Hadamard gate creates an **equal superposition** from a definite state. It is the gateway to quantum computing.

```
Matrix:                Action:
H = (1/√2) [1   1]    |0⟩ → (1/√2)(|0⟩ + |1⟩) = |+⟩
           [1  -1]    |1⟩ → (1/√2)(|0⟩ - |1⟩) = |−⟩
```

**Key properties:**
- Applying H twice returns to the original state: H·H = I
- Creates the superposition needed to start most quantum algorithms
- On the Bloch sphere: 180° rotation around the axis halfway between X and Z

### 3.2.5 S Gate (Phase Gate, √Z)

```
Matrix:         Action:
S = [1  0]      |0⟩ → |0⟩
    [0  i]      |1⟩ → i|1⟩
```

Applies a 90° phase rotation. Note: S·S = Z

### 3.2.6 T Gate (π/8 Gate, √S)

```
Matrix:              Action:
T = [1       0  ]   |0⟩ → |0⟩
    [0  e^(iπ/4)]   |1⟩ → e^(iπ/4)|1⟩
```

Applies a 45° phase rotation. Note: T·T = S

> 💡 The **universal gate set** {H, T, CNOT} can approximate ANY quantum computation to arbitrary accuracy!

### 3.2.7 Rotation Gates (Rx, Ry, Rz)

These gates rotate the qubit by an arbitrary angle θ around the specified axis:

```
Rx(θ) = [cos(θ/2)    -i·sin(θ/2)]
        [-i·sin(θ/2)   cos(θ/2)  ]

Ry(θ) = [cos(θ/2)   -sin(θ/2)]
        [sin(θ/2)    cos(θ/2) ]

Rz(θ) = [e^(-iθ/2)     0     ]
        [   0       e^(iθ/2)  ]
```

**Special cases:**
- Rx(π) = X (up to global phase)
- Ry(π) = Y (up to global phase)
- Rz(π) = Z (up to global phase)

### Summary: Single-Qubit Gates at a Glance

| Gate | Matrix | Bloch Sphere Action | Key Use |
|------|--------|-------------------|---------|
| **X** | Pauli-X | 180° around X | Bit flip (NOT) |
| **Y** | Pauli-Y | 180° around Y | Bit + phase flip |
| **Z** | Pauli-Z | 180° around Z | Phase flip |
| **H** | Hadamard | 180° around (X+Z)/√2 | Create superposition |
| **S** | Phase | 90° around Z | Quarter turn phase |
| **T** | π/8 | 45° around Z | Fine phase control |
| **Rx(θ)** | X-rotation | θ around X | Arbitrary X rotation |
| **Ry(θ)** | Y-rotation | θ around Y | Arbitrary Y rotation |
| **Rz(θ)** | Z-rotation | θ around Z | Arbitrary Z rotation |

---

## 3.3 The Bloch Sphere

The **Bloch sphere** is a 3D visualization of a single qubit's state:

```
                    |0⟩ (North Pole)
                     ●
                    /|\
                   / | \
                  /  |  \
       |−⟩ ●----/---●---\----● |+⟩   (Equator = superposition states)
                  \  |  /
                   \ | /
                    \|/
                     ●
                    |1⟩ (South Pole)
```

**Mapping:**
- **North pole** → |0⟩
- **South pole** → |1⟩
- **Positive X** → |+⟩ = (|0⟩ + |1⟩)/√2
- **Negative X** → |−⟩ = (|0⟩ - |1⟩)/√2
- **Positive Y** → |+i⟩ = (|0⟩ + i|1⟩)/√2
- **Negative Y** → |−i⟩ = (|0⟩ - i|1⟩)/√2

**Every quantum gate = a rotation on this sphere!**

**General qubit state in spherical coordinates:**

```
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)·sin(θ/2)|1⟩

where:
  θ = polar angle (0 to π)    → determines probability of 0 vs 1
  φ = azimuthal angle (0 to 2π) → determines the phase
```

---

## 3.4 Multi-Qubit Gates

### 3.4.1 CNOT Gate (Controlled-NOT, CX) — The Entanglement Gate

The CNOT gate is the most important multi-qubit gate. It has two inputs:
- **Control qubit**: unchanged
- **Target qubit**: flipped IF control is |1⟩

```
Truth Table:           Matrix (4×4):
|Control, Target⟩      CNOT = [1 0 0 0]
|0,0⟩ → |0,0⟩                [0 1 0 0]
|0,1⟩ → |0,1⟩                [0 0 0 1]
|1,0⟩ → |1,1⟩  ← flipped!   [0 0 1 0]
|1,1⟩ → |1,0⟩  ← flipped!

Circuit notation:
q_0: ──■──     (control: dot)
       |
q_1: ──⊕──     (target: circle with plus)
```

**Creating entanglement with H + CNOT:**

```
q_0: ─[H]──■──     Step 1: H puts q_0 in superposition: (|0⟩+|1⟩)/√2
            |       Step 2: CNOT entangles: (|00⟩+|11⟩)/√2 = Bell State!
q_1: ──────⊕──
```

### 3.4.2 CZ Gate (Controlled-Z)

Applies a Z gate to the target IF the control is |1⟩.

```
Matrix:              Action:
CZ = [1 0 0  0]     |00⟩ → |00⟩
     [0 1 0  0]     |01⟩ → |01⟩
     [0 0 1  0]     |10⟩ → |10⟩
     [0 0 0 -1]     |11⟩ → -|11⟩  ← phase flip!
```

> 💡 CZ is symmetric — it doesn't matter which qubit is "control" and which is "target."

### 3.4.3 Toffoli Gate (CCX, Controlled-Controlled-NOT)

The **quantum AND gate** — flips the target IF both controls are |1⟩.

```
Circuit notation:
q_0: ──■──     (control 1)
       |
q_1: ──■──     (control 2)
       |
q_2: ──⊕──     (target: flipped only if q_0=1 AND q_1=1)

Truth table (only the flip case):
|1,1,0⟩ → |1,1,1⟩
|1,1,1⟩ → |1,1,0⟩
```

### 3.4.4 SWAP Gate

Exchanges the states of two qubits.

```
Matrix:              Circuit decomposition:
SWAP = [1 0 0 0]    q_0: ──⊕──■──⊕──     (3 CNOTs = 1 SWAP)
       [0 0 1 0]           |   |  |
       [0 1 0 0]    q_1: ──■──⊕──■──
       [0 0 0 1]

Action: |ψ₁, ψ₂⟩ → |ψ₂, ψ₁⟩
```

### Summary: Multi-Qubit Gates

| Gate | Qubits | Action | Key Use |
|------|--------|--------|---------|
| **CNOT (CX)** | 2 | Flip target if control=1 | Entanglement |
| **CZ** | 2 | Phase flip if both=1 | Phase-based entanglement |
| **Toffoli (CCX)** | 3 | Flip target if both controls=1 | Quantum AND |
| **SWAP** | 2 | Exchange two qubit states | Qubit routing |
| **Fredkin (CSWAP)** | 3 | Swap targets if control=1 | Controlled swap |

---

## 3.5 Reading Quantum Circuits

Quantum circuits are read **left to right** (time flows left → right):

```
        ┌───┐          ┌─┐
q_0: ───┤ H ├────■─────┤M├───
        └───┘  ┌─┴─┐   └╥┘
q_1: ─────────┤ X  ├────╫────
              └───┘    ║
c_0: ══════════════════╩════
```

Reading this circuit:
1. **q_0** starts in |0⟩ (always, by default)
2. **H gate** applied to q_0 → creates superposition
3. **CNOT** with q_0 as control, q_1 as target → creates entanglement
4. **M** (measurement) on q_0 → result stored in classical bit c_0

### Circuit Conventions

| Symbol | Meaning |
|--------|---------|
| Single line (`─`) | Qubit wire (quantum information) |
| Double line (`═`) | Classical wire (measurement results) |
| Box with letter | Gate (H, X, Z, etc.) |
| Filled dot (`●`) | Control qubit |
| Circle with plus (`⊕`) | Target of CNOT |
| Meter symbol (`M`) | Measurement |
| Dashed line | Barrier (visual separator, no physical effect) |

---

## 3.6 Universal Gate Sets

A **universal gate set** is a small set of gates that can approximate **any** quantum operation to arbitrary accuracy.

| Universal Set | Gates | Notes |
|--------------|-------|-------|
| **Standard** | {H, T, CNOT} | Most common theoretical set |
| **IBM native** | {√X, Rz, CNOT} | What IBM hardware physically implements |
| **Continuous** | {Ry, Rz, CNOT} | Any rotation + entanglement |
| **Clifford+T** | {H, S, CNOT, T} | Common in error correction |

> 💡 The **Solovay-Kitaev theorem** proves that any quantum gate can be approximated to accuracy ε using O(log^c(1/ε)) gates from a universal set. This means we don't need infinitely many different gates!

---

## 3.7 Gate Identities and Useful Relationships

```
Basic identities:
  X·X = I       (applying X twice = doing nothing)
  H·H = I       (applying H twice = doing nothing)
  S·S = Z       (two S gates = one Z gate)
  T·T = S       (two T gates = one S gate)
  T⁴  = Z       (four T gates = one Z gate)

Hadamard conjugation:
  H·X·H = Z     (H converts X to Z and vice versa)
  H·Z·H = X
  H·Y·H = -Y

CNOT identities:
  CNOT · CNOT = I                    (self-inverse)
  (H⊗H) · CNOT · (H⊗H) = CNOT_reversed   (swap control/target)
```

---

## 📝 Chapter 3 Summary

1. **Quantum gates** are reversible unitary operations that transform qubit states
2. **Single-qubit gates** (X, Y, Z, H, S, T, rotations) manipulate individual qubits
3. **The Hadamard gate** is the most important — it creates superposition
4. **The CNOT gate** is the primary entangling gate — essential for multi-qubit algorithms
5. **The Bloch sphere** visualizes any single-qubit state as a point on a sphere
6. **Universal gate sets** like {H, T, CNOT} can build any quantum computation
7. Circuits are read left to right; gates are represented as matrices

---

## ⏭️ Next Chapter

**[Chapter 4: IBM Qiskit Framework →](04-qiskit-framework.md)**

We'll install Qiskit and start writing real quantum programs — building circuits, running simulations, and executing on IBM quantum hardware!

---

*[← Previous: Chapter 2 - Quantum Mechanics](02-quantum-mechanics.md) · [Back to Main README](../README.md)*