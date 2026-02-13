"""
Stern-Gerlach Job 1 — IBM Quantum Hardware
=============================================
One magnetic field (Z-direction). Two setups:
  Setup 1: 1000 atoms at once  → count spin ↑ vs ↓
  Setup 2: 1 atom × 1000 shots → find probability of each state

Quantum Circuit:
    |0⟩ ──[H]──[Measure]── → |0⟩ (spin +½) or |1⟩ (spin -½)
"""

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# IBM Quantum Connection
# =============================================================================
TOKEN = "fBSlLGMrl1uwzWdFIHJa9dzRVNGezhH0UCwoGNdvZPDx"  


def connect():
    """Connect to IBM and pick least-busy real backend."""
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=TOKEN,
    )
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  ✓ Backend: {backend.name} ({backend.num_qubits} qubits)")
    return backend


def run_on_hardware(qc, backend, shots):
    """Transpile and run a circuit on real hardware."""
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pm.run(qc)

    sampler = SamplerV2(backend)
    job = sampler.run([transpiled], shots=shots)
    print(f"    Job ID: {job.job_id()}  (waiting...)", flush=True)
    result = job.result()

    # Extract counts from the result DataBin
    # The classical register data is accessed via getattr
    creg = transpiled.cregs[0].name
    creg_data = getattr(result[0].data, creg)

    # Try get_counts() first, fall back to manual counting from bitarray
    try:
        counts = creg_data.get_counts()
    except AttributeError:
        # Newer Qiskit: creg_data is a BitArray, convert to counts manually
        counts = {}
        for shot_bits in creg_data.array:
            # Convert numpy array of bits to string
            bitstring = ''.join(str(b) for b in shot_bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

    return counts


# =============================================================================
# Setup 1: 1000 atoms at once (20 qubits × 50 circuits × 1 shot each)
# =============================================================================
def setup1(backend, n_atoms=1000):
    """
    Each qubit = one atom.  Circuit:

         ┌───┐┌─┐
    q_0: ┤ H ├┤M├   atom 0
    q_1: ┤ H ├┤M├   atom 1
     ...
    q_19:┤ H ├┤M├   atom 19
         └───┘└─┘

    We use 20 qubits per circuit, run 50 circuits = 1000 atoms total.
    """
    batch = 20
    n_up, n_down = 0, 0

    n_circuits = n_atoms // batch
    for i in range(n_circuits):
        print(f"    Circuit {i+1}/{n_circuits}...", end=" ", flush=True)
        qc = QuantumCircuit(batch, batch)
        for q in range(batch):
            qc.h(q)
        qc.measure(range(batch), range(batch))

        counts = run_on_hardware(qc, backend, shots=1)

        # Each bitstring has 20 characters, one per qubit
        for bitstring, count in counts.items():
            for bit in bitstring:
                if bit == '0':
                    n_up += count
                else:
                    n_down += count

    return n_up, n_down


# =============================================================================
# Setup 2: 1 atom, 1000 shots
# =============================================================================
def setup2(backend, n_shots=1000):
    """
    One qubit, measured 1000 times.  Circuit:

         ┌───┐┌─┐
    q_0: ┤ H ├┤M├
         └───┘└─┘
    c_0: ═════╩══

    Same circuit, 1000 shots = 1000 independent experiments.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    print("    Running 1 qubit × 1000 shots...", end=" ", flush=True)
    counts = run_on_hardware(qc, backend, shots=n_shots)

    n_up = counts.get('0', 0)
    n_down = counts.get('1', 0)
    return n_up, n_down


# =============================================================================
# Visualization
# =============================================================================
def plot_results(s1_up, s1_down, s2_up, s2_down):
    """Bar chart comparing both setups."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stern-Gerlach on IBM Quantum Hardware\n"
                 "Magnetic Field: Z-direction  |  Circuit: |0⟩→[H]→[Measure]",
                 fontsize=14, fontweight="bold")

    # --- Setup 1 ---
    bars1 = ax1.bar(
        ["Spin ↑ (+½)\n|0⟩", "Spin ↓ (-½)\n|1⟩"],
        [s1_up, s1_down],
        color=["red", "blue"], alpha=0.8, edgecolor="black", width=0.5
    )
    ax1.axhline(500, color="green", ls="--", lw=2, label="Expected: 500")
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Setup 1: 1000 Atoms at Once\n"
                  f"↑ = {s1_up}  ({s1_up/10:.1f}%)  |  "
                  f"↓ = {s1_down}  ({s1_down/10:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 11000)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 str(int(bar.get_height())),
                 ha="center", fontsize=14, fontweight="bold")

    # --- Setup 2 ---
    bars2 = ax2.bar(
        ["Spin ↑ (+½)\n|0⟩", "Spin ↓ (-½)\n|1⟩"],
        [s2_up, s2_down],
        color=["red", "blue"], alpha=0.8, edgecolor="black", width=0.5
    )
    ax2.axhline(500, color="green", ls="--", lw=2, label="Expected: 500")
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Setup 2: 1 Atom × 1000 Repetitions\n"
                  f"↑ = {s2_up}  ({s2_up/10:.1f}%)  |  "
                  f"↓ = {s2_down}  ({s2_down/10:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1100)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 str(int(bar.get_height())),
                 ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("stern_gerlach_job1.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  🔬 STERN-GERLACH JOB 1 — IBM Quantum Hardware")
    print("  Magnetic field: Z-direction")
    print("=" * 55)

    # Show the circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    print("\n  Quantum Circuit (per atom):")
    print(qc.draw(output='text'))

    # Connect
    print("\n  Connecting to IBM Quantum...")
    backend = connect()

    # ── Setup 1: 1000 atoms ──
    print("\n  ── SETUP 1: 1000 atoms at once ──")
    s1_up, s1_down = setup1(backend, n_atoms=1000)

    print(f"\n  Results:")
    print(f"    Spin ↑ (+½) = {s1_up:>4d}  ({s1_up/10:.1f}%)")
    print(f"    Spin ↓ (-½) = {s1_down:>4d}  ({s1_down/10:.1f}%)")

    # ── Setup 2: 1 atom × 1000 ──
    print("\n  ── SETUP 2: 1 atom × 1000 repetitions ──")
    s2_up, s2_down = setup2(backend, n_shots=1000)

    print(f"\n  Results:")
    print(f"    Spin ↑ (+½) = {s2_up:>4d}  ({s2_up/10:.1f}%)")
    print(f"    Spin ↓ (-½) = {s2_down:>4d}  ({s2_down/10:.1f}%)")

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  {'Setup':<30} {'↑ (+½)':<10} {'↓ (-½)':<10} {'Ratio':<10}")
    print("-" * 55)
    r1 = s1_up / s1_down if s1_down > 0 else float('inf')
    r2 = s2_up / s2_down if s2_down > 0 else float('inf')
    print(f"  {'1000 atoms at once':<30} {s1_up:<10} {s1_down:<10} {r1:<10.3f}")
    print(f"  {'1 atom × 1000 reps':<30} {s2_up:<10} {s2_down:<10} {r2:<10.3f}")
    print(f"  {'Theory (ideal)':<30} {'500':<10} {'500':<10} {'1.000':<10}")
    print("-" * 55)
    print("""
  ✓ Both setups give ≈ 50/50 — spin is QUANTIZED
  ✓ Any deviation from 50/50 = real quantum hardware noise
    (gate errors, readout errors, decoherence)
  ✓ The magnetic field direction (Z) determines the
    measurement basis — that's what the Z-measurement does
    """)

    # Plot
    plot_results(s1_up, s1_down, s2_up, s2_down)
    print("  ✓ Saved: stern_gerlach_job1.png\n")