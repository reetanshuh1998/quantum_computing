"""
Stern-Gerlach Job 1 — Qiskit Quantum Simulator
==================================================
One magnetic field (Z-direction). Two setups:
  Setup 1: N atoms at once     → count spin ↑ vs ↓
  Setup 2: 1 atom × N shots    → find probability of each state

Quantum Circuit:
    |0⟩ ──[H]──[Measure]── → |0⟩ (spin +½) or |1⟩ (spin -½)
"""

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ⬇⬇⬇  CHANGE THESE TWO VALUES TO TEST DIFFERENT CASES  ⬇⬇⬇
# =============================================================================
N_ATOMS = 1000        # Setup 1: number of atoms sent at once
N_REPETITIONS = 1000  # Setup 2: number of times 1 atom is repeated
# =============================================================================
BATCH_SIZE = 20       # qubits per circuit (keep ≤ 25 for fast simulation)


def run_on_simulator(qc, shots):
    """Run a circuit on Qiskit's ideal statevector simulator."""
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()
    creg = qc.cregs[0].name
    counts = getattr(result[0].data, creg).get_counts()
    return counts


# =============================================================================
# Setup 1: N_ATOMS atoms at once
# =============================================================================
def setup1():
    """
    Each qubit = one atom.  Circuit per batch:

         ┌───┐┌─┐
    q_0: ┤ H ├┤M├   atom 0
    q_1: ┤ H ├┤M├   atom 1
     ...
    q_N: ┤ H ├┤M├   atom N
         └───┘└─┘
    """
    n_up, n_down = 0, 0
    n_circuits = (N_ATOMS + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division

    for i in range(n_circuits):
        # Last batch may be smaller
        batch = min(BATCH_SIZE, N_ATOMS - i * BATCH_SIZE)
        print(f"    Circuit {i+1}/{n_circuits} ({batch} qubits)...", end=" ", flush=True)

        qc = QuantumCircuit(batch, batch)
        for q in range(batch):
            qc.h(q)
        qc.measure(range(batch), range(batch))

        counts = run_on_simulator(qc, shots=1)
        print("done")

        for bitstring, count in counts.items():
            for bit in bitstring:
                if bit == '0':
                    n_up += count
                else:
                    n_down += count

    return n_up, n_down


# =============================================================================
# Setup 2: 1 atom × N_REPETITIONS shots
# =============================================================================
def setup2():
    """
    One qubit, measured N_REPETITIONS times.  Circuit:

         ┌───┐┌─┐
    q_0: ┤ H ├┤M├
         └───┘└─┘
    c_0: ═════╩══
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    print(f"    Running 1 qubit × {N_REPETITIONS} shots...", end=" ", flush=True)
    counts = run_on_simulator(qc, shots=N_REPETITIONS)
    print("done")

    n_up = counts.get('0', 0)
    n_down = counts.get('1', 0)
    return n_up, n_down


# =============================================================================
# Visualization
# =============================================================================
def plot_results(s1_up, s1_down, s2_up, s2_down):
    """Bar chart comparing both setups."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stern-Gerlach — Qiskit Quantum Simulator\n"
                 "Magnetic Field: Z-direction  |  Circuit: |0⟩→[H]→[Measure]",
                 fontsize=14, fontweight="bold")

    # --- Setup 1 ---
    expected_1 = N_ATOMS / 2
    bars1 = ax1.bar(
        ["Spin ↑ (+½)\n|0⟩", "Spin ↓ (-½)\n|1⟩"],
        [s1_up, s1_down],
        color=["red", "blue"], alpha=0.8, edgecolor="black", width=0.5
    )
    ax1.axhline(expected_1, color="green", ls="--", lw=2,
                label=f"Expected: {expected_1:.0f}")
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Setup 1: {N_ATOMS} Atoms at Once\n"
                  f"↑ = {s1_up}  ({s1_up/N_ATOMS*100:.1f}%)  |  "
                  f"↓ = {s1_down}  ({s1_down/N_ATOMS*100:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylim(0, N_ATOMS * 1.1)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + N_ATOMS * 0.015,
                 str(int(bar.get_height())),
                 ha="center", fontsize=14, fontweight="bold")

    # --- Setup 2 ---
    expected_2 = N_REPETITIONS / 2
    bars2 = ax2.bar(
        ["Spin ↑ (+½)\n|0⟩", "Spin ↓ (-½)\n|1⟩"],
        [s2_up, s2_down],
        color=["red", "blue"], alpha=0.8, edgecolor="black", width=0.5
    )
    ax2.axhline(expected_2, color="green", ls="--", lw=2,
                label=f"Expected: {expected_2:.0f}")
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Setup 2: 1 Atom × {N_REPETITIONS} Repetitions\n"
                  f"↑ = {s2_up}  ({s2_up/N_REPETITIONS*100:.1f}%)  |  "
                  f"↓ = {s2_down}  ({s2_down/N_REPETITIONS*100:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(0, N_REPETITIONS * 1.1)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + N_REPETITIONS * 0.015,
                 str(int(bar.get_height())),
                 ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("stern_gerlach_job1_sim.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  🔬 STERN-GERLACH JOB 1 — Qiskit Quantum Simulator")
    print("  Magnetic field: Z-direction")
    print("  Backend: StatevectorSampler (ideal, no noise)")
    print(f"  Setup 1: {N_ATOMS} atoms at once")
    print(f"  Setup 2: 1 atom × {N_REPETITIONS} repetitions")
    print("=" * 58)

    # Show the circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    print("\n  Quantum Circuit (per atom):")
    print(qc.draw(output='text'))

    # ── Setup 1 ──
    print(f"\n  ── SETUP 1: {N_ATOMS} atoms at once ──")
    s1_up, s1_down = setup1()

    print(f"\n  Results:")
    print(f"    Spin ↑ (+½) = {s1_up:>6d}  ({s1_up/N_ATOMS*100:.1f}%)")
    print(f"    Spin ↓ (-½) = {s1_down:>6d}  ({s1_down/N_ATOMS*100:.1f}%)")

    # ── Setup 2 ──
    print(f"\n  ── SETUP 2: 1 atom × {N_REPETITIONS} repetitions ──")
    s2_up, s2_down = setup2()

    print(f"\n  Results:")
    print(f"    Spin ↑ (+½) = {s2_up:>6d}  ({s2_up/N_REPETITIONS*100:.1f}%)")
    print(f"    Spin ↓ (-½) = {s2_down:>6d}  ({s2_down/N_REPETITIONS*100:.1f}%)")

    # ── Summary ──
    print("\n" + "=" * 58)
    print("  SUMMARY")
    print("=" * 58)
    print(f"  {'Setup':<35} {'↑ (+½)':<10} {'↓ (-½)':<10} {'Ratio':<10}")
    print("-" * 58)
    r1 = s1_up / s1_down if s1_down > 0 else float('inf')
    r2 = s2_up / s2_down if s2_down > 0 else float('inf')
    e1 = N_ATOMS // 2
    e2 = N_REPETITIONS // 2
    print(f"  {f'{N_ATOMS} atoms at once':<35} {s1_up:<10} {s1_down:<10} {r1:<10.3f}")
    print(f"  {f'1 atom × {N_REPETITIONS} reps':<35} {s2_up:<10} {s2_down:<10} {r2:<10.3f}")
    print(f"  {'Theory (ideal)':<35} {e1:<10} {e2:<10} {'1.000':<10}")
    print("-" * 58)
    print(f"""
  ✓ Both setups give ≈ 50/50 — spin is QUANTIZED
  ✓ Quantum circuit math:
      |0⟩ → H → (1/√2)|0⟩ + (1/√2)|1⟩ → Measure
      P(|0⟩) = |1/√2|² = 50%
      P(|1⟩) = |1/√2|² = 50%
  ✓ Try changing N_ATOMS and N_REPETITIONS at the top!
    """)

    # Plot
    plot_results(s1_up, s1_down, s2_up, s2_down)
    print("  ✓ Saved: stern_gerlach_job1_sim.png\n")