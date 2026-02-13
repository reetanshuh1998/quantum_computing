"""
Bell Experiment — Qiskit Quantum Simulator
=============================================
Tests non-locality of quantum mechanics using Bell states and CHSH inequality.

Classical physics (local hidden variables):  |S| ≤ 2
Quantum mechanics predicts:                  |S| = 2√2 ≈ 2.828

If |S| > 2, local realism is VIOLATED → quantum entanglement is real!

Bell Circuit:
    q_0: ──[H]──●──     (Alice's qubit)
                │
    q_1: ──────[X]──     (Bob's qubit)

    This creates: |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)
    Perfectly entangled — measuring one instantly determines the other.
"""

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ⬇⬇⬇  CHANGE THIS VALUE TO TEST DIFFERENT SAMPLE SIZES  ⬇⬇⬇
# =============================================================================
N_SHOTS = 10000   # number of measurements per setting
# =============================================================================


def create_bell_pair():
    """
    Create a Bell state (EPR pair).

         ┌───┐
    q_0: ┤ H ├──●──   Alice
         └───┘┌─┴─┐
    q_1: ─────┤ X ├   Bob
              └───┘

    Result: |Φ⁺⟩ = (1/√2)(|00⟩ + |11⟩)

    Math:
      |00⟩ → H⊗I → (1/√2)(|0⟩+|1⟩)|0⟩ = (1/√2)(|00⟩+|10⟩)
           → CNOT → (1/√2)(|00⟩+|11⟩) = |Φ⁺⟩
    """
    qc = QuantumCircuit(2)
    qc.h(0)      # Hadamard on Alice's qubit
    qc.cx(0, 1)  # CNOT: Alice controls Bob
    return qc


def build_chsh_circuit(alice_angle, bob_angle):
    """
    Build a complete CHSH measurement circuit.

    1. Create Bell pair |Φ⁺⟩
    2. Alice rotates her qubit by alice_angle
    3. Bob rotates his qubit by bob_angle
    4. Both measure in Z-basis

    Rotating before measurement = measuring in a rotated basis.
    Ry(θ) then Z-measure is equivalent to measuring along angle θ.
    """
    qc = QuantumCircuit(2, 2)

    # Step 1: Create entangled Bell pair
    qc.h(0)
    qc.cx(0, 1)

    # Step 2: Rotate measurement bases
    qc.ry(alice_angle, 0)   # Alice's measurement direction
    qc.ry(bob_angle, 1)     # Bob's measurement direction

    # Step 3: Measure both
    qc.measure([0, 1], [0, 1])

    return qc


def run_circuit(qc, shots):
    """Run circuit on ideal simulator, return counts dict."""
    sampler = StatevectorSampler()
    job = sampler.run([qc], shots=shots)
    result = job.result()
    creg = qc.cregs[0].name
    counts = getattr(result[0].data, creg).get_counts()
    return counts


def compute_correlation(counts, shots):
    """
    Compute correlation E(a,b) from measurement results.

    E = P(same) - P(different)
      = (N_00 + N_11 - N_01 - N_10) / N_total

    Same outcome (00, 11) → +1
    Different outcome (01, 10) → -1
    """
    n_00 = counts.get('00', 0)
    n_01 = counts.get('01', 0)
    n_10 = counts.get('10', 0)
    n_11 = counts.get('11', 0)

    E = (n_00 + n_11 - n_01 - n_10) / shots
    return E, n_00, n_01, n_10, n_11


def run_bell_test():
    """
    Run the full CHSH Bell test.

    CHSH uses 4 measurement settings:
      Alice: a1 = 0°,     a2 = 90° (π/2)
      Bob:   b1 = 45°,    b2 = 135° (3π/4)

    These are the angles that MAXIMIZE the Bell violation.

    S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)

    Classical limit:  |S| ≤ 2
    Quantum value:    |S| = 2√2 ≈ 2.828
    """
    # Optimal CHSH angles (in radians for Ry gate)
    a1 = 0              # Alice setting 1: 0°
    a2 = np.pi / 2      # Alice setting 2: 90°
    b1 = np.pi / 4      # Bob setting 1: 45°
    b2 = 3 * np.pi / 4  # Bob setting 2: 135°

    settings = [
        ("a1, b1", "0°, 45°",   a1, b1),
        ("a1, b2", "0°, 135°",  a1, b2),
        ("a2, b1", "90°, 45°",  a2, b1),
        ("a2, b2", "90°, 135°", a2, b2),
    ]

    results = []
    print(f"\n  Running {N_SHOTS} shots per setting...\n")

    for name, angles_str, a, b in settings:
        qc = build_chsh_circuit(a, b)
        counts = run_circuit(qc, N_SHOTS)
        E, n00, n01, n10, n11 = compute_correlation(counts, N_SHOTS)
        results.append({
            "name": name,
            "angles": angles_str,
            "E": E,
            "counts": {"00": n00, "01": n01, "10": n10, "11": n11},
        })
        print(f"    {name} ({angles_str:>10}):  E = {E:+.4f}  "
              f"  |00⟩={n00}  |01⟩={n01}  |10⟩={n10}  |11⟩={n11}")

    # Compute CHSH value: S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)
    S = results[0]["E"] - results[1]["E"] + results[2]["E"] + results[3]["E"]

    return results, S


def run_correlation_sweep():
    """
    Sweep Bob's angle from 0° to 360° while Alice stays at 0°.
    Shows the sinusoidal correlation pattern.
    """
    alice_angle = 0
    bob_angles_deg = np.arange(0, 361, 15)
    bob_angles_rad = np.radians(bob_angles_deg)

    measured_E = []
    theory_E = []

    print("\n  Correlation sweep (Alice=0°, Bob=0°→360°)...")
    for deg, rad in zip(bob_angles_deg, bob_angles_rad):
        qc = build_chsh_circuit(alice_angle, rad)
        counts = run_circuit(qc, N_SHOTS)
        E, _, _, _, _ = compute_correlation(counts, N_SHOTS)
        measured_E.append(E)

        # Theory: E(a,b) = -cos(a - b) for Bell state |Φ⁺⟩ with Ry rotations
        theory_E.append(-np.cos(rad))

    return bob_angles_deg, np.array(measured_E), np.array(theory_E)


# =============================================================================
# Visualization
# =============================================================================
def plot_results(results, S, angles_deg, measured_E, theory_E):
    """Create the full visualization."""

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Bell Experiment — Testing Quantum Non-Locality",
                 fontsize=16, fontweight="bold", y=0.99)

    # ---- Panel 1: Circuit diagram ----
    ax1 = fig.add_subplot(3, 2, 1)
    qc = create_bell_pair()
    qc.draw('mpl', ax=ax1)
    ax1.set_title("Bell State Circuit: |Φ⁺⟩ = (1/√2)(|00⟩+|11⟩)",
                  fontsize=11, fontweight="bold")

    # ---- Panel 2: CHSH result ----
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.axis("off")

    violated = abs(S) > 2
    color = "green" if violated else "red"
    symbol = "✓ VIOLATED" if violated else "✗ NOT violated"

    ax2.text(0.5, 0.85, "CHSH Inequality Test", fontsize=14,
             fontweight="bold", ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.68, f"S = {S:+.4f}", fontsize=28,
             fontweight="bold", ha="center", color=color, transform=ax2.transAxes)
    ax2.text(0.5, 0.50, f"Classical limit: |S| ≤ 2.000",
             fontsize=12, ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.38, f"Quantum theory:  |S| = 2√2 ≈ 2.828",
             fontsize=12, ha="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.20, f"Bell inequality: {symbol}",
             fontsize=14, fontweight="bold", ha="center", color=color,
             transform=ax2.transAxes)
    ax2.text(0.5, 0.05, f"({N_SHOTS} shots per setting)",
             fontsize=10, ha="center", color="gray", transform=ax2.transAxes)

    # ---- Panel 3: Measurement counts for each setting ----
    ax3 = fig.add_subplot(3, 2, 3)
    setting_names = [r["name"] for r in results]
    n00 = [r["counts"]["00"] for r in results]
    n01 = [r["counts"]["01"] for r in results]
    n10 = [r["counts"]["10"] for r in results]
    n11 = [r["counts"]["11"] for r in results]

    x = np.arange(len(setting_names))
    w = 0.2
    ax3.bar(x - 1.5*w, n00, w, color="#2ecc71", label="|00⟩", alpha=0.8)
    ax3.bar(x - 0.5*w, n01, w, color="#e74c3c", label="|01⟩", alpha=0.8)
    ax3.bar(x + 0.5*w, n10, w, color="#3498db", label="|10⟩", alpha=0.8)
    ax3.bar(x + 1.5*w, n11, w, color="#f39c12", label="|11⟩", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{r['name']}\n({r['angles']})" for r in results], fontsize=8)
    ax3.set_ylabel("Counts")
    ax3.set_title("Measurement Outcomes per Setting", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # ---- Panel 4: Correlation values ----
    ax4 = fig.add_subplot(3, 2, 4)
    E_values = [r["E"] for r in results]
    colors = ["green" if e > 0 else "red" for e in E_values]
    bars = ax4.bar(setting_names, E_values, color=colors, alpha=0.8, edgecolor="black")
    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_ylabel("Correlation E(a,b)")
    ax4.set_title("Correlation per Setting", fontsize=11, fontweight="bold")
    ax4.set_ylim(-1.1, 1.1)
    ax4.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, E_values):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05 * np.sign(bar.get_height()),
                 f"{val:+.3f}", ha="center", fontsize=10, fontweight="bold")

    # ---- Panel 5: Correlation sweep ----
    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(angles_deg, theory_E, 'g--', lw=2.5, label=r"Theory: $-\cos(\theta)$")
    ax5.scatter(angles_deg, measured_E, color="purple", s=30, zorder=5,
                label="Measured (quantum circuit)")
    ax5.axhline(0, color="gray", ls=":", alpha=0.5)

    # Mark the CHSH optimal angles
    for angle, label, color in [(45, "b1=45°", "blue"), (135, "b2=135°", "red")]:
        ax5.axvline(angle, color=color, ls="--", alpha=0.5)
        ax5.text(angle + 3, 0.9, label, fontsize=8, color=color)

    ax5.set_xlabel("Bob's Measurement Angle θ (degrees)", fontsize=12)
    ax5.set_ylabel("Correlation E(0°, θ)", fontsize=12)
    ax5.set_title("Correlation vs. Measurement Angle (Alice fixed at 0°)",
                   fontsize=12, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-5, 365)
    ax5.set_ylim(-1.2, 1.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("bell_experiment.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  🔔 BELL EXPERIMENT — Testing Quantum Non-Locality")
    print("  Backend: StatevectorSampler (ideal, no noise)")
    print(f"  Shots per setting: {N_SHOTS}")
    print("=" * 62)

    # Show the Bell state circuit
    print("\n  Bell State Circuit (creates entangled pair):")
    bell = create_bell_pair()
    print(bell.draw(output='text'))

    # Show a full CHSH circuit
    print("\n  Full CHSH Circuit (with measurement rotations):")
    example = build_chsh_circuit(0, np.pi/4)
    print(example.draw(output='text'))

    # ── Run CHSH test ──
    print("\n  ── CHSH BELL TEST ──")
    results, S = run_bell_test()

    # ── Print CHSH result ──
    print(f"\n  CHSH Value:")
    print(f"    S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)")
    print(f"    S = ({results[0]['E']:+.4f}) - ({results[1]['E']:+.4f}) "
          f"+ ({results[2]['E']:+.4f}) + ({results[3]['E']:+.4f})")
    print(f"    S = {S:+.4f}")

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Classical limit:  |S| ≤ 2.000              │")
    print(f"  │  Quantum theory:   |S| = 2√2 ≈ 2.828       │")
    print(f"  │  Our result:       |S| = {abs(S):.4f}              │")
    if abs(S) > 2:
        print(f"  │                                             │")
        print(f"  │  ✓ BELL INEQUALITY VIOLATED!                │")
        print(f"  │  → Local hidden variables RULED OUT         │")
        print(f"  │  → Quantum entanglement is REAL             │")
        print(f"  │  → Nature is NON-LOCAL                      │")
    else:
        print(f"  │  ✗ Bell inequality not violated             │")
    print(f"  └─────────────────────────────────────────────┘")

    # ── Correlation sweep ──
    print("\n  ── CORRELATION SWEEP ──")
    angles_deg, measured_E, theory_E = run_correlation_sweep()

    # ── Summary ──
    print(f"""
  ══════════════════════════════════════════════════════
  WHAT THIS PROVES:
  ══════════════════════════════════════════════════════

  Einstein (EPR 1935):
    "Quantum mechanics must be incomplete.
     There must be local hidden variables."

  Bell (1964):
    "If local hidden variables exist, then |S| ≤ 2.
     Quantum mechanics predicts |S| = 2√2."

  Our experiment:
    |S| = {abs(S):.4f} > 2.000

  CONCLUSION:
    ✓ Einstein was WRONG — no local hidden variables
    ✓ Entangled particles are correlated beyond
      what any classical theory can explain
    ✓ Measuring Alice's qubit instantly affects
      what Bob can observe — NON-LOCALITY
  ══════════════════════════════════════════════════════
    """)

    # Plot
    plot_results(results, S, angles_deg, measured_E, theory_E)
    print("  ✓ Saved: bell_experiment.png\n")