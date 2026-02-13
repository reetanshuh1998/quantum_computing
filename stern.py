"""
Stern-Gerlach Experiment Simulation
=====================================
Simulates silver atoms (spin-1/2) passing through an inhomogeneous
magnetic field, demonstrating spatial quantization of angular momentum.

The beam splits into two discrete components corresponding to
spin-up (m_s = +1/2) and spin-down (m_s = -1/2) states.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# Physical Constants
# =============================================================================
MU_B = 9.274e-24        # Bohr magneton (J/T)
M_AG = 1.791e-25        # Mass of silver atom (kg)
K_B = 1.381e-23         # Boltzmann constant (J/K)
HBAR = 1.055e-34        # Reduced Planck constant (J·s)

# =============================================================================
# Experiment Parameters
# =============================================================================
T_OVEN = 1000           # Oven temperature (K)
DB_DZ = 100             # Magnetic field gradient (T/m)
L_MAGNET = 0.10         # Length of magnet region (m)
L_DRIFT = 0.20          # Drift distance after magnet (m)
N_ATOMS = 5000          # Number of simulated atoms
G_S = 2.0               # Electron spin g-factor
SPIN = 0.5              # Spin quantum number for silver's valence electron


def compute_beam_velocity(temperature, mass):
    """Compute most probable speed from Maxwell-Boltzmann distribution."""
    return np.sqrt(2 * K_B * temperature / mass)


def sample_velocities(n, v_mp):
    """
    Sample atomic speeds from a Maxwell-Boltzmann distribution.

    Parameters
    ----------
    n : int
        Number of atoms to sample.
    v_mp : float
        Most probable speed (m/s).

    Returns
    -------
    speeds : ndarray
        Array of sampled speeds.
    """
    # Maxwell-Boltzmann speed distribution: f(v) ~ v^2 * exp(-v^2 / v_mp^2)
    # chi(k=3) = sqrt(chisquare(df=3))
    # NumPy's Generator does NOT have a .chi() method,
    # so we use: sqrt(chisquare(df=3)) which is equivalent.
    rng = np.random.default_rng(42)
    speeds = np.sqrt(rng.chisquare(df=3, size=n)) * v_mp / np.sqrt(2)
    return speeds


def compute_deflection(m_s, speeds, db_dz, l_magnet, l_drift, g_s=2.0):
    """
    Compute vertical deflection of atoms after passing through the magnet.

    The force on the atom is: F_z = -m_s * g_s * mu_B * (dB/dz)
    Deflection in magnet:     z1 = 0.5 * a * t_magnet^2
    Additional drift:         z2 = v_z * t_drift

    Parameters
    ----------
    m_s : float
        Magnetic spin quantum number (+1/2 or -1/2).
    speeds : ndarray
        Longitudinal speeds of atoms (m/s).
    db_dz : float
        Magnetic field gradient (T/m).
    l_magnet : float
        Length of magnet region (m).
    l_drift : float
        Free drift distance after magnet (m).
    g_s : float
        Electron spin g-factor.

    Returns
    -------
    z_total : ndarray
        Total vertical deflections (m).
    """
    force = m_s * g_s * MU_B * db_dz
    acceleration = force / M_AG

    t_magnet = l_magnet / speeds
    z_in_magnet = 0.5 * acceleration * t_magnet ** 2
    v_z = acceleration * t_magnet

    t_drift = l_drift / speeds
    z_in_drift = v_z * t_drift

    z_total = z_in_magnet + z_in_drift
    return z_total


def simulate_stern_gerlach(n_atoms=N_ATOMS):
    """
    Run the full Stern-Gerlach simulation.

    Returns
    -------
    results : dict
        Dictionary containing all simulation data.
    """
    v_mp = compute_beam_velocity(T_OVEN, M_AG)
    speeds = sample_velocities(n_atoms, v_mp)

    # Each atom is randomly in spin-up or spin-down state
    rng = np.random.default_rng(123)
    spin_states = rng.choice([-SPIN, +SPIN], size=n_atoms)

    # Add small random transverse spread (collimation imperfection)
    z_initial = rng.normal(0, 0.0002, size=n_atoms)  # 0.2 mm spread

    # Compute deflections
    deflections = compute_deflection(spin_states, speeds, DB_DZ, L_MAGNET, L_DRIFT)
    z_final = z_initial + deflections

    # Convert to mm for visualization
    z_final_mm = z_final * 1000

    results = {
        "speeds": speeds,
        "spin_states": spin_states,
        "z_initial": z_initial * 1000,
        "z_final_mm": z_final_mm,
        "v_mp": v_mp,
        "n_atoms": n_atoms,
        "deflections_mm": deflections * 1000,
    }
    return results


def plot_experiment_schematic(ax):
    """Draw a schematic of the experimental apparatus."""
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Stern-Gerlach Experiment — Schematic", fontsize=14, fontweight="bold")

    # Oven
    oven = patches.FancyBboxPatch(
        (0, -0.5), 0.8, 1.0, boxstyle="round,pad=0.05",
        facecolor="#d4533b", edgecolor="black", linewidth=1.5
    )
    ax.add_patch(oven)
    ax.text(0.4, 0, "Oven\n(Ag)", ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")

    # Collimator slit
    ax.plot([1.2, 1.2], [-0.6, -0.15], color="black", linewidth=3)
    ax.plot([1.2, 1.2], [0.15, 0.6], color="black", linewidth=3)
    ax.text(1.2, 0.8, "Slit", ha="center", fontsize=8)

    # Beam before magnet
    ax.annotate("", xy=(2.0, 0), xytext=(0.85, 0),
                arrowprops=dict(arrowstyle="->", color="orange", lw=2))

    # Magnet (N and S poles)
    magnet_n = patches.FancyBboxPatch(
        (2.0, 0.4), 1.5, 0.5, boxstyle="round,pad=0.03",
        facecolor="#4a90d9", edgecolor="black", linewidth=1.5
    )
    magnet_s = patches.FancyBboxPatch(
        (2.0, -0.9), 1.5, 0.5, boxstyle="round,pad=0.03",
        facecolor="#e74c3c", edgecolor="black", linewidth=1.5
    )
    ax.add_patch(magnet_n)
    ax.add_patch(magnet_s)
    ax.text(2.75, 0.65, "N", ha="center", va="center", fontsize=12,
            fontweight="bold", color="white")
    ax.text(2.75, -0.65, "S", ha="center", va="center", fontsize=12,
            fontweight="bold", color="white")
    ax.text(2.75, 1.1, "Inhomogeneous\nMagnetic Field", ha="center", fontsize=7,
            style="italic", color="gray")

    # Beam splitting inside magnet
    ax.plot([2.0, 3.5], [0, 0], color="orange", lw=1.5, ls="--", alpha=0.4)
    ax.plot([2.0, 3.5], [0, 0.15], color="red", lw=2)
    ax.plot([2.0, 3.5], [0, -0.15], color="blue", lw=2)

    # Beam after magnet (diverging)
    ax.plot([3.5, 4.8], [0.15, 0.55], color="red", lw=2)
    ax.plot([3.5, 4.8], [-0.15, -0.55], color="blue", lw=2)

    # Detector screen
    screen = patches.FancyBboxPatch(
        (4.8, -1.2), 0.15, 2.4, boxstyle="round,pad=0.02",
        facecolor="#2ecc71", edgecolor="black", linewidth=1.5, alpha=0.7
    )
    ax.add_patch(screen)
    ax.text(5.3, 0, "Detector\nScreen", ha="center", va="center", fontsize=8)

    # Labels for the two spots
    ax.annotate(r"$m_s = +\frac{1}{2}$  (spin up)", xy=(4.85, 0.55),
                xytext=(5.1, 1.2), fontsize=8, color="red",
                arrowprops=dict(arrowstyle="->", color="red"))
    ax.annotate(r"$m_s = -\frac{1}{2}$  (spin down)", xy=(4.85, -0.55),
                xytext=(5.1, -1.2), fontsize=8, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue"))


def plot_results(results):
    """
    Create a multi-panel figure showing simulation results.

    Panel 1: Experiment schematic
    Panel 2: Detector histogram (the classic two-band pattern)
    Panel 3: Deflection vs speed scatter plot
    Panel 4: Classical prediction vs quantum result comparison
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Stern-Gerlach Experiment Simulation", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: Schematic ---
    ax1 = fig.add_subplot(2, 2, 1)
    plot_experiment_schematic(ax1)

    # --- Panel 2: Detector Pattern (Histogram) ---
    ax2 = fig.add_subplot(2, 2, 2)
    z = results["z_final_mm"]
    spin = results["spin_states"]

    ax2.hist(z[spin > 0], bins=80, orientation="horizontal", alpha=0.7,
             color="red", label=r"$m_s = +1/2$ (spin up)", density=True)
    ax2.hist(z[spin < 0], bins=80, orientation="horizontal", alpha=0.7,
             color="blue", label=r"$m_s = -1/2$ (spin down)", density=True)
    ax2.axhline(0, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Probability Density", fontsize=11)
    ax2.set_ylabel("Vertical Position on Screen (mm)", fontsize=11)
    ax2.set_title("Detector Screen Pattern", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Deflection vs Speed ---
    ax3 = fig.add_subplot(2, 2, 3)
    colors = np.where(spin > 0, "red", "blue")
    ax3.scatter(results["speeds"], results["deflections_mm"],
                c=colors, alpha=0.15, s=3, rasterized=True)
    ax3.axhline(0, color="gray", ls="--", alpha=0.5)
    ax3.set_xlabel("Atomic Speed (m/s)", fontsize=11)
    ax3.set_ylabel("Deflection (mm)", fontsize=11)
    ax3.set_title("Deflection vs. Atomic Speed", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Add annotation
    v_mp = results["v_mp"]
    ax3.axvline(v_mp, color="green", ls=":", alpha=0.7)
    ax3.text(v_mp * 1.02, ax3.get_ylim()[1] * 0.85, f"$v_{{mp}}$ = {v_mp:.0f} m/s",
             fontsize=8, color="green")

    # --- Panel 4: Classical vs Quantum Comparison ---
    ax4 = fig.add_subplot(2, 2, 4)

    # Quantum result: two discrete peaks
    bins = np.linspace(z.min(), z.max(), 120)

    ax4.hist(z, bins=bins, orientation="horizontal", alpha=0.8,
             color="purple", density=True, label="Quantum (observed)")

    # Classical prediction: continuous spread
    z_classical = np.linspace(z.min(), z.max(), 500)
    # Classical: uniform distribution of magnetic moment orientations
    # leads to a broad continuous band
    classical_pdf = np.ones_like(z_classical)
    classical_pdf /= np.trapezoid(classical_pdf, z_classical)
    ax4.plot(classical_pdf, z_classical, color="orange", lw=2.5,
             ls="--", label="Classical (predicted)")

    ax4.set_xlabel("Probability Density", fontsize=11)
    ax4.set_ylabel("Vertical Position (mm)", fontsize=11)
    ax4.set_title("Classical vs. Quantum Prediction", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("stern_gerlach_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_summary(results):
    """Print a summary of the simulation parameters and results."""
    z = results["z_final_mm"]
    spin = results["spin_states"]

    print("=" * 60)
    print("   STERN-GERLACH EXPERIMENT SIMULATION — SUMMARY")
    print("=" * 60)
    print(f"\n  Oven Temperature:        {T_OVEN} K")
    print(f"  Most Probable Speed:     {results['v_mp']:.1f} m/s")
    print(f"  Magnetic Field Gradient: {DB_DZ} T/m")
    print(f"  Magnet Length:           {L_MAGNET * 100:.1f} cm")
    print(f"  Drift Distance:          {L_DRIFT * 100:.1f} cm")
    print(f"  Number of Atoms:         {results['n_atoms']}")
    print(f"\n  --- Results ---")
    print(f"  Spin-up  (m_s = +1/2):   mean z = {z[spin > 0].mean():+.3f} mm")
    print(f"  Spin-down (m_s = -1/2):  mean z = {z[spin < 0].mean():+.3f} mm")
    print(f"  Beam separation:         {abs(z[spin > 0].mean() - z[spin < 0].mean()):.3f} mm")
    print(f"  Peak width (std):        {z[spin > 0].std():.3f} mm")
    print("=" * 60)
    print("\n  ✓ Two discrete spots observed — angular momentum is QUANTIZED!")
    print("  ✗ Classical prediction of a continuous smear is WRONG.\n")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("\nRunning Stern-Gerlach simulation...\n")
    results = simulate_stern_gerlach(n_atoms=N_ATOMS)
    print_summary(results)
    plot_results(results)