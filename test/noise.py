from Helper_Functions import Grover_oracle, Grover_operator
import numpy as np
import math
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

marked_states = ["11010"]
target = marked_states[0]
oracle = Grover_oracle(marked_states)
operator = Grover_operator(oracle)

n = 5
M = len(marked_states)
N = 2 ** n
optimal_iterations = math.floor(math.pi / (4 * math.asin(math.sqrt(M / N))))

shots = 10000
iterations_to_test = range(0, optimal_iterations + 3)

# --- IDEAL (StatevectorSampler, no noise) ---
ideal_probs = []
ideal_sampler = StatevectorSampler()

for num_iters in iterations_to_test:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(num_iters):
        qc.compose(operator, inplace=True)
    qc.measure_all()

    result = ideal_sampler.run([qc], shots=shots).result()
    counts = result[0].data.meas.get_counts()
    prob = counts.get(target, 0) / shots
    ideal_probs.append(prob)

# --- NOISY (your results) ---
noisy_probs = [0.0317, 0.1028, 0.1034, 0.0703, 0.0497, 0.0333, 0.0274]

# --- PLOT ---
fig, ax = plt.subplots(figsize=(12, 7))

x = list(iterations_to_test)
bar_width = 0.35

bars1 = ax.bar([i - bar_width/2 for i in x], ideal_probs, bar_width,
               color='steelblue', edgecolor='black', label='Ideal (no noise)')
bars2 = ax.bar([i + bar_width/2 for i in x], noisy_probs, bar_width,
               color='salmon', edgecolor='black', label='Noisy (FakeAlgiers)')

# Random guess line
ax.axhline(y=1/N, color='gray', linestyle='--', label=f'Random guess = {1/N:.4f}')

# Labels on bars
for bar, p in zip(bars1, ideal_probs):
    ax.text(bar.get_x() + bar.get_width()/2, p + 0.02, f'{p:.2f}',
            ha='center', fontsize=9, fontweight='bold', color='steelblue')
for bar, p in zip(bars2, noisy_probs):
    ax.text(bar.get_x() + bar.get_width()/2, p + 0.02, f'{p:.2f}',
            ha='center', fontsize=9, fontweight='bold', color='red')

ax.set_xlabel('Number of Grover Iterations', fontsize=14)
ax.set_ylabel(f'Probability of |{target}⟩', fontsize=14)
ax.set_title("Grover's Algorithm: Ideal vs Noisy (FakeAlgiers)\n"
             "~540 CX gates destroy the quantum signal!", fontsize=15)
ax.set_xticks(x)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=13)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("noise.png")