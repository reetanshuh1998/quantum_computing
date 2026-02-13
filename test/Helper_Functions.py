# Grover Oracle for multiple marked states

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCMTGate, ZGate, MCXGate
import numpy as np

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# Imports from Qiskit Runtime
from qiskit_ibm_runtime import SamplerV2 as Sampler


def Grover_oracle(marked_states):
    """Build a Grover oracle for multiple marked states

    Here we assume all input marked states have the same number of bits

    Parameters:
        marked_states (str or list): Marked states of oracle

    Returns:
        QuantumCircuit: Quantum circuit representing Grover oracle
    """
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    # Compute the number of qubits in circuit
    num_qubits = len(marked_states[0])

    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [
            ind
            for ind in range(num_qubits)
            if rev_target.startswith("0", ind)
        ]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if zero_inds:
            qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        if zero_inds:
            qc.x(zero_inds)
    return qc


def Grover_operator(oracle: QuantumCircuit, insert_barriers: bool = False, name: str = "Q", reflection_qubits=None) -> QuantumCircuit:
    """Build the Grover operator given an oracle.
    Parameters:
        oracle (QuantumCircuit): Oracle circuit
        insert_barriers (bool): Whether to insert barriers between sections
        name (str): Name of the Grover operator circuit
        reflection_qubits (list): List of qubits to apply the reflection on. If None, applies to all qubits.
    Returns:
        QuantumCircuit: Grover operator circuit
    """

    circuit = oracle.copy_empty_like(name=name, vars_mode="drop")
    circuit.compose(oracle, inplace=True)

    if insert_barriers:
        circuit.barrier()

    if reflection_qubits is None:
        reflection_qubits = [
            i for i, qubit in enumerate(circuit.qubits)
        ]

    circuit.h(reflection_qubits)  # H is self-inverse

    if insert_barriers:
        circuit.barrier()

    # Add the zero reflection.
    num_reflection = len(reflection_qubits)

    circuit.x(reflection_qubits)

    if insert_barriers:
        circuit.barrier()

    if num_reflection == 1:
        circuit.z(
            reflection_qubits[0]
        )  # MCX does not support 0 controls, hence this is separate
    else:
        mcx = MCXGate(num_reflection - 1)

        circuit.h(reflection_qubits[-1])
        circuit.append(mcx, reflection_qubits)
        circuit.h(reflection_qubits[-1])

    if insert_barriers:
        circuit.barrier()

    circuit.x(reflection_qubits)

    if insert_barriers:
        circuit.barrier()

    circuit.h(reflection_qubits)

    # minus sign
    circuit.global_phase = np.pi

    return circuit


def Get_Data_from_Fake_backend(shots, circuit, fake_backend):
    """
    Run a circuit on a fake backend and return the result data for PUB 0

    Parameters:
        shots (int): Number of shots to run the circuit
        circuit (QuantumCircuit): Circuit to run
        fake_backend (FakeBackend): Fake backend to run the circuit on

    Returns:
        data_pub (dict): Result data for PUB 0

    """
    pm = generate_preset_pass_manager(
        backend=fake_backend, optimization_level=1)
    isa_circuit = pm.run(circuit)
    sampler = Sampler(mode=fake_backend)
    # Run using sampler
    result = sampler.run([isa_circuit], shots=shots).result()
    # Access result data for PUB 0
    data_pub = result[0].data
    return data_pub


def XOR(qc, a, b, output):
    qc.cx(a, output)
    qc.cx(b, output)


def Sudoku_oracle(var_qubits, clause_list, clause_qubits, output_qubit):
    qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit)

    # Compute clauses
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1

    # Flip 'output' bit if all clauses are satisfied
    mcx = MCXGate(len(clause_qubits))
    mcx_index = [i for i in range(len(var_qubits), len(
        var_qubits)+len(clause_qubits)+len(output_qubit))]

    qc.append(mcx, mcx_index)

    # Uncompute clauses to reset clause-checking bits to 0
    i = 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[i])
        i += 1
    return qc