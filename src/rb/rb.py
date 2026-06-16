"""
Single-qubit logical Pauli-frame randomized benchmarking (RB) on top of the
correlated noise models in src/noisemodel.

Protocol
--------
Starting from the code's prepared logical |0>_L, we apply a random sequence of
L logical Pauli gates {I, X, Y, Z}, each followed by `rounds_between_gates`
rounds of syndrome extraction. The logical Pauli gates are implemented as
*transversal* physical X / Z operators on the data qubits supporting the
corresponding logical operator. Because logical X and Z commute with every
stabilizer of a CSS code, inserting them mid-circuit does not disturb the
detector structure -- only the final logical observable's expected value
changes. The (X, Z) parity mod 2 of all applied gates is tracked exactly
(the single-qubit Pauli group mod phase is the Klein four-group, and every
non-identity element is self-inverse), so a single closing "recovery" gate --
equal to the accumulated frame -- exactly returns the logical observable to
its original deterministic value in the absence of errors. A logical error
is then detected by simply comparing the decoder's prediction against the
circuit's actual observable flip, exactly as in a standard memory experiment.

Per the chosen design, temporally-correlated noise is resampled fresh at
each gate boundary ("reset at each gate"): every block of
`rounds_between_gates` rounds is an independent draw from the noise model,
using the model's own `rounds=rounds_between_gates` pair/streak statistics.
This lets us reuse each noise model's existing, validated error-sampling
primitives (`sample_errors` / `sample_control_errors` /
`calc_marginals_per_round`) unmodified -- we only write new orchestration
code that drives a single, persistent `stim.FlipSimulator` across multiple
gate-separated blocks.
"""

import numpy as np
import stim

from ..noisemodel.long_time_pair_m import LongTimePairM
from ..noisemodel.long_time_pair_c import LongTimePairC
from ..noisemodel.long_time_pair_all import LongTimePairA
from ..noisemodel.noise_model_util import split_circuit

CODE_CIRCUITS = {
    "surface_code": "surface_code:rotated_memory_z",
    "repetition_code": "repetition_code:memory",
}

# Single-qubit Pauli group mod global phase ~= (Z2 x Z2), Pauli <-> (has_X, has_Z).
PAULI_TO_FRAME = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
FRAME_TO_PAULI = {v: k for k, v in PAULI_TO_FRAME.items()}


def gen_template_circuit(code, distance, rounds):
    return stim.Circuit.generated(CODE_CIRCUITS[code], rounds=rounds, distance=distance)


def _extract_observable_support(circuit):
    """Maps OBSERVABLE_INCLUDE(0)'s rec[] targets back to the (qubit, basis)
    pairs of whichever measurement instruction produced those records."""
    circuit = circuit.flattened()
    measurement_log = []
    observable_targets = None
    basis_of = {"M": "Z", "MZ": "Z", "MR": "Z", "MX": "X", "MY": "Y"}
    for instr in circuit:
        if instr.name in basis_of:
            basis = basis_of[instr.name]
            for t in instr.targets_copy():
                measurement_log.append((t.value, basis))
        elif instr.name == "OBSERVABLE_INCLUDE":
            observable_targets = instr.targets_copy()
    n = len(measurement_log)
    return sorted({measurement_log[n + t.value][0] for t in observable_targets})


def partition_qubits_repetition(circuit):
    """get_partitioned_qubit_coords() requires QUBIT_COORDS, which stim's
    repetition_code generator never emits, so we re-derive data/syndrome
    qubit indices directly from the M / MR instructions instead."""
    circuit = circuit.flattened()
    data_qubits = None
    syndrome_qubits = None
    for instr in circuit[::-1]:
        if instr.name == "M" and data_qubits is None:
            data_qubits = [t.value for t in instr.targets_copy()]
        elif instr.name == "MR" and syndrome_qubits is None:
            syndrome_qubits = [t.value for t in instr.targets_copy()]
        if data_qubits is not None and syndrome_qubits is not None:
            break
    return data_qubits, syndrome_qubits


def get_logical_pauli_support(code, distance):
    """Physical data qubits to hit with a transversal X / Z gate to realize
    the logical X / logical Z operator for the given code and distance."""
    if code == "surface_code":
        cx = stim.Circuit.generated("surface_code:rotated_memory_x", rounds=1, distance=distance)
        cz = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=1, distance=distance)
        return {"X": _extract_observable_support(cx), "Z": _extract_observable_support(cz)}
    elif code == "repetition_code":
        cz = stim.Circuit.generated("repetition_code:memory", rounds=1, distance=distance)
        z_support = _extract_observable_support(cz)
        data_qubits, _ = partition_qubits_repetition(cz)
        # This code only protects against X (bit-flip) errors via Z-type stabilizers:
        # logical Z has a trivial weight-1 representative (no distance protection),
        # while logical X requires the full transversal product over all data qubits
        # (weight d, the operator the code actually protects).
        return {"X": sorted(data_qubits), "Z": z_support}
    raise ValueError(f"Unknown code {code!r}")


def get_noise_targets(model, circuit, code):
    """Qubit-index list the noise model should act on. Bypasses
    get_partitioned_qubit_coords (which needs QUBIT_COORDS) for the
    repetition code."""
    if code == "repetition_code":
        data_qubits, syndrome_qubits = partition_qubits_repetition(circuit)
        if model.noisy_qubits == "data":
            return data_qubits
        elif model.noisy_qubits == "syndrome":
            return syndrome_qubits
        elif model.noisy_qubits == "all":
            return data_qubits + syndrome_qubits
        raise ValueError(model.noisy_qubits)
    return model.get_targets(circuit)


def _archetype(model):
    """Each noise model class injects errors at different points within a
    round; dispatch on the (small, fixed) set of patterns used in src/noisemodel."""
    if isinstance(model, LongTimePairA):
        return "A"
    if isinstance(model, LongTimePairC):
        return "C"
    if isinstance(model, LongTimePairM):
        return "M"
    return "base"


def supported_codes_for_model(model):
    """Class 2 (CX-correlated) and the combined 'All' model hard-code the
    rotated surface code's 4-CX-layer round structure and H-instruction-based
    role detection; the repetition code's rounds don't match that shape."""
    archetype = _archetype(model)
    if archetype in ("C", "A"):
        return {"surface_code"}
    return {"surface_code", "repetition_code"}


def _compute_block_errors(model, archetype, targets, n_qubits, rounds, batch_size):
    if archetype == "C":
        return model.sample_control_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)
    return model.sample_errors(targets=targets, n_qubits=n_qubits, rounds=rounds, batch_size=batch_size)


def _do_round(model, archetype, sim, fragment, errors, j):
    """Inject round j's error masks at the correct point(s) within `fragment`
    and execute it, mirroring each noise model class's own _sample_circuit."""
    if archetype == "base":
        if model.error_type == "depolarizing":
            X, Y, Z = errors
            sim.broadcast_pauli_errors(pauli="X", mask=X[j])
            sim.broadcast_pauli_errors(pauli="Y", mask=Y[j])
            sim.broadcast_pauli_errors(pauli="Z", mask=Z[j])
        else:
            sim.broadcast_pauli_errors(pauli=model.error_type, mask=errors[j])
        sim.do(fragment)

    elif archetype == "M":
        piece0, piece1 = fragment
        sim.do(piece0)
        sim.broadcast_pauli_errors(pauli=model.error_type, mask=errors[j])
        sim.do(piece1)

    elif archetype == "C":
        p0, p1, p2, p3, p4 = fragment
        for i, piece in enumerate([p0, p1, p2, p3]):
            sim.do(piece)
            X, Y, Z = errors[i]
            sim.broadcast_pauli_errors(pauli="X", mask=X[j])
            sim.broadcast_pauli_errors(pauli="Y", mask=Y[j])
            sim.broadcast_pauli_errors(pauli="Z", mask=Z[j])
        sim.do(p4)

    elif archetype == "A":
        p0, p1, p2, p3, p4, p5 = fragment
        d_errors, m_errors, c_errors = errors
        dX, dY, dZ = d_errors
        sim.broadcast_pauli_errors(pauli="X", mask=dX[j])
        sim.broadcast_pauli_errors(pauli="Y", mask=dY[j])
        sim.broadcast_pauli_errors(pauli="Z", mask=dZ[j])
        for i, piece in enumerate([p0, p1, p2, p3]):
            sim.do(piece)
            cX, cY, cZ = c_errors[i]
            sim.broadcast_pauli_errors(pauli="X", mask=cX[j])
            sim.broadcast_pauli_errors(pauli="Y", mask=cY[j])
            sim.broadcast_pauli_errors(pauli="Z", mask=cZ[j])
        sim.do(p4)
        sim.broadcast_pauli_errors(pauli="X", mask=m_errors[j])
        sim.do(p5)


def _run_rounds(model, archetype, sim, init_round_fragment, repeat_fragment, errors, num_rounds, first_round_uses_init):
    for j in range(num_rounds):
        fragment = init_round_fragment if (first_round_uses_init and j == 0) else repeat_fragment
        _do_round(model, archetype, sim, fragment, errors, j)


def _apply_pauli_gate(sim, gate, logical_support):
    if gate == "I":
        return
    frag = stim.Circuit()
    if gate in ("X", "Y"):
        frag.append("X", logical_support["X"])
    if gate in ("Z", "Y"):
        frag.append("Z", logical_support["Z"])
    sim.do(frag)


def _compose_frame(frame, pauli_frame):
    return (frame[0] ^ pauli_frame[0], frame[1] ^ pauli_frame[1])


def run_rb_batch(model, code, distance, rounds_between_gates, gate_sequence, batch_size, logical_support=None):
    """
    Simulate `batch_size` independent shots of a logical RB sequence.

    gate_sequence: list of 'I'/'X'/'Y'/'Z', the L random logical gates to apply
        (a closing recovery gate is appended automatically).

    Returns: detection_events (batch_size x num_detectors bool array),
             observable_flips (batch_size bool array), recovery_gate (str).
    """
    m = rounds_between_gates
    num_blocks = len(gate_sequence) + 1
    template = gen_template_circuit(code, distance, max(2, m + 1))

    archetype = _archetype(model)
    (circuit_init, circuit_init_round, circuit_repeat_block, circuit_final), _ = model.split_circuit(template)
    targets = get_noise_targets(model, template, code)
    n_qubits = template.num_qubits

    if logical_support is None:
        logical_support = get_logical_pauli_support(code, distance)

    sim = stim.FlipSimulator(batch_size=batch_size)
    sim.do(circuit_init)

    frame = (0, 0)

    errors0 = _compute_block_errors(model, archetype, targets, n_qubits, m, batch_size)
    _run_rounds(model, archetype, sim, circuit_init_round, circuit_repeat_block, errors0, m, first_round_uses_init=True)

    for gate in gate_sequence:
        _apply_pauli_gate(sim, gate, logical_support)
        frame = _compose_frame(frame, PAULI_TO_FRAME[gate])
        errors_k = _compute_block_errors(model, archetype, targets, n_qubits, m, batch_size)
        _run_rounds(model, archetype, sim, None, circuit_repeat_block, errors_k, m, first_round_uses_init=False)

    recovery_gate = FRAME_TO_PAULI[frame]
    _apply_pauli_gate(sim, recovery_gate, logical_support)

    sim.do(circuit_final)

    detection_events = sim.get_detector_flips().transpose()
    observable_flips = sim.get_observable_flips().flatten()

    return detection_events, observable_flips, recovery_gate


def _marginal_round_circuit(model, archetype, fragment, marginals, targets, j):
    out = stim.Circuit()
    if archetype == "base":
        if model.error_type == "depolarizing":
            out.append("DEPOLARIZE1", targets, marginals[j])
        else:
            out.append(model.error_type + "_ERROR", targets, marginals[j])
        out += fragment

    elif archetype == "M":
        p0, p1 = fragment
        out += p0
        out.append("X_ERROR", targets, marginals[j])
        out += p1

    elif archetype == "C":
        p0, p1, p2, p3, p4 = fragment
        for i, piece in enumerate([p0, p1, p2, p3]):
            out += piece
            out.append("DEPOLARIZE2", targets[i], marginals[j])
        out += p4

    elif archetype == "A":
        d_marg, m_marg, c_marg = marginals
        d_targets, m_targets, c_targets = targets
        out.append("DEPOLARIZE1", d_targets, d_marg[j])
        p0, p1, p2, p3, p4, p5 = fragment
        for i, piece in enumerate([p0, p1, p2, p3]):
            out += piece
            out.append("DEPOLARIZE2", c_targets[i], c_marg[j])
        out += p4
        out.append("X_ERROR", m_targets, m_marg[j])
        out += p5

    return out


def build_marginalised_rb_circuit(model, code, distance, rounds_between_gates, num_gates):
    """A noise-equivalent circuit (same detectors/observable, no logical gates
    since those don't affect detector structure) used only to build a matching
    detector_error_model for decoding -- mirrors run_rb_batch's block structure."""
    m = rounds_between_gates
    num_blocks = num_gates + 1
    template = gen_template_circuit(code, distance, max(2, m + 1))

    archetype = _archetype(model)
    (circuit_init, circuit_init_round, circuit_repeat_block, circuit_final), _ = model.split_circuit(template)
    targets = get_noise_targets(model, template, code)

    output = stim.Circuit()
    output += circuit_init

    for b in range(num_blocks):
        marginals = model.calc_marginals_per_round(m)
        is_first = b == 0
        for j in range(m):
            fragment = circuit_init_round if (is_first and j == 0) else circuit_repeat_block
            output += _marginal_round_circuit(model, archetype, fragment, marginals, targets, j)

    output += circuit_final
    return output


def gen_random_pauli_sequence(length, rng=None, paulis=("I", "X", "Y", "Z")):
    rng = rng or np.random.default_rng()
    return list(rng.choice(paulis, size=length))
