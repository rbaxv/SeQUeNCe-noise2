# File: sequence/kernel/quantum_manager.py
#
# This module defines the quantum manager class hierarchy, to track quantum states.
# It supports various formalisms (ket vector, density matrix, Fock density matrix, Bell diagonal).
#
# Changes in this version focus on:
# 1. Restoring the original class hierarchy (abstract QuantumManager, and concrete subclasses).
# 2. Adapting all quantum state manipulations to use QuTiP's Qobj objects for robustness and efficiency.
# 3. Ensuring compatibility with the new noise models and stochastic parameters in components
#    (Memory, Detector, BSM), which delegate noise application to components, while QuantumManager
#    handles ideal state evolution.
# 4. FIX: Add `timeline` reference to `QuantumManager` and its subclasses for proper operation.

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import numpy as np
import qutip
from typing import TYPE_CHECKING, Any, Union, List
import math

# Re-added specific numpy/scipy imports as used in original subclasses for matrix ops
from numpy import log, array, cumsum, zeros, outer, kron
from scipy.sparse import csr_matrix
from scipy.special import binom

# Type hints for other core components
if TYPE_CHECKING:
    from ..components.circuit import Circuit
    from ..kernel.timeline import Timeline # Added import for Timeline
    from .quantum_state import State as QuantumBaseState, KetState as OriginalKetState, \
                               DensityState as OriginalDensityState, BellDiagonalState as OriginalBellDiagonalState


# Formalism constants (aligned with original file for consistency)
KET_STATE_FORMALISM = "ket_vector"
DENSITY_MATRIX_FORMALISM = "density_matrix"
FOCK_DENSITY_MATRIX_FORMALISM = "fock_density"
BELL_DIAGONAL_STATE_FORMALISM = "bell_diagonal"


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QuantumManager(ABC):
    """
    Abstract base class to track and manage quantum states.
    All states stored are of a single formalism.

    Attributes:
        timeline (Timeline): The simulation timeline (FIX: new attribute).
        states (dict[int, QuantumManager.State]): mapping of state keys to quantum state objects (Qobj wrappers).
        _least_available (int): The next available unique integer key for a new quantum state.
        formalism (str): The quantum state representation formalism.
        truncation (int): Max number of photons in Fock state representation (for Fock formalisms).
        dim (int): Subsystem Hilbert space dimension. (e.g., 2 for qubit, truncation + 1 for Fock).
    """

    def __init__(self, timeline: "Timeline", formalism: str, truncation: int = 1): # FIX: Added timeline
        self.timeline = timeline # FIX: Store timeline reference
        self.states: dict[int, QuantumManager.State] = {}
        self._least_available: int = 0
        self.formalism: str = formalism
        self.truncation = truncation
        self.dim = self.truncation + 1

        logger.info(f"Initialized abstract QuantumManager for formalism: {self.formalism}, truncation: {self.truncation}")

    @abstractmethod
    def new(self, state_data: Any) -> int:
        pass

    def get(self, key: int) -> "QuantumManager.State":
        state = self.states.get(key)
        if state is None:
            logger.error(f"Attempted to get non-existent quantum state for key: {key}")
            raise KeyError(f"Quantum state with key {key} does not exist.")
        logger.debug(f"Retrieved quantum state for key: {key}")
        return state

    @abstractmethod
    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp: float = None) -> dict[int, Any]:
        assert len(keys) == circuit.num_qubits, f"Mismatch between circuit size ({circuit.num_qubits}) and supplied qubits ({len(keys)})"
        if len(circuit.measured_qubits) > 0:
            assert meas_samp is not None, "Must specify random sample when measuring qubits."
        pass

    def _prepare_circuit(self, circuit: "Circuit", keys: list[int]) -> tuple[qutip.Qobj, list[int], qutip.Qobj]:
        old_states: list[qutip.Qobj] = []
        all_keys: list[int] = []

        for key in keys:
            qstate_wrapper = self.states[key]
            if qstate_wrapper.keys[0] not in all_keys:
                old_states.append(qstate_wrapper.state)
                all_keys.extend(qstate_wrapper.keys)

        if len(old_states) == 0:
            raise ValueError("No quantum states found for provided keys.")
        
        if len(old_states) == 1:
            compound_qobj = old_states[0]
        else:
            compound_qobj = qutip.tensor(old_states)

        circ_unitary = circuit.get_unitary_qobj()

        if circuit.num_qubits < len(all_keys):
            num_padding_qubits = len(all_keys) - circuit.num_qubits
            identity_qobj = qutip.identity([2] * num_padding_qubits)
            circ_unitary = qutip.tensor(circ_unitary, identity_qobj)

        logger.warning(f"[{self.timeline.now()}] `_swap_qubits` is a legacy method from NumPy-based formalism. Its exact usage with QuTiP Qobjs may differ.")
        return compound_qobj, all_keys, circ_unitary

    def _swap_qubits(self, all_keys: list[int], target_keys: list[int]) -> tuple[list[int], qutip.Qobj]:
        logger.warning(f"[{self.timeline.now()}] `_swap_qubits` is a legacy method from NumPy-based formalism. Its exact usage with QuTiP Qobjs may differ.")
        num_total_qubits = len(all_keys)
        # This function should ideally build a QuTiP swap unitary if truly needed.
        # For current structure, `Circuit.run` is expected to handle qubit mapping.
        return all_keys, qutip.identity([2]*num_total_qubits)

    @abstractmethod
    def set(self, keys: list[int], state_data: Any) -> None:
        pass

    def remove(self, key: int) -> None:
        qstate_wrapper = self.states.get(key)
        if qstate_wrapper:
            for k in qstate_wrapper.keys:
                if k in self.states:
                    del self.states[k]
            logger.debug(f"Removed quantum state associated with keys {qstate_wrapper.keys}.")
        else:
            logger.warning(f"Attempted to remove non-existent quantum state for key: {key}.")

    def set_states(self, states: dict):
        self.states = states
        if self.states:
            self._least_available = max(self.states.keys()) + 1
        else:
            self._least_available = 0
        logger.debug(f"QuantumManager states directly set. Next available key: {self._least_available}.")

    class State:
        def __init__(self, keys: List[int], state: qutip.Qobj, truncation: int = 1):
            self.keys = sorted(keys)
            self.state = state
            self.truncation = truncation

        @property
        def state_data(self):
            if self.state.isket:
                return self.state.full().flatten().tolist()
            elif self.state.isdm:
                return self.state.full().tolist()
            else:
                return self.state.data

        @property
        def dims(self):
            return self.state.dims

        def __str__(self):
            return f"QuantumManager.State(keys={self.keys}, state_dims={self.state.dims[0]})"
        
        def __repr__(self):
            return str(self)

        def __eq__(self, other):
            if not isinstance(other, QuantumManager.State):
                return NotImplemented
            return set(self.keys) == set(other.keys) and self.state == other.state


class QuantumManagerKet(QuantumManager):
    def __init__(self, timeline: "Timeline", truncation: int = 1): # FIX: Added timeline
        super().__init__(timeline, KET_STATE_FORMALISM, truncation) # FIX: Pass timeline to super

    def new(self, state_data: list[complex] = None) -> int:
        if state_data is None:
            state_data = [complex(1), complex(0)]
        
        qobj_state = qutip.Qobj(state_data, dims=[[self.dim], [1]])

        key = self._least_available
        self._least_available += 1
        
        self.states[key] = self.State([key], qobj_state, self.truncation)
        logger.debug(f"New ket state created for key {key}: {qobj_state}")
        return key

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp: float = None) -> dict[int, Any]:
        super().run_circuit(circuit, keys, meas_samp)

        main_qstate_wrapper = self.get(keys[0])
        current_qobj = main_qstate_wrapper.state

        if not current_qobj.isket:
            logger.error(f"[{self.timeline.now()}] run_circuit on KetManager received a non-ket state.")
            raise TypeError("QuantumManagerKet can only run circuits on ket states.")

        new_qobj, measurement_results = circuit.run(current_qobj, meas_samp, self.formalism, keys_order=keys)

        main_qstate_wrapper.state = new_qobj
        
        if measurement_results:
            self._measure_post_circuit(new_qobj, keys, main_qstate_wrapper.keys, measurement_results)

        logger.debug(f"Circuit applied to ket states {keys}. New state:\n{new_qobj}")
        return measurement_results

    def set(self, keys: list[int], state_data: list[complex]) -> None:
        num_subsystems = int(round(log(len(state_data), self.dim)))
        assert self.dim ** num_subsystems == len(state_data), \
            f"Length of amplitudes ({len(state_data)}) should be d**n (d={self.dim}, n={num_subsystems})."
        assert num_subsystems == len(keys), \
            f"Number of subsystems ({num_subsystems}) must match number of keys ({len(keys)})."

        qobj_state = qutip.Qobj(state_data)
        if not qobj_state.isket:
            logger.error(f"[{self.timeline.now()}] Attempted to set non-ket state in KetManager: {state_data}")
            raise ValueError("QuantumManagerKet can only set ket states.")

        quantum_state_object = self.State(keys, qobj_state, self.truncation)
        for key in keys:
            self.states[key] = quantum_state_object
        logger.debug(f"Set ket state for keys {keys}. Current state:\n{qobj_state}")

    def set_to_zero(self, key: int):
        self.set([key], [complex(1), complex(0)])

    def set_to_one(self, key: int):
        self.set([key], [complex(0), complex(1)])

    def _measure_post_circuit(self, collapsed_qobj: qutip.Qobj, measured_keys: list[int],
                              all_keys: list[int], measurement_results: dict[int, int]) -> None:
        unmeasured_keys = [k for k in all_keys if k not in measured_keys]

        for key, result in measurement_results.items():
            single_qubit_ket = qutip.basis(self.dim, result)
            self.states[key] = self.State([key], single_qubit_ket, self.truncation)
            logger.debug(f"Measured qubit {key} result {result}. Set to {single_qubit_ket}.")
        
        if len(unmeasured_keys) > 0 and collapsed_qobj.dims[0] == ([self.dim] * len(unmeasured_keys)):
            remaining_state_wrapper = self.State(unmeasured_keys, collapsed_qobj, self.truncation)
            for key in unmeasured_keys:
                self.states[key] = remaining_state_wrapper
            logger.debug(f"Remaining ket state assigned to keys {unmeasured_keys}.")
        else:
            pass


class QuantumManagerDensity(QuantumManager):
    def __init__(self, timeline: "Timeline", truncation: int = 1): # FIX: Added timeline
        super().__init__(timeline, DENSITY_MATRIX_FORMALISM, truncation) # FIX: Pass timeline to super

    def new(self, state_data: Union[list[list[complex]], list[complex]] = None) -> int:
        if state_data is None:
            state_data = [[complex(1), complex(0)], [complex(0), complex(0)]]
        
        qobj_state = qutip.Qobj(state_data)
        if not qobj_state.isdm:
            qobj_state = qobj_state * qobj_state.dag()

        key = self._least_available
        self._least_available += 1
        self.states[key] = self.State([key], qobj_state, self.truncation)
        logger.debug(f"New density state created for key {key}:\n{qobj_state}")
        return key

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp: float = None) -> dict[int, Any]:
        super().run_circuit(circuit, keys, meas_samp)

        main_qstate_wrapper = self.get(keys[0])
        current_qobj = main_qstate_wrapper.state

        if not current_qobj.isdm:
            logger.error(f"[{self.timeline.now()}] run_circuit on DensityManager received a non-density matrix state.")
            raise TypeError("QuantumManagerDensity can only run circuits on density matrix states.")

        new_qobj, measurement_results = circuit.run(current_qobj, meas_samp, self.formalism, keys_order=keys)

        main_qstate_wrapper.state = new_qobj

        if measurement_results:
            self._measure_post_circuit(new_qobj, keys, main_qstate_wrapper.keys, measurement_results)

        logger.debug(f"Circuit applied to density states {keys}. New state:\n{new_qobj}")
        return measurement_results

    def set(self, keys: list[int], state_data: Union[list[list[complex]], list[complex]]) -> None:
        if isinstance(state_data[0], list):
            matrix_dim = len(state_data)
            num_subsystems = int(round(log(matrix_dim, self.dim)))
        else:
            vector_len = len(state_data)
            num_subsystems = int(round(log(vector_len, self.dim)))

        assert self.dim ** num_subsystems == (len(state_data) if not isinstance(state_data[0], list) else len(state_data) * len(state_data[0])), \
            f"Length/dimensions of state ({len(state_data)}) should be d**n (d={self.dim}, n={num_subsystems})."
        assert num_subsystems == len(keys), \
            f"Number of subsystems ({num_subsystems}) must match number of keys ({len(keys)})."

        qobj_state = qutip.Qobj(state_data)
        if not qobj_state.isdm:
            qobj_state = qobj_state * qobj_state.dag()

        quantum_state_object = self.State(keys, qobj_state, self.truncation)
        for key in keys:
            self.states[key] = quantum_state_object
        logger.debug(f"Set density state for keys {keys}. Current state:\n{qobj_state}")


    def set_to_zero(self, key: int):
        self.set([key], [[complex(1), complex(0)], [complex(0), complex(0)]])

    def set_to_one(self, key: int):
        self.set([key], [[complex(0), complex(0)], [complex(0), complex(1)]])

    def _measure_post_circuit(self, collapsed_qobj: qutip.Qobj, measured_keys: list[int],
                              all_keys: list[int], measurement_results: dict[int, int]) -> None:
        unmeasured_keys = [k for k in all_keys if k not in measured_keys]

        for key, result in measurement_results.items():
            single_qubit_dm = qutip.basis(self.dim, result) * qutip.basis(self.dim, result).dag()
            self.states[key] = self.State([key], single_qubit_dm, self.truncation)
            logger.debug(f"Measured qubit {key} result {result}. Set to {single_qubit_dm}.")
        
        if len(unmeasured_keys) > 0 and collapsed_qobj.isdm:
            if collapsed_qobj.dims[0] == ([self.dim] * len(unmeasured_keys)):
                remaining_state_wrapper = self.State(unmeasured_keys, collapsed_qobj, self.truncation)
                for key in unmeasured_keys:
                    self.states[key] = remaining_state_wrapper
                logger.debug(f"Remaining density state assigned to keys {unmeasured_keys}.")
            else:
                pass


class QuantumManagerDensityFock(QuantumManager):
    def __init__(self, timeline: "Timeline", truncation: int = 1): # FIX: Added timeline
        super().__init__(timeline, FOCK_DENSITY_MATRIX_FORMALISM, truncation=truncation) # FIX: Pass timeline to super

    def new(self, state_data: Union[str, list[complex], list[list[complex]]] = "gnd") -> int:
        key = self._least_available
        self._least_available += 1

        qobj_state = None
        if state_data == "gnd":
            gnd_ket = qutip.basis(self.dim, 0)
            qobj_state = gnd_ket * gnd_ket.dag()
        elif isinstance(state_data, list):
            qobj_state = qutip.Qobj(state_data)
            if not qobj_state.isdm:
                qobj_state = qobj_state * qobj_state.dag()
        else:
            logger.error(f"Unsupported Fock state data for new(): {state_data}")
            raise ValueError("Unsupported Fock state data.")

        self.states[key] = self.State([key], qobj_state, self.truncation)
        logger.debug(f"New Fock density state created for key {key}:\n{qobj_state}")
        return key

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp: float = None) -> dict[int, Any]:
        logger.error(f"[{self.timeline.now()}] run_circuit method of class QuantumManagerDensityFock called, which is not supported.")
        raise NotImplementedError("run_circuit method of class QuantumManagerDensityFock is not supported.")

    def _generate_swap_operator(self, num_systems: int, i: int, j: int) -> qutip.Qobj:
        size_total = self.dim ** num_systems
        swap_matrix = np.zeros((size_total, size_total), dtype=complex)

        for old_index in range(size_total):
            old_str = np.base_repr(old_index, base=self.dim).zfill(num_systems)
            
            new_str_list = list(old_str)
            new_str_list[i], new_str_list[j] = new_str_list[j], new_str_list[i]
            new_str = "".join(new_str_list)
            
            new_index = int(new_str, base=self.dim)
            swap_matrix[new_index, old_index] = 1

        return qutip.Qobj(swap_matrix, dims=[[self.dim] * num_systems, [self.dim] * num_systems])

    def _prepare_state(self, keys: list[int]) -> tuple[qutip.Qobj, list[int]]:
        old_qobjs: list[qutip.Qobj] = []
        all_keys: list[int] = []

        for key in keys:
            qstate_wrapper = self.states[key]
            if qstate_wrapper.keys[0] not in all_keys:
                old_qobjs.append(qstate_wrapper.state)
                all_keys.extend(qstate_wrapper.keys)

        if len(old_qobjs) == 0:
            raise ValueError("No quantum states found for provided keys.")
        
        if len(old_qobjs) == 1:
            compound_qobj = old_qobjs[0]
        else:
            compound_qobj = qutip.tensor(old_qobjs)
        
        if len(keys) > 1:
            current_permutation = list(range(len(all_keys)))
            ordered_all_keys = list(keys) + [k for k in all_keys if k not in keys]
            desired_idx_map = {k: i for i, k in enumerate(ordered_all_keys)}
            
            for i, target_key in enumerate(keys):
                current_pos_of_target_key = current_permutation.index(desired_idx_map[target_key])
                
                if current_pos_of_target_key != i:
                    idx_to_swap_with = current_permutation[i]
                    current_permutation[i], current_permutation[current_pos_of_target_key] = \
                        current_permutation[current_pos_of_target_key], current_permutation[i]
                    
                    swap_op_qobj = self._generate_swap_operator(len(all_keys), i, current_pos_of_target_key)
                    compound_qobj = swap_op_qobj * compound_qobj * swap_op_qobj.dag()

            all_keys = ordered_all_keys

        return compound_qobj, all_keys

    def _prepare_operator(self, all_keys: list[int], keys: list[int], operator_qobj: qutip.Qobj) -> qutip.Qobj:
        num_target_qubits = len(keys)
        num_total_qubits = len(all_keys)

        identity_qobj = qutip.identity([self.dim] * (num_total_qubits - num_target_qubits))
        prepared_operator = qutip.tensor(operator_qobj, identity_qobj)

        return prepared_operator

    def apply_operator(self, operator_data: Union[np.ndarray, qutip.Qobj], keys: list[int]):
        operator_qobj = qutip.Qobj(operator_data)

        prepared_state_qobj, all_keys = self._prepare_state(keys)
        prepared_operator_qobj = self._prepare_operator(all_keys, keys, operator_qobj)

        new_state_qobj = prepared_operator_qobj * prepared_state_qobj * prepared_operator_qobj.dag()
        self.set(all_keys, new_state_qobj)
        logger.debug(f"Applied operator to Fock state keys {keys}. New state:\n{new_state_qobj}")

    def set(self, keys: list[int], state_data: Union[list[list[complex]], list[complex]]) -> None:
        if isinstance(state_data[0], list):
            matrix_dim = len(state_data)
            num_subsystems = int(round(log(matrix_dim, self.dim)))
        else:
            vector_len = len(state_data)
            num_subsystems = int(round(log(vector_len, self.dim)))

        assert self.dim ** num_subsystems == (len(state_data) if not isinstance(state_data[0], list) else len(state_data) * len(state_data[0])), \
            f"Length/dimensions of state ({len(state_data)}) should be d**n (d={self.dim}, n={num_subsystems})."
        assert num_subsystems == len(keys), \
            f"Number of subsystems ({num_subsystems}) must match number of keys ({len(keys)})."

        qobj_state = qutip.Qobj(state_data)
        if not qobj_state.isdm:
            qobj_state = qobj_state * qobj_state.dag()

        quantum_state_object = self.State(keys, qobj_state, self.truncation)
        for key in keys:
            self.states[key] = quantum_state_object
        logger.debug(f"Set Fock density state for keys {keys}. Current state:\n{qobj_state}")

    def set_to_zero(self, key: int):
        gnd_ket = qutip.basis(self.dim, 0)
        self.set([key], (gnd_ket * gnd_ket.dag()).full().tolist())

    def build_ladder(self) -> tuple[qutip.Qobj, qutip.Qobj]:
        destroy_op = qutip.destroy(self.truncation + 1)
        create_op = qutip.create(self.truncation + 1)
        return create_op, destroy_op

    def measure(self, keys: list[int], povms: list[Union[np.ndarray, qutip.Qobj]], meas_samp: float) -> int:
        if not keys:
            raise ValueError("Keys list cannot be empty for measurement.")

        main_state_wrapper = self.get(keys[0])
        current_qobj = main_state_wrapper.state

        if not current_qobj.isdm:
            logger.error(f"[{self.timeline.now()}] measure on FockDensityManager received a non-density matrix state.")
            raise TypeError("QuantumManagerDensityFock can only measure density matrix states.")

        qobj_povms = [qutip.Qobj(p) for p in povms]

        probabilities = [np.trace(p * current_qobj).real for p in qobj_povms]
        probabilities = np.array(probabilities)

        prob_sum = np.sum(probabilities)
        if not math.isclose(prob_sum, 1.0, rel_tol=1e-9, abs_tol=1e-12) or np.any(probabilities < 0):
            logger.warning(f"POVM probabilities do not sum to 1 or contain negative values: {probabilities}. Normalizing.")
            probabilities[probabilities < 0] = 0
            probabilities = probabilities / np.sum(probabilities)

        outcome_idx = np.searchsorted(np.cumsum(probabilities), meas_samp)

        projected_state = (qobj_povms[outcome_idx] * current_qobj * qobj_povms[outcome_idx].dag()).unit()
        
        all_keys = main_state_wrapper.keys
        
        for key in keys:
            if key in self.states:
                del self.states[key]
            logger.debug(f"Cleared state for measured key {key}.")

        remaining_keys = [k for k in all_keys if k not in keys]
        if remaining_keys:
            original_qobj_dims = current_qobj.dims[0]
            original_num_qubits = len(original_qobj_dims)
            
            measured_indices_in_qobj = [main_state_wrapper.keys.index(k) for k in keys]
            
            indices_to_keep = [i for i in range(original_num_qubits) if i not in measured_indices_in_qobj]

            if indices_to_keep:
                remaining_qobj = projected_state.ptrace(indices_to_keep)
                remaining_state_wrapper = self.State(remaining_keys, remaining_qobj, self.truncation)
                for key in remaining_keys:
                    self.states[key] = remaining_state_wrapper
                logger.debug(f"Remaining Fock density state assigned to keys {remaining_keys}.")
            else:
                logger.debug(f"No remaining keys after measurement for Fock state.")
        else:
            logger.debug(f"All Fock state keys were measured.")

        logger.debug(f"Measurement on Fock keys {keys} yielded outcome {outcome_idx}.")
        return outcome_idx

    def _build_loss_kraus_operators(self, loss_rate: float, all_keys: list[int], key: int) -> list[qutip.Qobj]:
        assert 0 <= loss_rate <= 1
        kraus_ops = []

        subsystem_index = all_keys.index(key)
        num_total_systems = len(all_keys)
        
        for k in range(self.dim):
            total_kraus_op_matrix = np.zeros((self.dim ** num_total_systems, self.dim ** num_total_systems), dtype=complex)

            for n in range(k, self.dim):
                coeff = np.sqrt(math.comb(n, k)) * np.sqrt(((1 - loss_rate) ** (n - k)) * (loss_rate ** k))
                
                single_op_matrix = np.zeros((self.dim, self.dim), dtype=complex)
                single_op_matrix[n - k, n] = 1

                ops_list = [qutip.identity(self.dim)] * num_total_systems
                ops_list[subsystem_index] = qutip.Qobj(single_op_matrix, dims=[[self.dim],[self.dim]])
                
                total_op_qobj = qutip.tensor(ops_list)
                total_kraus_op_matrix += coeff * total_op_qobj.full()

            if np.any(total_kraus_op_matrix != 0):
                kraus_ops.append(qutip.Qobj(total_kraus_op_matrix, 
                                            dims=[[self.dim] * num_total_systems, [self.dim] * num_total_systems]))

        return kraus_ops

    def add_loss(self, key: int, loss_rate: float):
        qstate_wrapper = self.get(key)
        original_qobj = qstate_wrapper.state
        all_keys = qstate_wrapper.keys

        if not original_qobj.isdm:
            logger.error(f"[{self.timeline.now()}] add_loss on FockDensityManager requires density matrix state for key {key}.")
            raise TypeError("add_loss requires a density matrix state.")

        kraus_ops = self._build_loss_kraus_operators(loss_rate, all_keys, key)
        
        output_qobj = qutip.Qobj(np.zeros(original_qobj.shape), dims=original_qobj.dims)
        
        for kraus_op in kraus_ops:
            if kraus_op.dims != original_qobj.dims:
                logger.error(f"Kraus operator dims {kraus_op.dims} do not match state dims {original_qobj.dims}.")
                continue
            output_qobj += kraus_op * original_qobj * kraus_op.dag()

        qstate_wrapper.state = output_qobj
        logger.debug(f"Applied loss (rate {loss_rate:.2f}) to Fock state key {key}. New state:\n{output_qobj}")


class QuantumManagerBellDiagonal(QuantumManager):
    def __init__(self, timeline: "Timeline"): # FIX: Added timeline
        super().__init__(timeline, BELL_DIAGONAL_STATE_FORMALISM) # FIX: Pass timeline to super

    def new(self, state_data: Any = None) -> int:
        key = self._least_available
        self._least_available += 1
        logger.debug(f"Generated new Bell diagonal key: {key}. State not yet set.")
        return key

    def get(self, key: int) -> "QuantumManager.State":
        if key not in self.states or not isinstance(self.states[key], self.State):
            logger.error(f"Attempt to get Bell diagonal state for key {key} before it was entangled/set.")
            raise KeyError(f"Bell diagonal state with key {key} does not exist or is not a BellDiagonalState.")
        return super().get(key)

    def set(self, keys: list[int], diag_elems: list[float]) -> None:
        if len(keys) != 2:
            logger.warning(f"[{self.timeline.now()}] Bell diagonal quantum manager received invalid set request with {len(keys)} keys (expected 2). Clearing keys.")
            for key in keys:
                if key in self.states:
                    del self.states[key]
            return

        class BellDiagonalInnerState(self.State):
            def __init__(self, keys: List[int], diag_elems: List[float], truncation: int = 1):
                super().__init__(keys, qutip.Qobj(np.diag(diag_elems), dims=[[2,2],[2,2]]), truncation)
                self._diag_elems = diag_elems

            @property
            def diag_elems(self):
                return self._diag_elems

            def __str__(self):
                return f"BellDiagonalState(keys={self.keys}, diag_elems={self.diag_elems})"

        new_state_wrapper = BellDiagonalInnerState(keys, diag_elems, self.truncation)
        for key in keys:
            self.states[key] = new_state_wrapper
        logger.debug(f"Set Bell diagonal state for keys {keys}: {diag_elems}")

    def set_to_noiseless(self, keys: list[int]):
        self.set(keys, [float(1), float(0), float(0), float(0)])

    def run_circuit(self, circuit: "Circuit", keys: list[int], meas_samp: float = None) -> dict[int, Any]:
        logger.error(f"[{self.timeline.now()}] run_circuit method of class QuantumManagerBellDiagonal called, which is not supported.")
        raise NotImplementedError("run_circuit method of class QuantumManagerBellDiagonal is not supported.")

    def measure(self, keys: list[int], povms: list[Union[np.ndarray, qutip.Qobj]], meas_samp: float) -> int:
        logger.error(f"[{self.timeline.now()}] measure method of class QuantumManagerBellDiagonal called, which is not supported.")
        raise NotImplementedError("measure method of class QuantumManagerBellDiagonal is not supported directly. Outcomes are typically derived from analytical state evolution.")

