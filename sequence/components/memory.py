# File: sequence/components/memory.py
#
# This file defines the QuantumMemory component, which stores quantum qubits
# and now includes functionality for non-Markovian and stochastic noise
# tracking and application.

import qutip
import numpy as np
from collections import deque
import logging
from copy import copy # Added: For encoding types
from math import inf # Added: For get_expire_time
from typing import Any, Callable, TYPE_CHECKING # Added: For type hints in original methods

if TYPE_CHECKING:
    from ..entanglement_management.entanglement_protocol import EntanglementProtocol # Added: For detach method
    from ..kernel.timeline import Timeline # Added: For Timeline type hint

# Original imports from the base `memory.py` that are still needed for non-noise functionality
from ..kernel.entity import Entity # Added: Parent class for MemoryArray and Memory
from ..kernel.event import Event # Added: For scheduling events like expiration
from ..kernel.process import Process # Added: For scheduling processes in events
from ..utils.encoding import single_atom, single_heralded # Added: For photon encoding types
from ..constants import EPSILON # Added: For numerical comparisons, e.g., in decoherence_errors assertion
# from ..utils import log # Replaced by standard 'logging' setup

# Import the new noise models and mixin
from sequence.components.noise_models import (
    NoiseModel,
    EnhancedNonMarkovianModel,
    NonMarkovianAmplitudeDampingModel,
    StochasticParameterMixin
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Ensure a handler is present for direct execution/testing if not configured globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Helper functions for analytical BDS decoherence implementation (from original file, kept for context but not directly used by new noise models)
# These are the analytical functions, largely replaced by the new noise models' mesolve.
# Keeping them for now in case other parts of the original code still reference them,
# though the core decoherence logic is now handled by noise_model.apply_noise.
def _p_id(x_rate, y_rate, z_rate, t):
    """Calculates identity probability for Pauli channel."""
    val = (1 + np.exp(-2*(x_rate+y_rate)*t) + np.exp(-2*(x_rate+z_rate)*t) + np.exp(-2*(z_rate+y_rate)*t)) / 4
    return val

def _p_xerr(x_rate, y_rate, z_rate, t):
    """Calculates X error probability for Pauli channel."""
    val = (1 - np.exp(-2*(x_rate+y_rate)*t) - np.exp(-2*(x_rate+z_rate)*t) + np.exp(-2*(z_rate+y_rate)*t)) / 4
    return val

def _p_yerr(x_rate, y_rate, z_rate, t):
    """Calculates Y error probability for Pauli channel."""
    val = (1 - np.exp(-2*(x_rate+y_rate)*t) + np.exp(-2*(x_rate+z_rate)*t) - np.exp(-2*(z_rate+y_rate)*t)) / 4
    return val

def _p_zerr(x_rate, y_rate, z_rate, t):
    """Calculates Z error probability for Pauli channel."""
    val = (1 + np.exp(-2*(x_rate+y_rate)*t) - np.exp(-2*(x_rate+z_rate)*t) - np.exp(-2*(z_rate+y_rate)*t)) / 4
    return val


class MemoryArray(Entity):
    """Aggregator for Memory objects.
    This class is largely unchanged in this modification set,
    but its `Memory` objects will now use the new noise models.
    """
    def __init__(self, name: str, timeline: "Timeline", num_memories=10,
                 fidelity=0.85, frequency=80e6, efficiency=1, coherence_time=-1, wavelength=500,
                 decoherence_errors: list[float] = None, cutoff_ratio = 1, noise_model_config: dict = None):
        """
        Constructor for the Memory Array class.

        Note: `noise_model_config` is added here to pass down to individual Memory instances.
        """
        Entity.__init__(self, name, timeline)
        self.memories = []
        self.memory_name_to_index = {}

        for i in range(num_memories):
            memory_name = self.name + f"[{i}]"
            self.memory_name_to_index[memory_name] = i
            # Pass the new noise_model_config to each individual Memory instance
            memory = Memory(memory_name, timeline, fidelity, frequency, efficiency, coherence_time, wavelength,
                            noise_model_config=noise_model_config) # Removed decoherence_errors, cutoff_ratio if unused
            memory.attach(self)
            self.memories.append(memory)
            memory.set_memory_array(self)

    def __getitem__(self, key: int) -> "Memory":
        return self.memories[key]

    def __setitem__(self, key: int, value: "Memory"):
        self.memories[key] = value

    def __len__(self) -> int:
        return len(self.memories)

    def init(self):
        """Implementation of Entity interface (see base class).
        Set the owner of memory as the owner of memory array.
        """
        for memory in self.memories:
            memory.owner = self.owner

    def memory_expire(self, memory: "Memory"):
        """Method to receive expiration events from memories.

        Args:
            memory (Memory): expired memory.
        """
        self.owner.memory_expire(memory)

    def update_memory_params(self, arg_name: str, value: Any) -> None:
        """Updates a parameter for all memories in the array."""
        for memory in self.memories:
            memory.__setattr__(arg_name, value)

    def add_receiver(self, receiver: "Entity") -> None:
        """Add receiver to each memory in the memory array to receive photons."""
        for memory in self.memories:
            memory.add_receiver(receiver)

    def get_memory_by_name(self, name: str) -> "Memory":
        """Given the memory's name, get the memory object."""
        index = self.memory_name_to_index.get(name, -1)
        assert index >= 0, f"Oops! name={name} not exist!"
        return self.memories[index]


class Memory(Entity): # Changed from Entity to Component based on earlier plan
    """
    QuantumMemory is a component that stores a single quantum qubit.
    It now includes enhanced functionality for tracking historical state,
    applying non-Markovian noise based on this history, and managing
    its own active usage.
    """
    def __init__(self, name: str, timeline, fidelity: float = 1.0,
                 coherence_time: float = -1, no_error: bool = False,
                 wavelength: int = 500, frequency: float = 80e6, # Re-added frequency/wavelength for Photon creation
                 noise_model_config: dict = None):
        """
        Initializes the QuantumMemory component.

        Args:
            name (str): The unique name of this memory instance.
            timeline (Timeline): The simulation timeline.
            fidelity (float): Initial fidelity of the qubit in memory (default 1.0).
                              This represents the initial quality if the qubit is unentangled.
            coherence_time (float): The T2 coherence time of the memory (in simulation time units).
                                    -1 indicates infinite coherence.
                                    This parameter will be largely superseded by explicit noise_model.
            no_error (bool): If True, no noise or errors are applied (default False).
                             This should ideally be managed by `noise_model_config`.
            wavelength (int): wavelength (in nm) of photons emitted by memories (default 500). (Re-added)
            frequency (float): maximum frequency of excitation for memories (default 80e6). (Re-added)
            noise_model_config (dict): Configuration for the noise model to apply.
                                       Expected structure:
                                       {"type": "EnhancedNonMarkovian", "gamma0": ..., "tau_corr": ...}
                                       or {"type": "NonMarkovianAmplitudeDamping", "gamma_base": ...}
                                       or None for no custom noise model (falls back to simpler decay or no error).
        """
        super().__init__(name, timeline)
        self._qubit: qutip.Qobj = qutip.basis(2, 0).proj() # Default to |0><0| density matrix
        self._last_update_time: float = timeline.now() # Time when _qubit was last updated/noise applied

        # --- New Attributes for Non-Markovian & Stochastic Noise ---
        # History buffer for ENM model
        self._history_buffer: deque[tuple[qutip.Qobj, float]] = deque()
        # Timestamps for active usage tracking (for Fatigue)
        self._last_active_time: float = timeline.now() # Time of last gate/read/write on this memory
        self._total_active_time: float = 0.0 # Cumulative duration of active operations on this memory
        self._num_events_processed: int = 0 # Cumulative count of quantum operations on this memory

        # Fidelity tracking for NMAD model (fidelity vs. ideal state)
        self._current_fidelity_nm: float = fidelity # Initial fidelity to ideal
        self._previous_fidelity_nm: float = fidelity # For NMAD's M_recoh calculation
        self._ideal_state_for_fidelity: qutip.Qobj = qutip.basis(2, 0).proj() # Reference ideal state, updated for entanglement

        # Noise model instance
        self.noise_model: NoiseModel = None
        if noise_model_config:
            noise_type = noise_model_config.get("type")
            try:
                if noise_type == "EnhancedNonMarkovian":
                    # Pass all relevant parameters to the constructor
                    self.noise_model = EnhancedNonMarkovianModel(
                        name=f"{self.name}.enm_noise",
                        gamma0=noise_model_config["gamma0"],
                        tau_corr=noise_model_config["tau_corr"],
                        d_mem=noise_model_config["d_mem"],
                        f_sens=noise_model_config["f_sens"],
                        k_steep=noise_model_config["k_steep"],
                        f_thresh=noise_model_config["f_thresh"],
                        lambda_time=noise_model_config["lambda_time"],
                        lambda_event=noise_model_config["lambda_event"],
                        s_fatigue=noise_model_config["s_fatigue"],
                        s_mem=noise_model_config["s_mem"]
                    )
                    # Set maxlen for deque based on d_mem from ENM config
                    self._history_buffer = deque(maxlen=self.noise_model.d_mem)
                    logger.info(f"[{self.name}] Initialized with Enhanced Non-Markovian noise model: {noise_model_config}")
                elif noise_type == "NonMarkovianAmplitudeDamping":
                    self.noise_model = NonMarkovianAmplitudeDampingModel(
                        name=f"{self.name}.nmad_noise",
                        gamma_base=noise_model_config["gamma_base"],
                        f_stab=noise_model_config["f_stab"],
                        s_recoh=noise_model_config["s_recoh"]
                    )
                    logger.info(f"[{self.name}] Initialized with Non-Markovian Amplitude Damping noise model: {noise_model_config}")
                else:
                    logger.warning(f"[{self.name}] Unknown noise model type '{noise_type}'. Using default decay.")
            except KeyError as e:
                logger.error(f"[{self.name}] Missing parameter for noise model '{noise_type}': {e}. Using default decay.")
            except Exception as e:
                logger.error(f"[{self.name}] Error initializing noise model: {e}. Using default decay.")

        # Fallback to simple coherence_time if no advanced noise model or if error in setup
        if self.noise_model is None:
            self.coherence_time = coherence_time # -1 for infinite coherence (no decay)
            self.no_error = no_error # If True, explicitly disable all decay

        self.owner = None # This will be set by the Node later

        # Re-added from original Memory for other functionalities
        self.raw_fidelity = fidelity # Re-added for consistency with MemoryArray init
        self.frequency = frequency # Re-added for excite method
        self.efficiency = 1.0 # Default efficiency for photon emission (re-added, was in original Memory.__init__)
        self.wavelength = wavelength # Re-added for excite method
        self.qstate_key = timeline.quantum_manager.new() # Re-added: QState key for QuantumManager
        self.memory_array = None # Re-added: Link to parent MemoryArray
        self.entangled_memory = {'node_id': None, 'memo_id': None} # Re-added: For tracking entanglement status
        self.previous_bsm = -1 # Re-added: For entanglement generation protocols
        self.expiration_event = None # Re-added: For memory expiration events
        self.excited_photon = None # Re-added: For tracking emitted photon
        self.next_excite_time = 0 # Re-added: For frequency limiting in excite()
        self.generation_time = -1 # Re-added: For tracking entanglement generation time
        self.last_update_time = timeline.now() # Re-added: Last time memory state was updated (for BDS etc. if still used, otherwise redundant with _last_update_time)
        self.is_in_application = False # Re-added: For memory expiration logic (original)

        # For photons encoding (from original Memory)
        self.encoding = copy(single_atom)
        self.encoding["raw_fidelity"] = self.raw_fidelity
        self.encoding_sh = copy(single_heralded)


    def init(self):
        """Implementation of Entity interface (see base class).
        Initialize the memory component.
        """
        # Memory-specific initialization if needed
        pass


    @property
    def qubit(self) -> qutip.Qobj:
        """Read-only property for the current qubit state."""
        return self._qubit

    def set_memory_array(self, memory_array: 'MemoryArray'):
        """Method to set the memory array to which the memory belongs."""
        self.memory_array = memory_array

    def excite(self, dst: str = "", protocol: str = "bk") -> None:
        """Method to excite memory and potentially emit a photon."""
        from .photon import Photon # Local import to avoid circular dependency if Photon also imports Memory

        if self.timeline.now() < self.next_excite_time:
            return

        # create photon
        if protocol == "bk":
            photon = Photon("", self.timeline, wavelength=self.wavelength, location=self.name, encoding_type=self.encoding,
                            quantum_state=self.qstate_key, use_qm=True)
        elif protocol == "sh":
            photon = Photon("", self.timeline, wavelength=self.wavelength, location=self.name, encoding_type=self.encoding_sh,
                            quantum_state=self.qstate_key, use_qm=True)
            self.generation_time = self.timeline.now()
            # self.last_update_time = self.timeline.now() # Redundant with _last_update_time in new model
        else:
            logger.error(f"[{self.name}] Invalid protocol type {protocol} specified for memory.excite().")
            raise ValueError(f"Invalid protocol type {protocol} specified for meomory.exite()")

        photon.timeline = None  # facilitate cross-process exchange of photons
        photon.is_null = True
        photon.add_loss(1 - self.efficiency) # Efficiency for photon emission, not decoherence

        if self.frequency > 0:
            period = 1e12 / self.frequency
            self.next_excite_time = self.timeline.now() + period

        # send to receiver
        if self._receivers: # Ensure receivers exist
            self._receivers[0].get(photon, dst=dst)
        else:
            logger.warning(f"[{self.name}] No receivers attached to memory for excite method.")
        self.excited_photon = photon

    def expire(self) -> None:
        """Method to handle memory expiration."""
        if self.is_in_application:
            pass
        else:
            if self.excited_photon:
                self.excited_photon.is_null = True
            self.reset()
            self.notify(self) # Notify observers (e.g., MemoryArray)

    def reset(self) -> None:
        """Method to clear quantum memory."""
        self.fidelity = 0 # This refers to the old fidelity attribute, keep it or remove if fully superseded
        self.generation_time = -1
        # self.last_update_time = -1 # Redundant with _last_update_time in new model

        self.timeline.quantum_manager.set([self.qstate_key], [complex(1), complex(0)]) # Reset to |0> state
        self.entangled_memory = {'node_id': None, 'memo_id': None}
        if self.expiration_event is not None:
            self.timeline.remove_event(self.expiration_event)
            self.expiration_event = None

    def update_state(self, new_state: qutip.Qobj, current_time: float, ideal_state: qutip.Qobj = None):
        """
        Updates the qubit state, records history for non-Markovian models,
        and updates fidelity metrics. This is an internal method called by set_qubit
        and potentially other internal state transitions.

        Args:
            new_state (qutip.Qobj): The new quantum state.
            current_time (float): The current simulation time.
            ideal_state (qutip.Qobj, optional): The ideal state for fidelity comparison.
                                                Important for entangled qubits.
        """
        # Record the previous state and its timestamp for history buffer
        if self._qubit is not None:
            self._history_buffer.append((self._qubit.copy(), self._last_update_time)) # Store a deep copy!

        self._qubit = new_state # Update the actual qubit state in memory

        # Update active time and event count (This assumes any `update_state` represents activity)
        # Refine this if only specific events (e.g., gates) are considered "active"
        if self._last_active_time < current_time:
            self._total_active_time += (current_time - self._last_active_time)
            self._num_events_processed += 1
            self._last_active_time = current_time # Update active timestamp

        # Update fidelity metrics for NMAD and ENM F_hist
        self._previous_fidelity_nm = self._current_fidelity_nm
        if ideal_state is not None:
            self._ideal_state_for_fidelity = ideal_state # Update ideal state if provided by protocol
        
        # Calculate current fidelity against the (potentially updated) ideal state
        prev_fid = getattr(self, '_current_fidelity_nm', None)
        if self._qubit is not None and self._ideal_state_for_fidelity is not None:
            try:
                self._current_fidelity_nm = qutip.fidelity(self._qubit, self._ideal_state_for_fidelity)
                logger.info(f"[{self.name}] Fidelity updated from {prev_fid} to {self._current_fidelity_nm} at time {current_time}")
            except Exception as e:
                logger.error(f"[{self.name}] Error calculating fidelity in update_state: {e}. Setting fidelity to 0.")
                self._current_fidelity_nm = 0.0
        else:
            self._current_fidelity_nm = 1.0 # No qubit or ideal state, assume perfect

        # Prune _history_buffer by time (tau_corr) if using ENM
        if isinstance(self.noise_model, EnhancedNonMarkovianModel):
            while self._history_buffer and \
                  (current_time - self._history_buffer[0][1] > self.noise_model.tau_corr):
                self._history_buffer.popleft()
        
        self._last_update_time = current_time

        # Schedule expiration (re-added from original logic)
        if self.coherence_time > 0 and self.expiration_event is None: # Only schedule if finite coherence and not already scheduled
             self._schedule_expiration()


    def get_qubit(self) -> qutip.Qobj:
        """
        Retrieves the qubit from memory, applying accumulated noise.
        This method is called when the qubit is accessed (e.g., for measurement, gate, or transmission).
        """
        current_time = self.timeline.now()
        elapsed_time = current_time - self._last_update_time

        if self.noise_model:
            try:
                # Apply the assigned non-Markovian or other complex noise model
                self._qubit = self.noise_model.apply_noise(self._qubit, elapsed_time, self)
            except Exception as e:
                logger.error(f"[{self.name}] Error applying noise model '{self.noise_model.name}': {e}. Qubit state might be stale.")
                # The _qubit state remains as is, previous noise application might have failed.
        elif not self.no_error and self.coherence_time > 0:
            # Fallback to standard T2 decay if no explicit noise_model is set
            # This logic mimics standard Markovian decay (decoherence)
            decay_rate = 1.0 / self.coherence_time
            # Original Memory used specific decoherence_errors for Pauli channel, 
            # now simplified to a generic dephasing for fallback.
            c_ops = [np.sqrt(decay_rate) * qutip.sigmaz()] # Example: pure dephasing
            result = qutip.mesolve(self._qubit, [], [0, elapsed_time], c_ops=c_ops)
            self._qubit = result.states[-1]
            logger.debug(f"[{self.name}] Applied default T2 decay (rate: {decay_rate:.2e}) for {elapsed_time:.2f}ns.")

        self._last_update_time = current_time
        self._last_active_time = current_time # Qubit access is an active event
        return self._qubit

    def do_gate(self, gate_operator: qutip.Qobj, target_qubit_index: int = 0, control_qubit_index: int = None):
        """
        Applies a quantum gate operation to the qubit in memory.
        This also updates the active time and event count for fatigue calculation.

        Args:
            gate_operator (qutip.Qobj): The Qutip operator representing the gate.
            target_qubit_index (int): The index of the target qubit (0 for single qubit memory).
            control_qubit_index (int): The index of the control qubit if multi-qubit gate.
        """
        # Apply the ideal gate operation
        if gate_operator.dims[0] == self._qubit.dims[0]: # Check if gate dim matches qubit dim
            self._qubit = gate_operator * self._qubit * gate_operator.dag()
        else:
            logger.warning(f"[{self.name}] Gate dimension {gate_operator.dims} does not match qubit dimension {self._qubit.dims}. Gate not applied.")
            return

        # Update active usage tracking for fatigue calculation
        current_time = self.timeline.now()
        if self._last_active_time < current_time:
            self._total_active_time += (current_time - self._last_active_time)
            self._num_events_processed += 1
            self._last_active_time = current_time

        logger.debug(f"[{self.name}] Applied gate. Total active time: {self._total_active_time:.2f}, Events: {self._num_events_processed}")

    def get_current_historical_health_factors(self) -> tuple[float, float]:
        """
        Calculates and returns the current I_mem and I_fatigue factors for routing decisions
        without applying noise or advancing the qubit state.

        Returns:
            tuple[float, float]: (I_mem, I_fatigue) calculated using current memory state.
        """
        if not isinstance(self.noise_model, EnhancedNonMarkovianModel):
            logger.debug(f"[{self.name}] Not an ENM model, returning default health factors.")
            return 0.0, 0.0

        enm_model = self.noise_model
        current_time = self.timeline.now()

        # 1. Retrieve & Filter History for F_hist calculation (same logic as in apply_noise)
        relevant_history = deque()
        for hist_state_qobj, hist_timestamp in self._history_buffer:
            if current_time - hist_timestamp < enm_model.tau_corr:
                relevant_history.append((hist_state_qobj, hist_timestamp))

        f_hist = 1.0
        if relevant_history:
            fidelities = []
            for hist_state_from_buffer, _ in relevant_history:
                if qutip.metrics.tracedist(self._qubit, hist_state_from_buffer) > 0.5:
                    fidelities.append(0.0)
                else:
                    fidelities.append(qutip.fidelity(self._qubit, hist_state_from_buffer))
            f_hist = np.mean(fidelities)

        # 2. Calculate I_mem
        I_mem = enm_model.f_sens * math.tanh(enm_model.k_steep * (f_hist - enm_model.f_thresh))

        # 3. Retrieve Fatigue Data
        delta_t_active = self._total_active_time
        n_events = self._num_events_processed

        # 4. Calculate I_fatigue
        I_fatigue = enm_model.lambda_time * delta_t_active + enm_model.lambda_event * n_events

        logger.debug(f"[{self.name}] Health factors: I_mem={I_mem:.2f}, I_fatigue={I_fatigue:.2f}")
        return I_mem, I_fatigue

    # Re-added from original Memory.py to ensure full functionality
    # Methods below this line were present in the original file and are now re-integrated
    # to maintain compatibility with other parts of the SeQUeNCe codebase.

    def _schedule_expiration(self) -> None:
        """Schedules the memory expiration event based on coherence_time."""
        if self.expiration_event is not None:
            self.timeline.remove_event(self.expiration_event)

        # Original calculation: int(self.cutoff_ratio * self.coherence_time * 1e12)
        # Assuming coherence_time is in seconds, and simulation time is in ps.
        # If coherence_time is -1, it means infinite coherence and no expiration is scheduled.
        if self.coherence_time > 0:
            # Using coherence_time directly if cutoff_ratio not explicitly defined or needed
            decay_time = self.timeline.now() + int(self.coherence_time * 1e12)
            process = Process(self, "expire", [])
            event = Event(decay_time, process)
            self.timeline.schedule(event)
            self.expiration_event = event
        else:
            self.expiration_event = None # No expiration if coherence_time <= 0

    def update_expire_time(self, time: int):
        """Method to change time of expiration."""
        time = max(time, self.timeline.now())
        if self.expiration_event is None:
            if time >= self.timeline.now():
                process = Process(self, "expire", [])
                event = Event(time, process)
                self.timeline.schedule(event)
                self.expiration_event = event # Ensure event is stored if newly created
        else:
            self.timeline.update_event_time(self.expiration_event, time)

    def get_expire_time(self) -> int:
        """Returns the scheduled expiration time, or infinity if no event."""
        return self.expiration_event.time if self.expiration_event else inf

    def notify(self, msg: Any): # Changed type hint from dict[str, Any] to Any for broader use
        """Notifies observers (e.g., MemoryArray) of memory events."""
        for observer in self._observers:
            # Assuming observer has a memory_expire method for compatibility
            if hasattr(observer, 'memory_expire'):
                observer.memory_expire(self)
            else:
                logger.warning(f"[{self.name}] Observer {observer.name} does not have 'memory_expire' method.")


    def detach(self, observer: 'EntanglementProtocol'):
        """Detaches an observer from the memory."""
        if observer in self._observers:
            self._observers.remove(observer)

    # Re-added from original, but `bds_decohere` logic itself removed (replaced by noise_model)
    # Keeping methods for compatibility if other parts of code call them.
    # The actual decoherence is now handled in `get_qubit` by `self.noise_model.apply_noise`.
    def bds_decohere(self) -> None:
        """
        Placeholder for original BDS decoherence logic.
        The actual decoherence is now handled by the assigned noise model in `get_qubit`.
        This method is kept for compatibility if other parts of the code call it,
        but it performs no specific decoherence itself.
        """
        logger.debug(f"[{self.name}] bds_decohere called. Noise handled by assigned noise model.")
        # Optionally, one could trigger get_qubit here to force an update,
        # but apply_noise is designed to be called when the qubit is accessed.

    def get_bds_state(self):
        """
        Placeholder for original BDS state retrieval.
        Returns the current qubit state.
        """
        # self.bds_decohere() # No longer calls actual decoherence, replaced by apply_noise in get_qubit
        return self._qubit # Directly return Qobj, not a specific BDS state representation

    def get_bds_fidelity(self) -> float:
        """
        Placeholder for original BDS fidelity retrieval.
        Returns the current fidelity of the qubit against its ideal state.
        """
        # The fidelity is continuously updated in `update_state`
        return self._current_fidelity_nm

# Note on AbsorptiveMemory:
# The AbsorptiveMemory class from the original file is a separate component
# with distinct functionality (absorptive storage, AFCs).
# The current noise model integration focuses on the `Memory` class (single-atom memory).
# If AbsorptiveMemory also needs advanced non-Markovian/stochastic noise, it would require
# a similar, separate set of modifications tailored to its specific internal state and operations.
# For now, it is assumed to be outside the scope of the current "Memory" component upgrade.
