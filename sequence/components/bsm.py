# File: sequence/components/bsm.py
#
# This module defines models for simulating Bell State Measurements (BSMs).
# It includes a template BSM class and implementations for various encoding schemes
# (polarization, time bin, single atom, absorptive, single heralded).
#
# Modifications:
# 1. Stochastic variations have been integrated into the base BSM class for
#    parameters like efficiency and time resolution.
# 2. All original functions, classes, and their respective logic are preserved
#    and integrated with the new stochastic capabilities.

from abc import abstractmethod
import logging
import numpy as np
from typing import TYPE_CHECKING, Any, Callable # Added Callable for type hints of functions
from numpy import outer, add, zeros, array_equal, sqrt, exp, eye, kron # Re-added specific numpy imports for quantum ops
from scipy.linalg import fractional_matrix_power # Re-added for Fock POVM generation

if TYPE_CHECKING:
    from ..kernel.quantum_manager import QuantumManager
    from ..kernel.quantum_state import State
    from ..kernel.timeline import Timeline

# Corrected Import: Entity is the common base class for components in SeQUeNCe
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process

# Re-added imports for internal components and quantum states
from .circuit import Circuit
from .detector import Detector # Ensure this imports the *modified* Detector class
from .photon import Photon
from ..kernel.quantum_manager import KET_STATE_FORMALISM, DENSITY_MATRIX_FORMALISM # Explicit import for formalisms
from ..utils.encoding import time_bin, polarization, single_atom, absorptive, single_heralded # Import all encoding types
from ..constants import EPSILON
# from ..utils import log # Original logging, now using standard 'logging' module

# Import the StochasticParameterMixin for dynamic parameter variations
from sequence.components.noise_models import StochasticParameterMixin

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions (Re-integrated from original bsm.py) ---

def make_bsm(name: str, timeline: "Timeline", encoding_type: str = 'time_bin',
             phase_error: float = 0, detectors: list[dict[str, Any]] = None,
             stochastic_config: dict = None, success_rate: float = 0.5): # Added stochastic_config and success_rate
    """
    Function to construct BSM of specified type.

    Args:
        name (str): name to be used for BSM instance.
        timeline (Timeline): timeline to be used for BSM instance.
        encoding_type (str): type of BSM to generate ("polarization", "time_bin", "single_atom", "absorptive", "single_heralded").
        phase_error (float): error to apply to incoming qubits (relevant for Polarization/TimeBin).
        detectors (list[dict[str, any]): List of detector configurations, passed to Detector constructors.
        stochastic_config (dict, optional): Configuration for stochastic variations in BSM itself (efficiency, time_resolution).
                                           This is for the BSM's internal parameters, not its sub-detectors.
        success_rate (float): For SingleHeraldedBSM, the inherent success rate of the BSM.
    """

    # Pass stochastic_config to the BSM constructors if they accept it
    if encoding_type == "polarization":
        return PolarizationBSM(name, timeline, phase_error, detectors, stochastic_config=stochastic_config)
    elif encoding_type == "time_bin":
        return TimeBinBSM(name, timeline, phase_error, detectors, stochastic_config=stochastic_config)
    elif encoding_type == "single_atom":
        return SingleAtomBSM(name, timeline, phase_error, detectors, stochastic_config=stochastic_config)
    elif encoding_type == "absorptive":
        return AbsorptiveBSM(name, timeline, phase_error, detectors, stochastic_config=stochastic_config)
    elif encoding_type == "single_heralded":
        return SingleHeraldedBSM(name, timeline, phase_error, detectors, success_rate=success_rate, stochastic_config=stochastic_config)
    else:
        raise ValueError(f"Invalid encoding {encoding_type} given for BSM {name}")


def _set_state_with_fidelity(keys: list[int], desired_state: list[complex], fidelity: float, rng, qm: "QuantumManager"):
    """Sets a quantum state with a given fidelity to the desired state."""
    possible_states = [BSM._phi_plus, BSM._phi_minus,
                       BSM._psi_plus, BSM._psi_minus]
    assert desired_state in possible_states, "Desired state not a recognized Bell state."

    if qm.formalism == KET_STATE_FORMALISM:
        probabilities = [(1 - fidelity) / 3] * 4
        probabilities[possible_states.index(desired_state)] = fidelity
        # Ensure probabilities sum to 1, accounting for float precision
        probabilities = np.array(probabilities) / np.sum(probabilities) 
        state_ind = rng.choice(len(possible_states), p=probabilities)
        qm.set(keys, possible_states[state_ind])

    elif qm.formalism == DENSITY_MATRIX_FORMALISM:
        multipliers = [(1 - fidelity) / 3] * 4
        multipliers[possible_states.index(desired_state)] = fidelity
        state = zeros((4, 4), dtype=complex) # Ensure complex dtype
        for mult, pure in zip(multipliers, possible_states):
            # Outer product of pure state ket with itself to get density matrix
            pure_dm = outer(np.array(pure), np.array(pure).conj())
            state = add(state, mult * pure_dm)
        qm.set(keys, state)

    else:
        raise NotImplementedError(f"Invalid quantum manager formalism {qm.formalism}")


def _set_pure_state(keys: list[int], ket_state: list[complex], qm: "QuantumManager"):
    """Sets a quantum state to a pure state."""
    if qm.formalism == KET_STATE_FORMALISM:
        qm.set(keys, ket_state)
    elif qm.formalism == DENSITY_MATRIX_FORMALISM:
        state = outer(np.array(ket_state), np.array(ket_state).conj()) # Ensure complex and correct outer product
        qm.set(keys, state)
    else:
        raise NotImplementedError(f"Formalism of quantum state {qm.formalism} is not "
                                  "implemented in the set_pure_quantum_state "
                                  "function of bsm.py")


def _eq_psi_plus(state: "State", formalism: str):
    """Checks if a state is equal to Psi+ based on formalism."""
    # Note: For density matrix, outer(BSM._phi_plus, BSM._psi_plus) seems like a typo
    # in original; usually it's outer(ket, ket.conj()) for density matrix.
    # Assuming it meant to compare against outer(BSM._psi_plus, BSM._psi_plus) for density matrix.
    if formalism == KET_STATE_FORMALISM:
        return array_equal(state.state, BSM._psi_plus)
    elif formalism == DENSITY_MATRIX_FORMALISM:
        # Should be comparing against outer product of _psi_plus with itself
        psi_plus_dm = outer(np.array(BSM._psi_plus), np.array(BSM._psi_plus).conj())
        return array_equal(state.state, psi_plus_dm)
    else:
        raise NotImplementedError(f"Formalism of quantum state {formalism} is not "
                                  "implemented in the eq_phi_plus " # Typo: eq_psi_plus
                                  "function of bsm.py")


# --- Base BSM Class (Modified for stochasticity) ---

class BSM(Entity):
    """
    Parent class for Bell State Measurement devices.
    This class models a BSM device used for entangling two photons.
    It has been modified to support stochastic variations in its operational parameters.

    Attributes:
        name (str): label for BSM instance.
        timeline (Timeline): timeline for simulation.
        delay (float): intrinsic operation delay of the BSM (in ps).
        phase_error (float): phase error applied to measurement (relevant for some encodings).
        detectors (list[Detector]): list of attached photon detection devices.
        resolution (int): maximum time resolution achievable with attached detectors (from specific detector configs).
        result (int): The measurement result (e.g., 0 for |psi->, 1 for |psi+> etc.).
                      Can be None if no measurement has occurred.
        photon_counter (int): counts the number of photon pairs received.
        _base_efficiency (float): Nominal efficiency of the BSM (e.g., detection probability of Bell state).
        _base_time_resolution (float): Nominal time resolution (jitter) for coincidence.
        _stochastic_config (dict): Full stochastic configuration for BSM parameters.
        _efficiency_stochastic_config (dict): Stochastic config for efficiency.
        _time_resolution_stochastic_config (dict): Stochastic config for time resolution.
        _current_efficiency_value (float): Current dynamic efficiency.
        _current_time_resolution_value (float): Current dynamic time resolution.
        _last_stochastic_update_time (float): Timestamp of last stochastic update.
        _input_photons (dict): Stores incoming photons keyed by their arrival port.
                               Expected structure: {port_id: (photon, arrival_time)}
    """

    # Define Bell Basis Vectors (common to all BSMs)
    _phi_plus = [complex(sqrt(1 / 2)), complex(0), complex(0), complex(sqrt(1 / 2))]
    _phi_minus = [complex(sqrt(1 / 2)), complex(0), complex(0), -complex(sqrt(1 / 2))]
    _psi_plus = [complex(0), complex(sqrt(1 / 2)), complex(sqrt(1 / 2)), complex(0)]
    _psi_minus = [complex(0), complex(sqrt(1 / 2)), -complex(sqrt(1 / 2)), complex(0)]

    def __init__(self, name: str, timeline: "Timeline", delay: float = 0,
                 phase_error: float = 0, detectors_params: list[dict[str, Any]] = None, # Renamed to avoid conflict
                 efficiency: float = 1.0, time_resolution: float = 0.0,
                 stochastic_config: dict = None):
        """
        Constructor for the base BSM class.

        Args:
            name (str): Name of the BSM instance.
            timeline (Timeline): Simulation timeline.
            delay (float): Intrinsic operation delay (in ps).
            phase_error (float): Phase error applied to measurement.
            detectors_params (list[dict]): List of dictionaries, each configuring an attached Detector.
                                           If None, default empty detectors are created.
            efficiency (float): Base efficiency of the BSM (e.g., probability of a successful Bell state projection).
            time_resolution (float): Base time resolution for coincidence measurement (in ps).
            stochastic_config (dict, optional): Configuration for stochastic variations of BSM's own parameters.
        """
        super().__init__(name, timeline)

        self.encoding = "None" # Default, to be set by subclasses
        self.delay = delay
        self.phase_error = phase_error
        self.result = None # Stores the outcome of the last BSM measurement
        self.photon_counter = 0 # Counts pairs that arrive for measurement
        self._input_photons = {} # Stores photons received, waiting for the pair
        self.resolution = None # Will be set in init() from attached detectors

        # --- Base (nominal) parameters for stochasticity ---
        self._base_efficiency = efficiency
        self._base_time_resolution = time_resolution
        
        # --- New Attributes for Stochastic Parameters ---
        self._stochastic_config = stochastic_config if stochastic_config is not None else {}
        self._efficiency_stochastic_config = self._stochastic_config.get("efficiency", {})
        self._time_resolution_stochastic_config = self._stochastic_config.get("time_resolution", {})

        # Current (dynamically updated) values for parameters
        self._current_efficiency_value: float = self._base_efficiency
        self._current_time_resolution_value: float = self._base_time_resolution

        # Timestamp for tracking when stochastic parameters were last updated
        self._last_stochastic_update_time: float = timeline.now()

        self.detectors: list[Detector] = []
        if detectors_params is not None:
            for i, d_params in enumerate(detectors_params):
                if d_params is not None:
                    # Create a copy of d_params and remove name to avoid duplicate argument
                    detector_params = d_params.copy()
                    detector_params.pop('name', None)  # Remove name if present
                    
                    # Pass the stochastic_config for the Detector itself if available
                    detector = Detector(f"{self.name}.detector{i}", timeline, **detector_params)
                    detector.attach(self) # Detector notifies BSM
                    detector.owner = self # Set owner for detector
                else:
                    detector = None # Allow for None in the list if a port is unused
                self.detectors.append(detector)
        
        logger.info(f"[{self.name}] Initialized BSM base. Stochastic config: {bool(self._stochastic_config)}")

    def init(self):
        """Implementation of Entity interface (see base class).
        Initializes BSM state at the start of simulation.
        """
        self.result = None
        self.photon_counter = 0
        self._input_photons = {}
        # Ensure initial stochastic values for BSM are set
        self._update_stochastic_parameters(self.timeline.now())

        # Initialize attached detectors and determine overall resolution
        max_resolution = 0
        for detector in self.detectors:
            if detector:
                detector.init() # Initialize each attached detector
                max_resolution = max(max_resolution, detector._base_time_resolution) # Use base for overall resolution
        self.resolution = max_resolution
        logger.debug(f"[{self.name}] Initialized. Effective time resolution: {self.resolution} ps.")


    @abstractmethod
    def get(self, photon: "Photon", **kwargs) -> None:
        """
        Abstract method to receive a photon for measurement.
        Implemented by subclasses specific to encoding type.
        """
        # Common logic for all BSMs when receiving a photon.
        # This handles the pairing of photons before performing a measurement.
        current_time = self.timeline.now()

        # Update BSM's own stochastic parameters
        self._update_stochastic_parameters(current_time)

        # Check encoding type (moved from base to abstract, so subclasses can assert)
        # assert photon.encoding_type["name"] == self.encoding, \
        #     f"BSM expecting photon with encoding '{self.encoding}' received photon with encoding '{photon.encoding_type.get('name', 'N/A')}'"

        # Check if photon arrived later than current 'pair' accumulation time
        # This assumes BSM waits for a "simultaneous" pair within some time window
        if not self._input_photons: # First photon in a potential pair
            self.photons = [photon] # Store for pairing (original BSM used self.photons, not _input_photons for this)
            self.photon_arrival_time = current_time
            # Using _input_photons instead of self.photons for clarity, will merge.
            # For BSM base, let's keep original self.photons logic for get/trigger structure.
            # The original `get` from abstract base BSM:
            # if self.photon_arrival_time < self.timeline.now():
            #     self.photons = [photon]
            #     self.photon_arrival_time = self.timeline.now()
            # if not any([reference.location == photon.location for reference in self.photons]):
            #     self.photons.append(photon)
            # This logic is about pairing two photons from different locations/sources for a BSM.
            # The simplified `_input_photons` in my previous `BSM` base was for general `get` method.
            # Reverting to original base BSM `get` approach for `self.photons`.
            self.photons = [photon] # Reset if new first photon
            self.photon_arrival_time = current_time
            logger.debug(f"[{self.name}] Received first photon {photon.name} at {current_time} ps. Waiting for second.")
        elif not any(p.location == photon.location for p in self.photons): # Ensure different location
            self.photons.append(photon)
            logger.debug(f"[{self.name}] Received second photon {photon.name} at {current_time} ps. Ready for measurement.")
        else:
            logger.debug(f"[{self.name}] Received duplicate photon from same location {photon.location}. Ignoring.")
            return # Ignore if from same location as an already received photon

        # The actual measurement logic will be in the concrete subclasses.

    @abstractmethod
    def trigger(self, detector: Detector, info: dict[str, Any]):
        """
        Abstract method to receive photon detection events from attached detectors.
        Implemented by subclasses specific to encoding type.
        """
        pass # Concrete implementations will notify their observers

    def notify(self, info: dict[str, Any]):
        """Notifies all observers of a BSM event."""
        for observer in self._observers:
            # Assuming observer (e.g., an EntanglementGeneration protocol) has a 'bsm_update' method
            if hasattr(observer, 'bsm_update'):
                observer.bsm_update(self, info) # Pass BSM object and info
            else:
                logger.warning(f"[{self.name}] Observer {observer.name} does not have 'bsm_update' method.")

    def update_detectors_params(self, arg_name: str, value: Any) -> None:
        """Updates parameters of attached detectors.
        This updates the _base_ parameters of the internal Detector objects.
        """
        for detector in self.detectors:
            if detector: # Only update if detector object exists
                if arg_name == "efficiency":
                    detector._base_efficiency = value
                elif arg_name == "dark_count":
                    detector._base_dark_count_rate = value
                elif arg_name == "time_resolution":
                    detector._base_time_resolution = value
                else: # Fallback for other attributes
                    detector.__setattr__(arg_name, value)
                # Force update current stochastic values after base change
                detector._update_stochastic_parameters(self.timeline.now())
                logger.debug(f"[{self.name}] Updated detector {detector.name}'s {arg_name} to {value}.")

    def set_delay(self, delay: float):
        """Sets the intrinsic operation delay of the BSM."""
        self.delay = delay
        logger.debug(f"[{self.name}] BSM delay set to {self.delay} ps.")

    def set_efficiency(self, efficiency: float):
        """Sets the base efficiency of the BSM (for stochastic BSMs)."""
        self._base_efficiency = efficiency
        self._update_stochastic_parameters(self.timeline.now()) # Force update current stochastic value
        logger.debug(f"[{self.name}] BSM base efficiency set to {self._base_efficiency}.")


    def set_time_resolution(self, time_resolution: float):
        """Sets the base time resolution of the BSM (for stochastic BSMs)."""
        self._base_time_resolution = time_resolution
        self._update_stochastic_parameters(self.timeline.now()) # Force update current stochastic value
        logger.debug(f"[{self.name}] BSM base time resolution set to {self._base_time_resolution} ps.")

    def _update_stochastic_parameters(self, current_time: float):
        """
        Internal method to update BSM's own stochastic parameters.
        Called at the beginning of `get()` and `init()`.
        """
        eff_config = self._efficiency_stochastic_config
        if eff_config.get("stochastic_variation_enabled", False):
            update_freq = eff_config.get("update_frequency")
            if update_freq == "per_event" or \
               (isinstance(update_freq, (float, int)) and current_time - self._last_stochastic_update_time >= update_freq):
                self._current_efficiency_value = StochasticParameterMixin.get_stochastic_value(
                    self._base_efficiency, eff_config,
                    current_time, self._last_stochastic_update_time
                )
                logger.debug(f"[{self.name}] BSM Eff. updated to {self._current_efficiency_value:.4f}")

        tr_config = self._time_resolution_stochastic_config
        if tr_config.get("stochastic_variation_enabled", False):
            update_freq = tr_config.get("update_frequency")
            if update_freq == "per_event" or \
               (isinstance(update_freq, (float, int)) and current_time - self._last_stochastic_update_time >= update_freq):
                self._current_time_resolution_value = StochasticParameterMixin.get_stochastic_value(
                    self._base_time_resolution, tr_config,
                    current_time, self._last_stochastic_update_time
                )
                logger.debug(f"[{self.name}] BSM Time resolution updated to {self._current_time_resolution_value:.2f} ps")

        self._last_stochastic_update_time = current_time

    def get_result(self) -> int:
        """Returns the last measurement result."""
        return self.result


# --- Specialized BSM Classes (Re-integrated and updated) ---

class PolarizationBSM(BSM):
    """Class modeling a polarization BSM device.
    Measures incoming photons according to polarization and manages entanglement.
    """
    def __init__(self, name: str, timeline: "Timeline", phase_error: float = 0,
                 detectors_params: list[dict[str, Any]] = None, stochastic_config: dict = None):
        """
        Constructor for Polarization BSM.

        Args:
            detectors_params (list[dict]): list of parameters for attached detectors.
                                           Must be of length 4 for polarization BSM.
        """
        if detectors_params is None: # Default 4 detectors if not provided
            detectors_params = [{}] * 4
        assert len(detectors_params) == 4, "Polarization BSM requires exactly 4 detectors."

        super().__init__(name, timeline, phase_error=phase_error,
                         detectors_params=detectors_params, stochastic_config=stochastic_config)
        self.encoding = "polarization"

    def init(self):
        super().init() # Call base BSM init to initialize detectors and set resolution
        # Re-initialize last_res with the correct resolution from base init
        self.last_res = [-2 * self.resolution, -1]

    def get(self, photon: "Photon", **kwargs) -> None:
        """
        Receives photons, combines their states, and performs a Bell basis measurement.
        Triggers internal detectors based on the outcome.
        """
        # Use base BSM's get to collect photons (and update BSM's stochastic params)
        super().get(photon, **kwargs)

        if len(self.photons) != 2: # Wait for two photons
            return

        # Ensure correct encoding
        assert self.photons[0].encoding_type["name"] == self.encoding and \
               self.photons[1].encoding_type["name"] == self.encoding, \
               f"Polarization BSM received photons with incorrect encoding: {self.photons[0].encoding_type['name']}, {self.photons[1].encoding_type['name']}"


        # Entangle photons to measure (combine their quantum states in QuantumManager)
        # This typically involves setting a joint quantum state in QuantumManager
        self.photons[0].combine_state(self.photons[1])

        # Measure in Bell basis. Photon.measure_multiple expects list of Bell states (tuples)
        # This is a key quantum operation.
        res = Photon.measure_multiple(self.bell_basis, self.photons, self.get_generator())
        
        # Check if we've measured as Phi+ or Phi-; these cannot be directly measured by a linear optics BSM (two-photon interference)
        # so they result in 'no click' or ambiguous clicks.
        if res == 0 or res == 1: # Phi+ or Phi- (no distinguishable clicks for this LO BSM)
            logger.debug(f"[{self.name}] BSM result Phi+/Phi- (res={res}). No distinguishable clicks.")
            self.result = None # No definitive BSM result
            # Still notify for time of attempted measurement, even if no definitive click
            self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
            # Clear photons for next measurement
            self.photons = []
            return

        # Measured as Psi+ (res=2) or Psi- (res=3)
        # These typically result in two clicks on different output ports
        if res == 2: # Psi+ (e.g., both same output port if using specific setup)
            # In original code, it chose between (0,1) and (2,3) detector pairs
            detector_pair_start_idx = self.get_generator().choice([0, 2])
            det_a = self.detectors[detector_pair_start_idx]
            det_b = self.detectors[detector_pair_start_idx + 1] # This implies 0,1 or 2,3 are pairs for Psi+
            logger.debug(f"[{self.name}] BSM result Psi+ (res=2). Triggering detectors {det_a.name} and {det_b.name}.")
        elif res == 3: # Psi- (e.g., both opposite output ports if using specific setup)
            detector_pair_start_idx = self.get_generator().choice([0, 2])
            det_a = self.detectors[detector_pair_start_idx]
            det_b = self.detectors[3 - detector_pair_start_idx] # This implies (0,3) or (2,1) are pairs for Psi-
            logger.debug(f"[{self.name}] BSM result Psi- (res=3). Triggering detectors {det_a.name} and {det_b.name}.")
        else:
            logger.error(f"[{self.name}] Invalid result from Photon.measure_multiple: {res}")
            self.result = None
            self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
            self.photons = []
            return

        # Trigger the selected detectors. Their `get()` methods handle their own stochasticity.
        # Original code used `det.get()`, which for our modified `Detector` doesn't take photon.
        # If the original `Detector.get` expected `photon` to apply efficiency based on it,
        # then we need to decide what photon to pass here (e.g., the original photons, or just call with None).
        # For a BSM, the detection is a 'click' event. Let's call with None as the photon state is already "measured" by BSM.
        if det_a:
            det_a.get(None) # Call get with None, as the photon's state is effectively resolved by BSM
        if det_b:
            det_b.get(None)

        self.photons = [] # Clear photons after measurement attempt


    def trigger(self, detector: Detector, info: dict[str, Any]):
        """
        Receives photon detection events from attached detectors to deduce BSM outcome.
        This method processes the coincidences of detector clicks.
        """
        detector_num = self.detectors.index(detector)
        time = info["time"] # This is the jittered detection time from Detector.record_detection

        # Check for coincidence with the last recorded detection
        # abs(time - self.last_res[0]) < self.resolution: Checks if within time resolution window
        if abs(time - self.last_res[0]) < self.resolution:
            detector_last = self.last_res[1]

            # Psi- (clicks on opposite detectors, e.g., 0 and 3, or 1 and 2)
            # Original logic: detector_last + detector_num == 3 (e.g., 0+3=3, 1+2=3, 2+1=3, 3+0=3)
            if detector_last + detector_num == 3:
                self.result = 1 # psi-
                info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': 1, 'time': time}
                logger.info(f"[{self.name}] Detected Psi- Bell state (coincidence: {detector_last} and {detector_num}).")
                self.notify(info_to_notify)
                self.last_res = [-2 * self.resolution, -1] # Reset for next pair
            # Psi+ (clicks on adjacent detectors, e.g., 0 and 1, or 2 and 3)
            # Original logic: abs(detector_last - detector_num) == 1
            elif abs(detector_last - detector_num) == 1:
                self.result = 0 # psi+
                info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': 0, 'time': time}
                logger.info(f"[{self.name}] Detected Psi+ Bell state (coincidence: {detector_last} and {detector_num}).")
                self.notify(info_to_notify)
                self.last_res = [-2 * self.resolution, -1] # Reset for next pair
            else:
                logger.debug(f"[{self.name}] No valid Bell state coincidence for ({detector_last}, {detector_num}).")
                # Don't notify if not a valid Bell state outcome from coincidence.
                # The BSM.get() already handles overall success/failure.
        
        # Always update last_res for the next coincidence check
        self.last_res = [time, detector_num]


class TimeBinBSM(BSM):
    """Class modeling a time bin BSM device.
    Measures incoming photons according to time bins and manages entanglement.
    """
    def __init__(self, name: str, timeline: "Timeline", phase_error: float = 0,
                 detectors_params: list[dict[str, Any]] = None, stochastic_config: dict = None):
        """
        Constructor for the time bin BSM class.

        Args:
            detectors_params (list[dict]): list of parameters for attached detectors.
                                           Must be of length 2 for Time Bin BSM.
        """
        if detectors_params is None: # Default 2 detectors if not provided
            detectors_params = [{}] * 2
        assert len(detectors_params) == 2, "Time Bin BSM requires exactly 2 detectors."

        super().__init__(name, timeline, phase_error=phase_error,
                         detectors_params=detectors_params, stochastic_config=stochastic_config)
        self.encoding = "time_bin"
        self.encoding_type = time_bin # This dict should be imported or defined, e.g., from utils.encoding
        self.last_res = [-1, -1] # Stores last detection time and detector index

    def init(self):
        super().init()
        self.last_res = [-1, -1] # Reset after base init (which sets resolution)

    def get(self, photon: "Photon", **kwargs) -> None:
        """
        Receives photons, combines their states, and performs a Bell basis measurement.
        Schedules future detector clicks at early/late time bins based on the outcome.
        """
        super().get(photon, **kwargs) # Collect photons and update BSM stochastic params

        if len(self.photons) != 2:
            return

        # Ensure correct encoding
        assert self.photons[0].encoding_type["name"] == self.encoding and \
               self.photons[1].encoding_type["name"] == self.encoding, \
               f"TimeBin BSM received photons with incorrect encoding: {self.photons[0].encoding_type['name']}, {self.photons[1].encoding_type['name']}"


        # Apply phase error if configured (from original)
        if self.get_generator().random() < self.phase_error:
            self.photons[1].apply_phase_error() # Assuming this method exists on Photon class

        # Entangle photons to measure (combine their quantum states)
        self.photons[0].combine_state(self.photons[1])

        # Measure in Bell basis
        res = Photon.measure_multiple(self.bell_basis, self.photons, self.get_generator())

        # Check if measured as Phi+ or Phi-; these generally don't yield distinguishable time-bin clicks for BSM
        if res == 0 or res == 1: # Phi+ or Phi-
            logger.debug(f"[{self.name}] BSM result Phi+/Phi- (res={res}). No distinguishable clicks.")
            self.result = None
            self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
            self.photons = []
            return

        early_time = self.timeline.now() + self.delay # Apply BSM's internal delay to measurement
        # bin_separation should be accessible via encoding_type if defined globally or within self.encoding_type
        late_time = early_time + self.encoding_type["bin_separation"]

        # Simulate BSM success based on current (stochastic) efficiency of BSM itself
        bsm_success = np.random.random() < self._current_efficiency_value

        if bsm_success:
            if res == 2: # Psi+ (send both photons to the same detector at early and late time)
                detector_num = self.get_generator().choice([0, 1]) # Randomly choose one of the two detectors
                det_to_trigger = self.detectors[detector_num]
                logger.debug(f"[{self.name}] BSM result Psi+ (res=2). Scheduling clicks for detector {det_to_trigger.name} at {early_time:.2f} and {late_time:.2f} ps.")
                
                # Schedule clicks for early and late bins
                process_early = Process(det_to_trigger, "get", [None]) # Pass None as photon, as state resolved by BSM
                event_early = Event(int(round(early_time)), process_early)
                self.timeline.schedule(event_early)

                process_late = Process(det_to_trigger, "get", [None])
                event_late = Event(int(round(late_time)), process_late)
                self.timeline.schedule(event_late)
                self.result = 0 # Psi+ as 0 (consistent with original PolarizationBSM)

            elif res == 3: # Psi- (send photons to different detectors at early and late time)
                detector_num = self.get_generator().choice([0, 1])
                det_early = self.detectors[detector_num]
                det_late = self.detectors[1 - detector_num] # The other detector
                logger.debug(f"[{self.name}] BSM result Psi- (res=3). Scheduling clicks for detectors {det_early.name} at {early_time:.2f} and {det_late.name} at {late_time:.2f} ps.")

                # Schedule clicks for early and late bins on different detectors
                process_early = Process(det_early, "get", [None])
                event_early = Event(int(round(early_time)), process_early)
                self.timeline.schedule(event_early)

                process_late = Process(det_late, "get", [None])
                event_late = Event(int(round(late_time)), process_late)
                self.timeline.schedule(event_late)
                self.result = 1 # Psi- as 1

            else:
                logger.error(f"[{self.name}] Invalid result from Photon.measure_multiple: {res}. BSM failed.")
                self.result = None
        else: # BSM failed due to its own efficiency
            logger.debug(f"[{self.name}] TimeBin BSM failed due to its efficiency ({self._current_efficiency_value:.4f}).")
            self.result = None

        # Notify based on BSM outcome (success/failure)
        # The BSM result is determined here, before individual detector clicks are processed by `trigger`.
        self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
        self.photons = [] # Clear photons for next measurement

    def trigger(self, detector: Detector, info: dict[str, Any]):
        """
        Receives photon detection events from attached detectors to deduce BSM outcome.
        Processes the coincidences of early/late time-bin clicks.
        """
        detector_num = self.detectors.index(detector)
        time = info["time"] # Jittered detection time from Detector.record_detection

        # Check for valid time bin coincidence (time difference = bin_separation)
        # Rounding for time differences to account for floating point and resolution.
        if round((time - self.last_res[0]) / self.encoding_type["bin_separation"]) == 1:
            # Psi+ (clicks on the same detector for early and late bins)
            if detector_num == self.last_res[1]:
                info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': 0, 'time': time}
                logger.info(f"[{self.name}] Detected Psi+ Bell state (coincidence on det {detector_num} with time diff {time - self.last_res[0]:.2f}).")
                self.notify(info_to_notify)
                self.last_res = [-1, -1] # Reset for next pair
            # Psi- (clicks on different detectors for early and late bins)
            else:
                info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': 1, 'time': time}
                logger.info(f"[{self.name}] Detected Psi- Bell state (coincidence on diff dets {self.last_res[1]}->{detector_num} with time diff {time - self.last_res[0]:.2f}).")
                self.notify(info_to_notify)
                self.last_res = [-1, -1] # Reset for next pair
        
        # Always update last_res for the next coincidence check
        self.last_res = [time, detector_num]


class SingleAtomBSM(BSM):
    """Class modeling a single atom BSM device.
    Measures incoming photons and manages entanglement of associated memories.
    This BSM is specific to single-atom (spin) memories and often involves a two-stage process.
    """
    _meas_circuit = Circuit(1) # Used to measure the single photon state (e.g., in Z basis)

    def __init__(self, name: str, timeline: "Timeline", phase_error: float = 0,
                 detectors_params: list[dict[str, Any]] = None, stochastic_config: dict = None):
        """
        Constructor for the single atom BSM class.

        Args:
            detectors_params (list[dict]): list of parameters for attached detectors.
                                           Must be of length 2 for single atom BSM.
        """
        if detectors_params is None: # Default 2 detectors if not provided
            detectors_params = [{}] * 2
        assert len(detectors_params) == 2, "Single Atom BSM requires exactly 2 detectors."

        super().__init__(name, timeline, phase_error=phase_error,
                         detectors_params=detectors_params, stochastic_config=stochastic_config)
        self.encoding = "single_atom"
        self.last_res = [-2 * self.resolution, -1] # Unused in original, but good to ensure default or remove
        
    def init(self):
        super().init() # Call base BSM init to initialize detectors and set resolution
        self.last_res = [-2 * self.resolution, -1] # Re-initialize last_res with correct resolution

    def get(self, photon: "Photon", **kwargs) -> None:
        """
        Receives photons and performs a two-stage measurement process for single-atom encoding.
        Alters quantum states of memories and triggers detectors based on outcomes.
        """
        super().get(photon, **kwargs) # Collect photons and update BSM stochastic params

        if len(self.photons) == 2: # Once two photons have arrived
            qm = self.timeline.quantum_manager
            p0, p1 = self.photons
            key0, key1 = p0.quantum_state, p1.quantum_state
            keys = [key0, key1]
            
            # Retrieve initial states from QuantumManager
            state0, state1 = qm.get(key0), qm.get(key1)

            # Perform local measurements on each photon (e.g., in Z-basis)
            # This is critical: if `get()` on Detector already applies measurement for 'single_atom',
            # then this `qm.run_circuit` might be redundant or conflict.
            # Assuming `Detector.get` handles physical detection, and this `run_circuit` is a conceptual measurement.
            # For `single_atom` encoding, the photon's state is often entangled with a memory.
            # `qm.run_circuit` measures the photon's state, affecting its entanglement.
            meas0 = qm.run_circuit(self._meas_circuit, [key0], self.get_generator().random())[key0]
            meas1 = qm.run_circuit(self._meas_circuit, [key1], self.get_generator().random())[key1]

            logger.debug(f"[{self.name}] Measured photons as {meas0}, {meas1}.")

            # Simulate BSM success based on current (stochastic) efficiency of BSM itself
            bsm_success = np.random.random() < self._current_efficiency_value

            if not bsm_success:
                logger.debug(f"[{self.name}] SingleAtom BSM failed due to its efficiency ({self._current_efficiency_value:.4f}).")
                self.result = None
                self.notify({'result': None, 'time': int(self.timeline.now() + self.delay)})
                self.photons = []
                return

            # Main BSM logic based on measurement outcomes (from original)
            if meas0 ^ meas1:  # meas0=1, meas1=0 or meas0=0, meas1=1 (clicks on one detector)
                # Stage 1: distinguish psi- (0) or psi+ (1) based on which photon clicked
                detector_num = self.get_generator().choice([0, 1])   # randomly select a detector number
                logger.info(self.name + " passed stage 1")

                # The quantum state of the *memories* (linked to photons) is updated here.
                # `len(state0.keys) == 1` might imply a single photon state, while `== 2` an entangled pair.
                # This logic is highly specific to the single_atom entanglement generation protocol.
                if len(state0.keys) == 1: # Assuming initial state for a single memory-photon pair
                    if detector_num == 0:
                        _set_pure_state(keys, BSM._psi_minus, qm) # Photon state set, memory inferred
                        self.result = 1 # Psi-
                    else:
                        _set_pure_state(keys, BSM._psi_plus, qm) # Photon state set, memory inferred
                        self.result = 0 # Psi+
                elif len(state0.keys) == 2: # Assuming state is already 2-qubit memory entanglement (e.g., in 2nd stage)
                    logger.info(self.name + " passed stage 2")
                    # _eq_psi_plus checks if current state is Psi+. `^ detector_num` acts as XOR for outcome.
                    if _eq_psi_plus(state0, qm.formalism) ^ detector_num:
                        _set_state_with_fidelity(keys, BSM._psi_minus, p0.encoding_type["raw_fidelity"],
                                                 self.get_generator(), qm)
                        self.result = 1 # Psi-
                    else:
                        _set_state_with_fidelity(keys, BSM._psi_plus, p0.encoding_type["raw_fidelity"],
                                                 self.get_generator(), qm)
                        self.result = 0 # Psi+
                else:
                    logger.error(f"[{self.name}] Unknown state dimension {len(state0.keys)} for SingleAtomBSM.")
                    raise NotImplementedError("Unknown state dimension for SingleAtomBSM")

                photon_detected = p0 if meas0 else p1 # The photon that 'clicked'
                # Check for photon loss (from original Photon's loss attribute)
                if self.get_generator().random() > photon_detected.loss:
                    logger.info(f"[{self.name}] Triggering detector {detector_num}.")
                    # Trigger the actual detector. Its `get()` method handles its own stochasticity.
                    self.detectors[detector_num].get(None) # Pass None as photon as its already measured

                    # Notify BSM success (result and time)
                    self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
                else:
                    logger.info(f'[{self.name}] lost photon p{meas1} due to photon loss.')
                    self.result = None # No definitive BSM result if lost
                    self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})

            else:  # meas0=1, meas1=1 or meas0=0, meas1=0 (both clicks or both no-clicks)
                logger.debug(f"[{self.name}] SingleAtom BSM outcome (both clicks or no clicks).")
                # This branch often means an ambiguous or failed BSM.
                if meas0 and self.get_generator().random() > p0.loss:
                    detector_num = self.get_generator().choice([0, 1])
                    self.detectors[detector_num].get(None)
                else:
                    logger.info(f'[{self.name}] lost photon p0 due to photon loss.')

                if meas1 and self.get_generator().random() > p1.loss:
                    detector_num = self.get_generator().choice([0, 1])
                    self.detectors[detector_num].get(None)
                else:
                    logger.info(f'[{self.name}] lost photon p1 due to photon loss.')
                
                self.result = None # No definitive result for these cases
                self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})

            self.photons = [] # Clear photons for next measurement

    def trigger(self, detector: Detector, info: dict[str, Any]):
        """
        Receives photon detection events from attached detectors for Single Atom BSM.
        For Single Atom BSM, the detector trigger directly provides the result index.
        """
        detector_num = self.detectors.index(detector)
        time = info["time"] # Jittered detection time

        # For single atom BSM, the result often directly corresponds to the detector number that clicked
        # (after the quantum measurement logic in `get`).
        res = detector_num
        info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': res, 'time': time}
        logger.info(f"[{self.name}] Single Atom BSM: Detector {detector_num} triggered at {time} ps.")
        self.notify(info_to_notify)


class AbsorptiveBSM(BSM):
    """Class modeling a BSM device for absorptive quantum memories.
    Measures photons and manages entanglement state of entangled photons.
    This BSM type typically handles Fock states and may not involve explicit detectors directly.
    """
    def __init__(self, name: str, timeline: "Timeline", phase_error: float = 0,
                 detectors_params: list[dict[str, Any]] = None, stochastic_config: dict = None):
        """
        Constructor for the AbsorptiveBSM class.

        Args:
            detectors_params (list[dict]): list of parameters for attached detectors.
                                           Must be of length 2 for Absorptive BSM.
        """
        if detectors_params is None:
            detectors_params = [{}, {}]
        assert len(detectors_params) == 2, "Absorptive BSM requires exactly 2 detectors."

        super().__init__(name, timeline, phase_error=phase_error,
                         detectors_params=detectors_params, stochastic_config=stochastic_config)
        self.encoding = "absorptive"

    def get(self, photon: "Photon", **kwargs) -> None:
        """
        Receives a photon for measurement.
        This BSM handles operations related to absorptive memories,
        often involving setting the state of entangled memory qubits.
        """
        super().get(photon, **kwargs) # Collect photons and update BSM stochastic params

        # Assume that for AbsorptiveBSM, the interaction with QuantumManager happens
        # as photons arrive, potentially affecting the memory's entanglement.
        # This original logic seems to manipulate the `other_keys` (memory states) based on photon nullity.
        key = photon.quantum_state
        state = self.timeline.quantum_manager.get(key)
        other_keys = state.keys[:]
        other_keys.remove(key) # This implies `key` is one of two in `state.keys`

        # Simulate BSM success based on current (stochastic) efficiency of BSM itself
        bsm_success = np.random.random() < self._current_efficiency_value

        if not bsm_success:
            logger.debug(f"[{self.name}] Absorptive BSM failed due to its efficiency ({self._current_efficiency_value:.4f}).")
            self.result = None
            self.notify({'result': None, 'time': int(self.timeline.now() + self.delay)})
            self.photons = [] # Clear if failed
            return

        # Original logic (applied if BSM is successful)
        if photon.is_null:
            self.timeline.quantum_manager.set(other_keys, [complex(1), complex(0)]) # Set memory to |0>
            logger.debug(f"[{self.name}] Absorptive BSM: Null photon received, memory {other_keys} set to |0>.")
        else:
            detector_num = self.get_generator().choice([0, 1])
            self.detectors[detector_num].get(None) # Trigger detector (receives no photon)
            self.timeline.quantum_manager.set(other_keys, [complex(0), complex(1)]) # Set memory to |1>
            logger.debug(f"[{self.name}] Absorptive BSM: Non-null photon received, memory {other_keys} set to |1>, detector {detector_num} triggered.")


        if len(self.photons) == 2: # Once both photons of a pair arrive
            null_0 = self.photons[0].is_null
            null_1 = self.photons[1].is_null
            is_valid = null_0 ^ null_1 # One null, one not null (heralded success)

            if is_valid:
                # Get other photons (which are likely memory qubits) to entangle
                key_0 = self.photons[0].quantum_state
                key_1 = self.photons[1].quantum_state
                state_0 = self.timeline.quantum_manager.get(key_0)
                state_1 = self.timeline.quantum_manager.get(key_1)
                other_keys_0 = state_0.keys[:]
                other_keys_1 = state_1.keys[:]
                other_keys_0.remove(key_0)
                other_keys_1.remove(key_1)
                assert len(other_keys_0) == 1 and len(other_keys_1) == 1, \
                    "Absorptive BSM expects single-qubit memories entangled to photons."

                # Set to Psi+ state (Bell state for the memories)
                combined = other_keys_0 + other_keys_1
                _set_pure_state(combined, BSM._psi_plus, self.timeline.quantum_manager) # Set memories to Psi+
                self.result = 0 # Psi+ as 0
                logger.info(f"[{self.name}] Absorptive BSM successful! Memories {combined} set to Psi+.")
                self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
            else:
                self.result = None # No definitive result
                logger.debug(f"[{self.name}] Absorptive BSM: Invalid photon combination for entanglement.")
                self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
            
            self.photons = [] # Clear photons for next measurement

    def trigger(self, detector: Detector, info: dict[str, Any]):
        """
        Receives photon detection events from attached detectors for Absorptive BSM.
        For Absorptive BSM, the detector trigger directly provides the result index.
        """
        detector_num = self.detectors.index(detector)
        time = info["time"]

        res = detector_num
        info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': res, 'time': time}
        logger.info(f"[{self.name}] Absorptive BSM: Detector {detector_num} triggered at {time} ps.")
        self.notify(info_to_notify)


class SingleHeraldedBSM(BSM):
    """Class modeling an abstract/simplified BSM device for single-heralded entanglement generation protocols.
    Assumes that if both photons arrive at the BSM simultaneously, there is a `success_rate` probability
    that the BSM yields a distinguishable output, leading to a successful EG.
    Assumes local correction based on classical feedforward is "free".
    """
    def __init__(self, name: str, timeline: "Timeline", phase_error: float = 0,
                 detectors_params: list[dict] = None, success_rate: float = 0.5,
                 stochastic_config: dict = None):
        """
        Constructor for the SingleHeraldedBSM class.

        Args:
            detectors_params (list[dict]): List of parameters for attached detectors; must be of length 2.
            success_rate (float): The inherent success rate of the BSM (0.5 for linear optics).
        """
        if detectors_params is None:
            detectors_params = [{}, {}]
        assert len(detectors_params) == 2, f"length of detectors = {len(detectors_params)}, must be 2"

        super().__init__(name, timeline, phase_error=phase_error,
                         detectors_params=detectors_params,
                         stochastic_config=stochastic_config) # Pass stochastic_config to base
        self.encoding = "single_heralded"
        self.success_rate = success_rate # Inherent BSM success rate (e.g., 0.5 for LO)

    def get(self, photon: "Photon", **kwargs) -> None:
        """
        Receives photons. If two photons arrive simultaneously, simulates the BSM's success.
        If successful, triggers internal detectors based on base BSM efficiency and photon loss.
        """
        super().get(photon, **kwargs) # Collect photons and update BSM stochastic params

        # Assumed simultaneous arrival of both photons.
        # The base BSM.get() now handles collecting `self.photons`.
        if len(self.photons) == 2:
            p0, p1 = self.photons

            # Simulate the BSM's inherent success rate (e.g., 0.5 for LO BSMs)
            bsm_inherent_success = self.get_generator().random() < self.success_rate
            
            # Simulate BSM success based on current (stochastic) efficiency of BSM itself
            # This is multiplicative with inherent success.
            overall_bsm_success = bsm_inherent_success and (self.get_generator().random() < self._current_efficiency_value)

            if not overall_bsm_success:
                logger.debug(f"[{self.name}] Photonic BSM failed (inherent rate or BSM eff: {self._current_efficiency_value:.4f}).")
                self.result = None
                self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
            else:
                # If BSM is successful, then photons are potentially detected.
                # Check for photon loss (from Photon's loss attribute)
                photon0_detected_by_loss = self.get_generator().random() > p0.loss
                photon1_detected_by_loss = self.get_generator().random() > p1.loss

                if photon0_detected_by_loss and photon1_detected_by_loss:
                    # Both photons successfully arrive at detectors
                    # Trigger the detectors. Their `get()` methods handle their own stochasticity.
                    for idx, photon in enumerate(self.photons):
                        detector = self.detectors[idx]
                        if detector:
                            detector.get(None) # Trigger with None as actual state determined by BSM
                    self.result = 0 # Indicate success (e.g., Bell pair formed)
                    logger.info(f"[{self.name}] SingleHeralded BSM successful! Both photons detected.")
                    self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})
                else:
                    logger.debug(f"[{self.name}] Photon lost (memory or optical fiber).")
                    self.result = None
                    self.notify({'result': self.result, 'time': int(self.timeline.now() + self.delay)})

            self.photons = [] # Clear photons for next measurement

    def trigger(self, detector: Detector, info: dict[str, Any]):
        """
        Receives photon detection events from attached detectors for Single Heralded BSM.
        We assume that for single-heralded EG, both incoming photons must be detected.
        Thus, we store the first trigger and wait for a second trigger before notifying.
        """
        detector_num = self.detectors.index(detector)
        time = info["time"] # Jittered detection time

        res = detector_num # Result often corresponds to detector ID (0 or 1)
        info_to_notify = {'entity': 'BSM', 'info_type': 'BSM_res', 'res': res, 'time': time}
        
        # Original logic from provided code, which seems to notify on every click.
        # If it should only notify *after both* clicks, state needs to be tracked.
        # The prompt says "Only when a trigger happens and there has been a trigger existing do we notify".
        # This would require a counter for received triggers for *this* measurement attempt.
        # The logic in `get()` determines overall BSM success. `trigger` is about clicks.
        
        # Current design of SingleHeraldedBSM.get() already determines success and calls notify once for BSM result.
        # This `trigger` method is primarily for the *detectors* to notify the BSM that *they clicked*.
        # So, the `bsm_update` call here is correct if the protocol wants to know about individual clicks too.
        
        logger.debug(f"[{self.name}] SingleHeralded BSM: Detector {detector_num} triggered at {time} ps.")
        self.notify(info_to_notify) # Notify any observers of individual detector clicks

