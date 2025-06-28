# File: sequence/components/detector.py
#
# This file defines the Detector component, which simulates photon detection,
# and also the QSDetector hierarchy for various qubit encoding measurements.
#
# Modifications have been integrated to:
# 1. Incorporate stochastic variations in Detector parameters (efficiency, dark count rate, time resolution).
# 2. Preserve all original classes (Detector, QSDetector, QSDetectorPolarization,
#    QSDetectorTimeBin, QSDetectorFockDirect, QSDetectorFockInterference, FockDetector)
#    and their functionalities.

from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Any, TYPE_CHECKING, Callable
from numpy import eye, kron, exp, sqrt
from scipy.linalg import fractional_matrix_power
from math import factorial

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline # For Timeline type hint in __init__

# Corrected Import: Entity is the common base class for components in SeQUeNCe
from ..kernel.entity import Entity 
from ..kernel.event import Event
from ..kernel.process import Process
from .photon import Photon # Local import for Photon type hint in get()
from .circuit import Circuit # Added: For Detector._meas_circuit

# Import helper utilities
from ..utils.encoding import time_bin, fock
from ..constants import EPSILON

# Import the StochasticParameterMixin for dynamic parameter variations
from sequence.components.noise_models import StochasticParameterMixin

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Detector(Entity): # Changed: Inherit from Entity instead of Component
    """
    Single photon detector device.
    This class models a single photon detector, for detecting photons.
    Can be attached to many different devices to enable different measurement options.
    Now includes stochastic variations for its operational parameters: efficiency,
    dark count rate, and time resolution.

    Attributes:
        name (str): label for detector instance.
        timeline (Timeline): timeline for simulation.
        efficiency (float): The base quantum efficiency of the detector (0 to 1).
        dark_count (float): The base dark count rate in Hz.
        count_rate (float): maximum detection rate; defines detector cooldown time.
        time_resolution (int): minimum resolving power of photon arrival time (in ps).
        photon_counter (int): counts number of detection events.
        _meas_circuit (Circuit): Internal circuit for single atom encoding measurement.
        _base_efficiency (float): Nominal efficiency.
        _base_dark_count_rate (float): Nominal dark count rate.
        _base_time_resolution (float): Nominal time resolution.
        _stochastic_config (dict): Full stochastic configuration for all parameters.
        _efficiency_stochastic_config (dict): Stochastic config for efficiency.
        _dark_count_rate_stochastic_config (dict): Stochastic config for dark count rate.
        _time_resolution_stochastic_config (dict): Stochastic config for time resolution.
        _current_efficiency_value (float): Current dynamic efficiency.
        _current_dark_count_rate_value (float): Current dynamic dark count rate.
        _current_time_resolution_value (float): Current dynamic time resolution.
        _last_stochastic_update_time (float): Timestamp of last stochastic update.
        next_detection_time (float): The earliest time this detector can record a new detection
                                     due to cooldown.
    """

    _meas_circuit = Circuit(1)
    _meas_circuit.measure(0)

    def __init__(self, name: str, timeline: "Timeline", efficiency: float = 0.9,
                 dark_count: float = 0, count_rate: float = 25e6, time_resolution: int = 150,
                 stochastic_config: dict = None):
        super().__init__(name, timeline) # Entity.__init__(self, name, timeline)

        # Base (nominal) parameters from original Detector
        self._base_efficiency = efficiency
        self._base_dark_count_rate = dark_count
        self._base_time_resolution = time_resolution
        self.count_rate = count_rate
        
        # --- Stochastic Parameters Configuration ---
        self._stochastic_config = stochastic_config if stochastic_config is not None else {}
        self._efficiency_stochastic_config = self._stochastic_config.get("efficiency", {})
        self._dark_count_rate_stochastic_config = self._stochastic_config.get("dark_count_rate", {})
        self._time_resolution_stochastic_config = self._stochastic_config.get("time_resolution", {})

        # Current (dynamically updated) values for parameters
        self._current_efficiency_value: float = self._base_efficiency
        self._current_dark_count_rate_value: float = self._base_dark_count_rate
        self._current_time_resolution_value: float = self._base_time_resolution

        # Original internal state variables
        self.next_detection_time = -1 # Cooldown time tracker
        self.photon_counter = 0 # Simple counter for detected events
        self._last_stochastic_update_time: float = timeline.now() # Timestamp for stochastic updates

        logger.info(f"[{self.name}] Initialized Detector. "
                    f"Stochastic config: {bool(self._stochastic_config)}")

    def init(self):
        """Implementation of Entity interface (see base class).
        Initializes detector state at the start of simulation.
        """
        self.next_detection_time = -1
        self.photon_counter = 0
        # If base dark count is > 0, schedule the first dark count event
        # The rate will be dynamically updated by _update_stochastic_parameters later
        if self._base_dark_count_rate > 0:
            self.add_dark_count()
        
        # Ensure initial stochastic values are set
        self._update_stochastic_parameters(self.timeline.now())


    def get(self, photon: Photon = None, **kwargs) -> None:
        """
        Method to receive a photon for measurement.
        This method updates detector parameters based on stochastic models before detection.

        Args:
            photon (Photon): photon to detect.
            **kwargs: Additional arguments (e.g., source information).

        Side Effects:
            May notify upper entities of a detection event.
            May modify quantum state of photon if single_atom encoding.
        """
        self.photon_counter += 1 # Increment total received photons (not necessarily detected)
        current_time = self.timeline.now()

        # Update stochastic parameters to their current values
        self._update_stochastic_parameters(current_time)

        detected = False
        # 1. Simulate detection of actual photon
        if photon and not photon.is_null:
            # If using single_atom encoding, run circuit measurement first
            if photon.encoding_type and photon.encoding_type.get("name") == "single_atom":
                # Assuming quantum_state is the key for the Qutip state in QuantumManager
                key = photon.quantum_state
                # Run the measurement circuit (e.g., measuring in Z-basis to check for |1>)
                # `res` will contain the outcome (e.g., 0 for |0>, 1 for |1>)
                try:
                    res = self.timeline.quantum_manager.run_circuit(Detector._meas_circuit, [key], self.get_generator().random())
                    # If measured |0>, it's not a 'detection' in the classical sense for some protocols.
                    # This logic depends on the protocol's interpretation of a 'click'.
                    # Original logic: if not res[key], return. This means only |1> counts as a detection.
                    if not res[key]: # If the measurement outcome for `key` is 0 (i.e., |0>), then return.
                        logger.debug(f"[{self.name}] Photon's single_atom state measured as |0>. Not considered a detection.")
                        return # Exit, no detection recorded
                except Exception as e:
                    logger.error(f"[{self.name}] Error running measurement circuit for photon {photon.name}: {e}")
                    # Decide how to handle this error: consider it lost, or proceed. For now, exit.
                    return


            # If photon state implies it *should* trigger, apply detector efficiency
            if np.random.random() < self._current_efficiency_value:
                detected = True
                logger.debug(f"[{self.name}] Photon detected (efficiency: {self._current_efficiency_value:.4f}).")
            else:
                logger.debug(f"[{self.name}] Photon lost due to detector inefficiency (efficiency: {self._current_efficiency_value:.4f}).")
        else:
            logger.debug(f"[{self.name}] Received null photon or no photon.")


        # 2. Simulate dark counts
        # Dark counts occur at a rate (Hz). For a discrete event sim, schedule next dark count.
        # This `get` call itself may be a dark count event.
        # The `add_dark_count` method schedules future dark count *events*.
        # This `if` block handles if *this specific `get` call* is a dark count that results in detection.
        # Original: dark_count is rate in 1/s, convert to probability for this instant of time.
        # Simplified: if dark_count > 0, it means dark counts are possible.
        # The actual occurrence is managed by the `add_dark_count` scheduled events.
        # If this `get` call is *triggered* by a dark count event, then `detected` is already True.
        # If this `get` call is triggered by a photon but the photon was NOT detected,
        # we do NOT *also* have a dark count *at the same instant* in a Poisson process.
        # Dark counts are separate events. So, this `get` logic only applies to actual photons.


        # Record detection time with time resolution (jitter) if detected
        if detected:
            self.record_detection() # Use original record_detection to handle cooldown and notify
        else:
            logger.debug(f"[{self.name}] No detection event registered for this photon.")

    def add_dark_count(self) -> None:
        """
        Method to schedule false positive detection events as a Poisson process.
        The rate of this Poisson process is dynamically updated by `_current_dark_count_rate_value`.

        Side Effects:
            May schedule future `record_detection` calls.
            May schedule future calls to itself (`add_dark_count`).
        """
        # Update stochastic parameter *before* scheduling next event
        current_time = self.timeline.now()
        self._update_stochastic_parameters(current_time) # Update all stochastic params
        
        # Use the current, stochastic dark count rate
        current_dark_count_rate_hz = self._current_dark_count_rate_value # In Hz (1/s)

        if current_dark_count_rate_hz <= 0:
            logger.debug(f"[{self.name}] Dark count rate is zero or negative. Not scheduling dark count events.")
            return

        # Time to next event in seconds (from exponential distribution)
        # Using 1e12 to convert to picoseconds (SeQUeNCe's typical time unit)
        time_to_next_sec = self.get_generator().exponential(1 / current_dark_count_rate_hz)
        time_to_next_ps = int(time_to_next_sec * 1e12)
        
        time = time_to_next_ps + current_time # Time of next dark count event

        # Schedule the next dark count event (recursively)
        process_add_dc = Process(self, "add_dark_count", [])
        event_add_dc = Event(time, process_add_dc)
        self.timeline.schedule(event_add_dc)

        # Schedule the detection recording for this dark count event
        # Dark counts directly lead to a "record_detection" event
        process_record_dc = Process(self, "record_detection", [])
        event_record_dc = Event(time, process_record_dc)
        self.timeline.schedule(event_record_dc) # This schedules the actual "click" for the future

        logger.debug(f"[{self.name}] Scheduled next dark count at {time:.2f} ps (rate: {current_dark_count_rate_hz:.2e} Hz).")


    def record_detection(self):
        """
        Method to record a detection event (either from a photon or a dark count).
        Will respect detector cooldown time and apply time resolution jitter.
        Will notify observers with the actual (jittered) detection time.
        """
        now = self.timeline.now()

        # Update stochastic time resolution before applying jitter
        self._update_stochastic_parameters(now) # Important to get latest time_resolution

        # Check if the detector is out of cooldown (ready to detect)
        if now >= self.next_detection_time:
            # Apply time resolution jitter
            detection_time_jitter = np.random.normal(0, self._current_time_resolution_value)
            actual_detection_time = now + detection_time_jitter
            
            # Ensure detection time is rounded to a multiple of time_resolution
            # Original code implies resolution is used for rounding detection time.
            rounded_detection_time = round(actual_detection_time / self._current_time_resolution_value) * self._current_time_resolution_value
            
            logger.info(f"[{self.name}] Detection recorded at {rounded_detection_time:.2f} ps. "
                        f"(Jitter: {detection_time_jitter:.2f} ps, Resolution: {self._current_time_resolution_value:.2f} ps)")
            
            # Notify observers (e.g., QSDetector, BSM) of the detection
            self.notify({'time': int(rounded_detection_time)}) # Convert to int if timeline expects int times

            # Update next_detection_time based on detector count_rate (cooldown)
            cooldown_period_ps = int(1e12 / self.count_rate) # Convert Hz to ps
            self.next_detection_time = now + cooldown_period_ps
            logger.debug(f"[{self.name}] Detector cooldown until {self.next_detection_time:.2f} ps (period: {cooldown_period_ps} ps).")
        else:
            logger.debug(f"[{self.name}] Detection attempted, but detector is in cooldown. Next detection possible at {self.next_detection_time:.2f} ps.")

    def notify(self, info: dict[str, Any]):
        """Custom notify function (calls `trigger` method of observers)."""
        for observer in self._observers:
            # Assuming observer (e.g., QSDetector) has a 'trigger' method
            if hasattr(observer, 'trigger'):
                observer.trigger(self, info)
            else:
                logger.warning(f"[{self.name}] Observer {observer.name} does not have 'trigger' method.")

    def _update_stochastic_parameters(self, current_time: float):
        """
        Internal method to update all stochastic parameters based on their configurations and the current time.
        This method is called at the beginning of `get()` and `add_dark_count()`.
        """
        # Determine if a full update is needed based on `_last_stochastic_update_time`
        # and individual parameter's `update_frequency`.
        # This implementation updates ALL parameters if *any* of them needs an update
        # based on its `update_frequency` (unless it's "per_event").
        # If "per_event", it updates regardless of last update time.

        # --- Update Efficiency ---
        eff_config = self._efficiency_stochastic_config
        if eff_config.get("stochastic_variation_enabled", False):
            update_freq = eff_config.get("update_frequency")
            if update_freq == "per_event" or \
               (isinstance(update_freq, (float, int)) and current_time - self._last_stochastic_update_time >= update_freq):
                self._current_efficiency_value = StochasticParameterMixin.get_stochastic_value(
                    self._base_efficiency, eff_config,
                    current_time, self._last_stochastic_update_time
                )
                logger.debug(f"[{self.name}] Eff. updated to {self._current_efficiency_value:.4f}")

        # --- Update Dark Count Rate ---
        dc_config = self._dark_count_rate_stochastic_config
        if dc_config.get("stochastic_variation_enabled", False):
            update_freq = dc_config.get("update_frequency")
            if update_freq == "per_event" or \
               (isinstance(update_freq, (float, int)) and current_time - self._last_stochastic_update_time >= update_freq):
                self._current_dark_count_rate_value = StochasticParameterMixin.get_stochastic_value(
                    self._base_dark_count_rate, dc_config,
                    current_time, self._last_stochastic_update_time
                )
                logger.debug(f"[{self.name}] Dark count updated to {self._current_dark_count_rate_value:.2e} Hz")

        # --- Update Time Resolution ---
        tr_config = self._time_resolution_stochastic_config
        if tr_config.get("stochastic_variation_enabled", False):
            update_freq = tr_config.get("update_frequency")
            if update_freq == "per_event" or \
               (isinstance(update_freq, (float, int)) and current_time - self._last_stochastic_update_time >= update_freq):
                self._current_time_resolution_value = StochasticParameterMixin.get_stochastic_value(
                    self._base_time_resolution, tr_config,
                    current_time, self._last_stochastic_update_time
                )
                logger.debug(f"[{self.name}] Time resolution updated to {self._current_time_resolution_value:.2f} ps")

        # Update the overall timestamp for *timed* stochastic updates
        # This ensures that if a parameter's `update_frequency` is e.g. 1000.0,
        # it only updates after that time has passed since this _last_stochastic_update_time.
        self._last_stochastic_update_time = current_time


class QSDetector(Entity, ABC):
    """Abstract QSDetector parent class.
    Provides a template for objects measuring qubits in different encoding schemes.
    """
    def __init__(self, name: str, timeline: "Timeline"):
        super().__init__(name, timeline) # Entity.__init__(self, name, timeline)
        self.components = [] # List of internal hardware components (e.g., beamsplitters, switches)
        self.detectors = [] # List of attached Detector instances
        self.trigger_times = [] # Tracks simulation time of detection events for each detector (list of lists)

    def init(self):
        """Initializes all internal components and detectors."""
        for component in self.components:
            component.attach(self) # QSDetector acts as observer for its components
            if hasattr(component, 'init'): # Call init on sub-components if they have it
                component.init()
            component.owner = self.owner # Pass down the ultimate owner (e.g., a Node)

    def update_detector_params(self, detector_id: int, arg_name: str, value: Any) -> None:
        """Updates a base parameter of an attached detector (e.g., efficiency, dark_count).
        Note: This updates the _base_ value. For stochastic changes, update stochastic_config directly.
        """
        assert 0 <= detector_id < len(self.detectors), "Invalid detector_id"
        detector = self.detectors[detector_id]
        # Direct setattr might bypass stochastic config updates for base values.
        # It's better to update via stochastic_config if using stochastic params.
        # This method's usage might need review if it's meant to control stochastic parameters.
        if arg_name == "efficiency":
            detector._base_efficiency = value
        elif arg_name == "dark_count": # Original name was dark_count
            detector._base_dark_count_rate = value
        elif arg_name == "time_resolution":
            detector._base_time_resolution = value
        else:
            detector.__setattr__(arg_name, value) # Fallback for other attributes

    @abstractmethod
    def get(self, photon: Photon, **kwargs) -> None:
        """Abstract method for receiving photons for measurement."""
        pass

    def trigger(self, detector: Detector, info: dict[str, Any]) -> None:
        """
        Receives a detection trigger from an attached Detector.
        This method is called by the Detector's `notify` method.
        """
        try:
            detector_index = self.detectors.index(detector)
            self.trigger_times[detector_index].append(info['time'])
            logger.debug(f"[{self.name}] Received trigger from {detector.name} at {info['time']} ps.")
        except ValueError:
            logger.error(f"[{self.name}] Received trigger from unknown detector {detector.name}.")
        except IndexError:
            logger.error(f"[{self.name}] Detector index {detector_index} out of bounds for trigger_times.")


    def set_detector(self, idx: int,  efficiency: float = 0.9, dark_count: float = 0, # Renamed to dark_count_rate in Detector
                     count_rate: float = 25e6, time_resolution: int = 150, stochastic_config: dict = None):
        """
        Method to set the properties of an attached detector.
        Allows updating base values and providing a new stochastic_config.

        Args:
            idx (int): the index of attached detector whose properties are going to be set.
            For other parameters see the `Detector` class.
            stochastic_config (dict, optional): New stochastic config for this detector.
        """
        assert 0 <= idx < len(self.detectors), "`idx` must be a valid index of attached detector."

        detector = self.detectors[idx]
        # Update base values
        detector._base_efficiency = efficiency
        detector._base_dark_count_rate = dark_count
        detector._base_time_resolution = time_resolution
        detector.count_rate = count_rate
        detector._base_time_resolution = time_resolution # Corrected assignment
        
        # Update stochastic configuration (merging or replacing as needed)
        if stochastic_config is not None:
            detector._stochastic_config.update(stochastic_config) # Merge new config
            detector._efficiency_stochastic_config = detector._stochastic_config.get("efficiency", {})
            detector._dark_count_rate_stochastic_config = detector._stochastic_config.get("dark_count_rate", {})
            detector._time_resolution_stochastic_config = detector._stochastic_config.get("time_resolution", {})
            # Force an update of current stochastic values
            detector._update_stochastic_parameters(self.timeline.now())


    def get_photon_times(self) -> list[list[int]]:
        """Returns collected trigger times and resets the buffer."""
        times = self.trigger_times
        self.trigger_times = [[] for _ in range(len(self.detectors))] # Reset for next interval
        return times

    @abstractmethod
    def set_basis_list(self, basis_list: list[int], start_time: int, frequency: float) -> None:
        """Abstract method to set measurement basis list for sub-components."""
        pass


class QSDetectorPolarization(QSDetector):
    """QSDetector to measure polarization encoded qubits.
    There are two detectors.
    Detectors[0] and detectors[1] are directly connected to the beamsplitter.
    """
    def __init__(self, name: str, timeline: "Timeline"):
        super().__init__(name, timeline)
        from .beam_splitter import BeamSplitter # Local import

        for i in range(2):
            # Pass new stochastic_config here if needed, otherwise Detector defaults
            d = Detector(name + f".detector{i}", timeline)
            self.detectors.append(d)
            d.attach(self) # Detector observes QSDetectorPolarization

        self.splitter = BeamSplitter(name + ".splitter", timeline)
        self.splitter.add_receiver(self.detectors[0])
        self.splitter.add_receiver(self.detectors[1])
        
        # Initialize trigger_times to match the number of detectors
        self.trigger_times = [[] for _ in range(len(self.detectors))]

        self.components = [self.splitter] + self.detectors

    def init(self) -> None:
        """Initializes all internal components and detectors."""
        assert len(self.detectors) == 2, "Polarization detector requires exactly 2 detectors."
        super().init()

    def get(self, photon: Photon, **kwargs) -> None:
        """Method to receive a photon for measurement.
        Forwards the photon to the internal polarization beamsplitter.
        """
        self.splitter.get(photon)

    def set_basis_list(self, basis_list: list[int], start_time: int, frequency: float) -> None:
        """Sets the measurement basis list for the internal beamsplitter."""
        self.splitter.set_basis_list(basis_list, start_time, frequency)

    def update_splitter_params(self, arg_name: str, value: Any) -> None:
        """Updates a parameter of the internal beamsplitter."""
        self.splitter.__setattr__(arg_name, value)


class QSDetectorTimeBin(QSDetector):
    """QSDetector to measure time bin encoded qubits.
    There are three detectors.
    The switch is connected to detectors[0] and the interferometer.
    The interferometer is connected to detectors[1] and detectors[2].
    """
    def __init__(self, name: str, timeline: "Timeline"):
        super().__init__(name, timeline)
        from .switch import Switch # Local import
        from .interferometer import Interferometer # Local import

        self.switch = Switch(name + ".switch", timeline)
        self.detectors = [Detector(name + f".detector{i}", timeline) for i in range(3)]
        self.switch.add_receiver(self.detectors[0])
        
        # Original: time_bin["bin_separation"] might need to be imported or configured
        self.interferometer = Interferometer(name + ".interferometer", timeline, time_bin["bin_separation"])
        self.interferometer.add_receiver(self.detectors[1])
        self.interferometer.add_receiver(self.detectors[2])
        self.switch.add_receiver(self.interferometer)

        self.components = [self.switch, self.interferometer] + self.detectors
        self.trigger_times = [[] for _ in range(len(self.detectors))]

    def init(self):
        """Initializes all internal components and detectors."""
        assert len(self.detectors) == 3, "Time-bin detector requires exactly 3 detectors."
        super().init()

    def get(self, photon: Photon, **kwargs) -> None:
        """Method to receive a photon for measurement.
        Forwards the photon to the internal fiber switch.
        """
        self.switch.get(photon)

    def set_basis_list(self, basis_list: list[int], start_time: int, frequency: float) -> None:
        """Sets the measurement basis list for the internal switch."""
        self.switch.set_basis_list(basis_list, start_time, frequency)

    def update_interferometer_params(self, arg_name: str, value: Any) -> None:
        """Updates a parameter of the internal interferometer."""
        self.interferometer.__setattr__(arg_name, value)


class QSDetectorFockDirect(QSDetector):
    """QSDetector to directly measure photons in Fock state.
    Usage: to measure diagonal elements of effective density matrix.
    """
    def __init__(self, name: str, timeline: "Timeline", src_list: list[str]):
        super().__init__(name, timeline)
        assert len(src_list) == 2, "FockDirect detector requires exactly 2 sources."
        self.src_list = src_list

        for i in range(2):
            d = Detector(name + f".detector{i}", timeline)
            self.detectors.append(d)
            d.attach(self) # Detector observes this QSDetector
        self.components = self.detectors # Only detectors as components for this type

        self.trigger_times = [[] for _ in range(len(self.detectors))]
        self.arrival_times = [[], []] # Tracks arrival times at input ports

        self.povms = [None] * 4 # POVM operators for measurement

    def init(self):
        """Initializes POVMs and internal components."""
        self._generate_povms()
        super().init()

    def _generate_povms(self):
        """Method to generate POVM operators corresponding to photon detector having 0 and 1 click."""
        truncation = self.timeline.quantum_manager.truncation
        # Assuming quantum_manager has build_ladder and supports Fock state operations
        create, destroy = self.timeline.quantum_manager.build_ladder()

        # POVMs for Detector 0
        efficiency0 = self.detectors[0]._base_efficiency # Use base efficiency for POVM generation
        destroy0 = destroy * sqrt(efficiency0)
        create0 = create * sqrt(efficiency0)
        
        # Summation for POVM (I - exp(a^dag a (eff-1)) ) for 0-click
        # This is a bit more complex, often simplified as:
        # P_k = |k><k| * (1-eff)^k * eff
        # P_0 = |0><0| + sum_{n=1 to inf} |n><n| * (1-eff)^n
        # P_1 = |1><1| * eff * (1-eff)^0 + |2><2| * eff * (1-eff)^1 + ...
        # The original code's `series_elem_list` is a specific analytical form for this.
        series_elem_list0_1 = [((-1)**i) * fractional_matrix_power(create0, i+1).dot(
            fractional_matrix_power(destroy0, i+1)) / factorial(i+1) for i in range(truncation)]
        povm0_1 = sum(series_elem_list0_1) # POVM for 1-click on detector 0
        povm0_0 = eye(truncation + 1) - povm0_1 # POVM for 0-click on detector 0

        # POVMs for Detector 1
        efficiency1 = self.detectors[1]._base_efficiency
        destroy1 = destroy * sqrt(efficiency1)
        create1 = create * sqrt(efficiency1)
        series_elem_list1_1 = [((-1)**i) * fractional_matrix_power(create1, i+1).dot(
            fractional_matrix_power(destroy1, i+1)) / factorial(i+1) for i in range(truncation)]
        povm1_1 = sum(series_elem_list1_1) # POVM for 1-click on detector 1
        povm1_0 = eye(truncation + 1) - povm1_1 # POVM for 0-click on detector 1

        # Store POVMs in a specific order for measurement logic
        self.povms = [povm0_0, povm0_1, povm1_0, povm1_1]

    def get(self, photon: Photon, **kwargs) -> None:
        """Receives a photon for measurement, performing Fock state measurement."""
        src = kwargs.get("src")
        if src is None:
            logger.error(f"[{self.name}] QSDetectorFockDirect.get called without 'src' in kwargs.")
            return

        assert photon.encoding_type and photon.encoding_type.get("name") == "fock", "Photon must be in Fock representation."
        
        try:
            input_port = self.src_list.index(src)
        except ValueError:
            logger.error(f"[{self.name}] Source '{src}' not in registered src_list {self.src_list}.")
            return

        arrival_time = self.timeline.now()
        self.arrival_times[input_port].append(arrival_time)

        key = photon.quantum_state
        samp = self.get_generator().random()
        
        # Measure using POVMs based on input port
        if input_port == 0:
            result = self.timeline.quantum_manager.measure([key], self.povms[0:2], samp)
        elif input_port == 1:
            result = self.timeline.quantum_manager.measure([key], self.povms[2:4], samp)
        else:
            logger.error(f"[{self.name}] Input port {input_port} out of range for QSDFockDirect.")
            raise Exception(f"too many input ports for QSDFockDirect {self.name}")

        assert result in [0, 1], f"The measurement outcome {result} is not valid (expected 0 or 1)."
        
        # If result is 1 (detection click), record it.
        # Note: The `Detector` object (`self.detectors[input_port]`) handles its own efficiency,
        # dark counts, and cooldown. This `record_detection` directly triggers a "click" event.
        if result == 1:
            self.detectors[input_port].record_detection() # Trigger actual SPD detection logic

    def set_basis_list(self, basis_list: list[int], start_time: int, frequency: float) -> None:
        """Does nothing for this class (not applicable for Fock direct measurement)."""
        logger.debug(f"[{self.name}] set_basis_list called, but not applicable for FockDirect.")
        pass


class QSDetectorFockInterference(QSDetector):
    """QSDetector with two input ports and two photon detectors behind beamsplitter.
    The detectors will physically measure the two beamsplitter output photonic modes' Fock states.
    POVM operators which apply to pre-beamsplitter photonic state are used.
    NOTE: in the current implementation, to realize interference, we require that Photons arrive at both input ports
    simultaneously, and at most 1 Photon instance can be input at an input port at a time.
    Usage: to realize Bell state measurement (BSM) and to measure off-diagonal elements of the effective density matrix.
    """
    def __init__(self, name: str, timeline: "Timeline", src_list: list[str], phase: float = 0):
        super().__init__(name, timeline)
        assert len(src_list) == 2, "FockInterference detector requires exactly 2 sources."
        self.src_list = src_list
        self.phase = phase # Relative phase between two input optical paths

        for i in range(2):
            d = Detector(name + f".detector{i}", timeline)
            self.detectors.append(d)
            d.attach(self) # Detector observes this QSDetector
        self.components = self.detectors # Only detectors are internal components

        self.trigger_times = [[] for _ in range(len(self.detectors))]
        self.detect_info = [[], []] # Tracks detection information (time, outcome) for each detector
        self.arrival_times = [[], []] # Tracks arrival times at input modes

        self.temporary_photon_info = [{}, {}] # Stores info for simultaneous photon arrival detection

        self.povms = [None] * 4 # POVM operators for 00, 01, 10, 11 clicks

    def init(self):
        """Initializes POVMs and internal components."""
        self._generate_povms()
        super().init()

    def _generate_transformed_ladders(self):
        """Method to generate transformed creation/annihilation operators by the beamsplitter.
        Will be used to construct POVM operators.
        """
        truncation = self.timeline.quantum_manager.truncation
        identity = eye(truncation + 1)
        create, destroy = self.timeline.quantum_manager.build_ladder()
        phase = self.phase
        efficiency1 = sqrt(self.detectors[0]._base_efficiency) # Use base efficiency for POVMs
        efficiency2 = sqrt(self.detectors[1]._base_efficiency)

        # Modified mode operators in Heisenberg picture by beamsplitter transformation
        # considering inefficiency and ignoring relative phase (as per original logic)
        create1 = (kron(efficiency1 * create, identity) + np.exp(1j * phase) * kron(identity, efficiency2 * create)) / sqrt(2)
        destroy1 = create1.conj().T
        create2 = (kron(efficiency1 * create, identity) - np.exp(1j * phase) * kron(identity, efficiency2 * create)) / sqrt(2)
        destroy2 = create2.conj().T

        return create1, destroy1, create2, destroy2

    def _generate_povms(self):
        """Method to generate POVM operators corresponding to photon detector having 00, 01, 10 and 11 click(s).
        Will be used to generated outcome probability distribution.
        """
        truncation = self.timeline.quantum_manager.truncation
        create1, destroy1, create2, destroy2 = self._generate_transformed_ladders()

        # For detector1 (index 0)
        series_elem_list1 = [((-1)**i) * fractional_matrix_power(create1, i+1).dot(
            fractional_matrix_power(destroy1, i+1)) / factorial(i+1) for i in range(truncation)]
        povm1_1 = sum(series_elem_list1) # POVM for 1-click on detector 0
        povm0_1 = eye((truncation+1) ** 2) - povm1_1 # POVM for 0-click on detector 0

        # For detector2 (index 1)
        series_elem_list2 = [((-1)**i) * fractional_matrix_power(create2, i+1).dot(
            fractional_matrix_power(destroy2, i+1)) / factorial(i+1) for i in range(truncation)]
        povm1_2 = sum(series_elem_list2) # POVM for 1-click on detector 1
        povm0_2 = eye((truncation+1) ** 2) - povm1_2 # POVM for 0-click on detector 1

        # POVM operators for 4 possible outcomes (00, 01, 10, 11)
        # These are tensor products of single-detector POVMs
        povm00 = povm0_1 @ povm0_2 # No click on detector 0, No click on detector 1
        povm01 = povm0_1 @ povm1_2 # No click on detector 0, Click on detector 1
        povm10 = povm1_1 @ povm0_2 # Click on detector 0, No click on detector 1
        povm11 = povm1_1 @ povm1_2 # Click on detector 0, Click on detector 1

        # Store in specific order for quantum_manager.measure
        self.povms = [povm00, povm01, povm10, povm11]

    def get(self, photon: Photon, **kwargs) -> None:
        """Receives a photon for measurement, handling interference."""
        src = kwargs.get("src")
        if src is None:
            logger.error(f"[{self.name}] QSDetectorFockInterference.get called without 'src' in kwargs.")
            return

        assert photon.encoding_type and photon.encoding_type.get("name") == "fock", "Photon must be in Fock representation."
        
        try:
            input_port = self.src_list.index(src)
        except ValueError:
            logger.error(f"[{self.name}] Source '{src}' not in registered src_list {self.src_list}.")
            return

        arrival_time = self.timeline.now()
        self.arrival_times[input_port].append(arrival_time)

        assert not self.temporary_photon_info[input_port], \
            f"[{self.name}] At most 1 Photon instance should arrive at input port {input_port} at a time."
        
        self.temporary_photon_info[input_port]["photon"] = photon
        self.temporary_photon_info[input_port]["time"] = arrival_time

        dict0 = self.temporary_photon_info[0]
        dict1 = self.temporary_photon_info[1]

        # If both input ports have received a photon simultaneously, perform measurement
        if dict0 and dict1:
            if dict0["time"] != dict1["time"]:
                logger.warning(f"[{self.name}] Photons arrived at different times ({dict0['time']} vs {dict1['time']}). "
                                "Interference requires simultaneous arrival. Not performing measurement.")
                # Reset temporary info or handle as non-interfering. For now, reset.
                self.temporary_photon_info = [{}, {}]
                return

            photon0 = dict0["photon"]
            photon1 = dict1["photon"]
            key0 = photon0.quantum_state
            key1 = photon1.quantum_state

            samp = self.get_generator().random()
            
            # Measure using POVMs on the combined quantum state
            # quantum_manager.measure expects list of keys, list of POVMs, and random sample
            result = self.timeline.quantum_manager.measure([key0, key1], self.povms, samp)

            assert result in list(range(len(self.povms))), f"Measurement outcome {result} not valid."
            
            detection_time = self.timeline.now()
            info0 = {"time": detection_time, "outcome": 0} # Default no click
            info1 = {"time": detection_time, "outcome": 0} # Default no click

            # Based on the measurement result, trigger individual detectors and update detect_info
            if result == 0: # 00 click: no click on either detector
                logger.debug(f"[{self.name}] BSM outcome 00 (no clicks).")
            elif result == 1: # 01 click: click on detector 1 only
                self.detectors[1].record_detection()
                info1["outcome"] = 1
                logger.debug(f"[{self.name}] BSM outcome 01 (detector 1 click).")
            elif result == 2: # 10 click: click on detector 0 only
                self.detectors[0].record_detection()
                info0["outcome"] = 1
                logger.debug(f"[{self.name}] BSM outcome 10 (detector 0 click).")
            elif result == 3: # 11 click: click on both detectors
                self.detectors[0].record_detection()
                self.detectors[1].record_detection()
                info0["outcome"] = 1
                info1["outcome"] = 1
                logger.debug(f"[{self.name}] BSM outcome 11 (both clicks).")

            self.detect_info[0].append(info0)
            self.detect_info[1].append(info1)
            self.temporary_photon_info = [{}, {}] # Reset for next pair

    def get_photon_times(self) -> list[list[int]]:
        """Method to get detector trigger times and detection information.
        Will clear `trigger_times` and `detect_info`.
        """
        # Return trigger_times (from detectors) and detect_info (from BSM outcomes)
        trigger_times_copy = [list(times) for times in self.trigger_times] # Get a copy
        detect_info_copy = [list(info) for info in self.detect_info] # Get a copy

        self.trigger_times = [[] for _ in range(len(self.detectors))] # Reset for next interval
        self.detect_info = [[] for _ in range(len(self.detectors))]
        
        # Original code returned (trigger_times, detect_info) or just trigger_times.
        # This depends on what upstream expects. Returning trigger_times for consistency with QSDetector base.
        # The user's query implied only trigger_times from get_photon_times().
        return trigger_times_copy # Returning trigger_times directly as per QSDetector base

    def set_basis_list(self, basis_list: list[int], start_time: int, frequency: float) -> None:
        """Does nothing for this class (not applicable for Fock interference measurement)."""
        logger.debug(f"[{self.name}] set_basis_list called, but not applicable for FockInterference.")
        pass

    def set_phase(self, phase: float):
        """Sets the relative phase between the two input optical paths and regenerates POVMs."""
        self.phase = phase
        self._generate_povms()


class FockDetector(Detector): # Inherits from Detector, so it also gets stochastic params
    """Class modeling a Fock detector.
    A Fock detector can detect the number of photons in a given mode.
    This class is often used for specific testing or custom measurements,
    not typically part of the standard QSDetector hierarchy.

    Attributes:
        name (str): name of the detector
        timeline (Timeline): the simulation timeline
        efficiency (float): the efficiency of the detector (base value)
        wavelength (int): wave length in nm
        photon_counter (int): counting photon for the non-ideal detector (actual clicks)
        photon_counter2 (int): counting photon for the ideal detector (theoretical clicks)
        measure_protocol: Custom measurement protocol, if any.
    """

    def __init__(self, name: str, timeline: "Timeline", efficiency: float, wavelength: int = 0, stochastic_config: dict = None):
        # Pass stochastic_config to base Detector class
        super().__init__(name, timeline, efficiency=efficiency, stochastic_config=stochastic_config)
        self.name = name # Re-assigning name, though super() already does it.
        self.wavelength = wavelength
        self.encoding_type = fock # Specific to Fock encoding
        
        # Counters for specific use cases (ideal vs non-ideal)
        self.photon_counter = 0 # Non-ideal clicks (affected by efficiency)
        self.photon_counter2 = 0 # Ideal clicks (not affected by efficiency)
        self.measure_protocol = None # Placeholder for a measurement protocol

    def init(self):
        """Initializes internal counters and calls parent init to handle stochastic setup."""
        super().init() # Calls Detector.init() to set up dark counts etc.
        self.photon_counter = 0
        self.photon_counter2 = 0
        # self.measure_protocol = None # Re-initialize if needed


    def get(self, photon: Photon = None, **kwargs) -> int: # Changed return type to int from None
        """
        Not ideal detector. There is a chance for photon loss due to efficiency.
        Uses the base Detector's `_update_stochastic_parameters` and `_current_efficiency_value`.
        """
        super()._update_stochastic_parameters(self.timeline.now()) # Update stochastic parameters
        
        if photon and not photon.is_null and np.random.random() < self._current_efficiency_value:
            self.photon_counter += 1
            logger.debug(f"[{self.name}] Fock (real) detected {self.photon_counter} photons.")
        else:
            logger.debug(f"[{self.name}] Fock (real) did not detect photon.")
        return self.photon_counter # Return count, even if 0

    def get_2(self, photon: Photon = None, **kwargs) -> int: #IDEAL
        """Ideal detector, no photon loss."""
        # This method bypasses the efficiency logic of the base Detector's get(),
        # thus it remains "ideal" in terms of efficiency.
        if photon and not photon.is_null: # Ensure it's a valid photon
            self.photon_counter2 += 1
            logger.debug(f"[{self.name}] Fock (ideal) detected {self.photon_counter2} photons.")
        return self.photon_counter2

    def getx2(self, photon: Photon = None, **kwargs) -> int:
        """Simulates two detection attempts for a single photon (non-ideal)."""
        super()._update_stochastic_parameters(self.timeline.now())
        
        if photon and not photon.is_null:
            if np.random.random() < self._current_efficiency_value:
                self.photon_counter += 1
            if np.random.random() < self._current_efficiency_value: # Second attempt
                self.photon_counter += 1
        return self.photon_counter

    def get_2x2(self, photon: Photon = None, **kwargs) -> int: #IDEAL
        """Simulates two ideal detection attempts for a single photon."""
        if photon and not photon.is_null:
            self.photon_counter2 += 2
        return self.photon_counter2

    def measure(self, photon: Photon) -> tuple[int, int, int, int]:
        """
        Custom measure method for Fock detectors, comparing ideal vs real counts.
        Note: This logic is likely intended to be called *after* `get` or `get_2`
        have been called for one or more photons. It's not a detector input method.
        """
        # These are local temporary counters for this specific `measure` call
        detector_photon_counter_ideal = 0
        spd_ideal = 0
        detector_photon_counter_real = 0
        spd_real = 0

        # Logic based on the accumulated counts in photon_counter and photon_counter2
        if self.photon_counter2 >= 1: # If at least one ideal photon was registered
            detector_photon_counter_ideal = 1 # A "click" for ideal
        if self.photon_counter2 >= 1: # Redundant check with above, assuming it's a different concept
            spd_ideal = 1 # Another click concept

        if self.photon_counter >= 1: # If at least one real photon was registered
            detector_photon_counter_real = 1
        if self.photon_counter >= 1: # Redundant check
            spd_real = 1

        # Reset internal counters after measurement if this is a "readout"
        # self.photon_counter = 0
        # self.photon_counter2 = 0

        return detector_photon_counter_ideal, spd_ideal, detector_photon_counter_real, spd_real 
    
    def received_message(self, src: str, msg: Any):
        """Handles incoming messages (placeholder)."""
        logger.debug(f"[{self.name}] Received message from {src}: {msg}")

