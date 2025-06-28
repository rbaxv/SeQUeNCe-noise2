# File: sequence/components/noise_models.py
#
# This file defines the core noise models used in the SeQUeNCe quantum network simulator,
# including Enhanced Non-Markovian (ENM), Non-Markovian Amplitude Damping (NMAD),
# and a utility mixin for stochastic parameter variations.

import qutip
from collections import deque
import numpy as np
import math
import logging

# Set up logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add a handler if not already configured (e.g., for direct script execution)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NoiseModel:
    """
    Abstract base class for all noise models in SeQUeNCe.
    All custom noise models should inherit from this class.
    """
    def __init__(self, name: str):
        self.name = name

    def apply_noise(self, qubit_state: qutip.Qobj, elapsed_time: float, *args, **kwargs) -> qutip.Qobj:
        """
        Applies noise to a qubit state over an elapsed time.
        This method must be implemented by subclasses.

        Args:
            qubit_state (qutip.Qobj): The current quantum state of the qubit (density matrix).
            elapsed_time (float): The time duration over which the noise is applied (in simulation time units).
            *args, **kwargs: Additional arguments specific to the noise model (e.g., memory object for history).

        Returns:
            qutip.Qobj: The evolved quantum state after applying noise.
        """
        raise NotImplementedError(f"apply_noise method not implemented for {self.__class__.__name__}")

class EnhancedNonMarkovianModel(NoiseModel):
    """
    Implements an Enhanced Non-Markovian (ENM) noise model that accounts for both
    memory (correlation with past states) and fatigue (cumulative operations) effects.

    The effective noise rate (gamma_eff) is modulated by:
    gamma_eff = gamma0 * exp(s_fatigue * I_fatigue - s_mem * I_mem)

    Where:
    - I_mem captures memory effects based on historical fidelity.
    - I_fatigue captures fatigue effects based on active time and event count.
    """
    def __init__(self, name: str, gamma0: float, tau_corr: float, d_mem: int,
                 f_sens: float, k_steep: float, f_thresh: float,
                 lambda_time: float, lambda_event: float, s_fatigue: float, s_mem: float):
        """
        Initializes the Enhanced Non-Markovian (ENM) noise model.

        Args:
            name (str): Unique name for the noise model instance.
            gamma0 (float): Base noise strength (e.g., 1e-4 /ns).
                            Typical values might range from 1e-5 to 1e-3, representing a base decay rate.
            tau_corr (float): Correlation time window (in simulation time units, e.g., ns).
                              States older than this time are ignored for memory calculation.
                              Values like 100 ns - 1000 ns are common for memory effects.
                              A larger tau_corr means more history is considered relevant.
            d_mem (int): Maximum number of historical states to store in the memory buffer.
                         Trade-off between accuracy and memory usage. Values like 10-100 are reasonable.
                         This sets the `maxlen` for the deque in Memory.
            f_sens (float): Fidelity sensitivity. How strongly historical fidelity influences I_mem.
                            Higher values (e.g., 0.5 - 2.0) mean greater impact.
            k_steep (float): Steepness factor for the tanh function in I_mem calculation.
                             Controls how sharply I_mem changes around F_thresh.
                             Higher values (e.g., 5.0 - 15.0) create a sharper transition.
            f_thresh (float): Fidelity threshold. Historical fidelity below this value starts to penalize I_mem.
                              Values (e.g., 0.8 - 0.95) depend on desired fidelity performance.
                              Adjusting this shifts the "onset" of the memory-induced degradation.
            lambda_time (float): Weight for accumulated active time in fatigue (units of 1/time^2, e.g., 1e-8 /ns^2).
                                 Determines how quickly fatigue builds with active usage duration.
                                 Small values (e.g., 1e-8 to 1e-6) are typically sufficient due to cumulative nature.
            lambda_event (float): Weight for number of processed events in fatigue (unitless, e.g., 1e-3 per event).
                                  Determines how quickly fatigue builds with each operation.
                                  Values (e.g., 1e-4 to 1e-2) indicate event impact.
            s_fatigue (float): Scaling factor for the overall fatigue effect in the exponent.
                               Higher values (e.g., 1.0 - 5.0) mean fatigue leads to faster degradation.
            s_mem (float): Scaling factor for the overall memory effect in the exponent.
                           Higher values (e.g., 1.0 - 5.0) mean good memory leads to stronger mitigation.
        """
        super().__init__(name)
        self.gamma0 = gamma0
        self.tau_corr = tau_corr
        self.d_mem = d_mem
        self.f_sens = f_sens
        self.k_steep = k_steep
        self.f_thresh = f_thresh
        self.lambda_time = lambda_time
        self.lambda_event = lambda_event
        self.s_fatigue = s_fatigue
        self.s_mem = s_mem

    def apply_noise(self, qubit_state: qutip.Qobj, elapsed_time: float, memory_obj: 'Memory') -> qutip.Qobj:
        """
        Applies non-Markovian noise to a qubit state based on its memory's history and fatigue.

        Args:
            qubit_state (qutip.Qobj): The current quantum state of the qubit (density matrix).
            elapsed_time (float): The time duration over which the noise is applied.
            memory_obj ('Memory'): A reference to the Memory instance, providing access to history
                                   attributes like _history_buffer, _total_active_time, _num_events_processed.

        Returns:
            qutip.Qobj: The evolved quantum state after applying the dynamically calculated noise.
        """
        if elapsed_time <= 0:
            return qubit_state

        try:
            current_time = memory_obj._timeline.now()

            # 1. Retrieve & Filter History for F_hist calculation
            # memory_obj._history_buffer is already constrained by d_mem in Memory.update_state.
            # Here, we filter further by tau_corr.
            relevant_history = deque()
            for hist_state_qobj, hist_timestamp in memory_obj._history_buffer:
                if current_time - hist_timestamp < self.tau_corr:
                    relevant_history.append((hist_state_qobj, hist_timestamp))

            # 2. Calculate F_hist(S_{current}) - Historical Fidelity
            # F_hist(S_current) = (1/|H_t|) * sum_{ (S',t') in H_t } fidelity(S_current, S')
            f_hist = 1.0  # Default to 1.0 if no relevant history
            if relevant_history:
                fidelities = []
                for hist_state_from_buffer, _ in relevant_history:
                    # Performance optimization: Quick check with trace distance before full fidelity
                    # Trace distance ranges from 0 (identical) to 1 (maximally different).
                    # If states are far apart (trace_dist > a threshold, e.g., 0.5), fidelity will be low.
                    # Fidelity = 1 - trace_dist for pure states, but more complex for mixed.
                    # A threshold like (1 - self.f_thresh + some_buffer) can pre-filter.
                    # This is a heuristic to skip expensive fidelity calculation for very different states.
                    if qutip.metrics.tracedist(qubit_state, hist_state_from_buffer) > 0.5: # Example threshold
                        fidelities.append(0.0) # Assume low fidelity
                    else:
                        fidelities.append(qutip.fidelity(qubit_state, hist_state_from_buffer))
                f_hist = np.mean(fidelities)
                
            # 3. Calculate I_mem (Memory Influence Factor)
            # I_mem = f_sens * tanh[k_steep * (F_hist - F_thresh)]
            I_mem = self.f_sens * math.tanh(self.k_steep * (f_hist - self.f_thresh))

            # 4. Retrieve Fatigue Data from Memory Object
            delta_t_active = memory_obj._total_active_time
            n_events = memory_obj._num_events_processed

            # 5. Calculate I_fatigue (Fatigue Influence Factor)
            # I_fatigue = lambda_time * Delta_t_active + lambda_event * N_events
            I_fatigue = self.lambda_time * delta_t_active + self.lambda_event * n_events

            # 6. Calculate gamma_eff (Effective Noise Rate)
            # gamma_eff = gamma0 * exp(s_fatigue * I_fatigue - s_mem * I_mem)
            gamma_eff = self.gamma0 * math.exp(self.s_fatigue * I_fatigue - self.s_mem * I_mem)
            gamma_eff = max(0.0, gamma_eff) # Ensure decay rate is non-negative

            # 7. Apply Noise using Lindblad Master Equation
            # For amplitude damping (qubit decaying from |1> to |0>)
            # This is commonly `sqrt(gamma) * sigma_minus`.
            c_ops = [np.sqrt(gamma_eff) * qutip.sigmam()]

            # For depolarizing noise, use three collapse operators:
            # c_ops = [np.sqrt(gamma_eff/3) * qutip.sigmax(),
            #          np.sqrt(gamma_eff/3) * qutip.sigmay(),
            #          np.sqrt(gamma_eff/3) * qutip.sigmaz()]
            # The type of noise (amplitude damping, depolarizing) should be a config parameter.
            # For this example, assuming amplitude damping as the primary non-Markovian channel.

            # Solve Master Equation for the elapsed time
            result = qutip.mesolve(qubit_state, [], [0, elapsed_time], c_ops=c_ops,
                                   options=qutip.Options(nsteps=10000)) # Increase nsteps for stability
            return result.states[-1] # Return the final state after evolution

        except Exception as e:
            logger.warning(f"[{self.name}] Enhanced Non-Markovian noise application failed: {e}. "
                           "Falling back to base Markovian decay for this step.")
            # Fallback: Apply a simple Markovian decay if the complex calculation fails
            # This uses gamma0 as a fallback rate, assuming amplitude damping.
            c_ops_fallback = [np.sqrt(self.gamma0) * qutip.sigmam()]
            result_fallback = qutip.mesolve(qubit_state, [], [0, elapsed_time], c_ops=c_ops_fallback)
            return result_fallback.states[-1]


class NonMarkovianAmplitudeDampingModel(NoiseModel):
    """
    Implements a Non-Markovian Amplitude Damping (NMAD) model that
    modulates the damping rate based on recent fidelity changes (recoherence effects).

    The effective damping rate (gamma_prime_eff) is:
    gamma_prime_eff = gamma_base * max(0, 1 - s_recoh * M_recoh)

    Where:
    - M_recoh captures the degree of fidelity "recoherence" or "stability."
    """
    def __init__(self, name: str, gamma_base: float, f_stab: float, s_recoh: float):
        """
        Initializes the Non-Markovian Amplitude Damping (NMAD) noise model.

        Args:
            name (str): Unique name for the noise model instance.
            gamma_base (float): Base amplitude damping rate (e.g., 1e-4 /ns).
                                 This is the maximum damping rate.
            f_stab (float): Fidelity stability threshold (e.g., 0.95).
                            If current_fidelity / previous_fidelity is above this, recoherence starts to kick in.
                            Higher f_stab means the system needs to be more stable to experience recoherence.
            s_recoh (float): Recoherence scaling parameter (e.g., 2.0 - 10.0).
                             Determines the strength of the recoherence effect.
                             Higher s_recoh means stronger reduction in damping rate.
        """
        super().__init__(name)
        self.gamma_base = gamma_base
        self.f_stab = f_stab
        self.s_recoh = s_recoh

    def apply_noise(self, qubit_state: qutip.Qobj, elapsed_time: float, memory_obj: 'Memory') -> qutip.Qobj:
        """
        Applies Non-Markovian Amplitude Damping noise based on fidelity changes.

        Args:
            qubit_state (qutip.Qobj): The current quantum state of the qubit (density matrix).
            elapsed_time (float): The time duration over which the noise is applied.
            memory_obj ('Memory'): A reference to the Memory instance, providing access to
                                   _current_fidelity_nm and _previous_fidelity_nm.

        Returns:
            qutip.Qobj: The evolved quantum state after applying dynamic amplitude damping.
        """
        if elapsed_time <= 0:
            return qubit_state

        try:
            # 1. Retrieve Fidelities from Memory Object
            current_fidelity = memory_obj._current_fidelity_nm
            previous_fidelity = memory_obj._previous_fidelity_nm

            # 2. Calculate M_recoh (Recoherence Measure)
            # M_recoh = max(0, fidelity(S_current, S_prev) - F_stab)
            # Here, S_current is `qubit_state`. S_prev is the state that led to `previous_fidelity_nm`.
            # A direct fidelity(S_current, S_prev) is ideal but `S_prev` isn't directly passed here.
            # We use the ratio of fidelities to the ideal state as an approximation of stability.
            fidelity_ratio = 1.0 # Default if previous_fidelity is zero
            if previous_fidelity > 1e-9: # Avoid division by zero for initial states
                fidelity_ratio = current_fidelity / previous_fidelity
            
            M_recoh = max(0.0, fidelity_ratio - self.f_stab)

            # 3. Calculate gamma_prime_eff (Effective Damping Rate)
            # gamma_prime_eff = gamma_base * max(0, 1 - s_recoh * M_recoh)
            gamma_prime_eff = self.gamma_base * max(0.0, 1 - self.s_recoh * M_recoh)
            gamma_prime_eff = max(0.0, gamma_prime_eff) # Ensure decay rate is non-negative

            # 4. Apply Damping
            # Amplitude damping collapse operator
            c_ops = [np.sqrt(gamma_prime_eff) * qutip.sigmam()]
            
            result = qutip.mesolve(qubit_state, [], [0, elapsed_time], c_ops=c_ops,
                                   options=qutip.Options(nsteps=10000)) # Increase nsteps for stability
            return result.states[-1]

        except Exception as e:
            logger.warning(f"[{self.name}] Non-Markovian Amplitude Damping application failed: {e}. "
                           "Falling back to base Markovian damping for this step.")
            # Fallback: Apply base amplitude damping if calculation fails
            c_ops_fallback = [np.sqrt(self.gamma_base) * qutip.sigmam()]
            result_fallback = qutip.mesolve(qubit_state, [], [0, elapsed_time], c_ops=c_ops_fallback)
            return result_fallback.states[-1]

class StochasticParameterMixin:
    """
    A mixin class providing a static method to generate dynamically varied
    parameter values based on configured stochastic models (Gaussian jitter,
    linear drift, periodic).
    """
    @staticmethod
    def get_stochastic_value(base_value: float, config: dict, current_time: float, last_update_time: float) -> float:
        """
        Calculates a stochastically varied parameter value.

        Args:
            base_value (float): The nominal or configured base value of the parameter.
            config (dict): A dictionary containing the stochastic configuration for this parameter.
                           Expected keys: "stochastic_variation_enabled", "variation_type",
                           "sigma_jitter", "drift_rate", "amplitude", "period", "offset",
                           "min_clip", "max_clip".
            current_time (float): The current simulation time (from Timeline).
            last_update_time (float): The last time this parameter's stochastic value was updated.
                                      Used for drift calculations.

        Returns:
            float: The dynamically varied parameter value.
        """
        if not config.get("stochastic_variation_enabled", False):
            return base_value

        variation_type = config.get("variation_type")
        value = base_value

        if variation_type == "gaussian_jitter":
            sigma = config.get("sigma_jitter", 0.0)
            value = np.random.normal(base_value, sigma)
        elif variation_type == "linear_drift":
            drift_rate = config.get("drift_rate", 0.0)
            # Drift applies over the total simulation time, not just elapsed since last update,
            # unless the intent is to simulate "per-step" drift.
            # Assuming drift over total time for a smoother trend.
            # If `last_update_time` is always the start of sim for drift,
            # then `current_time` itself is enough.
            value = base_value + drift_rate * current_time
        elif variation_type in ["periodic", "periodic_variation"]:
            amplitude = config.get("amplitude", 0.0)
            period = config.get("period", 1.0)
            offset = config.get("offset", 0.0) # Phase offset
            if period <= 0:
                logger.warning(f"Periodic variation for {config} has non-positive period. Using base value.")
                value = base_value
            else:
                value = base_value + amplitude * math.sin(2 * math.pi * (current_time + offset) / period)
        else:
            logger.warning(f"Unknown stochastic variation_type: {variation_type}. Using base value.")
            value = base_value

        # Apply clipping to keep value within physical bounds
        min_clip = config.get("min_clip", -np.inf)
        max_clip = config.get("max_clip", np.inf)
        value = np.clip(value, min_clip, max_clip)

        return value

