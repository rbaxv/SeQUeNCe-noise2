# File: run_simulation_example.py
#
# This script demonstrates how to set up and run a quantum network simulation
# using the modified SeQUeNCe components (Memory, Detector, BSM) that
# incorporate non-Markovian noise models and stochastic parameter variations.
#
# It utilizes the Timeline for event scheduling and the
# SimulationTopologyBuilder for network construction from a configuration.
#
# To run this script, ensure you have the modified SeQUeNCe codebase
# with the changes we've implemented.

import logging
import json
import os
import qutip # Used for quantum state analysis (e.g., fidelity calculation)
from typing import Optional
import numpy as np

# --- Import Core SeQUeNCe Components ---
from sequence.kernel.timeline import Timeline
from sequence.network_management.simulation_topology_builder import SimulationTopologyBuilder
from sequence.kernel.quantum_manager import QuantumManagerKet
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.network_management.reservation import Reservation
from sequence.message import Message

# Configure logging for better visibility during simulation run
logging.basicConfig(level=logging.INFO, # Set to INFO for clean output
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("SimulationRunner")

# --- 1. Define Network Configuration (as a Python dictionary for simplicity) ---
# In a real scenario, this would typically be loaded from a JSON file.
# This configuration includes examples of noise_model for Memory and
# stochastic parameters for Detector and BSM.

# Ensure constants are defined or imported if needed by components
class Constants:
    CARRIAGE_RETURN = '\r'
    SLEEP_SECONDS = 0.1
    NANOSECONDS_PER_MILLISECOND = 1_000_000.0
    PICOSECONDS_PER_NANOSECOND = 1_000.0

# Temporarily patch constants if not in sequence/constants.py
try:
    from sequence.constants import CARRIAGE_RETURN, SLEEP_SECONDS, \
                                   NANOSECONDS_PER_MILLISECOND, PICOSECONDS_PER_NANOSECOND
except ImportError:
    # If not found, use a fallback for constants defined above
    import sys
    sys.modules['sequence.constants'] = Constants()
    logger.warning("sequence.constants not found, using internal default constants for demo.")


# Example Network Configuration with Noise and Stochastic Parameters
NETWORK_CONFIG = {
    "quantum_routers": [
        {
            "name": "NodeA",
            "memory_array": {
                "name": "NodeA.mem_array",
                "num_memories": 2, # Two memories for testing
                "coherence_time": 1000000.0, # Long coherence for base
                "fidelity": 0.98,
                "noise_model": {
                    "type": "EnhancedNonMarkovian",
                    "gamma0": 1e-4,      # Initial decay rate (e.g., 10^-4 1/ps)
                    "tau_corr": 5000.0,  # Correlation time (e.g., 5000 ps = 5 ns)
                    "d_mem": 20,         # Depth of memory history buffer
                    "f_sens": 0.1,       # Sensitivity to historical fidelity deviations
                    "k_steep": 5.0,      # Steepness of fidelity function
                    "f_thresh": 0.9,     # Fidelity threshold for non-Markovian effects
                    "lambda_time": 1e-9, # Fatigue rate per unit time (1/ps)
                    "lambda_event": 5e-5, # Fatigue rate per event
                    "s_fatigue": 0.5,    # Sensitivity to fatigue (exponent)
                    "s_mem": 0.2         # Sensitivity to memory history (exponent)
                }
            },
            "detectors": [ # Standalone detector on NodeA
                {
                    "name": "NodeA.det0",
                    "efficiency": 0.95,
                    "dark_count": 100, # Hz
                    "time_resolution": 50, # ps
                    "efficiency_stochastic": {
                        "stochastic_variation_enabled": True,
                        "variation_type": "gaussian_jitter",
                        "sigma_jitter": 0.01, # 1% std dev
                        "min_clip": 0.8,
                        "max_clip": 1.0,
                        "update_frequency": "per_event"
                    },
                    "dark_count_stochastic": {
                        "stochastic_variation_enabled": True,
                        "variation_type": "linear_drift",
                        "drift_rate": 0.0001, # Linear increase per ps
                        "drift_period": 1000000, # 1 microsecond period for drift cycle
                        "min_clip": 50,
                        "max_clip": 200,
                        "update_frequency": 100000 # Update every 100 ns
                    }
                }
            ]
        },
        {
            "name": "NodeB",
            "memory_array": {
                "name": "NodeB.mem_array",
                "num_memories": 2,
                "coherence_time": 1000000.0,
                "fidelity": 0.98,
                "noise_model": {
                    "type": "NonMarkovianAmplitudeDamping",
                    "gamma_base": 1e-5,    # Base amplitude damping rate (1/ps)
                    "f_stab": 0.05,        # Fidelity stabilization factor
                    "s_recoh": 0.7         # Recoherence sensitivity
                }
            }
        },
        {
            "name": "RouterC",
            "bsms": [
                {
                    "name": "RouterC.bsm0",
                    "encoding_type": "polarization",
                    "efficiency": 0.9, # BSM's own base efficiency
                    "time_resolution": 100, # BSM's own base time resolution
                    "stochastic_config": { # Stochastic for BSM's parameters
                        "efficiency": {
                            "stochastic_variation_enabled": True,
                            "variation_type": "periodic_variation",
                            "amplitude": 0.05,
                            "period": 500000, # 500 ns period
                            "offset": 0.0,
                            "min_clip": 0.7,
                            "max_clip": 1.0,
                            "update_frequency": "per_event"
                        }
                    },
                    "detectors": [ # Internal detectors for BSM (will use their own stochastic)
                        {"name": "RouterC.bsm0.det0", "efficiency": 0.99, "dark_count": 50, "time_resolution": 30},
                        {"name": "RouterC.bsm0.det1", "efficiency": 0.99, "dark_count": 50, "time_resolution": 30},
                        {"name": "RouterC.bsm0.det2", "efficiency": 0.99, "dark_count": 50, "time_resolution": 30},
                        {"name": "RouterC.bsm0.det3", "efficiency": 0.99, "dark_count": 50, "time_resolution": 30}
                    ]
                }
            ]
        }
    ],
    "links": [
        {
            "name": "LinkAC",
            "source": "NodeA",
            "target": "RouterC",
            "length": 20000, # 20 km
            "attenuation": 0.0002 # dB/km
        },
        {
            "name": "LinkBC",
            "source": "NodeB",
            "target": "RouterC",
            "length": 20000, # 20 km
            "attenuation": 0.0002 # dB/km
        }
    ]
}

# --- 2. Data Collection Setup ---
# A simple list to store log entries for post-simulation analysis
simulation_logs = []

def collect_data_callback(current_time: int):
    try:
        # Get the memory arrays directly as entities
        mem_array_a = timeline_instance.get_entity_by_name("NodeA.MemoryArray")
        mem_array_b = timeline_instance.get_entity_by_name("NodeB.MemoryArray")
        
        mem0_a = None
        mem0_b = None
        
        # Get the first memory from each array
        if mem_array_a and hasattr(mem_array_a, 'memories') and len(mem_array_a.memories) > 0:
            mem0_a = mem_array_a.memories[0]
            
        if mem_array_b and hasattr(mem_array_b, 'memories') and len(mem_array_b.memories) > 0:
            mem0_b = mem_array_b.memories[0]
        
        # Try different fidelity attributes that might exist
        fidelity_a = None
        fidelity_b = None
        
        if mem0_a:
            # Try various fidelity attributes
            for attr_name in ['_current_fidelity_nm', 'fidelity', 'raw_fidelity', 'current_fidelity']:
                if hasattr(mem0_a, attr_name):
                    fidelity_a = getattr(mem0_a, attr_name)
                    break
        
        if mem0_b:
            # Try various fidelity attributes
            for attr_name in ['_current_fidelity_nm', 'fidelity', 'raw_fidelity', 'current_fidelity']:
                if hasattr(mem0_b, attr_name):
                    fidelity_b = getattr(mem0_b, attr_name)
                    break
        
        # Always log the data, even if None, so we can see what's happening
        simulation_logs.append({
            "time_ps": current_time,
            "NodeA_fidelity": fidelity_a,
            "NodeB_fidelity": fidelity_b,
        })
        
        # Log when we find actual fidelity values
        if fidelity_a is not None or fidelity_b is not None:
            logger.info(f"FIDELITY DATA: Time {current_time} ps: NodeA={fidelity_a}, NodeB={fidelity_b}")
            
    except Exception as e:
        logger.error(f"Error during data collection callback at {current_time} ps: {e}")

# --- 3. Main Simulation Execution Logic ---
def run_simulation(stop_time_ps: int, random_seed: Optional[int] = None):
    """
    Executes the quantum network simulation.
    Args:
        stop_time_ps (int): The simulation stop time in picoseconds.
        random_seed (Optional[int]): Seed for reproducibility.
    """
    global timeline_instance # Make timeline_instance accessible to the callback
    timeline_instance = Timeline(stop_time=stop_time_ps, seed=random_seed)

    # Set up QuantumManager before building network components
    quantum_manager = QuantumManagerKet(timeline_instance)
    timeline_instance.set_quantum_manager(quantum_manager)

    builder = SimulationTopologyBuilder(timeline_instance)
    
    # Save config to a temporary file, then load it (simulating loading from a file)
    temp_config_path = "temp_network_config.json"
    try:
        with open(temp_config_path, "w") as f:
            json.dump(NETWORK_CONFIG, f, indent=4)
        builder.load_config(temp_config_path)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path) # Clean up temporary file

    builder.build_network()

    # Initialize all components (e.g., nodes, routers, their internal components)
    timeline_instance.init()

    # --- FORCE MEMORY UPDATE FOR TESTING ---
    mem_array_a = timeline_instance.get_entity_by_name("NodeA.MemoryArray")
    mem_array_b = timeline_instance.get_entity_by_name("NodeB.MemoryArray")
    if mem_array_a and hasattr(mem_array_a, 'memories') and len(mem_array_a.memories) > 0:
        mem0_a = mem_array_a.memories[0]
        # Create a random pure state
        rand_state_a = qutip.rand_ket(2).proj()
        mem0_a.update_state(rand_state_a, timeline_instance.now(), ideal_state=qutip.basis(2, 0).proj())
        logger.info(f"[FORCE UPDATE] NodeA.MemoryArray[0] forced update_state at {timeline_instance.now()} ps")
    if mem_array_b and hasattr(mem_array_b, 'memories') and len(mem_array_b.memories) > 0:
        mem0_b = mem_array_b.memories[0]
        rand_state_b = qutip.rand_ket(2).proj()
        mem0_b.update_state(rand_state_b, timeline_instance.now(), ideal_state=qutip.basis(2, 0).proj())
        logger.info(f"[FORCE UPDATE] NodeB.MemoryArray[0] forced update_state at {timeline_instance.now()} ps")
    # --- END FORCE MEMORY UPDATE ---

    # --- NEW: Schedule Initial Events to Kickstart Simulation ---
    try:
        node_a = timeline_instance.get_entity_by_name("NodeA")
        node_b = timeline_instance.get_entity_by_name("NodeB")
        router_c = timeline_instance.get_entity_by_name("RouterC")

        logger.info("Scheduling initial entanglement generation attempts.")

        # Create dummy reservations for demonstration
        # In a real scenario, an application would generate this
        dummy_reservation_a = Reservation("NodeA", "RouterC", timeline_instance.now(), timeline_instance.stop_time, 1, 0.9, 1, 1001)
        dummy_reservation_b = Reservation("NodeB", "RouterC", timeline_instance.now(), timeline_instance.stop_time, 1, 0.9, 1, 1002)

        # Schedule an event for NodeA to request entanglement
        class RequestEntanglementEvent(Event):
            def __init__(self, timeline, node, dst_node_name, reservation, time):
                super().__init__(time, Process(self, "execute_request", "RequestEntanglement"))
                self.node = node
                self.dst_node_name = dst_node_name
                self.reservation = reservation

            def execute_request(self):
                logger.info(f"DEBUG: execute_request called for {self.node.name} at time {timeline_instance.now()} ps")
                timeline_instance.logger.info(f"[{self.node.name}] Requesting entanglement with {self.dst_node_name} for reservation ID {self.reservation.request_id} at {timeline_instance.now()} ps.")
                # This calls the NetworkManager of the node
                if hasattr(self.node, 'reserve_net_resource'):
                     logger.info(f"DEBUG: Calling reserve_net_resource on {self.node.name}")
                     self.node.reserve_net_resource(self.dst_node_name,
                                                    self.reservation.start_time,
                                                    self.reservation.end_time,
                                                    self.reservation.memory_size,
                                                    self.reservation.fidelity,
                                                    self.reservation.entanglement_number,
                                                    self.reservation.request_id)
                     logger.info(f"DEBUG: reserve_net_resource call completed for {self.node.name}")
                else:
                    timeline_instance.logger.error(f"Node {self.node.name} does not have 'reserve_net_resource' method.")

        # Schedule the requests for NodeA and NodeB
        # Give them a small initial delay
        request_start_time = 1000 # ps
        logger.info(f"DEBUG: Scheduling events at time {request_start_time} ps, stop_time is {timeline_instance.stop_time} ps")
        
        event_a = RequestEntanglementEvent(timeline_instance, node_a, "RouterC", dummy_reservation_a, request_start_time)
        event_b = RequestEntanglementEvent(timeline_instance, node_b, "RouterC", dummy_reservation_b, request_start_time + 500) # Offset for NodeB
        
        logger.info(f"DEBUG: Created event_a: time={event_a.time}, process.owner={event_a.process.owner.name}")
        logger.info(f"DEBUG: Created event_b: time={event_b.time}, process.owner={event_b.process.owner.name}")
        
        timeline_instance.schedule(event_a)
        timeline_instance.schedule(event_b)
        
        logger.info(f"DEBUG: After scheduling, timeline has {len(timeline_instance.events)} events in queue")
        if timeline_instance.events:
            next_event = timeline_instance.events[0]
            logger.info(f"DEBUG: Next event in queue: time={next_event.time}, owner={next_event.process.owner.name if hasattr(next_event.process, 'owner') else 'None'}")

        logger.info("Initial entanglement requests scheduled.")

    except Exception as e:
        logger.error(f"Error scheduling initial events: {e}")

    # Register the data collection callback
    timeline_instance.add_callback(collect_data_callback)

    # Schedule a simple periodic event to make the simulation actually run
    def periodic_logger(current_time: int):
        """Simple periodic event to keep the simulation running."""
        if current_time % 1000000 == 0:  # Every 1 microsecond
            logger.info(f"Simulation time: {current_time} ps")
    
    timeline_instance.add_callback(periodic_logger)

    # Print all event times and owners in the queue before running the simulation
    logger.info("DEBUG: Listing all events in the timeline queue before run:")
    for idx, event in enumerate(timeline_instance.events):
        owner = getattr(event.process, 'owner', None)
        owner_name = owner.name if owner and hasattr(owner, 'name') else str(owner)
        logger.info(f"DEBUG: Event {idx}: time={event.time}, owner={owner_name}, activation={getattr(event.process, 'activation', None)}")

    logger.info("Starting simulation run...")
    timeline_instance.run()
    logger.info("Simulation run finished.")

    # --- 4. Post-Simulation Analysis (Example) ---
    logger.info("--- Simulation Results Summary ---")
    logger.info(f"Total events scheduled: {timeline_instance.schedule_counter}")
    logger.info(f"Total events executed: {timeline_instance.run_counter}")
    logger.info(f"Final simulation time: {timeline_instance.now()} ps")

    # Example: Print collected memory fidelity logs
    logger.info(f"Collected {len(simulation_logs)} memory state update logs.")
    for entry in simulation_logs:
        logger.info(f"Time: {entry['time_ps']} ps, NodeA Fidelity: {entry['NodeA_fidelity']}, NodeB Fidelity: {entry['NodeB_fidelity']}")

    # --- Plotting Section ---
    try:
        import matplotlib.pyplot as plt
        
        logger.info(f"DEBUG: Total simulation_logs entries: {len(simulation_logs)}")
        
        # Show some sample data
        for i, entry in enumerate(simulation_logs[:5]):  # Show first 5 entries
            logger.info(f"DEBUG: Entry {i}: {entry}")
        
        # Filter out None values for plotting
        valid_entries = [entry for entry in simulation_logs if entry["NodeA_fidelity"] is not None or entry["NodeB_fidelity"] is not None]
        logger.info(f"DEBUG: Valid entries with non-None fidelity: {len(valid_entries)}")
        
        if not valid_entries:
            logger.warning("No valid fidelity data found. Creating empty plot with debug info.")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"No fidelity data collected\nTotal entries: {len(simulation_logs)}\nCheck debug logs for details", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.xlabel("Time (ps)")
            plt.ylabel("Fidelity")
            plt.title("Memory Fidelity Over Time (No Data)")
            plt.tight_layout()
            plt.savefig("fidelity_plot.png")
            plt.show()
            plt.close()
            logger.info("Empty plot saved as fidelity_plot.png")
            return
        
        times = [entry["time_ps"] for entry in valid_entries]
        fidelity_a = [entry["NodeA_fidelity"] for entry in valid_entries]
        fidelity_b = [entry["NodeB_fidelity"] for entry in valid_entries]
        
        logger.info(f"DEBUG: Plotting {len(times)} data points")
        logger.info(f"DEBUG: Time range: {min(times)} to {max(times)} ps")
        logger.info(f"DEBUG: NodeA fidelity range: {min(fidelity_a) if fidelity_a else 'N/A'} to {max(fidelity_a) if fidelity_a else 'N/A'}")
        logger.info(f"DEBUG: NodeB fidelity range: {min(fidelity_b) if fidelity_b else 'N/A'} to {max(fidelity_b) if fidelity_b else 'N/A'}")
        
        plt.figure(figsize=(10, 6))
        if any(f is not None for f in fidelity_a):
            plt.plot(times, fidelity_a, label="NodeA Memory 0 Fidelity", marker='o', markersize=3)
        if any(f is not None for f in fidelity_b):
            plt.plot(times, fidelity_b, label="NodeB Memory 0 Fidelity", marker='s', markersize=3)
        
        plt.xlabel("Time (ps)")
        plt.ylabel("Fidelity")
        plt.title("Memory Fidelity Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("fidelity_plot.png")
        plt.show()  # Display the plot window
        plt.close()
        logger.info("Fidelity plot saved as fidelity_plot.png")
    except Exception as e:
        logger.error(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()


# --- Run the simulation ---
if __name__ == "__main__":
    # Example: Run for 50 seconds (50 billion picoseconds) with a fixed seed
    STOP_TIME_PS = 50_000_000_000  # Increased to 50 seconds (50 billion ps) for comprehensive run
    RANDOM_SEED = 42  # For reproducibility

    run_simulation(STOP_TIME_PS, RANDOM_SEED)

