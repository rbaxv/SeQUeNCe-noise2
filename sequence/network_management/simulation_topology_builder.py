# File: sequence/network_management/simulation_topology_builder.py (Conceptual file path after rename)
#
# This file defines the SimulationTopologyBuilder class, which is responsible for
# building and configuring a quantum network simulation based on a provided
# network configuration. It acts as the central point for parsing high-level
# network definitions and instantiating all the necessary simulation components.
#
# Modifications in this version focus on:
# 1. Parsing and passing new noise model configurations to Memory components.
# 2. Parsing and passing stochastic parameter configurations to Detector components.
# 3. Ensuring that BSMs also receive relevant stochastic configurations for their internal parameters.
# 4. Renaming the class from NetworkManager to SimulationTopologyBuilder to clarify its role.

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline
    # Corrected import path for Node and QuantumRouter
    from ..topology.node import Node, QuantumRouter # Now importing QuantumRouter from node.py
    from ..components.bsm import BSM
    from ..components.memory import Memory, MemoryArray
    from ..components.detector import Detector
    from ..components.optical_channel import QuantumChannel


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SimulationTopologyBuilder: # Renamed from NetworkManager
    """
    Manages the construction and configuration of a quantum network simulation.
    It takes a network configuration (e.g., from a JSON file) and
    instantiates all nodes, links, and their internal components,
    passing relevant parameters including new noise models and stochastic settings.

    Attributes:
        _network_config (dict): The loaded network configuration.
        _timeline (Timeline): The simulation timeline instance.
    """

    def __init__(self, timeline: "Timeline"):
        """
        Constructor for the SimulationTopologyBuilder.

        Args:
            timeline (Timeline): The simulation timeline to which components will be added.
        """
        self._network_config: Dict[str, Any] = {}
        self._timeline = timeline
        logger.info("SimulationTopologyBuilder initialized.")

    def load_config(self, config_path: str) -> None:
        """
        Loads the network configuration from a JSON file.

        Args:
            config_path (str): The path to the JSON configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                self._network_config = json.load(f)
            logger.info(f"Network configuration loaded from: {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {config_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading config: {e}")
            raise

    def build_network(self) -> None:
        """
        Builds the quantum network based on the loaded configuration.
        This method orchestrates the creation of nodes, routers, and links,
        and their internal components (memories, detectors, BSMs).
        """
        if not self._network_config:
            logger.warning("No network configuration loaded. Skipping network build.")
            return

        logger.info("Building network from configuration...")

        # Create Nodes
        nodes_config = self._network_config.get("nodes", [])
        for node_params in nodes_config:
            self._create_node(node_params)
        logger.info(f"Created {len(self._timeline.entities)} entities (nodes, etc.).")

        # Create Quantum Routers (if defined separately)
        routers_config = self._network_config.get("quantum_routers", [])
        for router_params in routers_config:
            self._create_quantum_router(router_params)

        # Create Links (Optical Channels)
        links_config = self._network_config.get("links", [])
        for link_params in links_config:
            self._create_link(link_params)

        logger.info("Network build complete.")


    def _create_node(self, node_params: Dict[str, Any]) -> "Node":
        """
        Creates a Node component and its internal components (MemoryArray, Memory).
        Passes noise model configurations to Memory components.
        """
        from ..topology.node import Node # Local import from the correct file

        node_name = node_params["name"]
        logger.debug(f"Creating Node: {node_name}")
        
        node = Node(node_name, self._timeline)
        # Entity.__init__ already calls timeline.add_entity(self)

        # Handle memory array and individual memories
        memory_array_config = node_params.get("memory_array", {})
        if memory_array_config:
            self._create_memory_array(node, memory_array_config)
            
        return node


    def _create_memory_array(self, node: "Node", config: Dict[str, Any]) -> "MemoryArray":
        """
        Creates a MemoryArray and its constituent Memory objects.
        Crucially, this method now extracts and passes the `noise_model`
        configuration to the `Memory` constructors.
        """
        from ..components.memory import MemoryArray # Local import

        array_name = config.get("name", f"{node.name}.mem_array")
        num_memories = config.get("num_memories", 10)
        fidelity = config.get("fidelity", 1.0)
        frequency = config.get("frequency", 80e6)
        efficiency = config.get("efficiency", 1.0)
        coherence_time = config.get("coherence_time", -1)
        no_error = config.get("no_error", False)
        
        noise_model_config = config.get("noise_model")

        logger.debug(f"Creating MemoryArray for {node.name}: {array_name} with {num_memories} memories.")

        memory_array = MemoryArray(array_name, self._timeline,
                                   num_memories=num_memories,
                                   fidelity=fidelity,
                                   frequency=frequency,
                                   efficiency=efficiency,
                                   coherence_time=coherence_time,
                                   noise_model_config=noise_model_config) 
        # Entity.__init__ already calls timeline.add_entity(self)
        node.add_component(memory_array)
        
        return memory_array


    def _create_quantum_router(self, router_params: Dict[str, Any]) -> "QuantumRouter":
        """
        Creates a QuantumRouter component and its internal components (BSM, Detector).
        Passes stochastic parameter configurations to Detector and BSM components.
        """
        from ..topology.node import QuantumRouter # Local import from the correct file

        router_name = router_params["name"]
        logger.debug(f"Creating Quantum Router: {router_name}")
        
        router = QuantumRouter(router_name, self._timeline)
        # Entity.__init__ already calls timeline.add_entity(self)

        bsms_config = router_params.get("bsms", [])
        for bsm_params in bsms_config:
            self._create_bsm(router, bsm_params)
        
        detectors_config = router_params.get("detectors", [])
        for det_params in detectors_config:
            self._create_detector(router, det_params)

        return router


    def _create_bsm(self, owner_entity: Union["Node", "QuantumRouter"], config: Dict[str, Any]) -> "BSM":
        """
        Creates a BSM component and its internal Detector components.
        Passes BSM's own stochastic configurations and passes Detector-specific
        stochastic configurations to the Detectors it instantiates.
        """
        from ..components.bsm import BSM, make_bsm # Local import

        bsm_name = config.get("name", f"{owner_entity.name}.bsm")
        encoding_type = config.get("encoding_type", "time_bin")
        phase_error = config.get("phase_error", 0.0)
        delay = config.get("delay", 0)
        efficiency = config.get("efficiency", 1.0)
        time_resolution = config.get("time_resolution", 0.0)
        success_rate = config.get("success_rate", 0.5)

        bsm_stochastic_config = config.get("stochastic_config", {})
        
        # If efficiency and time_resolution are provided but not in stochastic_config,
        # add them to the stochastic_config for the BSM
        if efficiency != 1.0 and "efficiency" not in bsm_stochastic_config:
            bsm_stochastic_config["efficiency"] = {
                "stochastic_variation_enabled": False,
                "base_value": efficiency
            }
        if time_resolution != 0.0 and "time_resolution" not in bsm_stochastic_config:
            bsm_stochastic_config["time_resolution"] = {
                "stochastic_variation_enabled": False,
                "base_value": time_resolution
            }

        bsm_detectors_params = config.get("detectors", [])

        logger.debug(f"Creating BSM: {bsm_name} of type {encoding_type}.")

        bsm = make_bsm(bsm_name, self._timeline,
                       encoding_type=encoding_type,
                       phase_error=phase_error,
                       detectors=bsm_detectors_params,
                       stochastic_config=bsm_stochastic_config,
                       success_rate=success_rate)

        # Entity.__init__ already calls timeline.add_entity(self)
        logger.debug(f"Checking {owner_entity.name} (type: {type(owner_entity)}) for add_bsm method.")
        if hasattr(owner_entity, 'add_bsm'):
            owner_entity.add_bsm(bsm)
        else:
            logger.warning(f"Owner entity {owner_entity.name} of type {type(owner_entity)} does not have an 'add_bsm' method for {bsm_name}. BSM added to timeline only.")

        return bsm


    def _create_detector(self, owner_entity: Union["Node", "QuantumRouter", "BSM"], config: Dict[str, Any]) -> "Detector":
        """
        Creates a standalone Detector component.
        Crucially, this method now extracts and passes the `*_stochastic`
        configurations to the `Detector` constructor.
        """
        from ..components.detector import Detector # Local import

        detector_name = config.get("name", f"{owner_entity.name}.detector")
        efficiency = config.get("efficiency", 0.9)
        dark_count = config.get("dark_count", 0)
        count_rate = config.get("count_rate", 25e6)
        time_resolution = config.get("time_resolution", 150)

        detector_stochastic_config = {}
        if config.get("efficiency_stochastic"):
            detector_stochastic_config["efficiency"] = config["efficiency_stochastic"]
        if config.get("dark_count_stochastic"):
            detector_stochastic_config["dark_count_rate"] = config["dark_count_stochastic"]
        if config.get("time_resolution_stochastic"):
            detector_stochastic_config["time_resolution"] = config["time_resolution_stochastic"]
        
        if config.get("stochastic_config"):
            detector_stochastic_config.update(config["stochastic_config"])

        logger.debug(f"Creating Detector: {detector_name}. Stochastic config: {bool(detector_stochastic_config)}")

        detector = Detector(detector_name, self._timeline,
                            efficiency=efficiency,
                            dark_count=dark_count,
                            count_rate=count_rate,
                            time_resolution=time_resolution,
                            stochastic_config=detector_stochastic_config)
        
        # Entity.__init__ already calls timeline.add_entity(self)
        logger.debug(f"Checking {owner_entity.name} (type: {type(owner_entity)}) for add_detector method.")
        if hasattr(owner_entity, 'add_detector'):
            owner_entity.add_detector(detector)
        else:
            logger.warning(f"Owner entity {owner_entity.name} of type {type(owner_entity)} does not have an 'add_detector' method for {detector_name}. Detector added to timeline only.")

        return detector


    def _create_link(self, link_params: Dict[str, Any]) -> "QuantumChannel":
        """
        Creates an OpticalChannel (Link) component between two nodes.
        This method assumes OpticalChannel does not currently have stochastic noise.
        """
        from ..components.optical_channel import QuantumChannel # Use QuantumChannel instead of OpticalChannel
        from ..topology.node import Node # Local import for Node type check

        link_name = link_params.get("name")
        source_node_name = link_params["source"]
        target_node_name = link_params["target"]
        # Convert length from km to meters if needed
        length_km = link_params.get("length", 10_000)  # default 10 km
        if length_km > 1000:  # Heuristic: if value is very large, it's probably in meters already
            distance = length_km
        else:
            distance = length_km * 1_000  # convert km to m
        attenuation = link_params.get("attenuation", 0.0002)
        delay = link_params.get("delay", 0)
        polarization_fidelity = link_params.get("polarization_fidelity", 1.0)
        from sequence.constants import SPEED_OF_LIGHT
        light_speed = link_params.get("light_speed", SPEED_OF_LIGHT)

        logger.debug(f"Creating Link: {link_name} from {source_node_name} to {target_node_name}.")

        source_node = self._timeline.get_entity_by_name(source_node_name)
        target_node = self._timeline.get_entity_by_name(target_node_name)

        if not isinstance(source_node, Node) or not isinstance(target_node, Node):
            logger.error(f"Cannot create link. Source ({source_node_name}) or target ({target_node_name}) is not a valid Node.")
            raise ValueError(f"Invalid node(s) for link: {source_node_name}, {target_node_name}")

        link = QuantumChannel(link_name, self._timeline,
                              attenuation=attenuation,
                              distance=distance,
                              polarization_fidelity=polarization_fidelity,
                              light_speed=light_speed)
        
        # Entity.__init__ already calls timeline.add_entity(self)

        link.set_ends(source_node, target_node)

        # source_node.add_link(link)
        # target_node.add_link(link)

        return link

