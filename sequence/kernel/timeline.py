# File: sequence/kernel/timeline.py
#
# This module defines the Timeline class, which provides an interface for the simulation kernel
# and drives event execution. All entities are required to have an attached timeline for simulation.
#
# Modifications in this version focus on:
# 1. Integrating a global, reproducible random number generator (RNG) using numpy.random.
# 2. Providing a consistent way for all components to access this RNG.
# 3. Ensuring comprehensive initialization and management of various types of entities.
# 4. Re-integrating full simulation control (stop time, counters, progress bar).
# 5. Maintaining compatibility with the new QuantumManager structure.

import heapq
import numpy as np
import logging
from typing import Any, Callable, List, Dict, Tuple, TYPE_CHECKING, Optional
from math import inf # Re-added for stop_time
from datetime import timedelta # Re-added for human-readable time
from sys import stdout # Re-added for progress bar
from time import sleep, time_ns # Re-added for progress bar
from _thread import start_new_thread # Re-added for progress bar


# Type hinting for other core components
if TYPE_CHECKING:
    from .event import Event
    from .process import Process
    from .quantum_manager import QuantumManager
    from .entity import Entity # Re-added for generic entity management
    from ..topology.node import Node # For specific node registration, if needed
    from ..topology.quantum_router import QuantumRouter # For specific router registration, if needed

# Runtime imports for isinstance checks
from ..topology.node import Node, QuantumRouter

# Re-added constants used by original Timeline (assuming they exist in constants.py)
try:
    from ..constants import CARRIAGE_RETURN, SLEEP_SECONDS, \
                            NANOSECONDS_PER_MILLISECOND, PICOSECONDS_PER_NANOSECOND
except ImportError:
    # Define defaults if constants.py not available for testing/standalone
    CARRIAGE_RETURN = '\r'
    SLEEP_SECONDS = 0.1
    NANOSECONDS_PER_MILLISECOND = 1_000_000.0
    PICOSECONDS_PER_NANOSECOND = 1_000.0
    logging.warning("Missing ../constants.py. Using default values for progress bar constants.")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Timeline:
    """
    The central orchestrator of the simulation. It manages events, processes,
    and the global simulation time. It also provides a global random number
    generator for reproducible simulations.

    Attributes:
        _time (int): Current simulation time in picoseconds (ps).
        events (list[Event]): A min-heap (priority queue) of scheduled events.
        _callbacks (list[Callable]): List of callback functions executed at each time step.
        _nodes (dict[str, Node]): Dictionary of all quantum nodes in the network (specific type of entity).
        _quantum_routers (dict[str, QuantumRouter]): Dictionary of all quantum routers (specific type of entity).
        entities (dict[str, Entity]): A comprehensive dictionary of all registered entities, including nodes and routers.
        _random_generator (np.random.Generator): The global random number generator for the simulation.
        _quantum_manager (QuantumManager): The manager for all quantum states in the simulation.
        stop_time (int): The simulation time (in ps) at which the simulation should stop.
        schedule_counter (int): Counter for the total number of events scheduled.
        run_counter (int): Counter for the total number of events executed.
        is_running (bool): Flag indicating if the simulation is currently active.
        show_progress (bool): Flag to control the display of a progress bar.
    """

    def __init__(self, stop_time: int = inf, seed: Optional[int] = None):
        """
        Constructor for the Timeline.

        Args:
            stop_time (int): Stop time (in ps) of simulation (default inf).
            seed (int, optional): Seed for the global random number generator.
                                  If None, a random seed will be used.
        """
        self._time: int = 0
        self.events: List["Event"] = [] # Min-heap for events
        self._callbacks: List[Callable[[float], None]] = [] # Callbacks for timeline events
        
        # --- Entity Management (Re-integrated from original Timeline) ---
        self.entities: Dict[str, "Entity"] = {} # Generic entity dictionary
        self._nodes: Dict[str, "Node"] = {} # Specific node dictionary for faster lookup/type
        self._quantum_routers: Dict[str, "QuantumRouter"] = {} # Specific router dictionary

        self.stop_time: int = stop_time # Re-added: Simulation stop time
        self.schedule_counter: int = 0 # Re-added: Counter for scheduled events
        self.run_counter: int = 0 # Re-added: Counter for executed events
        self.is_running: bool = False # Re-added: Flag for simulation running state
        self.show_progress: bool = False # Re-added: Progress bar control

        # --- Global Random Number Generator (RNG) ---
        if seed is not None:
            self._random_generator: np.random.Generator = np.random.default_rng(seed)
            logger.info(f"Timeline initialized with RNG seed: {seed}")
        else:
            self._random_generator: np.random.Generator = np.random.default_rng()
            logger.info("Timeline initialized with random RNG seed.")

        self._quantum_manager: Optional["QuantumManager"] = None # QuantumManager initialized later externally


    def init(self) -> None:
        """
        Initializes the timeline and all its registered entities.
        This method should be called after all entities (nodes, routers, etc.)
        have been added to the timeline and QuantumManager has been set.
        """
        self._time = 0
        self.events = []
        self.schedule_counter = 0 # Reset counters on init
        self.run_counter = 0
        self.is_running = False
        
        # Initialize all registered entities (re-integrated from original Timeline)
        logger.info("Timeline initializing all entities.")
        for entity_name, entity in self.entities.items():
            if hasattr(entity, 'init'):
                entity.init()
            logger.debug(f"Entity '{entity_name}' initialized.")

        # QuantumManager itself doesn't have an `init` in the last version,
        # but if it were to, it would be called here.
        if self._quantum_manager:
            logger.debug("QuantumManager initialized (if applicable).")
        
        logger.info("Timeline initialization complete.")


    def now(self) -> int:
        """
        Returns the current simulation time.
        Returns:
            int: Current simulation time in picoseconds.
        """
        return self._time

    def get_generator(self) -> np.random.Generator:
        """
        Returns the global random number generator for the simulation.
        All components requiring randomness should use this method to ensure reproducibility.
        Returns:
            np.random.Generator: The global numpy random number generator.
        """
        return self._random_generator

    def set_quantum_manager(self, quantum_manager: "QuantumManager") -> None:
        """
        Sets the QuantumManager instance for the timeline.
        Args:
            quantum_manager (QuantumManager): The QuantumManager instance.
        """
        self._quantum_manager = quantum_manager
        logger.info(f"QuantumManager set on Timeline: {quantum_manager.formalism} formalism.")

    @property
    def quantum_manager(self) -> "QuantumManager":
        """
        Property to get the QuantumManager instance.
        Raises:
            RuntimeError: If QuantumManager has not been set.
        """
        if self._quantum_manager is None:
            logger.error("Attempted to access QuantumManager before it was set on the Timeline.")
            raise RuntimeError("QuantumManager has not been set on the Timeline.")
        return self._quantum_manager

    def schedule(self, event: "Event") -> None:
        """
        Schedules an event to be executed at a specific time.
        Events are stored in a min-heap, ordered by time.
        Args:
            event (Event): The event to schedule.
        """
        # Ensure owner is a live object if passed as string (re-integrated original logic)
        if isinstance(event.process.owner, str):
            event.process.owner = self.get_entity_by_name(event.process.owner)
            if event.process.owner is None:
                logger.error(f"Attempted to schedule event for non-existent entity: {event.process.owner}. Event ignored.")
                return # Do not schedule if owner not found

        heapq.heappush(self.events, event)
        self.schedule_counter += 1 # Re-added: Increment scheduled event counter
        logger.debug(f"Scheduled event at {event.time} ps: {event.process.owner.name}.{event.process.activation}")

    def update_event_time(self, event: "Event", new_time: int) -> None:
        """
        Updates the scheduled time of an existing event.
        Args:
            event (Event): The event whose time needs to be updated.
            new_time (int): The new time for the event.
        Raises:
            ValueError: If the event is not found in the scheduled events.
        """
        # The `heapq` module does not support efficient arbitrary element removal or key updates.
        # The most robust way with `heapq` is to mark the old event as invalid and push a new one.
        # However, to explicitly remove and update as per the original `EventList` functionality,
        # we perform a less efficient find-remove-add operation.
        
        # Check if event is in heap (this is O(N))
        if event in self.events:
            self.events.remove(event) # O(N)
            heapq.heapify(self.events) # Rebuild heap O(N)
            event.time = new_time # Update the time on the original event object
            heapq.heappush(self.events, event) # O(logN)
            logger.debug(f"Updated event {event.process.owner.name}.{event.process.activation} to {new_time} ps.")
        else:
            logger.warning(f"Attempted to update time for unscheduled event: {event.process.owner.name}.{event.process.activation}.")
            # If not found, schedule it as a new event if it's a valid update scenario
            event.time = new_time # Ensure event object itself is updated
            heapq.heappush(self.events, event) # Add it if it wasn't there
            logger.debug(f"Event not found, scheduled as new at {new_time} ps.")


    def remove_event(self, event: "Event") -> None:
        """
        Removes a scheduled event.
        Args:
            event (Event): The event to remove.
        Raises:
            ValueError: If the event is not found in the scheduled events.
        """
        try:
            if event in self.events: # This check is O(N)
                self.events.remove(event) # O(N)
                heapq.heapify(self.events) # Rebuild heap O(N)
                logger.debug(f"Removed event: {event.process.owner.name}.{event.process.activation} at {event.time} ps.")
            else:
                logger.warning(f"Attempted to remove unscheduled event: {event.process.owner.name}.{event.process.activation}.")
        except ValueError:
            logger.error(f"Event {event.process.owner.name}.{event.process.activation} not found in scheduled events for removal.")
            raise ValueError(f"Event {event} not found in scheduled events for removal.")


    def add_entity(self, entity: "Entity") -> None:
        """
        Registers a generic entity with the timeline.
        This is the primary method for adding any component to the simulation.
        Args:
            entity (Entity): The entity to register.
        """
        if entity.name in self.entities:
            logger.warning(f"Entity with name '{entity.name}' already registered.")
        entity.timeline = self # Ensure entity knows its timeline
        self.entities[entity.name] = entity
        logger.debug(f"Entity '{entity.name}' registered with timeline.")

        # For backward compatibility with specific register methods, if they are still called
        if isinstance(entity, Node):
            self._nodes[entity.name] = entity
        if isinstance(entity, QuantumRouter):
            self._quantum_routers[entity.name] = entity

    def remove_entity_by_name(self, name: str) -> None:
        """
        Removes a registered entity by name.
        Args:
            name (str): The name of the entity to remove.
        """
        entity = self.entities.pop(name, None)
        if entity:
            entity.timeline = None # Clear timeline reference from entity
            logger.debug(f"Entity '{name}' removed from timeline.")
            # Also remove from specific caches if present
            self._nodes.pop(name, None)
            self._quantum_routers.pop(name, None)
        else:
            logger.warning(f"Attempted to remove non-existent entity: {name}.")

    def get_entity_by_name(self, name: str) -> Optional["Entity"]:
        """
        Retrieves a registered entity by name.
        Args:
            name (str): The name of the entity.
        Returns:
            Optional[Entity]: The registered Entity object, or None if not found.
        """
        entity = self.entities.get(name)
        if entity is None:
            logger.debug(f"Entity '{name}' not found.")
        return entity

    # Specific register/get methods for Node and QuantumRouter (kept for backward compatibility/clarity)
    # These will internally call `add_entity`.
    def register_node(self, node: "Node") -> None:
        """Registers a quantum node with the timeline (calls add_entity)."""
        self.add_entity(node)

    def get_node(self, name: str) -> "Node":
        """Retrieves a registered quantum node by name (calls get_entity_by_name)."""
        node = self.get_entity_by_name(name)
        if not isinstance(node, Node):
            raise KeyError(f"Entity '{name}' is not a Node.")
        return node

    def register_quantum_router(self, router: "QuantumRouter") -> None:
        """Registers a quantum router with the timeline (calls add_entity)."""
        self.add_entity(router)

    def get_quantum_router(self, name: str) -> "QuantumRouter":
        """Retrieves a registered quantum router by name (calls get_entity_by_name)."""
        router = self.get_entity_by_name(name)
        if not isinstance(router, QuantumRouter):
            raise KeyError(f"Entity '{name}' is not a QuantumRouter.")
        return router

    def run(self) -> None:
        """
        Runs the simulation by executing scheduled events until no more events remain
        or the stop time is reached. Includes progress bar display.
        """
        logger.info("Simulation started.")
        start_exec_time_ns = time_ns() # For execution time tracking
        self.is_running = True

        if self.show_progress:
            self.progress_bar() # Start progress bar in a separate thread

        while len(self.events) > 0:
            event = heapq.heappop(self.events) # Get the next event (lowest time)
            logger.info(f"DEBUG: Popped event at time {event.time} ps, current time {self._time} ps, stop_time {self.stop_time} ps")

            if event.time >= self.stop_time: # Re-added: Check against stop_time
                logger.info(f"DEBUG: Event time {event.time} >= stop_time {self.stop_time}, rescheduling and breaking")
                self.schedule(event)  # Return event to list if it exceeds stop_time
                logger.info(f"Simulation stopped due to stop_time ({self.stop_time} ps) reached.")
                break
            
            # Re-added: assertion from original (ensure time doesn't go backwards)
            assert self._time <= event.time, f"Invalid event time for process scheduled on {event.process.owner.name} (Current: {self._time}, Event: {event.time})."
            
            # Re-added: check for invalid events (e.g., lazily deleted)
            if hasattr(event, 'is_invalid') and event.is_invalid():
                logger.debug(f"Skipping invalid event: {event.process.owner.name}.{event.process.activation}")
                continue

            self._time = event.time # Advance simulation time
            
            logger.debug(f"Executing event #{self.run_counter}: process owner={event.process.owner.name}, method={event.process.activation} at {self._time} ps.")
            try:
                # Execute the process associated with the event
                event.process.run() # Original calls process.run(), which then calls getattr on owner
                self.run_counter += 1 # Re-added: Increment executed event counter
            except AttributeError:
                logger.error(f"Method '{event.process.activation}' not found on owner '{event.process.owner.name}' for event at {self._time} ps.")
            except Exception as e:
                logger.error(f"Error executing event at {self._time} ps ({event.process.owner.name}.{event.process.activation}): {e}")

            # Execute any registered callbacks at the current time
            for callback in self._callbacks:
                try:
                    callback(self._time)
                except Exception as e:
                    logger.error(f"Error executing timeline callback at {self._time} ps: {e}")

        self.is_running = False
        time_elapsed_ns = time_ns() - start_exec_time_ns
        logger.info(f"Simulation finished at {self._time} ps. "
                    f"Execution Time: {self.ns_to_human_time(time_elapsed_ns)}; "
                    f"Scheduled Event: {self.schedule_counter}; "
                    f"Executed Event: {self.run_counter}.")

    def stop(self) -> None:
        """Method to stop simulation prematurely."""
        logger.info("Timeline is stopped by explicit call.")
        self.stop_time = self.now() # Set stop_time to current time to halt next loop iteration

    def add_callback(self, callback: Callable[[float], None]) -> None:
        """
        Registers a callback function to be executed at each time step during `run()`.
        Args:
            callback (Callable[[float], None]): A function that takes the current time (float) as an argument.
        """
        self._callbacks.append(callback)
        logger.debug(f"Callback {callback.__name__} added to timeline.")

    def reset(self) -> None:
        """
        Resets the timeline to its initial state, clearing all events,
        entities, and resetting time.
        The RNG is NOT reset here, as it's typically managed once per Timeline instance.
        To reset RNG, create a new Timeline with a seed.
        """
        self._time = 0
        self.events = []
        self._callbacks = []
        self.entities = {} # Clear all entities
        self._nodes = {} # Clear specific caches
        self._quantum_routers = {}
        self._quantum_manager = None
        self.stop_time = inf # Reset stop time
        self.schedule_counter = 0 # Reset counters
        self.run_counter = 0
        self.is_running = False
        self.show_progress = False
        logger.info("Timeline reset to initial state.")

    @staticmethod
    def seed(seed: int) -> None:
        """
        Sets the seed for the global NumPy random generator.
        Note: For reproducibility, it's generally better to pass the seed
        directly to the Timeline constructor. This static method is for
        backward compatibility if needed to set a global NumPy seed.
        """
        np.random.seed(seed)
        logger.warning("Using static `Timeline.seed()` is deprecated. Pass seed to `Timeline.__init__` for modern NumPy RNG.")

    def progress_bar(self):
        """Method to draw progress bar in a separate thread."""
        start_new_thread(self._print_progress_info, ())

    def _print_progress_info(self):
        """Internal method for progress bar thread."""
        start_time_ns = time_ns()

        while self.is_running:
            execution_time_ns = time_ns() - start_time_ns
            human_exec_time = self.ns_to_human_time(execution_time_ns)
            
            current_sim_time_ns = self.convert_to_nanoseconds(self._time) * 1000 # Convert ps to ns
            human_sim_time = self.ns_to_human_time(current_sim_time_ns)
            
            # Convert stop_time (ps) to nanoseconds for display
            stop_time_display = 'NaN'
            if self.stop_time != float('inf'):
                stop_time_ns = self.convert_to_nanoseconds(self.stop_time) * 1000
                stop_time_display = self.ns_to_human_time(stop_time_ns)

            process_bar_str = (f'{CARRIAGE_RETURN}Execution time: {human_exec_time};     '
                               f'Simulation time: {human_sim_time} / {stop_time_display}')

            print(f'{process_bar_str}', end=CARRIAGE_RETURN)
            stdout.flush()
            sleep(SLEEP_SECONDS)

        # Print final state of progress bar before thread exits
        execution_time_ns = time_ns() - start_time_ns
        human_exec_time = self.ns_to_human_time(execution_time_ns)
        current_sim_time_ns = self.convert_to_nanoseconds(self._time) * 1000
        human_sim_time = self.ns_to_human_time(current_sim_time_ns)
        stop_time_display = 'NaN'
        if self.stop_time != float('inf'):
            stop_time_ns = self.convert_to_nanoseconds(self.stop_time) * 1000
            stop_time_display = self.ns_to_human_time(stop_time_ns)
        
        final_process_bar_str = (f'{CARRIAGE_RETURN}Execution time: {human_exec_time};     '
                                 f'Simulation time: {human_sim_time} / {stop_time_display}')
        print(f'{final_process_bar_str}', end='\n') # Final newline for a clean output
        stdout.flush()


    @staticmethod
    def ns_to_human_time(nanoseconds: float) -> str:
        """Returns a string in the form [D day[s], ][H]H:MM:SS[.UUUUUU]."""
        # Ensure nanoseconds is treated as milliseconds for timedelta
        milliseconds = nanoseconds / NANOSECONDS_PER_MILLISECOND * 1000 # Corrected calculation for timedelta
        return str(timedelta(milliseconds=milliseconds))

    @staticmethod
    def convert_to_nanoseconds(picoseconds: int) -> float:
        """Converts picoseconds to nanoseconds."""
        return picoseconds / PICOSECONDS_PER_NANOSECOND

