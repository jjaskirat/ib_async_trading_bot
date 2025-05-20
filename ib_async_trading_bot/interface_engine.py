import asyncio
import aiofiles # Required for async file operations: pip install aiofiles
import logging
import math
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from threading import Thread
from typing import Optional, Tuple, Any
from concurrent.futures import Future # To store the result/status of the async task

from ib_async_trading_bot.objects import InterfaceObject, InterfaceTriggerObject
from ib_async_trading_bot.logger import get_logger

# -----------------Define Logger------------------------
logger = get_logger('interface_engine')


# Original threaded decorator (might still be useful for other blocking tasks)
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    return wrapper

class State(Enum):
    """Enumeration of possible system states."""
    UNSTARTED = auto()
    DOWNLOAD = auto()
    DOWNLOADING = auto() # New state: Async download in progress
    LOITERING = auto()
    TRIGGERED = auto()
    ERROR = auto()
    CLEANUP = auto()
    DONE = auto()
    # TASK_IN_PROGRESS might be redundant now or needs clearer definition
    # TASK_IN_PROGRESS = auto()

class Trigger(Enum):
    """Enumeration of possible triggers."""
    INITIALIZE = auto()
    DOWNLOAD = auto() # Trigger to start the download process
    TRADE = auto()
    CLEANUP = auto()
    # TASK_IN_PROGRESS = auto()
    
class InterfaceEngine(ABC):
    """
    Abstract base class for trading interface engines using a state machine pattern.
    Includes asynchronous download handling.
    """

    def __init__(self):
        super().__init__()
        self._state = State.UNSTARTED
        self._last_action_time: Optional[float] = None
        self._fps = 10  # Adjusted FPS allows frequent checks without busy-waiting
        self._cooldown = 1500  # 25 minutes in seconds

        # Asyncio related attributes
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[Thread] = None
        self._download_future: Optional[Future] = None # Monitors the async task
        self._data_queue: Optional[asyncio.Queue] = None # Queue for streamed data chunks

        # self.heartbeat = HeartBeat() # Turtle might cause issues

    @property
    def state(self) -> State:
        """Get the current state of the state machine."""
        return self._state

    @state.setter
    def state(self, new_state: State):
        """Safely transition to a new state with logging."""
        if self._state != new_state:
            logger.info(f"State transition: {self._state.name} -> {new_state.name}")
            self._state = new_state
        # else: # Optional: log if attempting to set the same state
        #     logger.debug(f"Already in state: {new_state.name}")

    def _start_async_loop(self):
        """Starts the asyncio event loop in a separate thread."""
        if self._async_loop is None or not self._async_loop.is_running():
            logger.info("Starting asyncio event loop thread.")
            self._async_loop = asyncio.new_event_loop()
            def run_loop(loop):
                asyncio.set_event_loop(loop)
                try:
                    loop.run_forever()
                finally:
                    # Ensure cleanup happens before the loop closes fully
                    if hasattr(loop, "shutdown_asyncgens"): # >= Python 3.6
                         loop.run_until_complete(loop.shutdown_asyncgens())
                    tasks = asyncio.all_tasks(loop)
                    if tasks:
                         # Give tasks a chance to finish/cancel
                         for task in tasks:
                             if not task.done():
                                 task.cancel()
                         loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

                    loop.close()
                    logger.info("Asyncio event loop closed.")

            self._loop_thread = Thread(target=run_loop, args=(self._async_loop,), daemon=True)
            self._loop_thread.start()
        else:
            logger.debug("Asyncio loop already running.")

    def _stop_async_loop(self):
        """Stops the asyncio event loop if it's running."""
        if self._async_loop and self._async_loop.is_running():
            logger.info("Stopping asyncio event loop.")
            # Schedule loop.stop() to run on the loop's thread
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            if self._loop_thread:
                 self._loop_thread.join(timeout=5) # Wait for thread to finish
                 if self._loop_thread.is_alive():
                      logger.warning("Asyncio loop thread did not exit cleanly after stop signal.")
            self._async_loop = None
            self._loop_thread = None
        elif self._async_loop:
             logger.info("Asyncio event loop was not running, ensuring it's cleared.")
             self._async_loop = None # Ensure it's cleared if stopped externally
             self._loop_thread = None


    def run(self, interface_obj: InterfaceObject, mode: str = 'demo'):
        """ Main execution loop for the state machine. """
        self._start_async_loop() # Ensure loop is running
        try:
            while self._state != State.DONE:
                self._step(interface_obj, mode)
                # Sleep is important to prevent busy-waiting and allow other threads
                # (like the asyncio loop thread) to run.
                time.sleep(1 / self._fps)

        except KeyboardInterrupt:
            logger.info("============ Interrupted by user ============")
            # Initiate cleanup if not already cleaning up or done
            if self.state not in [State.CLEANUP, State.DONE]:
                self.state = State.CLEANUP

        except Exception as e:
            logger.error(f"============ Unhandled Error in run loop: {e} ============", exc_info=True)
            # Initiate error handling/cleanup
            if self.state not in [State.ERROR, State.CLEANUP, State.DONE]:
                self.state = State.ERROR

        finally:
            logger.info("Starting final shutdown sequence...")
            # Ensure cleanup runs if the loop exited unexpectedly without reaching DONE
            if self.state not in [State.CLEANUP, State.DONE]:
                 logger.warning(f"Loop exited unexpectedly in state {self.state.name}. Forcing cleanup.")
                 self.state = State.CLEANUP

            # Allow cleanup/error states to complete their steps
            while self.state not in [State.DONE]:
                logger.debug(f"Running final step in state {self.state.name}...")
                self._step(interface_obj, mode)
                time.sleep(0.1) # Short sleep during final steps

            self._stop_async_loop() # Stop the loop cleanly
            # self.heartbeat.cleanup_all() # Defer heartbeat cleanup or handle potential errors
            logger.info("============ Engine run finished ============")

    def _step(self, interface_obj: InterfaceObject, mode: str):
        """Execute one step of the state machine."""
        # Check for triggers (example - adapt as needed)
        # if self._state == State.LOITERING:
        #     trigger = self.check_to_trigger(interface_obj, mode)
        #     if trigger and not self._in_cooldown():
        #         self._last_action_time = time.time()
        #         self._handle_trigger(interface_obj, mode, trigger) # Use trigger handler
        #         return # Skip normal state handling if trigger was handled

        # State machine dispatch
        state_handlers = {
            State.UNSTARTED: self._handle_unstarted,
            State.DOWNLOAD: self._handle_download,        # State to initiate download
            State.DOWNLOADING: self._handle_downloading,  # State while download runs async
            State.LOITERING: self._handle_loitering,
            State.ERROR: self._handle_error,
            State.CLEANUP: self._handle_cleanup,
            # State.TRIGGERED might not be needed if handled by _handle_trigger
        }

        handler = state_handlers.get(self._state)
        if handler:
            try:
                handler(interface_obj, mode)
            except Exception as e:
                 logger.error(f"Error executing handler for state {self.state.name}: {e}", exc_info=True)
                 self.state = State.ERROR # Transition to error on handler failure
        elif self.state != State.DONE: # Log if state isn't handled and isn't DONE
             logger.warning(f"No handler defined for state: {self.state.name}")


    # --- State Handlers ---

    def _handle_unstarted(self, interface_obj: InterfaceObject, mode: str):
        """Handle the unstarted initialization state."""
        # self.state = State.TASK_IN_PROGRESS # Avoid ambiguous state
        try:
            logger.info("Initializing interface...")
            self.initialize(interface_obj) # Should be relatively quick or use @threaded if blocking
            logger.info("Initialization complete.")
            # Decide next state based on mode
            if mode == 'demo':
                 logger.info("Demo mode - skipping download, moving to loitering.")
                 self.state = State.LOITERING # Go directly to loitering in demo
                 # Or trigger a simulated trade:
                 # self._handle_trigger(interface_obj, mode, InterfaceTriggerObject(trigger=Trigger.TRADE, config={}))
            else:
                 logger.info("Deploy mode - moving to download state.")
                 self.state = State.DOWNLOAD # Start download process
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.state = State.ERROR


    def _handle_download(self, interface_obj: InterfaceObject, mode: str):
        """Handle the DOWNLOAD state: Initiate the asynchronous download task."""
        if not self._async_loop or not self._async_loop.is_running():
            logger.error("Asyncio loop not running. Cannot start download.")
            self.state = State.ERROR
            return
        if self._download_future and not self._download_future.done():
             logger.warning("Download already in progress. Ignoring request.")
             self.state = State.DOWNLOADING # Ensure state reflects reality
             return

        logger.info("Initiating asynchronous download task...")
        self._data_queue = asyncio.Queue(maxsize=100) # Create a queue with a max size
        # Example: Define arguments for the download method
        # These would likely come from interface_obj or config
        download_args = ("my_data.csv",) # e.g., filename
        # download_args = ("https://example.com/some_data",) # e.g., URL

        coro = self.download(interface_obj, self._data_queue, *download_args)

        # Schedule the coroutine to run on the asyncio event loop thread
        try:
            self._download_future = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
            logger.info(f"Download task scheduled. Future: {self._download_future}")
            self.state = State.DOWNLOADING # Transition to waiting state
        except Exception as e:
            logger.error(f"Failed to schedule download task: {e}", exc_info=True)
            self._data_queue = None # Clean up queue if scheduling failed
            self.state = State.ERROR

    def _handle_downloading(self, interface_obj: InterfaceObject, mode: str):
        """Handle the DOWNLOADING state: Monitor task and process data from queue."""
        if self._download_future is None or self._data_queue is None:
            logger.error("In DOWNLOADING state without a valid future or queue. Moving to ERROR.")
            self.state = State.ERROR
            return

        # --- Process data currently in the queue (non-blocking) ---
        processed_count = 0
        try:
            while not self._data_queue.empty():
                # Use get_nowait for non-blocking check from this (main) thread
                chunk = self._data_queue.get_nowait()
                if chunk is None:
                    # Sentinel received, indicates successful completion by the download task.
                    # We still wait for the future to be marked done below.
                    logger.debug("Received None sentinel from download queue.")
                else:
                    # --- !!! PROCESS THE DATA CHUNK HERE !!! ---
                    # This is the core of consuming the stream synchronously.
                    # Replace logger call with actual processing logic.
                    logger.debug(f"Processing data chunk: {len(chunk)} bytes (Queue size: {self._data_queue.qsize()})")
                    # Example: self.process_chunk(chunk)
                    # --- !!! ----------------------------- !!! ---
                    processed_count += 1
                # Notify the queue that the item is processed (important for queue management)
                self._data_queue.task_done()
        except asyncio.QueueEmpty:
            logger.debug("Download queue empty for now.")
            pass # This is expected when no new data is available yet
        except Exception as e:
             logger.error(f"Error processing data chunk from download queue: {e}", exc_info=True)
             # Decide if processing error should halt everything
             self.state = State.ERROR # Option: Transition to ERROR on processing failure

        # Optional: Log if processing happened
        # if processed_count > 0:
        #    logger.debug(f"Processed {processed_count} chunks.")


        # --- Check if the download task (Future) has finished ---
        if self._download_future.done():
            logger.info("Download future is marked as done.")
            try:
                # Check for exceptions raised within the async download task
                result = self._download_future.result(timeout=0) # Timeout 0 raises if not done, but we checked
                logger.info(f"Async download task completed successfully.")
                # Check if queue is fully processed (optional, depends on logic)
                if not self._data_queue.empty():
                     logger.warning("Download task finished, but queue still has items. Processing remaining...")
                     # Could add logic here to process remaining items, or error out.
                     # For now, we assume the sentinel was the last item for success.

                self.state = State.LOITERING # Download successful, move on
            except asyncio.CancelledError:
                 logger.warning("Download task was cancelled.")
                 self.state = State.LOITERING # Or ERROR/CLEANUP depending on why cancelled
            except Exception as e:
                logger.error(f"Async download task failed with exception: {e}", exc_info=True)
                self.state = State.ERROR # Transition to error state on task failure
            finally:
                 # Clean up resources for this download task
                 self._download_future = None
                 # Ensure queue is cleared before setting to None if error occurred mid-stream
                 # while self._data_queue and not self._data_queue.empty():
                 #     try: self._data_queue.get_nowait(); self._data_queue.task_done()
                 #     except asyncio.QueueEmpty: break
                 self._data_queue = None
                 logger.debug("Cleaned up download future and queue references.")
        # else:
        #     logger.debug(f"Download task still running (Queue size: {self._data_queue.qsize()})...")


    def _handle_loitering(self, interface_obj: InterfaceObject, mode: str):
        """Handle the waiting/loitering state."""
        logger.debug("Loitering...")
        # Check for triggers or timed actions here
        # trigger = self.check_to_trigger(interface_obj, mode)
        # if trigger and not self._in_cooldown():
        #     self._handle_trigger(interface_obj, mode, trigger)
        #     return

        if mode == 'demo':
             # Example demo behavior: After some time, simulate cleanup
             if self._last_action_time is None: self._last_action_time = time.time()
             if time.time() - self._last_action_time > 15: # Wait 15s in loitering
                 logger.info("Demo mode: Triggering cleanup after loitering.")
                 self.state = State.CLEANUP


    def _handle_error(self, interface_obj: InterfaceObject, mode: str):
        """Handle error state by transitioning to cleanup."""
        logger.error("Entering ERROR state. Transitioning to CLEANUP.")
        # Potentially log more details or attempt specific recovery here
        # If download future exists and caused the error, ensure it's cancelled/cleaned
        if self._download_future and not self._download_future.done():
            logger.info("Attempting to cancel ongoing download task due to error state.")
            if self._async_loop and self._async_loop.is_running():
                self._async_loop.call_soon_threadsafe(self._download_future.cancel)

        # Always move to cleanup from error state in this basic model
        self.state = State.CLEANUP

    def _handle_cleanup(self, interface_obj: InterfaceObject, mode: str):
        """Handle cleanup and shutdown procedures."""
        logger.info("Running cleanup procedures...")
        try:
            # Cancel ongoing download if cleanup is initiated externally
            if self._download_future and not self._download_future.done():
                 logger.info("Cleanup initiated: Attempting to cancel ongoing download task.")
                 if self._async_loop and self._async_loop.is_running():
                      self._async_loop.call_soon_threadsafe(self._download_future.cancel)
                 # Give cancellation a moment (optional)
                 # time.sleep(0.1)

            # Call the subclass's specific cleanup implementation
            self.cleanup_all(interface_obj)
            logger.info("Subclass cleanup_all completed.")

            # Final state transition
            self.state = State.DONE
            logger.info("Cleanup finished. Engine state set to DONE.")

        except Exception as e:
            logger.error(f"Error during cleanup implementation: {e}", exc_info=True)
            # Even if cleanup fails, we might still need to attempt to stop the loop
            # and transition to DONE to prevent infinite loops.
            self.state = State.DONE # Force DONE state even on cleanup error
            logger.error("Forcing state to DONE after cleanup error.")


    # --- Trigger Handling (Example) ---
    def _handle_trigger(
        self,
        interface_obj: InterfaceObject,
        mode: str, # Added mode back if needed by triggered actions
        trigger_obj: InterfaceTriggerObject
    ):
        """Execute the appropriate action for the triggered operation."""
        # self.state = State.TASK_IN_PROGRESS # Avoid ambiguous state
        logger.info(f"Handling trigger: {trigger_obj.trigger.name}")
        handlers = {
            Trigger.INITIALIZE: self.initialize,
            Trigger.DOWNLOAD: lambda io, **cfg: self.state == State.DOWNLOAD, # Trigger download state
            Trigger.TRADE: self.trade,
            Trigger.CLEANUP: lambda io, **cfg: self.state == State.CLEANUP, # Trigger cleanup state
        }

        handler = handlers.get(trigger_obj.trigger)
        current_state_before_handler = self.state

        if handler:
            try:
                # Pass relevant args; assumes handler takes interface_obj and config kwargs
                # Note: some handlers here just trigger a state change
                handler(interface_obj, **trigger_obj.config)
                logger.info(f"Trigger action for {trigger_obj.trigger.name} executed.")
                # Transition back to loitering *only if* the handler didn't change the state
                # and the state isn't one that manages its own follow-up (like DOWNLOADING)
                if self.state == current_state_before_handler and self.state not in [State.DOWNLOADING, State.CLEANUP, State.ERROR]:
                     self.state = State.LOITERING
            except Exception as e:
                 logger.error(f"Error handling trigger {trigger_obj.trigger.name}: {e}", exc_info=True)
                 self.state = State.ERROR
        else:
            logger.warning(f"No handler defined for trigger: {trigger_obj.trigger.name}")
            # If no handler, likely return to loitering if we were there
            if current_state_before_handler == State.LOITERING:
                 self.state = State.LOITERING


    # --- Abstract Methods ---

    # @abstractmethod
    # def _get_application_manager(self): pass
    @abstractmethod
    def _get_all_actions(self): pass
    @abstractmethod
    def _get_all_positions(self): pass
    @abstractmethod
    def cleanup_all(self, interface_obj: InterfaceObject, *args): pass
    # @abstractmethod
    # def initialize(self, interface_obj: InterfaceObject, *args): pass

    @abstractmethod
    async def download(self, interface_obj: InterfaceObject, data_queue: asyncio.Queue, *args):
        """
        Asynchronously download data and put chunks into the data_queue.
        Subclasses MUST implement this method.
        It MUST put data chunks (bytes or other agreed type) into the queue
        using `await data_queue.put(chunk)`.
        It MUST put `None` into the queue upon successful completion to signal the end.
        It SHOULD handle internal exceptions and potentially put an Exception
        object or None into the queue to signal failure.
        """
        pass

    @abstractmethod
    def trade(self, interface_obj: InterfaceObject, *args): pass

# --- Concrete Method (if cleanup_all is always the same) ---
    def cleanup(self, interface_obj: InterfaceObject, *args):
        """Default cleanup caller."""
        self.cleanup_all(interface_obj, *args)