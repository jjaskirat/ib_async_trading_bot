import asyncio
from enum import Enum
import os
import pytz
import pandas as pd
from functools import partial
from threading import Thread # Keep for potentially threaded non-async tasks if needed
import time
from typing import Any, Optional
import logging # Use standard logging or ib_async's util
from datetime import datetime, time as datetime_time
import holidays

# Import necessary ib_async components
from ib_async import (
    IB,
    Forex,
    Stock,
    Order,
    util as ib_async_util,
    Contract # Explicitly import Contract for type checking if needed
)

# Import base class and RL components
from ib_async_trading_bot.logger import get_logger # Use your logger setup
from ib_async_trading_bot.objects import InterfaceObject, InterfaceTriggerObject
from ib_async_trading_bot.interface_engine import (
    InterfaceEngine,
    State,
    Trigger,
    )
from ib_async_trading_bot.environment import Actions, Positions
from ib_async_trading_bot.dataset import get_dataset
from ib_async_trading_bot.model_util import get_integrated_model

logger = get_logger('interface_engine_interactive_broker')

class InterfaceEngineInteractiveBrokers(InterfaceEngine):
    """
    Interactive Brokers-specific implementation using ib_async and asyncio.
    """
    def __init__(self):
        """ Initialize the engine and the IB connection object. """
        super().__init__() # Call base __init__
        self.api = IB() # Initialize IB instance
        self._live_bar_streams = {} # Track active real-time streams {symbol: BarDataList}
        self._state_machine_task = None # Task handle for the main state loop
        self._triggered_symbol = None # Stores symbol that caused TRIGGERED state
        self._interface_obj: Optional[InterfaceObject] = None # Store interface obj for internal use
        self._mode: str = 'demo' # Store mode
        self.tz_new_york = pytz.timezone("America/New_York") # Timezone for market hours
        self.nyse_holidays = holidays.NYSE(years=datetime.now().year) # Market holidays

        # Register IB event handlers
        self.api.connectedEvent += self._on_connect
        self.api.disconnectedEvent += self._on_disconnect
        self.api.errorEvent += self._on_error
        self.api.positionEvent += self._on_position_update # Handle async position updates

    # # --- RL Environment / Model Helpers (Adapt as needed) ---
    # def _get_application_manager(self):
    #     # return TradingBot()
    #     return TradingBot() # Use mock if TradingBot not available

    def _get_all_actions(self): return Actions
    def _get_all_positions(self): return Positions

    # --- IB Connection and Event Handlers ---
    def _on_connect(self):
        logger.info("IB API Connected.")
        # If initializing, signal connection success allows state to progress
        if self.state == State.UNSTARTED: # Or a dedicated CONNECTING state
             logger.info("Connection successful during initialization.")
             # State machine logic in _handle_unstarted will check isConnected

    def _on_disconnect(self):
        logger.warning("IB API Disconnected.")
        if self.state not in [State.CLEANUP, State.DONE, State.ERROR]:
             logger.error("Unexpected disconnection. Transitioning to ERROR state.")
             self.state = State.ERROR # Critical failure

    def _on_error(self, reqId, errorCode, errorString, contract=None):
        # Enhanced error logging based on severity
        ignore_codes = {2104, 2106, 2108, 2158, 2157, 2103, 2105, 2107} # Market data farm/status
        warning_codes = {326, 399} # Unable to connect / Order Warning
        connectivity_codes = {1100, 1101, 1102, 1300, 2110, 2119} # Connection loss related

        msg = f"IB Msg: ReqId {reqId}, Code {errorCode} - {errorString}"
        if contract: msg += f" (Contract: {contract.symbol if contract else 'N/A'})"

        if errorCode in ignore_codes:
             logger.debug(msg)
        elif errorCode in connectivity_codes:
             logger.error(msg + " [Connectivity Issue]")
             if self.state not in [State.CLEANUP, State.DONE, State.ERROR]:
                  self.state = State.ERROR
        elif errorCode in warning_codes:
             logger.warning(msg)
        else: # Treat other codes as errors
             logger.error(msg)
             # Decide if other errors trigger ERROR state
             # if self.state not in [State.CLEANUP, State.DONE, State.ERROR]:
             #      self.state = State.ERROR

    def _on_position_update(self, position):
        """ Handles individual position updates from IB. """
        symbol = self._get_symbol_from_contract(position.contract)
        if symbol and self._interface_obj:
            logger.debug(f"Received position update for {symbol}: Account={position.account}, Qty={position.position}, AvgCost={position.avgCost}")
            # Update our tracked positions
            self._update_single_position(symbol, position.position)
        else:
             logger.debug(f"Received position update for unknown/untracked contract: {position.contract}")

    # --- Contract and Order Creation ---
    def create_contract(self, symbol_pair, type='fx'):
        """ Creates an IB Contract object (sync). """
        logger.debug(f"Creating contract for {symbol_pair} (type: {type})")
        contract = None
        match type:
            case 'stk':
                contract = Stock(symbol=symbol_pair, exchange='SMART', currency='USD')
            case 'fx':
                # Ensure symbol_pair is in 'EURUSD' format for Forex
                if len(symbol_pair) == 6:
                    contract = Forex(symbol_pair)
                else:
                     logger.error(f"Invalid symbol format for Forex: {symbol_pair}")
            # Add other types (futures, options) if needed
            case _:
                logger.error(f"Unsupported contract type: {type}")

        return contract # Returns None if invalid

    async def qualify_contract(self, contract):
         """ Qualifies a contract using the async API call. """
         if not contract: return None
         if not self.api.isConnected():
             logger.error("Cannot qualify contract: Not connected.")
             return None
         try:
             qual_contracts = await self.api.qualifyContractsAsync(contract)
             if qual_contracts:
                 logger.debug(f"Contract qualified: {qual_contracts[0]}")
                 return qual_contracts[0] # Return the first qualified contract
             else:
                 logger.warning(f"Could not qualify contract: {contract}")
                 return None
         except Exception as e:
             logger.error(f"Error qualifying contract {contract}: {e}", exc_info=True)
             return None

    def create_buy_order(self, quantity=10):
        """ Creates a default BUY Market Order (sync). """
        logger.debug(f"Creating BUY Market order for qty: {quantity}")
        # Using Market order for simplicity; use MIDPRICE or LMT with appropriate params if needed
        return Order(action='BUY', orderType='MKT', totalQuantity=quantity, tif='DAY')

    def create_sell_order(self, quantity=10):
        """ Creates a default SELL Market Order (sync). """
        logger.debug(f"Creating SELL Market order for qty: {quantity}")
        return Order(action='SELL', orderType='MKT', totalQuantity=quantity, tif='DAY')


    # --- Data Handling ---
    async def download(self, interface_obj: InterfaceObject, data_queue: asyncio.Queue, stock_symbols: list, hist_duration="2 D", hist_barsize="1 min"):
        """
        Asynchronously download initial historical data. Puts results (dict per symbol)
        into the data_queue. Finishes by putting None.
        """
        logger.info(f"[ASYNC TASK] Starting historical download for: {stock_symbols}")
        if not self.api.isConnected():
            logger.error("[ASYNC TASK] Cannot download: Not connected.")
            await data_queue.put(None); return

        try:
            for symbol in stock_symbols:
                contract = self.create_contract(symbol) # Determine type if needed
                qualified_contract = await self.qualify_contract(contract)
                if not qualified_contract:
                    logger.warning(f"[ASYNC TASK] Skipping historical data for {symbol} (qualification failed).")
                    continue

                logger.debug(f"[ASYNC TASK] Requesting historical for {symbol} ({hist_duration}, {hist_barsize})...")
                try:
                    hist_data = await asyncio.wait_for(
                        self.api.reqHistoricalDataAsync(
                            qualified_contract, endDateTime='', durationStr=hist_duration,
                            barSizeSetting=hist_barsize, whatToShow='MIDPOINT', useRTH=True,
                            formatDate=1, keepUpToDate=False),
                        timeout=60.0 # 60 second timeout per request
                    )

                    df = ib_async_util.df(hist_data) if hist_data else pd.DataFrame()
                    logger.info(f"[ASYNC TASK] Received {len(df)} historical bars for {symbol}.")
                    await data_queue.put({'symbol': symbol, 'type': 'historical', 'data': df})

                except asyncio.TimeoutError:
                     logger.error(f"[ASYNC TASK] Timeout requesting historical data for {symbol}.")
                     await data_queue.put({'symbol': symbol, 'type': 'historical', 'data': pd.DataFrame()}) # Put empty on timeout
                except Exception as e:
                     logger.error(f"[ASYNC TASK] Error requesting historical data for {symbol}: {e}", exc_info=True)
                     await data_queue.put({'symbol': symbol, 'type': 'historical', 'data': pd.DataFrame()}) # Put empty on error

                await asyncio.sleep(0.2) # Throttle requests slightly

            await data_queue.put(None) # Signal completion
            logger.info("[ASYNC TASK] Historical download task finished.")

        except Exception as e:
            logger.error(f"[ASYNC TASK] Unhandled error in download task: {e}", exc_info=True)
            await data_queue.put(None) # Ensure None is sent on outer error


    async def start_realtime_streams(self, interface_obj: InterfaceObject, stock_symbols: list):
        """ Subscribe to real-time bars for given symbols. """
        logger.info(f"Starting real-time subscriptions for: {stock_symbols}")
        if not self.api.isConnected(): logger.error("Cannot start streams: Not connected."); return

        for symbol in stock_symbols:
            if symbol in self._live_bar_streams: logger.warning(f"Stream for {symbol} already active."); continue

            contract = self.create_contract(symbol)
            qualified_contract = await self.qualify_contract(contract)
            if not qualified_contract: logger.warning(f"Skipping real-time for {symbol} (qualification failed)."); continue

            try:
                logger.debug(f"Subscribing to real-time bars for {symbol}...")
                # Note: Use realTimeUpdatesOnly=True if you don't want snapshot
                live_bars = self.api.reqRealTimeBars(
                    qualified_contract, barSize=5, whatToShow='MIDPOINT',
                    useRTH=True
                )
                self._live_bar_streams[symbol] = live_bars # Store the BarDataList object

                # Create partial func for callback to include symbol context
                callback = partial(self._on_realtime_bar_update, interface_obj, symbol)
                live_bars.updateEvent += callback # Attach callback
                logger.info(f"Subscribed to real-time bars for {symbol}.")

            except Exception as e:
                logger.error(f"Error subscribing to real-time bars for {symbol}: {e}", exc_info=True)
            await asyncio.sleep(0.2) # Throttle subscriptions


    async def _on_realtime_bar_update(self, interface_obj: InterfaceObject, symbol: str, bars: list, hasNewBar: bool):
        """ Callback for real-time bar updates. """
        logger.info(f"Checking for new Bar for symbol  {symbol}")
        if not hasNewBar: return # Process only completed bars

        logger.debug(f"New bar for {symbol}. Bar count: {len(bars)}")

        # Store latest data (the full BarDataList)
        if symbol not in interface_obj.market_data.seq:
             interface_obj.market_data.seq[symbol] = [pd.DataFrame(), None] # Ensure entry exists with empty hist
        interface_obj.market_data.seq[symbol][1] = bars # Store/update the live BarDataList
        # logger.debug(f"Updated live BarDataList for {symbol} in interface_obj.")

        # Check if conditions met to trigger trade logic
        # Example threshold (adapt as needed)
        hist_df, _ = interface_obj.market_data.seq.get(symbol, (pd.DataFrame(), None))
        required_len = 129 # Example: Need 128 total bars
        current_total_len = len(hist_df) + len(bars)

        if current_total_len < required_len:
            logger.debug(f"Data length for {symbol} ({current_total_len}) < required ({required_len}). No trigger.")
            return

        # # --- Trigger State Change ---
        # if self.state == State.LOITERING: # Only trigger if loitering
        #     logger.info(f"Sufficient data for {symbol}. Triggering TRADE state.")
        #     self._triggered_symbol = symbol # Store context for _handle_triggered
        #     self._handle_triggered()
        # else:
        #     logger.warning(f"New bar for {symbol}, but state is {self.state.name}. Trigger suppressed.")
        await self.trade(interface_obj, symbol, 'deploy')


    # --- Position Handling ---
    def _get_symbol_from_contract(self, contract: Contract) -> Optional[str]:
        """ Extracts a usable symbol string from various IB contract types. """
        if isinstance(contract, Stock): return contract.symbol
        if isinstance(contract, Forex): return contract.pair()
        # Add Forex, Future, Option etc. handling here
        logger.warning(f"Cannot extract symbol from unsupported contract type: {type(contract)}")
        return None

    def _update_single_position(self, symbol: str, quantity: float):
        """ Updates the position enum in interface_obj based on quantity. """
        positions_enum = self._get_all_positions()
        if quantity > 0: new_pos = positions_enum.LONG
        elif quantity < 0: new_pos = positions_enum.SHORT
        else: new_pos = positions_enum.FLAT

        if self._interface_obj:
            current_pos = self._interface_obj.positions.get(symbol)
            if current_pos != new_pos:
                 logger.info(f"Position change detected for {symbol}: {current_pos.name if current_pos else 'None'} -> {new_pos.name}")
                 self._interface_obj.positions[symbol] = new_pos
            else:
                 logger.debug(f"Position for {symbol} remains {new_pos.name} ({quantity}).")
        else:
            logger.warning("Cannot update position: _interface_obj not set.")


    async def update_positions(self, interface_obj: InterfaceObject):
        """ Fetches all current positions asynchronously and updates interface_obj. """
        logger.info("Requesting all current positions...")
        if not self.api.isConnected(): logger.error("Cannot fetch positions: Not connected."); return

        try:
            # Use reqPositionsAsync for non-blocking request
            positions_raw = await asyncio.wait_for(self.api.reqPositionsAsync(), timeout=30.0)
            logger.info(f"Received {len(positions_raw)} total position entries.")

            # Process and update tracked positions
            current_tracked_positions = {}
            tracked_symbols = {s.stock_symbol for s in interface_obj.stocks_to_consider}

            for pos in positions_raw:
                symbol = self._get_symbol_from_contract(pos.contract)
                if symbol and symbol in tracked_symbols:
                     positions_enum = self._get_all_positions()
                     if pos.position > 0: current_tracked_positions[symbol] = positions_enum.LONG
                     elif pos.position < 0: current_tracked_positions[symbol] = positions_enum.SHORT
                     else: current_tracked_positions[symbol] = positions_enum.FLAT

            # Ensure all tracked symbols have an entry, default to FLAT
            for symbol in tracked_symbols:
                if symbol not in current_tracked_positions:
                    current_tracked_positions[symbol] = self._get_all_positions().FLAT

            # Update the main object - check for changes
            if interface_obj.positions != current_tracked_positions:
                logger.info(f"Updating tracked positions: {current_tracked_positions}")
                interface_obj.positions = current_tracked_positions
            else:
                 logger.debug("No change in tracked positions.")

        except asyncio.TimeoutError:
            logger.error("Timeout requesting positions.")
        except Exception as e:
             logger.error(f"Failed to fetch or process positions: {e}", exc_info=True)


    # --- Trading Logic ---
    async def trade(self, interface_obj: InterfaceObject, stock_symbol: str, mode: str):
        """ Main trade logic called when state is TRIGGERED. """
        logger.info(f"Executing trade logic for: {stock_symbol}, Mode: {mode}")

        # 1. Get Action Prediction
        model, env = self.get_env_and_model(interface_obj, stock_symbol)
        if not model or not env:
            logger.error(f"Could not get model/env for {stock_symbol}. Aborting trade.")
            return

        # 2. Get Current Position (ensure it's up-to-date)
        await self.update_positions(interface_obj) # Refresh positions just before trade
        position = interface_obj.positions.get(stock_symbol, self._get_all_positions().FLAT)
        logger.info(f"Current position for {stock_symbol}: {position.name}")

        # 3. Predict Action
        observation, info = env.reset(position=position) # Reset env with current pos
        action_code, _ = model.predict(observation)
        action = self._get_all_actions()(action_code) # Convert code to enum
        logger.info(f"Action prediction for {stock_symbol}: {action.name}")

        # 4. Execute Trade based on mode
        if mode == 'deploy':
            await self.trade_interface(interface_obj, stock_symbol, action, position)
        elif mode == 'demo':
            self.trade_env(interface_obj, stock_symbol, action, position)
        else:
            logger.warning(f"Unknown trade mode: {mode}")

        # 5. Optional: Update positions again after attempting trade
        # await asyncio.sleep(2) # Allow time for order to potentially register
        await self.update_positions(interface_obj)


    def trade_env(self, interface_obj, stock_symbol, action, position):
        """ Simulate trade action in demo mode (logging). """
        actions = self._get_all_actions()
        positions = self._get_all_positions()
        qty = 10000 # Example FX quantity - **ADJUST THIS BASED ON RISK/CONFIG**
        
        if action == actions.BUY and position != positions.LONG: # Buy only if not already Long
            logger.info(f"Placing BUY order for {stock_symbol}, qty: {qty}")
            position = positions.LONG
        elif action == actions.SELL and position != positions.SHORT: # Sell only if not already Short
            logger.info(f"Placing SELL order for {stock_symbol}, qty: {qty}")
            position = positions.SHORT
        elif action == actions.DOUBLE_BUY and position == positions.SHORT: # Cover short and go long
            logger.info(f"Placing DOUBLE BUY (2 orders) for {stock_symbol}, qty: {qty} each")
            position = positions.LONG
        elif action == actions.DOUBLE_SELL and position == positions.LONG: # Cover long and go short
            logger.info(f"Placing DOUBLE SELL (2 orders) for {stock_symbol}, qty: {qty} each")
            position = positions.SHORT
        elif action == actions.HOLD:
             logger.info(f"Action is HOLD for {stock_symbol}. No order placed.")
             position = positions.FLAT
        else:
             logger.warning(f"Action {action.name} vs Position {position.name} for {stock_symbol} results in no order.")
             
        interface_obj.positions[stock_symbol] = position


    async def trade_interface(self, interface_obj: InterfaceObject, stock_symbol: str, action: Actions, position: Positions):
        """ Executes the predicted action via the IB API. """
        logger.info(f"Executing action '{action.name}' for {stock_symbol} via IB API (Current Pos: {position.name})")
        actions = self._get_all_actions()
        positions = self._get_all_positions()
        qty = 10000 # Example FX quantity - **ADJUST THIS BASED ON RISK/CONFIG**

        order_placed = False
        if action == actions.BUY and position != positions.LONG: # Buy only if not already Long
            logger.info(f"Placing BUY order for {stock_symbol}, qty: {qty}")
            await self.place_order_wrapper(stock_symbol, 'BUY', qty)
            order_placed = True
        elif action == actions.SELL and position != positions.SHORT: # Sell only if not already Short
            logger.info(f"Placing SELL order for {stock_symbol}, qty: {qty}")
            await self.place_order_wrapper(stock_symbol, 'SELL', qty)
            order_placed = True
        elif action == actions.DOUBLE_BUY and position == positions.SHORT: # Cover short and go long
            logger.info(f"Placing DOUBLE BUY (2 orders) for {stock_symbol}, qty: {qty} each")
            await self.place_order_wrapper(stock_symbol, 'BUY', qty) # Close short
            await asyncio.sleep(0.5) # Slight delay
            await self.place_order_wrapper(stock_symbol, 'BUY', qty) # Open long
            order_placed = True
        elif action == actions.DOUBLE_SELL and position == positions.LONG: # Cover long and go short
            logger.info(f"Placing DOUBLE SELL (2 orders) for {stock_symbol}, qty: {qty} each")
            await self.place_order_wrapper(stock_symbol, 'SELL', qty) # Close long
            await asyncio.sleep(0.5) # Slight delay
            await self.place_order_wrapper(stock_symbol, 'SELL', qty) # Open short
            order_placed = True
        elif action == actions.HOLD:
             logger.info(f"Action is HOLD for {stock_symbol}. No order placed.")
        else:
             logger.warning(f"Action {action.name} vs Position {position.name} for {stock_symbol} results in no order.")

        if order_placed: logger.info(f"Order placement attempted for {stock_symbol}.")


    async def place_order_wrapper(self, stock_symbol: str, action: str, quantity: float) -> Optional[Any]:
        """ Helper to qualify contract and place a market order. """
        if not self.api.isConnected(): logger.error(f"Cannot place order: Not connected."); return None
        trade_info = None
        try:
            contract = self.create_contract(stock_symbol)
            qualified_contract = await self.qualify_contract(contract)
            if not qualified_contract: logger.error(f"Order failed: Could not qualify {stock_symbol}."); return None

            order = self.create_buy_order(quantity) if action == 'BUY' else self.create_sell_order(quantity)
            logger.info(f"Placing {action} {quantity} {stock_symbol} ({order.orderType})...")

            # placeOrder is synchronous in ib_async but should be called from the loop thread
            trade = self.api.placeOrder(qualified_contract, order)
            trade_info = trade # Store trade info

            # Log trade status updates asynchronously
            async def log_trade_status():
                 async for _, fill in trade.fillEvent: # New async way to get fills
                     logger.info(f"Order FILLED: {fill.execution}")
                 # Other status events can be monitored similarly if needed (trade.commissionReportEvent, etc)
                 # Log final status
                 await trade.statusEvent # Wait for final status
                 logger.info(f"Final trade status for {stock_symbol} ({trade.order.orderId}): {trade.orderStatus.status}")

            asyncio.create_task(log_trade_status()) # Run status logging concurrently

        except Exception as e:
            logger.error(f"Failed to place {action} order for {stock_symbol}: {e}", exc_info=True)
        return trade_info # Return the initial Trade object

    def get_env_and_model(self, interface_obj: InterfaceObject, stock_symbol: str):
        """ Gets RL env/model using historical and latest live data. (Adapted) """
        # This implementation needs careful review based on your actual data structures
        # and how DataManager/TradingBot process them. Using Mocks for now.
        logger.debug(f"Getting env/model for {stock_symbol}")
        # application_manager = self._get_application_manager()
        # data_manager = application_manager._get_data_manager()
        market_data_entry = interface_obj.market_data.seq.get(stock_symbol)

        if not market_data_entry or market_data_entry[0] is None or market_data_entry[0].empty:
            logger.warning(f"Insufficient historical data for {stock_symbol} in get_env_and_model.")
            # Attempt to use only live data if available? Requires env adaptation.
            if market_data_entry and market_data_entry[1] is not None and len(market_data_entry[1]) > 0:
                 logger.info("Using only live data for env/model.")
                 live_df = ib_async_util.df(market_data_entry[1])
                 if live_df is None or live_df.empty: return None, None # No data at all
                 combined_df = live_df.reset_index(drop=True)
            else:
                return None, None # No data available
        else:
            # Combine historical and live data
            historical_df = market_data_entry[0]
            live_bars = market_data_entry[1]
            latest_df = pd.DataFrame()
            if live_bars and len(live_bars) > 0:
                try: latest_df = ib_async_util.df(live_bars)
                except: pass # Ignore conversion errors
                if latest_df is None: latest_df = pd.DataFrame()

            # Clean column names if needed (ib_async sometimes adds '_')
            rename_cols = {c + '_': c for c in ['open','high','low','close'] if c + '_' in latest_df.columns}
            if rename_cols: latest_df = latest_df.rename(columns=rename_cols)

            if not latest_df.empty:
                # Ensure columns match for concat; use intersection
                common_cols = historical_df.columns.intersection(latest_df.columns)
                combined_df = pd.concat(
                    [historical_df[list(common_cols)], latest_df[list(common_cols)]]
                ).drop_duplicates().reset_index(drop=True)
            else:
                combined_df = historical_df.reset_index(drop=True)

        logger.debug(f"Combined DF len for {stock_symbol}: {len(combined_df)}")
        if combined_df.empty: logger.error(f"Combined DF empty for {stock_symbol}."); return None, None

        # Limit history length (e.g., last 128 bars for env state)
        history_len = 128
        # start_idx = max(0, len(combined_df) - history_len)
        final_slice = combined_df.loc[-history_len:]
        # print(final_slice)

        if final_slice.empty: logger.error(f"Final slice empty for {stock_symbol}."); return None, None
        logger.debug(f"Using final slice of {len(final_slice)} bars for env.")

        # Prepare dataset using DataManager (assuming it takes a DataFrame)
        dataset_for_env = get_dataset(
             interface_obj.application_obj.data_obj, final_slice
        )

        # Get model and env
        model, env = get_integrated_model(
            interface_obj.application_obj.model_engine_obj,
            interface_obj.application_obj.ai_model_obj,
            dataset=[dataset_for_env],
            mode='test'
        )
        return model, env


    # --- Cleanup ---
    async def cleanup_all(self, interface_obj: InterfaceObject):
        """ Perform complete cleanup (async). """
        logger.info("Starting async cleanup...")

        # 1. Cancel state machine task
        if self._state_machine_task and not self._state_machine_task.done():
             logger.info("Cancelling state machine task...")
             self._state_machine_task.cancel()
             try: await self._state_machine_task
             except asyncio.CancelledError: logger.info("State machine task cancelled.")

        # 2. Cancel real-time streams (optional, disconnect often sufficient)
        symbols = list(self._live_bar_streams.keys())
        if symbols:
             logger.info(f"Cleaning up real-time streams for {symbols} (via disconnect)...")
             # Explicit cancellation is complex; rely on disconnect for now.
             # for symbol in symbols:
             #     bars = self._live_bar_streams.pop(symbol)
             #     # Remove callbacks if possible / needed
             self._live_bar_streams.clear()

        # 3. Disconnect API
        if self.api.isConnected():
            logger.info("Disconnecting from IB API...")
            self.api.disconnect()
            # Wait briefly? Loop should end automatically.
        else: logger.info("IB API already disconnected.")

        # 4. Stop TWS Gateway (synchronous, consider executor if slow)
        # Ensure sudo permissions are set: user ALL=(ALL) NOPASSWD: /opt/ibc/stop.sh
        logger.info("Stopping TWS Gateway (sync)...")
        try:
            # Run in executor to avoid blocking the closing loop? Or accept brief block.
            await asyncio.get_event_loop().run_in_executor(None, os.system, 'sudo /opt/ibc/stop.sh')
            os.system('sudo /opt/ibc/stop.sh') # Simpler, potentially blocking
        except Exception as e:
            logger.error(f"Failed to execute TWS stop script: {e}")

        logger.info("Async cleanup finished.")

    # --- State Machine Logic (Async) ---
    async def _async_step(self, interface_obj: InterfaceObject, mode: str):
        """ Async version of the state machine step. """
        logger.debug(f"Async step check. State: {self.state.name}")

        # --- Handler Dispatch ---
        state_handlers = {
            State.UNSTARTED: self._handle_unstarted,
            State.DOWNLOAD: self._handle_download,
            State.DOWNLOADING: self._handle_downloading,
            State.LOITERING: self._handle_loitering,
            # State.TRIGGERED: self._handle_triggered,
            State.ERROR: self._handle_error,
            State.CLEANUP: self._handle_cleanup,
        }
        handler = state_handlers.get(self.state)

        if handler:
            try:
                # Await async handlers, run sync handlers directly
                if asyncio.iscoroutinefunction(handler): await handler(interface_obj, mode)
                else: handler(interface_obj, mode)
            except asyncio.CancelledError: logger.info(f"Handler for {self.state.name} cancelled."); raise
            except Exception as e:
                 logger.error(f"Error in handler for state {self.state.name}: {e}", exc_info=True)
                 self.state = State.ERROR
        elif self.state != State.DONE:
             logger.warning(f"No handler for state: {self.state.name}")

        # --- Time/Condition Based Triggers (if Loitering) ---
        if self.state == State.LOITERING:
             trigger_action = self._check_time_triggers() # Check for scheduled actions
             if trigger_action:
                  logger.info(f"Time-based trigger: {trigger_action.name}. Setting state.")
                  if trigger_action == Trigger.INITIALIZE: self.state = State.UNSTARTED # Re-initialize? Risky.
                  elif trigger_action == Trigger.DOWNLOAD: self.state = State.DOWNLOAD
                  elif trigger_action == Trigger.CLEANUP: self.state = State.CLEANUP
                  # Add other triggers if needed

    # --- State Handlers (Async adaptations) ---
    async def _handle_unstarted(self, interface_obj: InterfaceObject, mode: str):
        """ Try to initialize and connect. """
        logger.info("Handling UNSTARTED state...")
        if not self.api.isConnected():
            try:
                # Start TWS Gateway (sync for now)
                logger.info("Starting TWS Gateway (sync)...")
                # Ensure sudo: user ALL=(ALL) NOPASSWD: /opt/ibc/twsstart.sh
                os.system('sudo /opt/ibc/twsstart.sh')
                await asyncio.sleep(20) # Increased wait time for TWS startup

                logger.info(f"Connecting to IBKR @ {interface_obj.host}:{interface_obj.port}...")
                await self.api.connectAsync(interface_obj.host, interface_obj.port, clientId=interface_obj.client_id)
                # Wait a moment for connection events
                await asyncio.sleep(2)

                if self.api.isConnected():
                    logger.info("Connection successful. Requesting initial positions...")
                    await self.update_positions(interface_obj) # Get initial positions
                    logger.info("Moving to DOWNLOAD state.")
                    self.state = State.DOWNLOAD
                else:
                    logger.error("Connection failed after attempt. Moving to ERROR.")
                    self.state = State.ERROR

            except Exception as e:
                 logger.error(f"Error during initialization/connection: {e}", exc_info=True)
                 self.state = State.ERROR
        else:
             logger.info("Already connected. Moving from UNSTARTED to DOWNLOAD.")
             self.state = State.DOWNLOAD

    async def _handle_download(self, interface_obj: InterfaceObject, mode: str):
        """ Initiate async historical download. """
        if self._download_future and not self._download_future.done():
            logger.warning("Download task still running. Staying in DOWNLOADING.")
            self.state = State.DOWNLOADING; return

        logger.info("Handling DOWNLOAD state: Starting historical data fetch...")
        self._data_queue = asyncio.Queue(maxsize=100)
        stock_symbols = [s.stock_symbol for s in interface_obj.stocks_to_consider]
        if not stock_symbols: logger.warning("No stocks selected."); self.state = State.LOITERING; return

        coro = self.download(interface_obj, self._data_queue, stock_symbols)
        self._download_future = asyncio.create_task(coro, name="HistoricalDownloadTask")
        logger.info(f"Historical download task created ({self._download_future.get_name()}). Moving to DOWNLOADING.")
        self.state = State.DOWNLOADING

    async def _handle_downloading(self, interface_obj: InterfaceObject, mode: str):
        """ Monitor download task and process queue. """
        if not self._download_future or not self._data_queue:
            logger.error("Invalid state: In DOWNLOADING without future/queue."); self.state = State.ERROR; return

        logger.debug("Handling DOWNLOADING state...")
        # Process queue items
        try:
            while not self._data_queue.empty():
                item = self._data_queue.get_nowait() # Non-blocking get
                if item is None: logger.debug("Got None sentinel from queue."); continue # Wait for future

                if isinstance(item, dict) and 'symbol' in item and 'data' in item:
                    symbol, df = item['symbol'], item['data']
                    logger.debug(f"Processing hist data for {symbol} ({len(df)} rows). QSize={self._data_queue.qsize()}")
                    if symbol not in interface_obj.market_data.seq: interface_obj.market_data.seq[symbol] = [None, None]
                    interface_obj.market_data.seq[symbol][0] = df # Store hist df
                    logger.info(f"Stored historical data for {symbol}.")
                else: logger.warning(f"Unexpected item in download queue: {item}")
                self._data_queue.task_done()
        except asyncio.QueueEmpty: pass # Expected when queue is empty
        except Exception as e: logger.error(f"Error processing download queue: {e}", exc_info=True)

        # Check if download task finished
        if self._download_future.done():
            logger.info("Historical download future is done.")
            try:
                await self._download_future # Raise exception if task failed
                logger.info("Historical download task completed successfully.")
                stock_symbols = [s.stock_symbol for s in interface_obj.stocks_to_consider]
                if stock_symbols:
                     logger.info("Starting real-time streams...")
                     await self.start_realtime_streams(interface_obj, stock_symbols)
                self.state = State.LOITERING
            except asyncio.CancelledError: logger.warning("Download task cancelled."); self.state = State.CLEANUP
            except Exception as e: logger.error(f"Download task failed: {e}"); self.state = State.ERROR
            finally: self._download_future = None; self._data_queue = None
        # else: logger.debug("Download task still running...")


    async def _handle_loitering(self, interface_obj: InterfaceObject, mode: str):
        """ Wait for triggers (real-time callbacks or time-based). """
        logger.debug("Loitering... Awaiting triggers.")
        # Time checks happen in _async_step loop
        # Real-time callbacks trigger state change directly
        await asyncio.sleep(5) # Prevent busy-waiting; adjust as needed


    async def _handle_triggered(self, interface_obj: InterfaceObject, mode: str):
        """ Execute trade logic for the triggered symbol. """
        logger.info("Handling TRIGGERED state...")
        symbol = self._triggered_symbol
        if not symbol:
            logger.warning("TRIGGERED state entered without symbol context. Returning to LOITERING.")
            self.state = State.LOITERING; return

        logger.info(f"Processing trigger for symbol: {symbol}")
        try:
            await self.trade(interface_obj, symbol, mode)
        except Exception as e:
            logger.error(f"Error during trade execution for {symbol}: {e}", exc_info=True)
            self.state = State.ERROR
        finally:
            # Reset context and return to loitering IF NOT in error state
            self._triggered_symbol = None
            if self.state == State.TRIGGERED: # Only change if not already ERROR
                logger.info(f"Trade processing complete for {symbol}. Returning to LOITERING.")
                self.state = State.LOITERING


    async def _handle_error(self, interface_obj: InterfaceObject, mode: str):
        """ Log error and move to cleanup. """
        logger.error("Handling ERROR state. Moving to CLEANUP.")
        # Potential: Try emergency close of positions? Risky.
        # await self.clear_all_positions(interface_obj)
        self.state = State.CLEANUP

    async def _handle_cleanup(self, interface_obj: InterfaceObject, mode: str):
        """ Run async cleanup procedures. """
        logger.info("Handling CLEANUP state...")
        await self.cleanup_all(interface_obj)
        self.state = State.DONE # Final state

    # --- Time/Holiday Checks ---
    def _is_market_closed(self) -> bool:
        """ Checks for weekends and NYSE holidays. """
        now_local = datetime.now()
        today = now_local.date()
        # Check weekend (Saturday=5, Sunday=6)
        if today.weekday() >= 5: logger.debug('Weekend - Market Closed'); return True
        # Check NYSE holidays
        if today in self.nyse_holidays: logger.info(f'NYSE Holiday Today ({self.nyse_holidays.get(today)}). Market Closed.'); return True
        return False

    def _is_time_between(self, check_time: datetime_time, start: tuple, end: tuple) -> bool:
        """ Checks if a time falls within a start/end range. """
        start_time = datetime_time(start[0], start[1], tzinfo=self.tz_new_york)
        end_time = datetime_time(end[0], end[1], tzinfo=self.tz_new_york)
        check_time_aware = check_time.replace(tzinfo=self.tz_new_york) if check_time.tzinfo is None else check_time
        # Handle overnight ranges if needed (e.g., end < start) - simplified here
        return start_time <= check_time_aware <= end_time

    def _check_time_triggers(self) -> Optional[Trigger]:
        """ Checks current time against predefined action windows (NY Time). """
        if self._is_market_closed(): return Trigger.CLEANUP # Trigger cleanup if market closed unexpectedly? Or just wait?

        now_ny = datetime.now(self.tz_new_york).time()
        logger.debug(f"Checking time triggers. Current NY time: {now_ny.strftime('%H:%M:%S')}")

        # Example: Cleanup window shortly after midnight NY
        if self._is_time_between(now_ny, (0, 1), (0, 15)):
            logger.info("Within cleanup time window.")
            return Trigger.CLEANUP

        # Example: Initialization window before market open
        if self._is_time_between(now_ny, (6, 0), (6, 15)):
             logger.info("Within initialization time window.")
             # Should only trigger if state is suitable (e.g., LOITERING after cleanup)
             # Avoid re-initializing constantly. Check state first?
             if self.state == State.LOITERING: # Or maybe a dedicated post-cleanup state
                 return Trigger.INITIALIZE # Careful with triggering UNSTARTED
             else:
                 logger.debug("Initialize time window, but state not LOITERING.")


        # Example: Data download window (if needed daily outside streams)
        # if self._is_time_between(now_ny, (7, 0), (7, 15)):
        #     logger.info("Within download time window.")
        #     if self.state == State.LOITERING: return Trigger.DOWNLOAD

        return None # No time trigger active
    
    def run(self, interface_obj: InterfaceObject, mode: str = 'demo'):
        """ Sets up and runs the main ib_async event loop with the state machine task. """
        self._interface_obj = interface_obj
        self._mode = mode
        logger.info(f"Engine starting in '{mode}' mode...")

        async def main_async_runner():
            """ Contains the primary async logic: running the state machine steps. """
            logger.debug("Main async runner started.")
            # Start the state machine task
            self._state_machine_task = asyncio.create_task(
                _run_state_machine_steps(), name="StateMachineLoop"
            )
            try:
                await self._state_machine_task # Wait for state machine to finish (reaches DONE)
                logger.info("State machine task completed.")
            except asyncio.CancelledError:
                logger.info("Main async runner cancelled.")
            except Exception as e:
                 logger.critical(f"Fatal error in main_async_runner: {e}", exc_info=True)
                 # Ensure cleanup is triggered
                 if self.state not in [State.CLEANUP, State.DONE]: self.state = State.CLEANUP
                 # May need to manually run cleanup step if loop dies here
                 await self._handle_cleanup(self._interface_obj, self._mode)


        async def _run_state_machine_steps():
            """ The core loop calling _async_step periodically. """
            logger.info("State machine step loop starting.")
            try:
                while self.state != State.DONE:
                    await self._async_step(self._interface_obj, self._mode)
                    # Dynamic sleep based on state
                    sleep_time = 1.0 # Default check interval
                    if self.state == State.LOITERING: sleep_time = 5.0
                    elif self.state == State.DOWNLOADING: sleep_time = 0.5
                    elif self.state in [State.CLEANUP, State.ERROR]: sleep_time = 0.2
                    await asyncio.sleep(sleep_time)
                logger.info("State machine loop finished naturally (State = DONE).")
            except asyncio.CancelledError:
                logger.info("State machine step loop received cancellation.")
                # If cancelled, ensure cleanup happens
                if self.state != State.DONE:
                     logger.warning("Step loop cancelled before DONE, initiating cleanup...")
                     await self._handle_cleanup(self._interface_obj, self._mode) # Run final cleanup
                raise # Re-raise cancellation


        try:
            # ib_async run handles loop creation, running main_async_runner, and cleanup on exit
            self.api.run(main_async_runner())
        except (KeyboardInterrupt, SystemExit):
            logger.warning("Shutdown signal received (KeyboardInterrupt/SystemExit).")
            if self.state != State.DONE:
                self.state = State.CLEANUP
                self.api.run(main_async_runner())
            # ib_async run() should handle cancelling tasks and stopping the loop.
            # Cleanup is handled within main_async_runner or _run_state_machine_steps cancellation.
        except Exception as e:
             logger.critical(f"Fatal error during ib_async run: {e}", exc_info=True)
        finally:
             logger.info("ib_async run has exited.")
             # Any final synchronous cleanup outside the loop if absolutely necessary
        
        # self.api.run(main_async_runner())