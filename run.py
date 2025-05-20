from pprint import pprint

from ib_async_trading_bot.logger import get_logger
from ib_async_trading_bot.objects import (
    TradingBotSB3Object,
    InterfaceObject,
    StockSymbol
)
from ib_async_trading_bot.interface_engine_interactive_brokers import InterfaceEngineInteractiveBrokers

logger = get_logger()

# --- Main Execution Block Example ---
if __name__ == "__main__":
    engine = InterfaceEngineInteractiveBrokers()
    stock_symbol_list = ['EURUSD', 'EURCAD', 'EURAUD']
    stock_symbol_list = [StockSymbol(i) for i in stock_symbol_list]
    # stock_symbol_list = ['top_gainers_processed_test']
    application_obj = TradingBotSB3Object(
        stock_symbol_list=stock_symbol_list
    )
    interface_obj = InterfaceObject(
        application_obj = application_obj,
        stocks_to_consider=stock_symbol_list
    )

    logger.info("### Starting Interactive Brokers Interface Engine ###")
    try:
        engine.run(interface_obj, mode='demo') # Use 'deploy' for actual IB connection
    finally:
        logger.info("### Interactive Brokers Interface Engine Finished ###")