#!/usr/bin/python
# -*- coding: utf-8 -*-

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(dir(path[0])))
    __package__ = "setups"

import datetime
import warnings

from event import SignalEvent
from trader import CryptoBacktest
from strategies.crypto import CrossExchangeOLSMeanReversionStrategy
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio
from configuration import Configuration

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    configuration = Configuration({
      'csv_dir': '../../data',
      'result_dir': '../../results',
      'instruments': { 'bitmex': ['BTC/USD'], 'binance': ['BTC/USD']},
      'ohlcv_window': 60,
      'initial_capital': 100000.0,
      'heartbeat': 0.00,
      'start_date': datetime.datetime(2019, 3, 25, 0, 0, 0),
    })

    backtest = CryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        CryptoPortfolio, CrossExchangeOLSMeanReversionStrategy
    )

    backtest.start_trading()