#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(dir(path[0])))
    __package__ = "setups"

import math

from itertools import product
from strategies.crypto.mr import OLSMeanReversionStrategy
from event import SignalEvent
from trader import CryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio
from datetime import datetime, timedelta
from configuration import Configuration

if __name__ == "__main__":
    configuration = Configuration({
      'csv_dir' : '../../data',
      'result_dir' : '../../results',
      'instruments': { 'bitmex': ['BTC/USD', 'ETH/USD'] },
      'initial_capital' : 100000.0,
      'ohlcv_window': 60,
      'graph_refresh_period': 100,
      'heartbeat' : 0,
      'start_date' : datetime(2019, 3, 25, 0, 0, 0),
    })

    backtest = CryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        CryptoPortfolio, OLSMeanReversionStrategy
    )

    backtest.start_trading()