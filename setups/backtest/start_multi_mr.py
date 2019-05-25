#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(dir(path[0])))
    __package__ = "setups"

import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm

from itertools import product
from strategies.crypto import OLSMeanReversionStrategy
from event import SignalEvent
from trader import MultiCryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio
from configuration import MultiMRConfiguration

if __name__ == "__main__":
    strat_lookback = [50, 100, 200]
    strat_z_entry = [2.0, 3.0, 4.0]
    strat_z_exit = [0.5, 1.0, 1.5]
    strat_params_list = list(product(strat_lookback, strat_z_entry, strat_z_exit))
    strat_params_dict = [ dict(ols_window=sp[0], zscore_entry=sp[1], zscore_exit=sp[2]) for sp in strat_params_list ]

    #Create a list of dictionaries with the correct keyword/value pairs for the strategy parameters
    configuration = MultiMRConfiguration({
      'csv_dir': '../../data',
      'result_dir': '../../results',
      'instruments': { 'bitmex': [ 'BTC/USD', 'ETH/USD' ]},
      'heartbeat': 0,
      'ohlcv_window': 60.00,
      'initial_capital': 100000.0,
      'start_date': datetime.datetime(2019, 4, 1, 0, 0, 0),
      'strat_lookback': strat_lookback,
      'strat_z_entry': strat_z_entry,
      'strat_z_exit': strat_z_exit,
      'strat_params': strat_params_dict,
      'params_names': ['ols_window', 'zscore_entry', 'zscore_exit'],
    })

    backtest = MultiCryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        CryptoPortfolio, OLSMeanReversionStrategy
    )

    backtest.start_trading()