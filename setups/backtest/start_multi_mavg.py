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
import warnings

from itertools import product
from strategies.crypto import MovingAverageCrossoverStrategy
from event import SignalEvent
from backtest import MultiCryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def compute_param_grid(fixed_params_list, variable_params_list):
      fixed_params = [ [fixed_params_list[key]] for key in list(fixed_params_list.keys()) ]
      variable_params = [ variable_params_list[key] for key in list(variable_params_list.keys()) ]
      fixed_params_names = list(fixed_params_list.keys())
      variable_params_names = list(variable_params_list.keys())

      param_names = [ *fixed_params_names, *variable_params_names ]
      params = list(product(*fixed_params, *variable_params))
      strat_params = [ {param_names[i]: params[j][i] for i in range(len(params[0])) } for j in range(len(params)) ]

      return fixed_params_names, variable_params_names, strat_params

if __name__ == "__main__":
    fixed_params_list = {}
    variable_params_list = { 'short_window': [10, 20, 30], 'long_window': [40, 50, 60] }
    params = compute_param_grid(fixed_params_list, variable_params_list)

    configuration = {
      'csv_dir' : '../../data',
      'result_dir' : '../../results',
      'instruments': { 'bitmex': [ 'BTC/USD', 'ETH/USD' ]},
      'heartbeat' : 0.00,
      'ohclv_window': 60.00,
      'initial_capital' : 100000.0,
      'start_date' : datetime.datetime(2019, 4, 1, 0, 0, 0),
      'strat_params': params[2],
      'params_names': ['ols_window', 'zscore_entry', 'zscore_exit']
    }

    backtest = MultiCryptoBacktest(
        configuration, HistoricCSVCryptoDataHandler, SimulatedCryptoExchangeExecutionHandler,
        CryptoPortfolio, MovingAverageCrossoverStrategy
    )

    backtest.start_trading()