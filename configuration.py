#!/usr/bin/python
# -*- coding: utf-8 -*-

# event.py

from __future__ import print_function
from itertools import product

class Configuration(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """

    def __init__(self, configuration):
      self.result_dir = configuration['result_dir']
      self.instruments = configuration['instruments']
      self.heartbeat = configuration['heartbeat']
      self.conditions = None

      # In the case of a multi instrument backtest, the instruments keys is an array
      if isinstance(self.instruments, list):
        self.exchange_names = list(self.instruments[0].keys())
      else:
        self.exchange_names = list(self.instruments.keys())

      self.ohlcv_window = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '1d': 86400
      }[configuration['ohlcv_window']]

      if 'csv_dir' in configuration:
        self.csv_dir = configuration['csv_dir']

      if 'initial_capital' in configuration:
        self.initial_capital = configuration['initial_capital']

      if 'graph_refresh_period' in configuration:
        self.graph_refresh_period = configuration['graph_refresh_period']

      if 'feeds' in configuration:
        self.feeds = configuration['feeds']

      if 'assets' in configuration:
        self.assets = configuration['assets']

      if 'strategy_params' in configuration:
        self.strategy_params = configuration['strategy_params']

      if 'indicators' in configuration:
        self.indicators = configuration['indicators']
      else:
        self.indicators = []

      if 'default_position_size' in configuration:
        self.default_position_size = configuration['default_position_size']

      if 'show_charts' in configuration:
        self.show_charts = configuration['show_charts']
      else:
        self.show_charts = True

      if 'update_charts' in configuration:
        self.update_charts = configuration['update_charts']
      else:
        self.update_charts = True

      if 'start_date' in configuration:
        self.start_date = configuration['start_date']
      else:
        self.start_date = None

      if 'end_date' in configuration:
        self.end_date = configuration['end_date']
      else:
        self.end_date = None

      if 'start_dates' in configuration:
        self.start_dates = configuration['start_dates']
      else:
        self.start_dates = None

      if 'end_dates' in configuration:
        self.end_dates = configuration['end_dates']
      else:
        self.end_dates = None



      if 'default_leverage' in configuration:
        self.default_leverage = configuration['default_leverage']
      else:
        self.default_leverage = 1

      if 'save_to_db' in configuration:
        self.save_to_db = configuration['save_to_db']
      else:
        self.save_to_db = False

      # Initial bars represents the number of bars that are considered already
      # past when the backtest is started and will thus not be fed into the event loop
      if 'initial_bars' in configuration:
        self.initial_bars = configuration['initial_bars']
      else:
        self.initial_bars = {
          60: 300,
          3600: 300,
          86400: 10
        }[self.ohlcv_window]


class MultiMRConfiguration(Configuration):

    def __init__(self, configuration):
      super().__init__(configuration)

      params = configuration['strategy_params']
      params_names_list = list(params.keys())
      params_product_list = list(product(*params.values()))

      for key in params_names_list:
        self.__dict__[key] = params[key]

      self.params_names = params_names_list

      self.strat_params = []
      for p in params_product_list:
        dic = {}
        for (i,pn) in enumerate(params_names_list):
          dic[pn] = p[i]

        self.strat_params.append(dic)


class MultiConditionConfiguration(Configuration):

    def __init__(self, configuration):
      super().__init__(configuration)

      conditions = configuration['conditions']
      conditions_names_list = list(conditions.keys())
      conditions_product_list = list(product(*conditions.values()))

      for key in conditions_names_list:
        self.__dict__[key] = conditions[key]

      self.conditions_names = conditions_names_list

      self.strat_conditions = []
      for p in conditions_product_list:
        dic = {}
        for (i,pn) in enumerate(conditions_names_list):
          dic[pn] = p[i]

        self.strat_conditions.append(dic)