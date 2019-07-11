#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import product
from datetime import datetime

class Configuration(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """

    def __init__(self, configuration, default):
      self.heartbeat = configuration.get("heartbeat", default.get("heartbeat"))
      self.show_charts = configuration.get("show_charts", default.get("show_charts"))
      self.update_charts = configuration.get("update_charts", default.get("update_charts"))
      self.save_to_db = configuration.get("save_to_db", default.get("save_to_db"))
      self.graph_refresh_period = configuration.get("graph_refresh_period", default.get("graph_refresh_period"))
      self.csv_dir = configuration.get("csv_dir", default.get("csv_dir"))
      self.result_dir = configuration.get("result_dir", default.get("result_dir"))
      self.conditions = None

      timeframe = configuration.get("timeframe")
      self.timeframe = {
        '10s': 10,
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '1h': 3600,
        '1d': 86400
      }[timeframe]


      self.backtest_date = datetime.utcnow()
      self.instruments = configuration.get("instruments")
      self.strategy = configuration.get("strategy")
      self.initial_capital = configuration.get("initial_capital")
      self.feeds = configuration.get("feeds")
      self.assets = configuration.get("assets")
      self.indicators = configuration.get("indicators", [])
      self.default_position_size = configuration.get("default_position_size")
      self.start_date = configuration.get("start_date")
      self.end_date = configuration.get("end_date", None)
      self.start_dates = configuration.get("start_dates", None)
      self.end_dates = configuration.get("end_dates", None)
      self.use_stops = configuration.get("use_stops", True)
      self.take_profit_gap = configuration.get("take_profit_gap", 0.1)
      self.stop_loss_gap = configuration.get("stop_loss_gap", 0.1)
      self.default_leverage = configuration.get("default_leverage", 1)

      # There are three ways to set the backtest name.
      # 1) Add backtest_name field in the .json configuration file
      # 2) Add the backtest name through the command line argument -n.
      # 3) Otherwise the backtest name will default to the strategy name
      self.backtest_name = configuration.get("backtest_name", self.strategy)

      # In the case of a multi instrument backtest, the instruments keys is an array
      if isinstance(self.instruments, list):
        self.exchange_names = list(self.instruments[0].keys())
      else:
        self.exchange_names = list(self.instruments.keys())


      if 'strategy_params' in configuration:
        if configuration['backtester_type'] == "simple_backtest":
          self.params_names = list(configuration['strategy_params'].keys())
          self.strategy_params = configuration.get('strategy_params')
        elif configuration['backtester_type'] == "super_backtest":
          params_names, params_dict = self._compute_params_dict(configuration['strategy_params'])
          self.params_names = params_names
          self.strategy_params = params_dict


      # Initial bars is the number of bars that are considered already
      # past when the backtest is started and will thus not be fed into the event loop
      default_initial_bars = {
          10: 300,
          60: 300,
          300: 300,
          900: 300,
          3600: 300,
          86400: 10
      }[self.timeframe]

      self.initial_bars = configuration.get("initial_bars", default_initial_bars)

    def set_configuration_filename(self, filename):
      self.configuration_filename = filename

    def set_configuration_backtest_name(self, backtest_name):
      self.backtest_name = backtest_name

    def set_logger(self, logger):
      self.logger = logger

    def _compute_params_dict(self, params):
      params_names = list(params.keys())
      params_product_list = list(product(*params.values()))

      for key in params_names:
        self.__dict__[key] = params[key]

      params_dic = []
      for p in params_product_list:
        dic = {}
        for (i,pn) in enumerate(params_names):
          dic[pn] = p[i]

        params_dic.append(dic)

      return params_names, params_dic