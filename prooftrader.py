#!/usr/bin/env python3

import sys
import json
import logging
import warnings
import argparse
import importlib

from trader import *
from datahandler.crypto import HistoricCSVCryptoDataHandler, LiveDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler, LiveExecutionHandler
from portfolio import BitmexPortfolioBacktest, CryptoPortfolio, BitmexPortfolio
from strategies.crypto.multi_random import MultiRandomStrategy
from configuration import Configuration
from utils.log import get_logger

from strategies.crypto import *

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def parse_args():
  parser = argparse.ArgumentParser(description='Backtest')

  parser.add_argument('-l', '--live',
                      default=False,
                      action='store_true',
                      help='Add the live flag to activate live trading'
                     )

  parser.add_argument('-f', '--file',
                      type=str,
                      required=True,
                      help='The name of the configuration JSON file')

  parser.add_argument('-n', '--name',
                      type=str,
                      required=False,
                      help='The backtest name')

  parser.add_argument('-c', '--conditions',
                      type=str,
                      required=False,
                      help='The name of the python module containing list of conditions to be tested')

  return parser.parse_args()


def create_backtester(config_filepath, settings_filepath, backtest_name):
    with open(config_filepath) as config_file, open(settings_filepath) as settings_file:
      config = json.load(config_file)
      default_settings = json.load(settings_file)

      strategies = {
        "qda": QDAStrategy,
        "rsi": RSIStrategy,
        "random": MultiRandomStrategy,
        "mean_reversion": SingleOLSMeanReversionStrategy,
        "generalized_mean_reversion": OLSMeanReversionStrategy,
        "moving_average_crossover": MovingAverageCrossoverStrategy,
        "macd_crossover": MACDCrossover,
        "condition": ConditionBasedStrategy,
        "momentum": MomentumStrategy,
        "only_short_momentum": OnlyShortMomentumStrategy,
        "only_long_momentum": OnlyLongMomentumStrategy,
        "volume": VolumeStrategy
      }

      backtesters = {
        "simple_backtest": SimpleBacktest,
        "multi_params_backtest": MultiParamsBacktest,
        "multi_conditions_backtest": MultiConditionsBacktest,
        "super_backtest": SuperBacktest
      }

      portfolios = {
        "crypto_portfolio": CryptoPortfolio,
        "bitmex_portfolio": BitmexPortfolioBacktest
      }

      configuration = Configuration(config, default_settings)
      configuration.set_configuration_filename(config_filepath)

      logger = get_logger(configuration)
      configuration.set_logger(logger)


      if backtest_name is not None:
        configuration.set_configuration_backtest_name(backtest_name)

      # if args.conditions:
      #   configuration.conditions = condition_file.conditions

      backtester_class = backtesters[config['backtester_type']]
      portfolio_class = portfolios[config['portfolio_type']]
      strategy_class = strategies[config['strategy']]

      backtest = backtester_class(
          configuration,
          HistoricCSVCryptoDataHandler,
          SimulatedCryptoExchangeExecutionHandler,
          portfolio_class,
          strategy_class
      )

      return backtest





def create_livetrader(config_filepath, settings_filepath):
    with open(config_filepath) as config_file, open(settings_filepath) as settings_file:
      config = json.load(config_file)
      default_settings = json.load(settings_file)

      strategies = {
        "qda": QDAStrategy,
        "rsi": RSIStrategy,
        "random": MultiRandomStrategy,
        "mean_reversion": SingleOLSMeanReversionStrategy,
        "generalized_mean_reversion": OLSMeanReversionStrategy,
        "moving_average_crossover": MovingAverageCrossoverStrategy,
        "macd_crossover": MACDCrossover,
        "condition": ConditionBasedStrategy,
        "momentum": MomentumStrategy,
        "only_short_momentum": OnlyShortMomentumStrategy,
        "only_long_momentum": OnlyLongMomentumStrategy,
        "volume": VolumeStrategy
      }

      configuration = Configuration(config, default_settings)
      configuration.set_configuration_filename(config_filepath)

      logger = get_logger(configuration)
      configuration.set_logger(logger)

      strategy_class = strategies[config['strategy']]
      livetrader_class = CryptoLiveTrade
      portfolio_class = BitmexPortfolio

      trader = livetrader_class(
          configuration,
          LiveDataHandler,
          LiveExecutionHandler,
          portfolio_class,
          strategy_class
      )

      return trader


args = parse_args()

if args.conditions:
  condition_file = importlib.import_module(args.conditions)

if args.name:
  name = args.name

config_filepath = args.file
settings_filepath = "./settings/default_settings.json"

if args.live:
  trader = create_livetrader(config_filepath, settings_filepath)
  trader.start_trading()
else:
  backtester = create_backtester(config_filepath, settings_filepath, name)
  backtester.start_trading()