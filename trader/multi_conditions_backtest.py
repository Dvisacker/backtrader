#!/usr/bin/python
# -*- coding: utf-8 -*-

# backtest.py

from __future__ import print_function

import datetime
import time
import pprint
import os

from datetime import datetime
from distutils.dir_util import copy_tree

try:
    import Queue as queue
except ImportError:
    import queue


class MultiConditionsBacktest(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """

    def __init__(self, configuration, data_handler, execution_handler, portfolio, strategy):
        """
        Initialises the backtest.
        Parameters:
        csv_dir - The hard root to the CSV data directory.
        instruments - The list of symbol strings.
        intial_capital - The starting capital for the portfolio.
        heartbeat - Backtest "heartbeat" in seconds
        start_date - The start datetime of the strategy.
        data_handler - (Class) Handles the market data feed.
        execution_handler - (Class) Handles the orders/fills for trades.
        portfolio - (Class) Keeps track of portfolio current and prior positions.
        strategy - (Class) Generates signals based on market data.
        """
        self.configuration = configuration
        self.result_dir = configuration.result_dir
        self.result_filepath = os.path.join(self.result_dir, 'last/results.csv')

        self.heartbeat = configuration.heartbeat
        self.backtest_start_time = datetime.utcnow()

        self.start_dates = configuration.start_date
        self.end_dates = configuration.end_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy

        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0

        self.show_charts = configuration.show_charts
        self.update_charts = configuration.update_charts
        self.strategy_params = configuration.strategy_params


    def _generate_trading_instances(self, start_date, end_date):
        """
        Generates the trading instance objects from
        their class types.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        print("Instrument List: %s..." % strategy_instruments)

        configuration = self.configuration
        configuration.start_date = start_date
        configuration.end_date = end_date


        self.data_handler = self.data_handler_cls(self.events, configuration)
        self.strategy = self.strategy_cls(self.data_handler, self.events, configuration, **self.strategy_params)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, configuration)
        self.execution_handler = self.execution_handler_cls(self.events, configuration)

    def _run(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            i += 1
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars()
            else:
                break

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_timeindex(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_backtest_result_dataframe()
        stats = self._show_stats()
        return stats

    def _show_stats(self):
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_start_time))
        stats = self.portfolio.compute_stats(backtest_result_dir)

        print("Results: ")
        print("Total USD return: %s" % stats['Total USD Return'])
        print("Total BTC return: %s" % stats['Total BTC Return'])
        print("Sharpe Ratio: %s" % stats['Sharpe Ratio'])
        print("Max drawdown: %s" % stats['Max Drawdown'])
        print("BTC Max drawdown: %s" % stats['BTC Max Drawdown'])
        print("Drawdown Duration: %s" % stats['Drawdown Duration'])
        print("BTC Drawdown Duration: %s" % stats['BTC Drawdown Duration'])
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)

        print("Final Portfolios: ")
        print(self.portfolio.portfolio_dataframe.tail(10))

        return stats


    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        out = open(os.path.join(self.result_dir, 'scores.csv'), "w")
        out.write("%s,%s,%s,%s,%s\n" % ("Instrument(s)", "Total Returns", "Sharpe Ratio", "Max Drawdown", "Drawdown Duration"))

        num_backtest = len(self.start_dates)
        for i, (start, end) in enumerate(zip(self.start_dates, self.end_dates)):
          print("Strategy %s out of %s..." % (i+1, num_backtest))
          print("Start Date: %s, End Date: %s..." % (start, end))
          self._generate_trading_instances(start, end)
          self._run()
          stats = self._output_performance()

          total_USD_return = stats['Total USD Return']
          total_BTC_return = stats['Total BTC Return']
          sharpe_ratio = stats['Sharpe Ratio']
          max_drawdown = stats['Max Drawdown']
          btc_max_drawdown = stats['BTC Max Drawdown']
          drawdown_duration = stats['Drawdown Duration']
          btc_drawdown_duration = stats['BTC Drawdown Duration']

          out.write(
            "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (start, end, total_USD_return, total_BTC_return, sharpe_ratio, max_drawdown, btc_max_drawdown, drawdown_duration, btc_drawdown_duration, self.signals, self.orders, self.fills)
          )

          out.close

