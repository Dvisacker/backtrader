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


class MultiCryptoBacktest(object):
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
        self.result_dir = configuration.result_dir
        self.heartbeat = configuration.heartbeat
        self.strat_params = configuration.strat_params
        self.params_names = configuration.params_names
        self.num_params = len(self.params_names)
        self.configuration = configuration
        self.backtest_start_time = datetime.utcnow()

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.last_result_dir = os.path.join(self.result_dir, 'last')

        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = len(self.strat_params)


    def _generate_trading_instances(self, strategy_params_dict):
        """
        Generates the trading instance objects from
        their class types.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        print("Strategy parameter dict: %s..." % strategy_params_dict)

        self.data_handler = self.data_handler_cls(self.events, self.configuration)
        self.strategy = self.strategy_cls(self.data_handler, self.events, self.configuration, **strategy_params_dict)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.configuration)
        self.execution_handler = self.execution_handler_cls(self.events, self.configuration)

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
        self.portfolio.create_equity_curve_dataframe()

        print("Creating summary stats...")
        stats = self.portfolio.print_summary_stats()

        print("Creating equity curve...")
        print(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)

        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)

        return stats

    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """

        if self.num_params == 0:
          self._generate_trading_instances(None)
          self._run()
          stats = self._output_performance()
          pprint.pprint(stats)

        if self.num_params == 1:
          out = open(os.path.join(self.last_result_dir, 'opt.csv'), "w")

          out.write(
              "%s,%s,%s,%s,%s\n" % (
              self.params_names[0], "Total Returns", "Sharpe Ratio", "Max Drawdown", "Drawdown Duration")
          )

          spl = len(self.strat_params)
          for i, sp in enumerate(self.strat_params):
            print("Strategy %s out of %s..." % (i+1, spl))
            self._generate_trading_instances(sp)
            self._run()
            stats = self._output_performance()
            pprint.pprint(stats)

            tot_ret = float(stats[0][1].replace("%", ""))
            sharpe = float(stats[1][1])
            max_dd = float(stats[2][1].replace("%", ""))
            dd_dur = int(stats[3][1])

            out.write(
              "%s,%s,%s,%s,%s\n" % (
                sp[self.params_names[0]], tot_ret, sharpe, max_dd, dd_dur)
            )

            out.close

        if self.num_params == 2:
          out = open(os.path.join(self.last_result_dir, 'opt.csv'), "w")

          out.write(
              "%s,%s,%s,%s,%s,%s\n" % (
              self.params_names[0], self.params_names[1], "Total Returns", "Sharpe Ratio", "Max Drawdown", "Drawdown Duration")
          )

          spl = len(self.strat_params)
          for i, sp in enumerate(self.strat_params):
            print("Strategy %s out of %s..." % (i+1, spl))
            self._generate_trading_instances(sp)
            self._run()
            stats = self._output_performance()
            pprint.pprint(stats)

            tot_ret = float(stats[0][1].replace("%", ""))
            sharpe = float(stats[1][1])
            max_dd = float(stats[2][1].replace("%", ""))
            dd_dur = int(stats[3][1])

            # Write data matrix
            out.write(
              "%s,%s,%s,%s,%s,%s\n" % (
                sp[self.params_names[0]], sp[self.params_names[1]],
                tot_ret, sharpe, max_dd, dd_dur)
            )

            out.close


        if self.num_params == 3:
          out = open(os.path.join(self.last_result_dir, 'opt.csv'), "w")

          out.write(
              "%s,%s,%s,%s,%s,%s,%s\n" % (
              self.params_names[0], self.params_names[1], self.params_names[2], "Total Returns", "Sharpe Ratio", "Max Drawdown", "Drawdown Duration")
          )

          spl = len(self.strat_params)
          for i, sp in enumerate(self.strat_params):
            print("Strategy %s out of %s..." % (i+1, spl))
            self._generate_trading_instances(sp)
            self._run()
            stats = self._output_performance()
            pprint.pprint(stats)

            tot_ret = float(stats[0][1].replace("%", ""))
            sharpe = float(stats[1][1])
            max_dd = float(stats[2][1].replace("%", ""))
            dd_dur = int(stats[3][1])

            out.write(
              "%s,%s,%s,%s,%s,%s,%s\n" % (
                sp[self.params_names[0]], sp[self.params_names[1]], sp[self.params_names[2]],
                tot_ret, sharpe, max_dd, dd_dur)
            )

            out.close

        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_start_time))
        os.mkdir(backtest_result_dir)
        copy_tree(self.last_result_dir, backtest_result_dir)

