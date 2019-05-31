#!/usr/bin/python
# -*- coding: utf-8 -*-

# backtest.py

from __future__ import print_function

import datetime
import time
import pprint
import os

from datetime import datetime
from threading import Thread
from distutils.dir_util import copy_tree

try:
    import Queue as queue
except ImportError:
    import queue


class CryptoBacktest(object):
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
        self.heartbeat = configuration.heartbeat
        self.graph_refresh_period = configuration.graph_refresh_period
        self.backtest_start_time = datetime.now()

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy

        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")

        self.data_handler = self.data_handler_cls(self.events, self.configuration)
        self.strategy = self.strategy_cls(self.data_handler, self.events, self.configuration)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.configuration)
        self.execution_handler = self.execution_handler_cls(self.events, self.configuration)

    def _run(self):
        """
        Executes the backtest.
        """
        i = 0
        self.portfolio.initialize_graphs()

        while True:
            i += 1
            if (i % self.graph_refresh_period == 0):
              self._update_charts()

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

    def _update_charts(self):
        self.portfolio.update_graphs()


    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        # Create a timestamped directory for backtest results
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_start_time))
        os.mkdir(backtest_result_dir)
        self.portfolio.create_equity_curve_dataframe()

        print("Creating summary stats...")
        stats = self.portfolio.save_stats(backtest_result_dir)

        print("Total return: %s" % stats['Total Return'])
        print("Sharpe Ratio: %s" % stats['Sharpe Ratio'])
        print("Max drawdown: %s" % stats['Max Drawdown'])
        print("Drawdown Duration: %s" % stats['Drawdown Duration'])
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)

        print("Creating equity curve...")
        print(self.portfolio.equity_curve.tail(10))


        self.portfolio.output_graphs()

    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run()
        self._output_performance()