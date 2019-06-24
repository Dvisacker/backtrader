#!/usr/bin/python
# -*- coding: utf-8 -*-

# backtest.py

from __future__ import print_function

import os
import csv
import time
import pprint
import datetime

from datetime import datetime
from threading import Thread
from distutils.dir_util import copy_tree

try:
    import Queue as queue
except ImportError:
    import queue


class SimpleBacktest(object):
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
        self.last_result_dir = os.path.join(self.result_dir, 'last')

        self.heartbeat = configuration.heartbeat
        self.graph_refresh_period = configuration.graph_refresh_period
        self.backtest_start_time = datetime.utcnow()

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy

        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1

        self.show_charts = configuration.show_charts
        self.update_charts = configuration.update_charts
        self.strategy_params = configuration.strategy_params

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        print("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")

        self.data_handler = self.data_handler_cls(self.events, self.configuration)
        self.strategy = self.strategy_cls(self.data_handler, self.events, self.configuration, **self.strategy_params)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.configuration)
        self.execution_handler = self.execution_handler_cls(self.data_handler, self.events, self.configuration)

    def _run(self):
        """
        Executes the backtest.
        """
        i = 0
        if self.update_charts:
          self.portfolio.initialize_graphs()

        while True:
            i += 1

            if (self.update_charts and i % self.graph_refresh_period == 0):
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
                            self.execution_handler.fill_triggered_orders(event)

                        elif event.type == 'SIGNAL':
                            event.print_signals()
                            self.signals += 1
                            self.portfolio.update_signals(event)

                        elif event.type == 'ORDER':
                            event.print_order()
                            self.orders += 1
                            self.execution_handler.execute_order(event)

                        elif event.type == 'FILL':
                            event.print_fill()
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def _update_charts(self):
        self.portfolio.update_charts()

    def _process_results(self):
        """
        Outputs the strategy performance from the backtest.
        """
        # Create a timestamped directory for backtest results
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_start_time))
        os.mkdir(backtest_result_dir)
        self.portfolio.create_backtest_result_dataframe()

        self._open_results_in_excel()
        stats = self._show_stats()
        self._save_results(stats)
        self._show_charts()

    def _show_charts(self):
        if self.show_charts:
          self.portfolio.output_graphs()

    def _save_results(self, stats):
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_start_time))
        self.portfolio.save_results(backtest_result_dir)

        fieldnames = [ 'Backtest Timestamp', 'Total USD Return', 'Total BTC Return', 'Sharpe Ratio', 'BTC Sharpe Ratio',
        'Max Drawdown', 'BTC Max Drawdown', 'Drawdown Duration', 'BTC Drawdown Duration',
        'Avg. winning trade', 'Median duration', 'Avg. losing trade', 'Median returns winning', 'Largest losing trade',
        'Gross loss', 'Largest winning trade', 'Avg duration', 'Avg returns losing', 'Median returns losing', 'Profit factor',
        'Winning round trips', 'Percent profitable', 'Total profit', 'Shortest duration', 'Median returns all round trips',
        'Losing round trips', 'Longest duration', 'Avg returns all round trips', 'Gross profit', 'Avg returns winning',
        'Total number of round trips', 'Ratio Avg. Win:Avg. Loss', 'Avg. trade net profit', 'Even round trips']

        out = open(os.path.join(self.last_result_dir, 'scores.csv'), "w")

        try:
          with out as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            general_stats = stats['general']
            pnl_stats = stats['pnl']['All trades'].to_dict()
            summary_stats = stats['summary']['All trades'].to_dict()
            duration_stats = stats['duration']['All trades'].to_dict()
            return_stats = stats['returns']['All trades'].to_dict()

            row = { 'Backtest Timestamp': self.backtest_start_time, **general_stats, **pnl_stats, **summary_stats, **duration_stats, **return_stats }
            writer.writerow(row)
        except IOError:
          print('I/O Error')

    def _show_stats(self):
        print("Creating summary stats...")
        stats = self.portfolio.compute_stats()

        global_stats = stats['general']
        pnl_stats = stats['pnl']
        trade_summary_stats = stats['summary']
        trade_duration_stats = stats['duration']
        trade_returns_stats = stats['returns']

        print('\nGLOBAL STATS\n')
        print("Total USD return: %s" % global_stats['Total USD Return'])
        print("Total BTC return: %s" % global_stats['Total BTC Return'])
        print("Sharpe Ratio: %s" % global_stats['Sharpe Ratio'])
        print("Max drawdown: %s" % global_stats['Max Drawdown'])
        print("BTC Max drawdown: %s" % global_stats['BTC Max Drawdown'])
        print("Drawdown Duration: %s" % global_stats['Drawdown Duration'])
        print("BTC Drawdown Duration: %s" % global_stats['BTC Drawdown Duration'])
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)

        print('\nPNL STATS\n')
        print(pnl_stats)

        print('\nTRADE SUMMARY STATS\n')
        print(trade_summary_stats)

        print('\nTRADE DURATION STATS\n')
        print(trade_duration_stats)

        print('\nTRADE RETURNS STATS\n')
        print(trade_returns_stats)

        print("\nBEFORE AND AFTER: \n")
        print(self.portfolio.portfolio_dataframe.head(1))
        print(self.portfolio.portfolio_dataframe.tail(1))

        return stats

    def _open_results_in_excel(self):
        print("Opening results in excel")

        os.system("open -a 'Microsoft Excel.app' '%s'" % self.result_filepath)

    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run()
        self._process_results()
