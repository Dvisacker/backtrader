#!/usr/bin/python
# -*- coding: utf-8 -*-

# simple_backtest.py

from __future__ import print_function

import os
import csv
import pdb
import time
import pprint
import datetime
import subprocess

from datetime import datetime, timedelta
from threading import Thread
from utils.helpers import format_instrument_list
from distutils.dir_util import copy_tree
from utils.log import logger

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

        self.backtest_name = configuration.backtest_name
        self.configuration_filename = configuration.configuration_filename
        self.start_date = configuration.start_date
        self.end_date = configuration.end_date
        self.instruments = configuration.instruments
        self.backtest_date = datetime.utcnow()
        self.graph_refresh_period = configuration.graph_refresh_period
        self.heartbeat = configuration.heartbeat
        self.default_leverage = configuration.default_leverage
        self.strategy = configuration.strategy

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

        self.logger = configuration.logger

        self._close_excel()
        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        self.logger.info("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")

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

            # time.sleep(self.heartbeat)

    def _update_charts(self):
        self.portfolio.update_charts()

    def _process_results(self):
        """
        Outputs the strategy performance from the backtest.
        """
        # Create a timestamped directory for backtest results
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_date))
        os.mkdir(backtest_result_dir)
        self.portfolio.create_backtest_result_dataframe()

        stats = self._show_stats()
        self._save_results(stats)
        self._open_results_in_excel()
        self._show_charts()

    def _show_charts(self):
        if self.show_charts:
          self.portfolio.output_graphs()

    def _save_results(self, stats):
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_date))
        self.portfolio.save_results(backtest_result_dir)

        backtest_scores = os.path.join(self.last_result_dir, 'scores.csv')
        all_backtest_scores = os.path.join(self.result_dir, 'all/scores.csv')

        fieldnames = [ 'Backtest Name', 'Backtest Date', 'Strategy', 'Start Date', 'End Date', 'Instrument(s)', 'Params'] +\
        [ 'Number of signals', 'Number of orders', 'Number of trades', 'Total USD Return', 'Total BTC Return',
        'Sharpe Ratio', 'BTC Sharpe Ratio', 'Max Drawdown', 'BTC Max Drawdown', 'Drawdown Duration', 'BTC Drawdown Duration',
        'Monthly BTC Return', 'Yearly BTC Return', 'Avg. winning trade', 'Median duration', 'Avg. losing trade', 'Median returns winning', 'Largest losing trade',
        'Gross loss', 'Largest winning trade', 'Avg duration', 'Avg returns losing', 'Median returns losing', 'Profit factor',
        'Winning round trips', 'Percent profitable', 'Total profit', 'Shortest duration', 'Median returns all round trips',
        'Losing round trips', 'Longest duration', 'Avg returns all round trips', 'Gross profit', 'Avg returns winning',
        'Total number of round trips', 'Ratio Avg. Win:Avg. Loss', 'Avg. trade net profit', 'Even round trips',
        'Configuration Filename', 'Leverage']

        all_backtest_scores_exists = os.path.isfile(all_backtest_scores)

        try:
          with open(backtest_scores, "w") as a, open(all_backtest_scores, "a") as b:
            writer_a = csv.DictWriter(a, fieldnames=fieldnames)
            writer_b = csv.DictWriter(b, fieldnames=fieldnames)

            writer_a.writeheader()
            if not all_backtest_scores_exists:
              writer_b.writeheader()

            general_stats = stats['general']
            pnl_stats = stats['pnl']['All trades'].to_dict()
            summary_stats = stats['summary']['All trades'].to_dict()
            duration_stats = stats['duration']['All trades'].to_dict()
            return_stats = stats['returns']['All trades'].to_dict()
            params = '/'.join([ '{}:{}'.format(item[0], item[1]) for item in self.strategy_params.items() ])

            row = { 'Backtest Name': self.backtest_name,
                    'Backtest Date': self.backtest_date,
                    'Strategy': self.strategy.strategy_name,
                    'Start Date': self.start_date,
                    'End Date': self.end_date,
                    'Instrument(s)': format_instrument_list(self.instruments),
                    'Params': params,
                    'Number of signals': self.signals,
                    'Number of orders': self.orders,
                    'Number of trades': self.fills,
                    **general_stats,
                    **pnl_stats,
                    **summary_stats,
                    **duration_stats,
                    **return_stats,
                    'Configuration Filename': self.configuration_filename,
                    'Leverage': self.default_leverage
                  }

            writer_a.writerow(row)
            writer_b.writerow(row)

        except IOError:
          self.logger.error('I/O Error')

    def _show_stats(self):
        self.logger.info("Creating summary stats...")
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
        all_backtest_scores = os.path.join(self.result_dir, 'all/scores.csv')
        os.system("open -a 'Microsoft Excel.app' '%s'" % all_backtest_scores)

    def _close_excel(self):
        subprocess.call(['osascript', '-e', 'tell application "Excel" to quit'])

    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run()
        self._process_results()