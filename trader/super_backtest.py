#!/usr/bin/python
# -*- coding: utf-8 -*-

# backtest.py

import os
import csv
import time
import pprint
import datetime
import subprocess

import pandas as pd

from datetime import datetime, timedelta
from distutils.dir_util import copy_tree
from utils.helpers import format_instrument_list
from utils.log import logger
from shutil import copyfile

try:
    import Queue as queue
except ImportError:
    import queue


class SuperBacktest(object):
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
        self.last_backtest_scores_path = os.path.join(self.result_dir, 'last/scores.csv')
        self.all_backtest_scores_path = os.path.join(self.result_dir, 'all/scores.csv')

        self.backtest_name = configuration.backtest_name
        self.configuration_filename = configuration.configuration_filename
        self.backtest_date = datetime.utcnow()
        self.start_dates = configuration.start_dates
        self.end_dates = configuration.end_dates
        self.instrument_list = configuration.instruments
        self.strategy_params = configuration.strategy_params
        self.params_names = configuration.params_names
        self.heartbeat = configuration.heartbeat
        self.default_leverage = configuration.default_leverage
        self.strategy = configuration.strategy

        self.num_periods = len(self.start_dates)
        self.num_instruments = len(self.instrument_list)
        self.num_params = len(self.strategy_params)
        self.num_backtests = self.num_periods * self.num_instruments * self.num_params

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

        self._close_excel()


    def _generate_trading_instances(self, start_date, end_date, instruments, params):
        """
        Generates the trading instance objects from
        their class types.
        """
        configuration = self.configuration
        configuration.start_date = start_date
        configuration.end_date = end_date
        configuration.instruments = instruments

        logger.info("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        logger.info("Start date: %s" % start_date)
        logger.info("End date: %s" % end_date)
        logger.info("Instrument(s): %s..." % instruments)
        logger.info("Params: %s..." % params)

        self.data_handler = self.data_handler_cls(self.events, configuration)
        self.strategy = self.strategy_cls(self.data_handler, self.events, configuration, **params)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, configuration)
        self.execution_handler = self.execution_handler_cls(self.data_handler, self.events, configuration)

    def _run(self):
        """
        Executes the backtest.
        """
        self.signals = 0
        self.orders = 0
        self.fills = 0
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

    def _process_results(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_backtest_result_dataframe()
        stats = self._show_stats()
        return stats

    def _show_stats(self):
        stats = self.portfolio.compute_stats()

        general_stats = stats['general']
        pnl_stats = stats['pnl']
        trade_summary_stats = stats['summary']
        trade_duration_stats = stats['duration']
        trade_returns_stats = stats['returns']

        print("Results: ")
        print("Total USD return: %s" % general_stats['Total USD Return'])
        print("Total BTC return: %s" % general_stats['Total BTC Return'])
        print("Sharpe Ratio: %s" % general_stats['Sharpe Ratio'])
        print("Max drawdown: %s" % general_stats['Max Drawdown'])
        print("BTC Max drawdown: %s" % general_stats['BTC Max Drawdown'])
        print("Drawdown Duration: %s" % general_stats['Drawdown Duration'])
        print("BTC Drawdown Duration: %s" % general_stats['BTC Drawdown Duration'])
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


    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        backtest_result_dir = os.path.join(self.result_dir, str(self.backtest_date))
        os.mkdir(backtest_result_dir)
        backtest_scores_path = os.path.join(backtest_result_dir, 'scores.csv')
        # all_backtest_scores_exists = os.path.isfile(self.all_backtest_scores_path)

        last_backtest_scores = open(self.last_backtest_scores_path, "w")
        backtest_scores = open(backtest_scores_path, "w")


        fieldnames = [ 'Backtest Name', 'Backtest Date', 'Strategy', 'Start Date', 'End Date', 'Instrument(s)', 'Params'] + \
        ['Number of signals', 'Number of orders', 'Number of trades', 'Total USD Return', 'Total BTC Return',
        'Sharpe Ratio', 'BTC Sharpe Ratio', 'Max Drawdown', 'BTC Max Drawdown', 'Drawdown Duration', 'BTC Drawdown Duration',
        'Monthly BTC Return', 'Yearly BTC Return', 'Avg. winning trade', 'Median duration', 'Avg. losing trade', 'Median returns winning', 'Largest losing trade',
        'Gross loss', 'Largest winning trade', 'Avg duration', 'Avg returns losing', 'Median returns losing', 'Profit factor',
        'Winning round trips', 'Percent profitable', 'Total profit', 'Shortest duration', 'Median returns all round trips',
        'Losing round trips', 'Longest duration', 'Avg returns all round trips', 'Gross profit', 'Avg returns winning',
        'Total number of round trips', 'Ratio Avg. Win:Avg. Loss', 'Avg. trade net profit', 'Even round trips',
        'Configuration Filename', 'Leverage']

        try:
          with last_backtest_scores as a, backtest_scores as b:
            writer_a = csv.DictWriter(a, fieldnames=fieldnames)
            writer_b = csv.DictWriter(b, fieldnames=fieldnames)
            writer_a.writeheader()
            writer_b.writeheader()

            # if not all_backtest_scores_exists:
            #     writer_c.writeheader()

            for i, (start, end) in enumerate(zip(self.start_dates, self.end_dates)):
              for j, instruments in enumerate(self.instrument_list):
                for k, params in enumerate(self.strategy_params):
                  num_backtest = i * (self.num_instruments * self.num_params + 1) + j * (self.num_params + 1) + k + 1
                  print("Strategy %s out of %s" % (num_backtest, self.num_backtests))
                  self._generate_trading_instances(start, end, instruments, params)
                  self._run()
                  stats = self._process_results()

                  general_stats = stats['general']
                  pnl_stats = stats['pnl']['All trades'].to_dict()
                  summary_stats = stats['summary']['All trades'].to_dict()
                  duration_stats = stats['duration']['All trades'].to_dict()
                  return_stats = stats['returns']['All trades'].to_dict()
                  params_value = '/'.join([ '{}:{}'.format(item[0], item[1]) for item in params.items() ])

                  row = {
                    'Backtest Name': self.backtest_name,
                    'Backtest Date': self.backtest_date,
                    'Strategy': self.strategy,
                    'Start Date': start,
                    'End Date': end,
                    'Instrument(s)': format_instrument_list(instruments),
                    'Params': params_value,
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
          print('I/O Error')

        all_scores_csv = pd.concat([ pd.read_csv(self.all_backtest_scores_path), pd.read_csv(backtest_scores_path)])
        all_scores_csv.to_csv(self.all_backtest_scores_path, columns=fieldnames, index=False, encoding='utf-8-sig')

        self._open_results_in_excel()

    def _open_results_in_excel(self):
      logger.info("Opening results in excel")

      os.system("open -a 'Microsoft Excel.app' '%s'" % self.last_backtest_scores_path)

    def _close_excel(self):
        subprocess.call(['osascript', '-e', 'tell application "Excel" to quit'])

