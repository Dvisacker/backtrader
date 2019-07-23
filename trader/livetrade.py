#!/usr/bin/python
# -*- coding: utf-8 -*-
# backtest.py


import io
import os
import time
import pprint
import logging
import datetime


from threading import Thread
from datetime import datetime
from distutils.dir_util import copy_tree
from exchanges import create_exchange_instances

try:
    import Queue as queue
except ImportError:
    import queue


class CryptoLiveTrade(object):
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
        self.backtest_date = configuration.backtest_date
        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy

        self.events = queue.Queue()
        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.heartbeat = configuration.heartbeat
        self.result_dir = configuration.result_dir
        self.exchange_names = configuration.exchange_names
        self.latest_timestamp = None
        self.logger = configuration.logger
        self.live_result_dir = os.path.join(self.result_dir, str(self.backtest_date))

        self._generate_trading_instances()



    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        self.logger.info("Creating DataHandler, Strategy, Portfolio and ExecutionHandler")
        self.exchanges = create_exchange_instances(self.exchange_names)
        self.data_handler = self.data_handler_cls(self.events, self.configuration, self.exchanges)
        self.strategy = self.strategy_cls(self.data_handler, self.events, self.configuration)

        time.sleep(1)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.configuration, self.exchanges)
        self.execution_handler = self.execution_handler_cls(self.events, self.configuration, self.exchanges)

    def _run(self):
        """
        Executes the backtest.
        """

        # Handle the events
        while True:
            try:
                event = self.events.get(False)
            except queue.Empty:
                pass
            else:
                if event is not None:
                    self.logger.info('Receiving {} event. Current Queue size: {}'.format(event.type, self.events.qsize()))
                    if event.type == 'MARKET':
                        # Avoid duplicate ticks
                        if self.latest_timestamp and event.timestamp <= self.latest_timestamp:
                          break

                        self.latest_timestamp = event.timestamp
                        self.data_handler.insert_new_bar_bitmex(event.data, event.timestamp)
                        self.strategy.calculate_signals(event)
                        self.portfolio.update_timeindex(event)
                        self.save_results()


                    elif event.type == 'SIGNAL':
                        event.print_signals()
                        self.signals += len(event.events)
                        self.portfolio.update_signals(event)

                    elif event.type == 'ORDER':
                        event.print_order()
                        self.orders += 1
                        self.execution_handler.execute_order(event)

                    elif event.type == 'FILL':
                        self.fills += 1
                        self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)


    def save_results(self):
        self.portfolio.update_result_dataframe()
        self.portfolio.save_result_dataframe(self.live_result_dir)

    def start_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        Thread(target = self._run).start()
