#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py
from __future__ import print_function

import os
import datetime
import functools
from math import floor
from pathlib import Path

try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()

from db.mongo_handler import MongoHandler
from event import FillEvent, OrderEvent, BulkOrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation, truncate, get_precision

class BitmexPortfolioBacktest(object):
    """
    The CryptoPortfolio is similar to the previous portfolio
    class. Instead of using the adjusted close data point, it uses
    the close datapoint
    """

    def __init__(self, data, events, configuration):
        """
        Initializes the portfolio with data and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).
        Parameters:
        data - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """

        self.data = data
        self.events = events
        self.instruments = configuration.instruments['bitmex']
        self.assets = configuration.assets['bitmex']
        self.start_date = configuration.start_date
        self.result_dir = configuration.result_dir
        self.default_position_size = configuration.default_position_size
        self.initial_capital = configuration.initial_capital

        self.current_positions = self.construct_current_positions()
        self.all_positions = [ self.current_positions ]
        self.current_holdings = self.construct_current_holdings()
        self.all_holdings = [ self.current_holdings ]

        self.db = MongoHandler()

    def construct_current_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """

        d = {}
        d['datetime'] = self.start_date
        d['total'] = 0.0
        d['total-USD'] = 0.0

        for s in self.instruments:
          price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
          d['bitmex-{}'.format(s)] = 0
          d['bitmex-{}-price'.format(s)] = price
          d['bitmex-{}-in-BTC'.format(s)] = 0
          d['bitmex-{}-in-USD'.format(s)] = 0
          d['bitmex-{}-leverage'.format(s)] = 1
          d['total'] += 0
          d['total-USD'] += 0

        return d

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = {}
        d['datetime'] = self.start_date

        price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")
        initial_btc = self.initial_capital / price

        d['bitmex-BTC-available-balance'] = initial_btc
        d['bitmex-BTC-balance'] = initial_btc
        d['bitmex-BTC-price'] = price
        d['bitmex-BTC-value'] = self.initial_capital #initial_btc * price
        d['bitmex-BTC-fill'] = ''
        d['total-USD'] = self.initial_capital
        d['fee'] = 0

        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])

        # Update positions
        # ================
        dp = {}
        dp['datetime'] = latest_datetime
        dp['total'] = 0
        dp['total-USD'] = 0

        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")

        for s in self.instruments:
          quantity = self.current_positions['bitmex-{}'.format(s)]
          price = self.current_positions['bitmex-{}-price'.format(s)]
          dp['bitmex-{}'.format(s)] = self.current_positions['bitmex-{}'.format(s)]
          dp['bitmex-{}-price'.format(s)] = self.current_positions['bitmex-{}-price'.format(s)]
          dp['bitmex-{}-in-BTC'.format(s)] = self.current_positions['bitmex-{}-in-BTC'.format(s)]
          dp['bitmex-{}-in-USD'.format(s)] = self.current_positions['bitmex-{}-in-USD'.format(s)]

          if 'bitmex-{}-leverage'.format(s) in self.current_positions:
            dp['bitmex-{}-leverage'.format(s)] = self.current_positions['bitmex-{}-leverage'.format(s)]
          else:
            dp['bitmex-{}-leverage'.format(s)] = 0

          dp['total'] += quantity * price
          dp['total-USD'] += quantity * price * btc_price

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = {}
        dh['datetime'] = latest_datetime
        dh['fee'] = self.current_holdings['fee']
        dh['total-USD'] = 0

        for s in self.assets:
          price = self.data.get_latest_bar_value('bitmex', '{}/USD'.format(s), "close")
          balance = self.current_holdings['bitmex-{}-balance'.format(s)]
          available_balance = self.current_holdings['bitmex-{}-available-balance'.format(s)]

          dh['bitmex-{}-available-balance'.format(s)] = available_balance
          dh['bitmex-{}-balance'.format(s)] = balance
          dh['bitmex-{}-price'.format(s)] = price
          dh['bitmex-{}-value'.format(s)] = balance * price
          dh['bitmex-{}-fill'.format(s)] = self.current_holdings['bitmex-{}-fill'.format(s)]
          dh['total-USD'] += balance * price

        dh['fee'] += 0.0

        # Append the current holdings
        self.all_holdings.append(dh)

        for s in self.assets:
          self.current_holdings['bitmex-{}-fill'.format(s)] = 0

        # self.write(dp, dh)

    # ======================
    # FILL/POSITION HANDLING
    # ======================
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL' and event.fill_type == 'Market':
          self.update_positions_from_fill(event)
          self.update_holdings_from_fill(event)
        elif event.type == 'FILL' and event.fill_type == 'ClosePosition':
          self.update_positions_from_exit(event)
          self.update_positions_from_fill(event)

    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value
        Parameters:
        fill - The Fill object to update the holdings with
        """
        fill_direction = { 'BUY': 1, 'SELL': -1 }[fill.direction]
        fill_price = fill.price
        fill_quantity = fill.quantity
        fill_symbol = fill.symbol

        # Check whether the fill is a buy or sell
        balance = self.current_holdings['bitmex-BTC-balance']
        available_balance = self.current_holdings['bitmex-BTC-available-balance']
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")
        entry_price = self.data.get_latest_bar_value('bitmex', fill_symbol, "close")

        # Update holdings list with new quantities
        self.current_holdings['bitmex-{}-entry-price'.format(fill_symbol)] = entry_price
        self.current_holdings['bitmex-BTC-available-balance'] = available_balance - fill_direction * fill_quantity / entry_price
        self.current_holdings['bitmex-BTC-price'] = btc_price
        self.current_holdings['bitmex-BTC-value'] = balance * btc_price
        self.current_holdings['bitmex-BTC-fill'] = fill_direction
        self.current_holdings['fee'] += fill.fee


    def update_holdings_from_exit(self, fill):
        fill_direction = { 'BUY': 1, 'SELL': -1 }[fill.direction]
        fill_quantity = fill.quantity
        fill_symbol = fill.symbol
        entry_price = self.current_holdings['bitmex-{}-entry-price'.format(fill_symbol)]
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")
        exit_price = self.data.get_latest_bar_value('bitmex', fill_symbol, "close")

        # We distinguish between the different contracts
        if (fill_symbol == 'BTC/USD'):
          self.current_holdings['bitmex-BTC-balance'] += fill_direction * fill_quantity * (1 / entry_price - 1 / exit_price)
          self.current_holdings['bitmex-BTC-available-balance'] += fill_direction * fill_quantity / exit_price
          self.current_holdings['bitmex-BTC-price'] = btc_price
          self.current_holdings['bitmex-BTC-value'] = self.current_holdings['bitmex-BTC-balance'] * btc_price
          self.current_holdings['fee'] += fill.fee

        else:
          self.current_holdings['bitmex-BTC-balance'] += fill_direction * fill_quantity * (exit_price - entry_price)
          self.current_holdings['bitmex-BTC-available-balance'] += fill_direction * fill_quantity / exit_price
          self.current_holdings['bitmex-BTC-price'] = btc_price
          self.current_holdings['bitmex-BTC-value'] = self.current_holdings['bitmex-BTC-balance'] * btc_price
          self.current_holdings['fee'] += fill.fee


    def update_positions_from_exit(self, fill):
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        fill_direction = { 'BUY': 1, 'SELL': -1 }[fill.direction]
        fill_symbol = fill.symbol
        quantity = fill.quantity

        self.current_positions['bitmex-{}'.format(fill_symbol)] += fill_direction * quantity
        self.current_positions['bitmex-BTC-price'] = btc_price

        for s in self.instruments:
          price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
          quantity = self.current_positions['bitmex-{}'.format(s)]
          self.current_positions['bitmex-{}-price'.format(s)] = price
          self.current_positions['bitmex-{}-in-BTC'.format(s)] = quantity * price
          self.current_positions['bitmex-{}-in-USD'.format(s)] = quantity * price * btc_price
          self.current_positions['bitmex-{}-leverage'.format(s)] = 1

    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        fill_direction = { 'BUY': 1, 'SELL': -1 }[fill.direction]
        fill_symbol = fill.symbol
        quantity = fill.quantity

        self.current_positions['bitmex-{}'.format(fill_symbol)] += fill_direction * quantity
        self.current_positions['bitmex-BTC-price'] = btc_price

        for s in self.instruments:
          price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
          quantity = self.current_positions['bitmex-{}'.format(s)]
          self.current_positions['bitmex-{}-price'.format(s)] = price
          self.current_positions['bitmex-{}-in-BTC'.format(s)] = quantity * price
          self.current_positions['bitmex-{}-in-USD'.format(s)] = quantity * price * btc_price
          self.current_positions['bitmex-{}-leverage'.format(s)] = 1

    def rebalance_portfolio(self, signals):
        """
        Rebalances the portfolio based on an array of signals. The amount of funds allocated to each
        signal depends on the relative strength given to each signal by the strategy module.
        Parameters:
        signals - Array of signal events
        """
        available_balance = self.current_holdings['bitmex-BTC-available-balance']
        exchange = 'bitmex'
        events = []
        default_position_size = self.default_position_size

        for sig in signals.events:
          sig.print_signal()
          price = self.data.get_latest_bar_value('bitmex', sig.symbol, "close")
          if not price:
            continue

          # We don't take into account take profits and stop losses for now
          if sig.signal_type == "EXIT":
            close_position_order = OrderEvent(exchange, sig.symbol, 'ClosePosition')
            events.append(close_position_order)
          else:
            direction = { 'LONG': 1, 'SHORT': -1 }[sig.signal_type]
            target_allocation = direction * default_position_size * sig.strength
            current_quantity = self.current_positions['bitmex-{}'.format(sig.symbol)]
            target_quantity = floor(target_allocation / price)
            side = 'buy' if (target_quantity - current_quantity) > 0 else 'sell'
            quantity = abs(target_quantity - current_quantity)

            if (target_allocation > available_balance):
                continue

            if (quantity == 0):
                continue

            order = OrderEvent(exchange, sig.symbol, 'Market', quantity, side, 1)
            events = [order]

        return events

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        pass


    def update_signals(self, events):
        """
        Acts on a SignalEvents to generate new orders
        based on the portfolio logic.
        """
        order_events = self.rebalance_portfolio(events)
        for event in order_events:
          self.events.put(event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

    def write(self, current_positions, current_holdings):
        """
        Saves the position and holdings updates to a database or to a file
        """
        self.db.insert_positions(current_positions)
        self.db.insert_holdings(current_holdings)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0+curve['returns']).cumprod()
        self.equity_curve = curve

    def print_summary_stats(self):
        """
        Print a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        return stats


    def output_summary_stats(self, backtest_result_dir):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        # We output both to the most recent backtest folder and to a backtest timestamped folder
        self.equity_curve.to_csv(os.path.join(self.result_dir, 'equity.csv'))
        self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))
        return stats

    def output_summary_stats_and_graphs(self, backtest_result_dir):
        """
        Creates a list of summary statistics and plots
        performance graphs
        """

        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        self.equity_curve.to_csv(os.path.join(self.result_dir, 'equity.csv'))
        self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))

        returns = self.equity_curve['returns']
        equity_curve = self.equity_curve['equity_curve']
        drawdown = self.equity_curve['drawdown']

        # Plot three charts: Equity curve,
        # period returns, drawdowns
        fig = plt.figure(figsize=(15,10))
        # Set the outer colour to white
        fig.patch.set_facecolor('white')
        # Plot the equity curve
        ax1 = fig.add_subplot(311, ylabel='Portfolio value, %')
        equity_curve.plot(ax=ax1, color="blue", lw=2.)
        plt.grid(True)

        # Plot the returns
        ax2 = fig.add_subplot(312, ylabel='Period returns, %')
        returns.plot(ax=ax2, color="black", lw=2.)
        plt.grid(True)

        # Plot the returns
        ax3 = fig.add_subplot(313, ylabel='Drawdowns, %')
        drawdown.plot(ax=ax3, color="red", lw=2.)
        plt.grid(True)

        # pf.show_perf_stats(returns)
        # pf.show_worst_drawdown_periods(returns)

        plt.figure(figsize = (15, 10))
        pf.plot_drawdown_underwater(returns).set_xlabel('Date')

        plt.figure(figsize = (15, 10))
        pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

        plt.figure(figsize = (15, 10))
        pf.plot_returns(returns).set_xlabel('Date')

        plt.figure(figsize = (15, 10))
        pf.plot_return_quantiles(returns).set_xlabel('Timeframe')

        plt.figure(figsize = (15, 10))
        pf.plot_monthly_returns_dist(returns).set_xlabel('Returns')

        plt.figure(figsize = (15, 10))
        pf.plot_rolling_volatility(returns, rolling_window=30).set_xlabel('date')

        plt.figure(figsize = (15, 10))
        pf.plot_rolling_sharpe(returns, rolling_window=30).set_xlabel('Date')

        pf.create_returns_tear_sheet(returns)
        return stats














