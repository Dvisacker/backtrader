#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py
from __future__ import print_function

import os
import datetime
from math import floor

try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()

from event import FillEvent, OrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation

class BitmexPortfolio(object):
    """
    The CryptoPortfolio is similar to the previous portfolio
    class. Instead of using the adjusted close data point, it uses
    the close datapoint
    """

    def __init__(self, data, events, configuration, exchanges):
        """
        Initialises the portfolio with data and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).
        Parameters:
        data - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """

        if 'instruments' not in configuration:
          raise AssertionError

        if 'assets' not in configuration:
          raise AssertionError

        if 'start_date' not in configuration:
          raise AssertionError

        if 'result_dir' not in configuration:
          raise AssertionError

        if 'bitmex' not in exchanges:
          raise AssertionError

        self.data = data
        self.events = events
        self.exchange = exchanges['bitmex']
        self.instruments = configuration['instruments']['bitmex']
        self.assets = configuration['assets']['bitmex']
        self.start_date = configuration['start_date']
        self.result_dir = configuration['result_dir']

        self.current_positions = self.construct_current_positions()
        self.all_positions = [ self.current_positions ]

        self.current_holdings = self.construct_current_holdings()
        self.all_holdings = [ self.current_holdings ]


    def construct_current_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        print(self.instruments)

        d = dict( (k,v) for k, v in [(s, 0) for s in self.instruments] )
        d['datetime'] = self.start_date

        positions = self.exchange.private_get_position()

        for p in positions:
          s = from_exchange_to_standard_notation('bitmex', p['symbol'])
          if s in self.instruments:
            d[s] = p['currentQty']

        return d

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict( (k,v) for k, v in [(s, 0.0) for s in self.assets] )
        d['commission'] = 0.0
        d['total'] = 0.0
        d['datetime'] = self.start_date

        response = self.exchange.fetch_balance()
        balances = response['total']
        symbols = list(balances.keys())

        for s in symbols:
          price = self.data.get_latest_bar_value('bitmex', '{}/USD'.format(s), "close")
          d[s] = balances[s]
          d['{} value'.format(s)] = balances[s] * price
          d['total'] += balances[s] * price
          d['commission'] += 0.0

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
        dp = dict( (k,v) for k, v in [(s, 0) for s in self.instruments] )
        dp['datetime'] = latest_datetime

        for s in self.instruments:
          dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = dict( (k,v) for k, v in [(s, 0) for s in self.assets] )
        dh['datetime'] = latest_datetime
        dh['commission'] = self.current_holdings['commission']


        # NOTE For now, let's assume that the balance are not modified.
        # Only the price in USD need to be recomputed every index.
        # for a in self.assets:
        # market_value = self.current_holdings[a]
        for s in self.assets:
          price = self.data.get_latest_bar_value('bitmex', '{}/USD'.format(s), "close")
          balance = self.current_holdings[s]
          self.current_holdings['{} value'.format(s)] = balance * price
          self.current_holdings['total'] += balance * price
          self.current_holdings['commission'] += 0.0

          # Append the current holdings
          dh = self.current_holdings
          self.all_holdings.append(dh)

    # ======================
    # FILL/POSITION HANDLING
    # ======================
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
          dp = self.construct_current_positions()
          dh = self.construct_current_holdings()

          self.current_holdings = dh
          self.current_positions = dp


    # def generate_order_with_stop_loss_and_take_profit(self, signal):


    def generate_naive_order(self, signal):
        """
        Simply files an Order object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.
        Parameters:
        signal - The tuple containing Signal information.
        """
        order = None
        exchange = signal.exchange
        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        mkt_quantity = 1
        cur_quantity = self.current_positions[symbol]
        order_type = 'Market'

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, mkt_quantity, 'buy')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, mkt_quantity, 'sell')

        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(exchange, symbol, order_type, abs(cur_quantity), 'sell')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(exchange, symbol, order_type, abs(cur_quantity), 'buy')

        return order

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)

    # ========================
    # POST-BACKTEST STATISTICS
    # ========================

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