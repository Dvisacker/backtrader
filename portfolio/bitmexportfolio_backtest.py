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

import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()

from db.mongo_handler import MongoHandler
from event import FillEvent, OrderEvent, BulkOrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation, truncate, get_precision
from utils.helpers import move_figure, plot

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

        self.current_portfolio = self.construct_current_portfolios()
        self.all_portfolios = []

        self.db = MongoHandler()
        self.legends_added = False

    def construct_current_portfolios(self):
      d = {}

      for s in self.instruments:
        price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
        d['bitmex-{}-price'.format(s)] = price
        d['bitmex-{}-position'.format(s)] = 0
        d['bitmex-{}-position-in-BTC'.format(s)] = 0
        d['bitmex-{}-position-in-USD'.format(s)] = 0
        d['bitmex-{}-leverage'.format(s)] = 1
        d['bitmex-{}-fill'.format(s)] = ''

      d['bitmex-total-position-in-BTC'] = 0
      d['bitmex-total-position-in-USD'] = 0

      btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
      initial_btc = self.initial_capital / btc_price

      d['bitmex-BTC-available-balance'] = initial_btc
      d['bitmex-BTC-balance'] = initial_btc
      d['bitmex-BTC-price'] = btc_price
      d['bitmex-BTC-balance-in-USD'] = self.initial_capital
      d['total'] = initial_btc
      d['total-in-USD'] = self.initial_capital
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
      df = {}
      df['datetime'] = latest_datetime
      df['bitmex-total-position-in-BTC'] = 0
      df['bitmex-total-position-in-USD'] = 0

      btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', "close")

      for s in self.instruments:
        quantity = self.current_portfolio['bitmex-{}-position'.format(s)]
        price = self.current_portfolio['bitmex-{}-price'.format(s)]
        df['bitmex-{}-position'.format(s)] = self.current_portfolio['bitmex-{}-position'.format(s)]
        df['bitmex-{}-price'.format(s)] = self.current_portfolio['bitmex-{}-price'.format(s)]
        # This needs to be updated i believe.
        df['bitmex-{}-position-in-BTC'.format(s)] = self.current_portfolio['bitmex-{}-position-in-BTC'.format(s)]
        df['bitmex-{}-position-in-USD'.format(s)] = self.current_portfolio['bitmex-{}-position-in-USD'.format(s)]
        df['bitmex-{}-fill'.format(s)] = self.current_portfolio['bitmex-{}-fill'.format(s)]

        if 'bitmex-{}-leverage'.format(s) in self.current_portfolio:
          df['bitmex-{}-leverage'.format(s)] = self.current_portfolio['bitmex-{}-leverage'.format(s)]
        else:
          df['bitmex-{}-leverage'.format(s)] = 0

        df['bitmex-total-position-in-BTC'] += quantity * price
        df['bitmex-total-position-in-USD'] += quantity * price * btc_price

      # Append the current positions
      # Update holdings
      # ===============
      df['fee'] = self.current_portfolio['fee']
      df['total'] = 0
      df['total-in-USD'] = 0

      for s in self.assets:
        price = self.data.get_latest_bar_value('bitmex', '{}/USD'.format(s), "close")
        balance = self.current_portfolio['bitmex-{}-balance'.format(s)]
        available_balance = self.current_portfolio['bitmex-{}-available-balance'.format(s)]

        df['bitmex-{}-available-balance'.format(s)] = available_balance
        df['bitmex-{}-balance'.format(s)] = balance
        df['bitmex-{}-balance-in-USD'.format(s)] = balance * price
        df['bitmex-{}-price'.format(s)] = price
        df['total'] += balance
        df['total-in-USD'] += balance * price

      df['fee'] += 0.0

      # Append the current holdings
      self.all_portfolios.append(df)

      for s in self.instruments:
        self.current_portfolio['bitmex-{}-fill'.format(s)] = ''

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
          self.update_portfolio_from_fill(event)
        elif event.type == 'FILL' and event.fill_type == 'ClosePosition':
          self.update_portfolio_from_exit(event)

    def update_portfolio_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        direction = { 'buy': 1, 'sell': -1 }[fill.direction]
        symbol = fill.symbol
        quantity = fill.quantity
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        entry_price = self.data.get_latest_bar_value('bitmex', symbol, 'close')

        self.current_portfolio['bitmex-{}-position'.format(symbol)] += direction * quantity
        self.current_portfolio['bitmex-{}-price'.format(symbol)] = entry_price
        self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = direction * quantity * entry_price
        self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = direction * quantity * entry_price * btc_price
        self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = 1
        self.current_portfolio['bitmex-{}-fill'.format(symbol)] = direction
        self.current_portfolio['bitmex-total-position-in-BTC'] += direction * quantity * entry_price * btc_price
        self.current_portfolio['bitmex-total-position-in-USD'] += direction * quantity * entry_price

        # Check whether the fill is a buy or sell
        balance = self.current_portfolio['bitmex-BTC-balance']

        # Update holdings list with new quantities
        self.current_portfolio['bitmex-{}-entry-price'.format(symbol)] = entry_price
        self.current_portfolio['bitmex-BTC-available-balance'] -= direction * quantity * entry_price
        self.current_portfolio['bitmex-BTC-price'] = btc_price
        self.current_portfolio['bitmex-BTC-balance-in-USD'] = balance * btc_price
        self.current_portfolio['total'] = balance
        self.current_portfolio['total-in-USD'] = balance * btc_price
        self.current_portfolio['fee'] += fill.fee

    def update_portfolio_from_exit(self, fill):
        symbol = fill.symbol
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        entry_price = self.current_portfolio['bitmex-{}-entry-price'.format(symbol)]
        price = self.data.get_latest_bar_value('bitmex', symbol, 'close') or 0

        quantity = self.current_portfolio['bitmex-{}-position'.format(symbol)]
        direction = 1 if quantity > 0 else -1

        self.current_portfolio['bitmex-{}-position'.format(symbol)] = 0
        self.current_portfolio['bitmex-{}-price'.format(symbol)] = price
        self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = 0
        self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = 0
        self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = 1
        self.current_portfolio['bitmex-{}-fill'.format(symbol)] = direction
        self.current_portfolio['bitmex-total-position-in-BTC'] += (-1) * quantity * price * btc_price
        self.current_portfolio['bitmex-total-position-in-USD'] += (-1) * quantity * price

        # We distinguish between the different contracts
        if (symbol == 'BTC/USD'):
          self.current_portfolio['bitmex-BTC-balance'] += direction * quantity * (1 / entry_price - 1 / price)
          self.current_portfolio['bitmex-BTC-available-balance'] += direction * quantity / price
          self.current_portfolio['bitmex-BTC-price'] = btc_price
          self.current_portfolio['bitmex-BTC-balance-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price

          # here, total, totalUSD are equal to the bitmex-BTC-Balance and bitmex-BTC-balance-in-USD fields
          self.current_portfolio['total'] += direction * quantity * (1 / entry_price - 1 / price)
          self.current_portfolio['total-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price
          self.current_portfolio['fee'] += fill.fee

        else:
          self.current_portfolio['bitmex-BTC-balance'] += direction * quantity * (price - entry_price)
          self.current_portfolio['bitmex-BTC-available-balance'] += direction * quantity / price
          self.current_portfolio['bitmex-BTC-price'] = btc_price
          self.current_portfolio['bitmex-BTC-balance-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price
          self.current_portfolio['fee'] += fill.fee

    def rebalance_portfolio(self, signals):
        """
        Rebalances the portfolio based on an array of signals. The amount of funds allocated to each
        signal depends on the relative strength given to each signal by the strategy module.
        Parameters:
        signals - Array of signal events
        """
        available_balance = self.current_portfolio['bitmex-BTC-available-balance']
        exchange = 'bitmex'
        events = []
        default_position_size = self.default_position_size

        for sig in signals.events:
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
            current_quantity = self.current_portfolio['bitmex-{}-position'.format(sig.symbol)]
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

    def initialize_graphs(self):
      plt.ion()

      # Plot three charts: Equity curve,
      # period returns, drawdowns
      fig = plt.figure(figsize=(10,10))
      # Set the outer colour to white
      fig.patch.set_facecolor('white')
      self.portfolio_value_ax = fig.add_subplot(211, ylabel='Portfolio value, %')
      self.prices_ax = fig.add_subplot(212, ylabel='Prices')


      self.price_axes = {}
      colors = ['red', 'blue', 'yellow', 'green', 'black']
      for (i, s) in enumerate(self.instruments):
          self.price_axes['bitmex-{}'.format(s)] = self.prices_ax.twinx()
          self.price_axes['bitmex-{}'.format(s)].tick_params(axis='y', labelcolor=colors[i])

      fig.tight_layout()

      fig = plt.figure(figsize=(10,10))
      move_figure(fig, 1000, 0)
      self.positions_and_available_balance_ax = fig.add_subplot(211, ylabel='Positions and available balance')
      self.currency_prices = fig.add_subplot(212, ylabel='Currency prices')

      self.update_graphs()


    def update_graphs(self):
      if not self.all_portfolios:
        return

      portfolios = pd.DataFrame(self.all_portfolios).copy()
      portfolios.set_index('datetime', inplace=True)

      btc_returns = portfolios['bitmex-BTC-balance'].pct_change()
      returns = portfolios['total-USD'].pct_change()
      equity = (1.0+returns).cumprod()
      drawdown, max_dd, dd_duration = create_drawdowns(equity)

      equity.plot(ax=self.portfolio_value_ax, color="blue", lw=1., label='Total Portfolio Value')

      # Plot the equity holdings
      portfolios['bitmex-BTC-available-balance'].plot(ax=self.positions_and_available_balance_ax, color="orange", lw=1., label="Available Balance")
      colors = ['red', 'blue', 'yellow', 'green', 'black']

      for (i, s) in enumerate(self.instruments):
        col = colors[i]
        ax = self.price_axes['bitmex-{}'.format(s)]
        price_label = 'bitmex-{} Price'.format(s).capitalize()
        position = 'bitmex-{} Position #'.format(s).capitalize()
        portfolios["bitmex-{}-price".format(s)].plot(ax=ax, lw=1., color=col, label=price_label)
        portfolios["bitmex-{}-in-USD".format(s)][-1000:].plot(ax=self.positions_and_available_balance_ax, lw=1., color=col, label=position)

      pf.plot_drawdown_underwater(returns, ax=self.currency_prices).set_xlabel('Date')
      plt.pause(0.001)
      plt.axis('tight')

      if not self.legends_added:
        self.portfolio_value_ax.legend(loc='upper left', frameon=False, markerscale=12)
        self.prices_ax.legend(loc='upper left', frameon=False, markerscale=12)
        self.positions_and_available_balance_ax.legend(loc='upper left', frameon=False, markerscale=12)

        for s in self.instruments:
            self.price_axes['bitmex-{}'.format(s)].legend(loc='upper left', frameon=False, markerscale=12)

      self.legends_added = True

    def write(self, current_portfolio):
        """
        Saves the position and holdings updates to a database or to a file
        """
        self.db.insert_portfolio(current_portfolio)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """

        portfolios = pd.DataFrame(self.all_portfolios)
        print(portfolios.head())
        portfolios.set_index('datetime', inplace=True)
        portfolios['returns'] = portfolios['total'].pct_change()
        portfolios['equity_curve'] = (1.0+portfolios['returns']).cumprod()
        self.equity_curve = portfolios

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
        self.equity_curve.to_csv(os.path.join(self.result_dir, 'last/equity.csv'))
        self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))
        return stats

    def output_graphs(self):
        plt.ioff()
        """
        Creates a list of summary statistics and plots
        performance graphs
        """
        curve = self.equity_curve
        total_return = curve['equity_curve'][-1]
        returns = curve['returns']
        pnl = curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)

        returns = curve['returns']
        equity_curve = curve['equity_curve']
        drawdown = curve['drawdown']

        # Plot three charts: Equity curve,
        # period returns, drawdowns
        fig = plt.figure(figsize=(15,10))
        # Set the outer colour to white
        fig.patch.set_facecolor('white')

        # Plot the equity curve
        ax1 = fig.add_subplot(311, ylabel='Portfolio value, %')
        equity_curve.plot(ax=ax1, color="blue", lw=1.)
        plt.grid(True)

        # Plot the returns
        prices_ax = fig.add_subplot(312, ylabel='Period returns, %')
        returns.plot(ax=prices_ax, color="black", lw=1.)
        plt.grid(True)

        # Plot the returns
        positions_and_available_balance_ax = fig.add_subplot(313, ylabel='Drawdowns, %')
        drawdown.plot(ax=positions_and_available_balance_ax, color="red", lw=1.)
        plt.grid(True)

        self.price_figure = {}
        for s in self.instruments:
          fig = plt.figure(figsize=(15,10))
          ax = fig.add_subplot(111, ylabel='bitmex-{} Price'.format(s))
          fill_id = 'bitmex-{}-fill'.format(s)
          price_id = 'bitmex-{}-price'.format(s)
          prices = curve[price_id]
          fills = curve[fill_id]
          buys = pd.Series({ x: prices[x] if fills[x] == "BUY" else np.NaN for x in curve.index })
          sells = pd.Series({ x: prices[x] if fills[x] == "SELL" else np.NaN for x in curve.index })
          prices.plot(ax=ax, color='blue', lw=1., label='bitmex-{} Price'.format(s))
          buys.plot(ax=ax, color='green', marker='o', label='Buys')
          sells.plot(ax=ax, color='red', marker='x', label='Sells')
          ax.legend(loc='best', frameon=False)

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

        fig = pf.create_returns_tear_sheet(returns, return_fig=True)
        fig.savefig('../../results/last/returns_tear_sheet.pdf')
        plt.close(fig)

        webbrowser.open_new(r'file:///Users/davidvanisacker/Programming/Trading/backtest/results/last/returns_tear_sheet.pdf')
        plot()

    def save_stats(self, backtest_result_dir):
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

        stats = {
          "Total Return": "%0.2f%%" % ((total_return - 1.0) * 100.0),
          "Sharpe Ratio": "%0.2f" % sharpe_ratio,
          "Max Drawdown": "%0.2f%%" % (max_dd * 100.0),
          "Drawdown Duration": "%d" % dd_duration
        }

        self.equity_curve.to_csv(os.path.join(self.result_dir, 'last/equity.csv'))
        self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))

        return stats















