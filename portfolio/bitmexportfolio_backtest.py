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
import matplotlib.gridspec as gridspec

from db.mongo_handler import MongoHandler
from event import FillEvent, OrderEvent, BulkOrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils import ceil_dt, from_exchange_to_standard_notation, from_standard_to_exchange_notation, truncate, get_precision
from utils.helpers import move_figure, plot, compute_all_indicators

from stats.trades import generate_trade_stats

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
        self.default_leverage = configuration.default_leverage
        self.save_to_db = configuration.save_to_db
        self.initial_capital = configuration.initial_capital
        self.indicators = configuration.indicators


        self.current_portfolio = self.construct_current_portfolios()
        self.all_portfolios = []

        self.all_transactions = []

        self.db = MongoHandler()
        self.legends_added = False


    def construct_current_portfolios(self):
      print('Building current portfolios')
      d = {}

      for s in self.instruments:
        price = self.data.get_latest_bar_value('bitmex', s, 'close') or 0
        d['bitmex-{}-price'.format(s)] = price
        d['bitmex-{}-position'.format(s)] = 0
        d['bitmex-{}-position-in-BTC'.format(s)] = 0
        d['bitmex-{}-position-in-USD'.format(s)] = 0
        d['bitmex-{}-leverage'.format(s)] = self.default_leverage
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
        price = self.data.get_latest_bar_value('bitmex', s, 'close')
        df['bitmex-{}-price'.format(s)] = price
        df['bitmex-{}-position-in-BTC'.format(s)] = quantity * price
        df['bitmex-{}-position-in-USD'.format(s)] = quantity * price  * btc_price
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

      if self.save_to_db:
        self.write_to_db(df)

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
        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])
        direction = { 'buy': 1, 'sell': -1 }[fill.direction]
        symbol = fill.symbol
        quantity = fill.quantity
        fee_rate = fill.fee

        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        previous_position = self.current_portfolio['bitmex-{}-position'.format(symbol)]
        new_position = previous_position + direction * quantity
        entry_price = self.data.get_latest_bar_value('bitmex', symbol, 'close')

        btc_value = entry_price * quantity

        btc_fee = btc_value * fee_rate
        balance = self.current_portfolio['bitmex-BTC-balance']
        leverage = self.default_leverage

        self.current_portfolio['bitmex-{}-position'.format(symbol)] += direction * quantity
        self.current_portfolio['bitmex-{}-price'.format(symbol)] = entry_price
        self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = direction * btc_value
        self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = direction * btc_value * btc_price
        self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = leverage
        self.current_portfolio['bitmex-{}-fill'.format(symbol)] = direction
        self.current_portfolio['bitmex-total-position-in-BTC'] += direction * btc_value * btc_price
        self.current_portfolio['bitmex-total-position-in-USD'] += direction * btc_value

        # Update holdings list with new quantities
        self.current_portfolio['bitmex-BTC-balance'] -= btc_fee
        self.current_portfolio['bitmex-{}-entry-price'.format(symbol)] = entry_price
        self.current_portfolio['bitmex-BTC-available-balance'] = self.current_portfolio['bitmex-BTC-available-balance'] - (abs(new_position) - abs(previous_position)) * (entry_price / leverage) - btc_fee
        self.current_portfolio['bitmex-BTC-price'] = btc_price
        self.current_portfolio['bitmex-BTC-balance-in-USD'] = (balance - btc_fee) * btc_price
        self.current_portfolio['total'] = (balance - btc_fee)
        self.current_portfolio['total-in-USD'] = (balance - btc_fee) * btc_price
        self.current_portfolio['fee'] += btc_fee

        txn = {}
        txn['datetime'] = latest_datetime
        txn['amount'] = direction * quantity
        txn['price'] = entry_price * btc_price
        txn['txn_dollars'] = direction * entry_price * btc_price * quantity
        txn['symbol'] = symbol

        self.all_transactions.append(txn)

    def update_portfolio_from_exit(self, fill):
        symbol = fill.symbol
        fee_rate = fill.fee

        latest_datetime = self.data.get_latest_bar_datetime('bitmex', self.instruments[0])
        btc_price = self.data.get_latest_bar_value('bitmex', 'BTC/USD', 'close')
        entry_price = self.current_portfolio['bitmex-{}-entry-price'.format(symbol)]
        price = self.data.get_latest_bar_value('bitmex', symbol, 'close') or 0
        quantity = self.current_portfolio['bitmex-{}-position'.format(symbol)]
        direction = -1 if quantity > 0 else 1
        leverage = self.default_leverage

        # We distinguish between the different contracts
        if (symbol == 'BTC/USD'):
          btc_value = quantity / price
          btc_fee = btc_value * fee_rate
          self.current_portfolio['bitmex-{}-position'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-price'.format(symbol)] = price
          self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = leverage
          self.current_portfolio['bitmex-{}-fill'.format(symbol)] = direction
          self.current_portfolio['bitmex-total-position-in-BTC'] -= btc_value
          self.current_portfolio['bitmex-total-position-in-USD'] -= btc_value * btc_price
          self.current_portfolio['bitmex-BTC-balance'] += quantity * (1 / entry_price - 1 / price) - btc_fee
          self.current_portfolio['bitmex-BTC-available-balance'] += abs(btc_value / leverage) - btc_fee
          self.current_portfolio['bitmex-BTC-price'] = btc_price
          self.current_portfolio['bitmex-BTC-balance-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price

          # here, total, totalUSD are equal to the bitmex-BTC-Balance and bitmex-BTC-balance-in-USD fields
          self.current_portfolio['total'] += direction * quantity * (1 / entry_price - 1 / price) - btc_fee
          self.current_portfolio['total-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price
          self.current_portfolio['fee'] += btc_fee

        else:
          btc_value = quantity * price
          btc_fee = btc_value * fee_rate
          self.current_portfolio['bitmex-{}-position'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-price'.format(symbol)] = price
          self.current_portfolio['bitmex-{}-position-in-BTC'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-position-in-USD'.format(symbol)] = 0
          self.current_portfolio['bitmex-{}-leverage'.format(symbol)] = leverage
          self.current_portfolio['bitmex-{}-fill'.format(symbol)] = direction
          self.current_portfolio['bitmex-total-position-in-BTC'] -= btc_value
          self.current_portfolio['bitmex-total-position-in-USD'] -= btc_value * btc_price

          self.current_portfolio['bitmex-BTC-balance'] += quantity * (price - entry_price) - btc_fee
          self.current_portfolio['bitmex-BTC-available-balance'] += abs(btc_value / leverage) - btc_fee
          self.current_portfolio['bitmex-BTC-price'] = btc_price
          self.current_portfolio['bitmex-BTC-balance-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price

          # here, total, totalUSD are equal to the bitmex-BTC-Balance and bitmex-BTC-balance-in-USD fields
          self.current_portfolio['total'] += direction * quantity * (price - entry_price) - btc_fee
          self.current_portfolio['total-in-USD'] = self.current_portfolio['bitmex-BTC-balance'] * btc_price
          self.current_portfolio['fee'] += btc_fee

        txn = {}
        txn['datetime'] = latest_datetime
        txn['amount'] = direction * quantity
        txn['price'] = price * btc_price
        txn['txn_dollars'] = direction * entry_price * btc_price * quantity
        txn['symbol'] = symbol

        self.all_transactions.append(txn)

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
            # Might want to throw an error here
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
                print('Available Balance exceeded')
                # Might want to throw an error here
                continue

            if (quantity == 0):
                # Might want to throw an error here
                continue

            order = OrderEvent(exchange, sig.symbol, 'Market', quantity, side, 1)
            events.append(order)

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

      self.update_charts()


    def update_charts(self):
      if not self.all_portfolios:
        return

      portfolios = pd.DataFrame(self.all_portfolios).copy()
      portfolios.set_index('datetime', inplace=True)

      btc_returns = portfolios['bitmex-BTC-balance'].pct_change()
      returns = portfolios['total-in-USD'].pct_change()
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
        portfolios["bitmex-{}-position-in-USD".format(s)][-1000:].plot(ax=self.positions_and_available_balance_ax, lw=1., color=col, label=position)

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

    def write_to_db(self, current_portfolio):
        """
        Saves the position and holdings updates to a database or to a file
        """
        self.db.insert_portfolio(current_portfolio)

    def create_backtest_result_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        portfolios = pd.DataFrame(self.all_portfolios)
        transactions = pd.DataFrame(self.all_transactions)
        indicators = compute_all_indicators(self.instruments, self.data, self.indicators)
        columns = ['datetime'] + [ 'bitmex-{}-position-in-USD'.format(s) for s in self.instruments ]
        positions = portfolios[columns]
        positions['cash'] = portfolios['bitmex-BTC-available-balance']
        portfolios.set_index('datetime', inplace=True)
        positions.set_index('datetime', inplace=True)

        if not transactions.empty:
          transactions.set_index('datetime', inplace=True)


        trades = pf.round_trips.extract_round_trips(transactions)

        portfolios['benchmark_equity'] = self.initial_capital * (portfolios['bitmex-BTC-price'] / portfolios['bitmex-BTC-price'].ix[0])
        portfolios['btc_benchmark_equity'] = self.initial_capital / portfolios['bitmex-BTC-price']

        portfolios['returns'] = portfolios['total-in-USD'].pct_change()
        portfolios['btc_returns'] = portfolios['total'].pct_change()
        portfolios['benchmark_returns'] = portfolios['benchmark_equity'].pct_change()
        portfolios['btc_benchmark_returns'] = portfolios['btc_benchmark_equity'].pct_change()

        portfolios['equity_curve'] = (1.0 + portfolios['returns']).cumprod()
        portfolios['btc_equity_curve'] = (1.0 + portfolios['btc_returns']).cumprod()
        portfolios['benchmark_equity_curve'] = (1.0 + portfolios['benchmark_returns']).cumprod()
        portfolios['btc_benchmark_equity_curve'] = (1.0 + portfolios['btc_benchmark_returns']).cumprod()

        self.portfolio_dataframe = portfolios
        self.positions_dataframe = positions
        self.transactions_dataframe = transactions
        self.indicators_dataframe = indicators
        self.trades_dataframe = trades

    def open_results_in_excel(self):
        file_path = os.path.join(self.result_dir, 'last/results.csv')
        os.system("open -a 'Microsoft Excel.app' '%s'" % file_path)

    def output_graphs(self):
        plt.ioff()
        """
        Creates a list of summary statistics and plots
        performance graphs
        """
        curve = self.portfolio_dataframe
        positions = self.positions_dataframe
        txns = self.transactions_dataframe

        returns = curve['returns']
        benchmark_returns = curve['benchmark_returns']
        equity_curve = curve['equity_curve']
        btc_equity_curve = curve['btc_equity_curve']
        drawdown = curve['drawdown']
        benchmark_equity_curve = curve['benchmark_equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(equity_curve)
        benchmark_drawdown, _, _ = create_drawdowns(benchmark_equity_curve)

        fig = plt.figure(figsize=(15,10))
        fig.patch.set_facecolor('white')
        gs = gridspec.GridSpec(3,1, width_ratios=[1], height_ratios=[1,1,1], hspace=0.2, wspace=0.4)

        usd_equity_ax = fig.add_subplot(gs[0], ylabel='USD Portfolio Value, %')
        usd_equity_ax.axes.get_xaxis().set_visible(False)
        equity_curve.plot(ax=usd_equity_ax, color="blue", lw=1., label='Backtest')
        benchmark_equity_curve.plot(ax=usd_equity_ax, color="gray", lw=1., label='Benchmark (Hold USD)')
        usd_equity_ax.legend(loc='best', frameon=False)
        plt.grid(True)

        btc_equity_ax = fig.add_subplot(gs[1], ylabel='BTC Portfolio Value, %')
        btc_equity_ax.axes.get_xaxis().set_visible(False)
        btc_equity_curve.plot(ax=btc_equity_ax, color="orange", lw=1.)
        plt.grid(True)

        drawdown_ax = fig.add_subplot(gs[2], ylabel='Drawdowns, %', xlabel='Date')
        drawdown.plot(ax=drawdown_ax, color="red", lw=1., label='Backtest')
        benchmark_drawdown.plot(ax=drawdown_ax, color="gray", lw=1., label='Benchmark (Hold USD)')
        drawdown_ax.legend(loc='best', frameon=False)
        plt.grid(True)

        self.price_figure = {}
        for s in self.instruments:
          fig = plt.figure(figsize=(15,10))
          ax = fig.add_subplot(111, ylabel='bitmex-{} Price'.format(s))
          fill_id = 'bitmex-{}-fill'.format(s)
          price_id = 'bitmex-{}-price'.format(s)
          prices = curve[price_id]
          fills = curve[fill_id]
          buys = pd.Series({ x: prices[x] if fills[x] == 1 else np.NaN for x in curve.index })
          sells = pd.Series({ x: prices[x] if fills[x] == -1 else np.NaN for x in curve.index })
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

        fig = plt.figure(figsize = (15,3))
        pf.plot_long_short_holdings(returns, positions)

        plt.figure(figsize = (15,10))
        pf.plot_slippage_sensitivity(returns, positions, txns)

        # plt.figure(figsize=(15,10))
        # pf.plot_gross_leverage(returns, positions)

        plt.figure(figsize = (15, 10))
        pf.plot_rolling_volatility(returns, rolling_window=30).set_xlabel('date')

        plt.figure(figsize = (15, 10))
        pf.plot_rolling_sharpe(returns, rolling_window=30).set_xlabel('Date')

        if txns.empty:
          txns = None

        rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.995',
            'figure.facecolor': '0.97',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.labelweight': 'bold',
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        }

        with pf.plotting.plotting_context(rc=rc):
          position_fig = pf.create_position_tear_sheet(returns, positions=positions, return_fig=True)
          position_fig.savefig('results/last/position_tear_sheet.pdf')
          plt.close(position_fig)

          returns_fig = pf.create_returns_tear_sheet(returns, positions=positions, transactions=txns, benchmark_rets=benchmark_returns, return_fig=True)
          returns_fig.savefig('results/last/returns_tear_sheet.pdf')
          plt.close(returns_fig)

          round_trip_fig = pf.create_round_trip_tear_sheet(returns, positions, txns, return_fig=True)
          round_trip_fig.savefig('results/last/round_trip_tear_sheet.pdf')
          plt.close(round_trip_fig)

          # transaction_fig = pf.create_txn_tear_sheet(returns, positions, transactions=txns, return_fig=True)
          # transaction_fig.savefig('results/last/transaction_tear_sheet.pdf')
          # plt.close(transaction_fig)

          # live_start_date = returns.index[-40]
          # bayesian_fig = pf.create_bayesian_tear_sheet(returns, live_start_date=live_start_date, return_fig=True)
          # bayesian_fig.savefig('results/last/bayesian_tear_sheet.pdf')
          # plt.close(bayesian_fig)

        webbrowser.open_new(r'file:///Users/davidvanisacker/Programming/Trading/backtest/results/last/returns_tear_sheet.pdf')
        webbrowser.open_new(r'file:///Users/davidvanisacker/Programming/Trading/backtest/results/last/position_tear_sheet.pdf')
        webbrowser.open_new(r'file:///Users/davidvanisacker/Programming/Trading/backtest/results/last/round_trip_tear_sheet.pdf')
        # webbrowser.open_new(r'file:///Users/davidvanisacker/Programming/Trading/backtest/results/last/transaction_tear_sheet.pdf')
        # webbrowser.open_new(r'file:///Users/davidvanisacker/Programming/Trading/backtest/results/last/bayesian_tear_sheet.pdf')
        plot()

    def save_results(self, backtest_result_dir):
        self.portfolio_dataframe.to_csv(os.path.join(self.result_dir, 'last/results.csv'))
        self.positions_dataframe.to_csv(os.path.join(self.result_dir, 'last/positions.csv'))
        self.transactions_dataframe.to_csv(os.path.join(self.result_dir, 'last/transactions.csv'))
        self.indicators_dataframe.to_csv(os.path.join(self.result_dir, 'last/indicators.csv'))
        self.trades_dataframe.to_csv(os.path.join(self.result_dir, 'last/trades.csv'))
        self.portfolio_dataframe.to_csv(os.path.join(backtest_result_dir, 'results.csv'))
        self.positions_dataframe.to_csv(os.path.join(backtest_result_dir, 'positions.csv'))
        self.transactions_dataframe.to_csv(os.path.join(backtest_result_dir, 'transactions.csv'))
        self.indicators_dataframe.to_csv(os.path.join(backtest_result_dir, 'indicators.csv'))
        self.trades_dataframe.to_csv(os.path.join(backtest_result_dir, 'trades.csv'))

    def compute_stats(self):
        """
        Creates a list of summary statistics and plots
        performance graphs
        """
        total_return = self.portfolio_dataframe['equity_curve'][-1]
        total_btc_return = self.portfolio_dataframe['btc_equity_curve'][-1]
        returns = self.portfolio_dataframe['returns']
        btc_returns = self.portfolio_dataframe['btc_returns']
        pnl = self.portfolio_dataframe['equity_curve']
        btc_pnl = self.portfolio_dataframe['btc_equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.portfolio_dataframe['drawdown'] = drawdown

        btc_sharpe_ratio = create_sharpe_ratio(btc_returns)
        btc_drawdown, btc_max_dd, btc_dd_duration = create_drawdowns(btc_pnl)
        self.portfolio_dataframe['btc_drawdown'] = btc_drawdown

        stats = generate_trade_stats(self.trades_dataframe)
        stats['general'] = {
          "Total USD Return": "%0.2f%%" % ((total_return - 1.0) * 100.0),
          "Total BTC Return": "%0.2f%%" % ((total_btc_return - 1.0) * 100.0),
          "Sharpe Ratio": "%0.2f" % sharpe_ratio,
          "BTC Sharpe Ratio": "%0.2f" % btc_sharpe_ratio,
          "Max Drawdown": "%0.2f%%" % (max_dd * 100.0),
          "BTC Max Drawdown": "%0.2f%%" % (btc_max_dd * 100.0),
          "Drawdown Duration": "%d" % dd_duration,
          "BTC Drawdown Duration": "%d" % btc_dd_duration
        }

        return stats

