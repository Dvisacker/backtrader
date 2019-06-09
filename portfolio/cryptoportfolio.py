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

import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyfolio as pf
import seaborn as sns; sns.set()

plt.style.use('ggplot')

from event import FillEvent, OrderEvent
from performance import create_sharpe_ratio, create_drawdowns
from utils.helpers import move_figure, plot

class CryptoPortfolio(object):
    """
    The CryptoPortfolio is similar to the previous portfolio
    class. Instead of using the adjusted close data point, it uses
    the close datapoint
    """

    def __init__(self, data, events, configuration):
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
        self.data = data
        self.events = events

        self.exchanges = configuration.exchange_names
        self.exchange = configuration.exchange_names[0]
        self.instruments = configuration.instruments
        self.start_date = configuration.start_date
        self.result_dir = configuration.result_dir
        self.initial_capital = configuration.initial_capital
        self.all_positions = []
        self.current_positions = self.construct_current_positions()
        self.all_holdings = []
        self.current_holdings = self.construct_current_holdings()

        self.legends_added = False


    def construct_current_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        for e in d:
          d[e] = dict((k,v) for k,v in [(s, 0) for s in self.instruments[e]])

        return d

    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        for e in d:
          d[e] = dict((k,v) for k,v in [(s, 0) for s in self.instruments[e]])

        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = {}
        for e in self.instruments:
          d[e] = { e: {} }
          for s in self.instruments[e]:
            d[e][s] = 0.0
            d['{}-{}-price'.format(e,s)] = 0
            d['{}-{}-fill'.format(e,s)] = ''

        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['fee'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = {}
        for e in self.instruments:
          d[e] = { e: {} }
          for s in self.instruments[e]:
              d[e][s] = 0.0
              d['{}-{}-price'.format(e,s)] = 0
              d['{}-{}-fill'.format(e,s)] = ''

        d['cash'] = self.initial_capital
        d['fee'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Adds a new record to the positions matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OHLCV).
        Makes use of a MarketEvent from the events queue.
        """
        default_symbol = self.instruments[self.exchange][0]
        default_exchange = self.exchange
        latest_datetime = self.data.get_latest_bar_datetime(default_exchange, default_symbol)

        # Update positions
        # ================
        dp = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        for e in dp:
          dp[e] = dict( (k,v) for k,v in [(s, 0) for s in self.instruments[e]])

        dp['datetime'] = latest_datetime

        for e in self.instruments:
          for s in self.instruments[e]:
            dp[e][s] = self.current_positions[e][s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        # ===============
        dh = dict( (k,v) for k, v in [(e, {}) for e in self.instruments] )
        for e in dh:
          dh[e] = dict( (k,v) for k, v in [(s, 0) for s in self.instruments[e]])

        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['fee'] = self.current_holdings['fee']
        dh['total'] = self.current_holdings['total']

        # NOTE This does seem to cover only the case where all the assets are traded against a similar quote currency.
        for e in self.instruments:
          for s in self.instruments[e]:
            # Approximation to the real value
            close_price = self.data.get_latest_bar_value(e, s, "close")
            market_value = self.current_positions[e][s] * close_price
            dh[e][s] = market_value
            dh['{}-{}-fill'.format(e,s)] = self.current_holdings['{}-{}-fill'.format(e,s)]
            dh['{}-{}-price'.format(e,s)] = close_price
            dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)

        # Reset the fill variable
        for e in self.instruments:
          for s in self.instruments[e]:
            self.current_holdings['{}-{}-fill'.format(e,s)] = ''

    # ======================
    # FILL/POSITION HANDLING
    # ======================

    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to
        reflect the new position.
        Parameters:
        fill - The Fill object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update positions list with new quantities
        self.current_positions[fill.exchange][fill.symbol] += fill_dir*fill.quantity

        if fill.leverage is not None:
          self.current_positions[fill.exchange][fill.symbol] = fill.leverage

    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to
        reflect the holdings value.
        Parameters:
        fill - The Fill object to update the holdings with.
        """

        symbol = fill.symbol
        exchange = fill.exchange

        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.data.get_latest_bar_value(exchange, symbol, "close")
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[exchange][symbol] += cost
        self.current_holdings['fee'] += fill.fee
        self.current_holdings['cash'] -= (cost + fill.fee)
        self.current_holdings['total'] -= (cost + fill.fee)
        self.current_holdings['{}-{}-fill'.format(exchange, symbol)] = fill.direction

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)


    def generate_order(self, signal):
        order = self.generate_naive_order(signal)
        return order

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

        cur_quantity = self.current_positions[exchange][symbol]
        cash = self.current_holdings['cash']
        price = self.data.get_latest_bar_value(exchange, symbol, "close")
        amount = cash / price
        order_type = 'MKT'

        # fill_cost = self.data.get_latest_bar_value(fill.exchange, fill.symbol, "close")
        # cost = fill_dir * fill_cost
        # available_quantity = cost / (fill_dir * fill_cost)

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, amount, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, amount, 'SELL')

        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(exchange, symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(exchange, symbol, order_type, abs(cur_quantity), 'BUY')

        return order


    def generate_bitmex_order(self, signal):
        """
        Bitmex works a bit differently than other exchanges since BTC is held by default
        Therefore:
        - To be short BTC: Short x2
        - To be long BTC: Simply hold (no positions)
        - To be neutral BTC: Short x1
        - To be long BTC x2: Long x1
        """

        order = None
        exchange = signal.exchange
        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength
        order_type = "MKT"

        current_quantity = self.current_positions[exchange][symbol]
        current_cash = self.current_positions['cash'] #in USD
        current_leverage = self.current_positions['leverage']

        new_position_size = 10

        if direction == "SHORT":
          new_quantity = -new_position_size
          new_leverage = 2

          position_size = new_leverage * ()

        elif direction == "LONG":
          new_quantity = 0
          new_leverage = 1 #doesn't matter

        elif direction == "EXIT":
          new_quantity = -new_position_size
          new_leverage = 1

        if (new_quantity - current_quantity) > 0:
          order_side = "BUY"
        elif (new_quantity - current_quantity) <= 0:
          order_side = "SELL"

        order = OrderEvent(exchange, symbol, order_type, abs(new_quantity - current_quantity), order_side, new_leverage)
        return order

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_order(event)
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

        # We format the balance columns that contain objects with symbols as keys to different columns
        # respectively named "exchange-symbol"
        for e in self.instruments:
          for s in self.instruments[e]:
            curve["{}-{}".format(e,s)] = curve[e].map(lambda x: x[s])

          curve.drop(columns=e, inplace=True)

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
        equity_curve = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(equity_curve)
        self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        return stats


    def initialize_graphs(self):
      plt.ion()

      # Plot three charts: Equity curve,
      # period returns, drawdowns
      fig = plt.figure(figsize=(10,10))
      # Set the outer colour to white
      fig.patch.set_facecolor('white')
      self.ax1 = fig.add_subplot(211, ylabel='Portfolio value, %')
      self.ax2 = fig.add_subplot(212, ylabel='Prices')


      self.price_axes = {}
      colors = ['red', 'blue', 'yellow', 'green', 'black']
      for e in self.instruments:
        for (i, s) in enumerate(self.instruments[e]):
          self.price_axes['{}-{}'.format(e,s)] = self.ax2.twinx()
          self.price_axes['{}-{}'.format(e,s)].tick_params(axis='y', labelcolor=colors[i])

      fig.tight_layout()

      fig = plt.figure(figsize=(10,10))
      move_figure(fig, 1000, 0)
      self.ax3 = fig.add_subplot(211, ylabel='Positions')
      self.ax4 = fig.add_subplot(212, ylabel='Currency prices')

      self.update_graphs()


    def update_graphs(self):
      if not self.all_holdings:
        return

      curve = pd.DataFrame(self.all_holdings).copy()

      for e in self.instruments:
          for s in self.instruments[e]:
            curve["{}-{}".format(e,s)] = curve[e].map(lambda x: x[s])

          curve.drop(columns=e, inplace=True)

      curve.set_index('datetime', inplace=True)
      returns = curve['total'].pct_change()
      cash = curve['cash']
      equity = (1.0+returns).cumprod()
      drawdown, max_dd, dd_duration = create_drawdowns(equity)

      equity.plot(ax=self.ax1, color="blue", lw=1., label='Total Portfolio Value')

      # Plot the equity curve
      cash.plot(ax=self.ax3, color="orange", lw=1., label="Cash")
      colors = ['red', 'blue', 'yellow', 'green', 'black']

      for e in self.instruments:
        for (i, s) in enumerate(self.instruments[e]):
          col = colors[i]
          ax = self.price_axes['{}-{}'.format(e,s)]
          price_label = '{}-{} Price'.format(e,s).capitalize()
          position = '{}-{} Position #'.format(e,s).capitalize()
          curve["{}-{}-price".format(e,s)].plot(ax=ax, lw=1., color=col, label=price_label)
          curve["{}-{}".format(e,s)][-1000:].plot(ax=self.ax3, lw=1., color=col, label=position)

      pf.plot_drawdown_underwater(returns, ax=self.ax4).set_xlabel('Date')
      plt.pause(0.001)
      plt.axis('tight')

      if not self.legends_added:
        self.ax1.legend(loc='upper left', frameon=False, markerscale=12)
        self.ax2.legend(loc='upper left', frameon=False, markerscale=12)
        self.ax3.legend(loc='upper left', frameon=False, markerscale=12)

        for e in self.instruments:
          for s in self.instruments[e]:
            self.price_axes['{}-{}'.format(e,s)].legend(loc='upper left', frameon=False, markerscale=12)


      self.legends_added = True

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
        ax2 = fig.add_subplot(312, ylabel='Period returns, %')
        returns.plot(ax=ax2, color="black", lw=1.)
        plt.grid(True)

        # Plot the returns
        ax3 = fig.add_subplot(313, ylabel='Drawdowns, %')
        drawdown.plot(ax=ax3, color="red", lw=1.)
        plt.grid(True)

        self.price_figure = {}
        for e in self.instruments:
          for s in self.instruments[e]:
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111, ylabel='{}-{} Price'.format(e,s))
            fill_id = '{}-{}-fill'.format(e,s)
            price_id = '{}-{}-price'.format(e,s)
            prices = curve[price_id]
            fills = curve[fill_id]
            buys = pd.Series({ x: prices[x] if fills[x] == "BUY" else np.NaN for x in curve.index })
            sells = pd.Series({ x: prices[x] if fills[x] == "SELL" else np.NaN for x in curve.index })
            prices.plot(ax=ax, color='blue', lw=1., label='{}-{} Price'.format(e,s))
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

        # root = 'file:///Users/davidvanisacker/Programming/Trading/backtest/'
        # result_dir = 'results/last/returns_tear_sheet.pdf'
        # tearsheet_path = os.path.join(root, result_dir)
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
