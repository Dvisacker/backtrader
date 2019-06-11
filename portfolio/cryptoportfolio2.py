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
from utils.helpers import move_figure

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
        self.assets = configuration.assets
        self.start_date = configuration.start_date
        self.result_dir = configuration.result_dir
        self.initial_capital = configuration.initial_capital
        self.all_positions = self.construct_all_positions()
        self.current_positions = self.construct_current_positions()
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()


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
            d['{}-{}-close'.format(e,s)] = 0

        for e in self.assets:
          d[e] = { e: {} }
          for s in self.assets[e]:
            d[e][s] = self.initial_assets[e][s]

        d['datetime'] = self.start_date
        d['fee'] = 0.0
        d['total'] = 0.0 # need to modify this
        d['fill'] = ''
        return [d]

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict( (k,v) for k,v in [(e, {}) for e in self.instruments])
        for e in d:
          d[e] = dict((k,v) for k,v in [(s, 0.0) for s in self.instruments[e]])

        d = dict( (k,v) for k,v in [(e, {}) for e in self.assets])
        for e in d:
          d[e] = dict((k,v) for k,v in [(s, 0.0) for s in self.assets[e]])

        d['fee'] = 0.0
        d['total'] = 0.0
        d['fill'] = ''
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

        dh = dict( (k,v) for k, v in [(e, {}) for e in self.assets] )
        for e in dh:
          dh[e] = dict( (k,v) for k, v in [(s, 0) for s in self.assets[e]])

        dh['datetime'] = latest_datetime
        dh['fee'] = self.current_holdings['fee']
        dh['total'] = self.current_holdings['total']

        # We assume the fill event comes after the update_timeindex call.
        dh['fill'] = self.current_holdings['fill']

        for e in self.assets:
          for s in self.assets[e]:
            instrument = '{}{}'.format(s, "USD")
            close_price = self.data.get_latest_bar_value(e, instrument, "close")
            market_value = self.current_holdings[e][s] * close_price
            dh[e][s] = market_value
            dh['total'] += market_value

        # NOTE This does seem to cover only the case where all the assets are traded against a similar quote currency.
        for e in self.instruments:
          for s in self.instruments[e]:
            # Approximation to the real value
            close_price = self.data.get_latest_bar_value(e, s, "close")
            market_value = self.current_positions[e][s] * close_price
            dh[e][s] = market_value
            dh['{}-{}-close'.format(e,s)] = close_price
            dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)

        # Reset the fill variable
        self.current_holdings['fill'] = ''

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

        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.data.get_latest_bar_value(fill.exchange, fill.symbol, "close")
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.exchange][fill.symbol] += cost
        self.current_holdings['fee'] += fill.fee
        self.current_holdings['cash'] -= (cost + fill.fee)
        self.current_holdings['total'] -= (cost + fill.fee)
        # print('Registering Fill at {}'.format(self.current_holdings['datetime']))
        self.current_holdings['fill'] = fill.direction

    def update_bitmex_holdings_from_fill(self, fill):
        # Check whether the fill is a buy or sell
        direction = fill.direction
        leverage = fill.leverage
        quantity = fill.quantity
        current_price = self.data.get_latest_bar_value(fill.exchange, fill.symbol, "close")

        available_balance = self.current_holdings['cash']
        new_available_balance = available_balance - fill.quantity / fill.leverage

        self.current_holdings['cash'] =
        self.current_holdings[fill.exchange][fill.symbol] += fill.quantity



        # In the case of bitmex: quantity = 100 for example

        # Open a short x (-1)
        cost = current_price * (1 / quantity)




        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.data.get_latest_bar_value(fill.exchange, fill.symbol, "close")
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.exchange][fill.symbol] += cost
        self.current_holdings['fee'] += fill.fee
        self.current_holdings['cash'] -= (cost + fill.fee)
        self.current_holdings['total'] -= (cost + fill.fee)
        self.current_holdings['fill'] = fill.direction

        # print('Registering Fill at {}'.format(self.current_holdings['datetime']))

    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)


    def generate_order(self, signal):
        if signal.exchange == "bitmex":
          order = self.generate_bitmex_order(signal)
        else:
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
        strength = signal.strength

        mkt_quantity = 300
        cur_quantity = self.current_positions[exchange][symbol]
        order_type = 'MKT'

        fill_cost = self.data.get_latest_bar_value(fill.exchange, fill.symbol, "close")
        cost = fill_dir * fill_cost



        available_quantity = cost / (fill_dir * fill_cost)

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(exchange, symbol, order_type, mkt_quantity, 'SELL')

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

        # order = OrderEvent(exchange, symbol, order_type, (new_quantity - current_quantity), new_leverage)
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
      self.ax1 = fig.add_subplot(311, ylabel='Portfolio value, %')
      self.ax2 = fig.add_subplot(312, ylabel='Currency prices')
      self.ax3 = fig.add_subplot(313, ylabel='Positions')

      fig = plt.figure(figsize=(10,10))
      move_figure(fig, 1000, 0)
      self.ax4 = fig.add_subplot(111, ylabel='Drawdown underwater')

      self.update_charts()
      self.ax1.legend(loc='upper left', frameon=False, markerscale=12)
      self.ax2.legend(loc='upper left', frameon=False, markerscale=12)
      self.ax3.legend(loc='upper left', frameon=False, markerscale=12)

      # self.ax2 = fig.add_subplot(512, ylabel='Period returns, %')
      # self.ax3 = fig.add_subplot(513, ylabel='Drawdowns, %')
      # self.ax5 = fig.add_subplot(514, ylabel=)
      # fig = plt.figure(figsize=(15,10))
      # self.ax5 = fig.add_subplot(111, ylabel='Drawdown periods')
      # fig = plt.figure(figsize=(15,10))
      # self.ax6 = fig.add_subplot(111, ylabel='Rolling volatlity')
      # fig = plt.figure(figsize=(15,10))
      # self.ax7 = fig.add_subplot(111, ylabel='Rolling sharpe')


    def update_charts(self):
      curve = pd.DataFrame(self.all_holdings).copy()

      for e in self.instruments:
          for s in self.instruments[e]:
            curve["{}-{}".format(e,s)] = curve[e].map(lambda x: x[s])

          curve.drop(columns=e, inplace=True)

      curve.set_index('datetime', inplace=True)
      returns = curve['total'].pct_change()
      cash = curve['cash']
      equity_curve = (1.0+returns).cumprod()
      drawdown, max_dd, dd_duration = create_drawdowns(equity_curve)

      buys = pd.Series({ x: equity_curve[x] if curve['fill'][x] == "BUY" else np.NaN for x in curve.index })
      sells = pd.Series({ x: equity_curve[x] if curve['fill'][x] == "SELL" else np.NaN for x in curve.index })

      # buys = pd.Series(curve['fill'].index.map(lambda x: equity_curve[x] if curve['fill'][x] == "BUY" else 0))
      # sells = pd.Series(curve['fill'].index.map(lambda x: equity_curve[x] if curve['fill'][x] == "SELL" else 0))


      # print(buys.tail())
      # print(sells.tail())

      # Plot the equity curve
      equity_curve.plot(ax=self.ax1, color="blue", lw=1., label='Total Portfolio Value')
      buys.plot(ax=self.ax1, color='green', lw=1., marker='o', label='Buys')
      sells.plot(ax=self.ax1, color='red', lw=1, marker='x', label='Red')
      cash.plot(ax=self.ax3, color="orange", lw=1., label="Cash")

      plt.grid(True)

      for e in self.instruments:
        for s in self.instruments[e]:
          price_label = '{}-{} Price'.format(e,s).capitalize()
          position = '{}-{} Position'.format(e,s).capitalize()

          curve["{}-{}-close".format(e,s)].plot(ax=self.ax2, color="red", lw=1., label=price_label)
          curve["{}-{}".format(e,s)].plot(ax=self.ax3, color="green", lw=1., label=position)

      pf.plot_drawdown_underwater(returns, ax=self.ax4).set_xlabel('Date')
      plt.pause(0.001)
      plt.axis('tight')

      # total = np.array(list(map(lambda x: x['total'], self.all_holdings)))
      # returns = np.array([100.0 * a1 / a2 - 100 for a1, a2 in zip(total[1:], total)])

      # sharpe_ratio = create_sharpe_ratio(returns)
      # equity_curve = np.cumprod(1 + returns)
      # drawdown, max_dd, dd_duration = create_drawdowns(equity_curve)

      # Plot the returns
      # returns.plot(ax=self.ax2, color="black", lw=1.)
      # plt.grid(True)

      # # # Plot the returns
      # drawdown.plot(ax=self.ax3, color="red", lw=1.)
      # plt.grid(True)
      # # pf.show_perf_stats(returns)
      # # pf.show_worst_drawdown_periods(returns)
      # pf.plot_drawdown_periods(returns, top=5, ax=self.ax5).set_xlabel('Date')
      # pf.plot_rolling_volatility(returns, rolling_window=30, ax=self.ax6).set_xlabel('Date')
      # pf.plot_rolling_sharpe(returns, rolling_window=30, ax=self.ax7).set_xlabel('Date')
      # plt.figure(figsize = (15, 10))

      # plt.figure(figsize = (15, 10))
      # pf.plot_returns(returns).set_xlabel('Date')

      # plt.figure(figsize = (15, 10))
      # pf.plot_return_quantiles(returns).set_xlabel('Timeframe')

      # plt.figure(figsize = (15, 10))
      # pf.plot_monthly_returns_dist(returns).set_xlabel('Returns')
      # pf.create_returns_tear_sheet(returns)


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

        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown

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

        fig = pf.create_returns_tear_sheet(returns, return_fig=True)
        fig.savefig('returns_tear_sheet.pdf')

        root = 'file:///Users/davidvanisacker/Programming/Trading/backtest/setups/backtest'
        result_dir = 'results/last/returns_tear_sheet.pdf'
        tearsheet_path = os.path.join(root, result_dir)
        webbrowser.open_new(tearsheet_path)
        plt.show()


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

            stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                    ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                    ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                    ("Drawdown Duration", "%d" % dd_duration)]

            self.equity_curve.to_csv(os.path.join(self.result_dir, 'last/equity.csv'))
            self.equity_curve.to_csv(os.path.join(backtest_result_dir, 'equity.csv'))

            return stats
