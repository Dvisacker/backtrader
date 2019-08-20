#!/usr/bin/python
# -*- coding: utf-8 -*-

# plot_performance.py

import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyfolio as pf


if __name__ == "__main__":
    data = pd.io.parsers.read_csv(
        "results.csv", header=0,
        parse_dates=True,
        index_col=0
    )

    # Plot three charts: Equity curve,
    # period returns, drawdowns
    fig = plt.figure(figsize = (15, 10))
    # Set the outer colour to white
    fig.patch.set_facecolor('white')

    equity_curve = data['equity_curve']
    returns = data['returns']
    drawdown = data['drawdown']

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

    pf.show_worst_drawdown_periods(returns)

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

    # plt.figure(figsize = (15, 30))
    # pf.create_returns_tear_sheet(returns)

    # Plot the figure
    # plt.show()