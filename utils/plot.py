import datetime
import numpy as numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import statsmodels.tsa.stattools as ts

from pandas.stats.api import ols


def plot_price_series(df, ts1, ts2):
  months = mdates.MonthLocator()
  fig, ax = plt.subplots()
  ax.plot(df.index, df[ts1], label=ts1)
  ax.plot(df.index, df[ts2], label=ts2)
  ax.xaxis.set_major_locator(months)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013,1,1))
  ax.grid(True)
  fig.autofmt_xdate()

  plt.xlabel('Month/Year')
  plt.ylabel('Price ($)')
  plt.title('%s and %s daily prices' %(ts1, ts2))
  plt.legend()
  plt.show()


def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()


def plot_residuals(df):
    months = mdates.MonthLocator()  # every month
    fig, ax = plt.subplots()
    ax.plot(df.index, df["res"], label="Residuals")
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
    ax.grid(True)
    fig.autofmt_xdate()

    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('Residual Plot')
    plt.legend()

    plt.plot(df["res"])
    plt.show()