import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pdb

def ofi_model(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)

    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True,
                            x='ofi', y='next_midprice_returns',
                            title = 'Returns/OFI Correlation',
                            alpha=0.5, figsize=(12,10))


    ofi_ = sm.add_constant(df['ofi'])
    ols = sm.OLS(df.next_midprice_returns, ofi_).fit()
    print(ols.summary2())
    plt.show()

def tfi_model(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)
    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='tfi', y='next_midprice_returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    tfi = sm.add_constant(df['tfi'])
    ols = sm.OLS(df.next_midprice_returns, tfi).fit()
    print(ols.summary2())
    plt.show()


def vfi_model(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)
    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='vfi', y='next_midprice_returns',
            title = 'VFI/Returns correlation', alpha=0.5, figsize=(12,10))

    vfi = sm.add_constant(df['vfi'])
    ols = sm.OLS(df.next_midprice_returns, vfi).fit()
    print(ols.summary2())
    plt.show()


def winsorized_tfi_model(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)
    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='tfi', y='next_midprice_returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    tfi = sm.add_constant(df['tfi'])
    ols = sm.OLS(df.next_midprice_returns, tfi).fit()
    print(ols.summary2())
    plt.show()


def ofi_comtemporeanous_returns(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)

    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='ofi', y='midprice_returns',
            title = 'Returns/OFI Correlation', alpha=0.5, figsize=(12,10))

    ofi_ = sm.add_constant(df['ofi'])
    ols = sm.OLS(df.midprice_returns, ofi_).fit()
    print(ols.summary2())
    plt.show()

def tfi_comtemporeanous_returns(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)
    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='tfi', y='midprice_returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    tfi = sm.add_constant(df['tfi'])
    ols = sm.OLS(df.midprice_returns, tfi).fit()
    print(ols.summary2())
    plt.show()

def vfi_comtemporeanous_returns(df, lag=1):
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)
    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='vfi', y='midprice_returns',
            title='VFI/Returns correlation', alpha=0.5, figsize=(12,10))

    vfi = sm.add_constant(df['vfi'])
    ols = sm.OLS(df.midprice_returns, vfi).fit()
    print(ols.summary2())
    plt.show()

def midprice_returns_lag_1(df):
    df['midprice_returns_lagged_1'] = df.midprice_returns.shift(1)
    df['midprice_returns_lagged_2'] = df.midprice_returns.shift(2)
    df['midprice_returns_lagged_3'] = df.midprice_returns.shift(3)

    df.dropna(inplace=True)
    print(df.head(10))
    df.plot(kind='scatter', grid=True, x='midprice_returns_lagged_1', y='midprice_returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    X = df[['midprice_returns_lagged_1']]
    X = sm.add_constant(X)

    ols = sm.OLS(df.midprice_returns, X).fit()
    print(ols.summary2())
    plt.show()

def midprice_returns_lag_3(df):
    df['midprice_returns_lagged_1'] = df.midprice_returns.shift(1)
    df['midprice_returns_lagged_2'] = df.midprice_returns.shift(2)
    df['midprice_returns_lagged_3'] = df.midprice_returns.shift(3)
    df.dropna(inplace=True)

    X = df[['midprice_returns_lagged_1', 'midprice_returns_lagged_2', 'midprice_returns_lagged_3']]
    X = sm.add_constant(X)

    ols = sm.OLS(df.midprice_returns, X).fit()
    print(ols.summary2())
    plt.show()


def flow_imbalance_1(df):
    df['tfi_lagged_1'] = df.tfi.shift(1)
    df['tfi_lagged_2'] = df.tfi.shift(2)
    df['tfi_lagged_3'] = df.tfi.shift(3)
    df['midprice_returns_lagged_1'] = df.midprice_returns.shift(1)
    df['midprice_returns_lagged_2'] = df.midprice_returns.shift(2)
    df['midprice_returns_lagged_3'] = df.midprice_returns.shift(3)
    df.dropna(inplace=True)

    X = df[['tfi_lagged_1', 'tfi_lagged_2', 'tfi_lagged_3',
            'midprice_returns_lagged_1', 'midprice_returns_lagged_2', 'midprice_returns_lagged_3']]
    X = sm.add_constant(X)

    ols = sm.OLS(df.midprice_returns, X).fit()
    print(ols.summary2())
    plt.show()

def flow_imbalance_2(bars, btc_bars):
    bars['tfi_lagged_1'] = bars.tfi.shift(1)
    bars['tfi_lagged_2'] = bars.tfi.shift(2)
    bars['tfi_lagged_3'] = bars.tfi.shift(3)
    bars['midprice_returns_lagged_1'] = bars.midprice_returns.shift(1)
    bars['midprice_returns_lagged_2'] = bars.midprice_returns.shift(2)
    bars['midprice_returns_lagged_3'] = bars.midprice_returns.shift(3)
    bars['btc_returns_lagged_1'] = btc_bars.midprice_returns.shift(1)
    bars['btc_returns_lagged_2'] = btc_bars.midprice_returns.shift(2)
    bars['btc_returns_lagged_3'] = btc_bars.midprice_returns.shift(3)
    bars.dropna(inplace=True)

    X = bars[['tfi_lagged_1', 'tfi_lagged_2', 'tfi_lagged_3',
            'midprice_returns_lagged_1', 'midprice_returns_lagged_2', 'midprice_returns_lagged_3',
            'btc_returns_lagged_1', 'btc_returns_lagged_2', 'btc_returns_lagged_3']]
    X = sm.add_constant(X)

    ols = sm.OLS(bars.midprice_returns, X).fit()
    print(ols.summary2())
    plt.show()

def winsorized_flow_imbalance_1(df):
    df['tfi_lagged_1'] = df.tfi.shift(1)
    df['tfi_lagged_2'] = df.tfi.shift(2)
    df['tfi_lagged_3'] = df.tfi.shift(3)
    df.dropna(inplace=True)

    threshold = abs(df['tfi_lagged_1']).quantile(0.95)
    new_df = df[abs(df['tfi_lagged_1']) > threshold]
    new_df.dropna(inplace=True)
    X = new_df[['tfi_lagged_1']]
    X = sm.add_constant(X)

    y = new_df['midprice_returns']

    ols = sm.OLS(y, X).fit()
    print(ols.summary2())
    plt.show()

def winsorized_flow_imbalance_2(df):
    df['tfi_lagged_1'] = df.tfi.shift(1)
    df['tfi_lagged_2'] = df.tfi.shift(2)
    df['tfi_lagged_3'] = df.tfi.shift(3)
    df.dropna(inplace=True)

    threshold = abs(df['tfi_lagged_1']).quantile(0.90)
    new_df = df[abs(df['tfi_lagged_1']) > threshold]
    new_df.dropna(inplace=True)
    X = new_df[['tfi_lagged_1', 'tfi_lagged_2', 'tfi_lagged_3']]
    X = sm.add_constant(X)

    y = new_df['midprice_returns']

    ols = sm.OLS(y, X).fit()
    print(ols.summary2())
    plt.show()

def winsorized_midprice_returns_lag_1(df):
    df['midprice_returns_lagged_1'] = df.midprice_returns.shift(1)
    df['midprice_returns_lagged_2'] = df.midprice_returns.shift(2)
    df['midprice_returns_lagged_3'] = df.midprice_returns.shift(3)
    df.dropna(inplace=True)

    threshold = abs(df['midprice_returns_lagged_1']).quantile(0.75)
    new_df = df[abs(df['midprice_returns_lagged_1']) > threshold]
    new_df.dropna(inplace=True)
    X = new_df[['midprice_returns_lagged_1']]
    X = sm.add_constant(X)

    new_df.plot(kind='scatter', grid=True, x='midprice_returns_lagged_1', y='midprice_returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    y = new_df['midprice_returns']

    ols = sm.OLS(y, X).fit()
    print(ols.summary2())
    plt.show()


def winsorized_midprice_returns_lag_3(df):
    df['midprice_returns_lagged_1'] = df.midprice_returns.shift(1)
    df['midprice_returns_lagged_2'] = df.midprice_returns.shift(2)
    df['midprice_returns_lagged_3'] = df.midprice_returns.shift(3)
    df.dropna(inplace=True)

    threshold = abs(df['midprice_returns_lagged_1']).quantile(0.90)
    new_df = df[abs(df['midprice_returns_lagged_1']) > threshold]
    new_df.dropna(inplace=True)
    X = new_df[['midprice_returns_lagged_1', 'midprice_returns_lagged_2', 'midprice_returns_lagged_3']]
    X = sm.add_constant(X)

    y = new_df['midprice_returns']

    ols = sm.OLS(y, X).fit()
    print(ols.summary2())
    plt.show()

def returns_lag_1(df):
    df['returns_lagged_1'] = df.returns.shift(1)
    df['returns_lagged_2'] = df.returns.shift(2)
    df['returns_lagged_3'] = df.returns.shift(3)

    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='returns_lagged_1', y='returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    X = df[['returns_lagged_1']]
    X = sm.add_constant(X)

    ols = sm.OLS(df.returns, X).fit()
    print(ols.summary2())
    plt.show()

def returns_lag_3(df):
    df['returns_lagged_1'] = df.returns.shift(1)
    df['returns_lagged_2'] = df.returns.shift(2)
    df['returns_lagged_3'] = df.returns.shift(3)
    df.dropna(inplace=True)

    X = df[['returns_lagged_1', 'returns_lagged_2', 'returns_lagged_3']]
    X = sm.add_constant(X)

    ols = sm.OLS(df.returns, X).fit()
    print(ols.summary2())
    plt.show()


def returns_lagged_correlation_1(df_1, df_2):
  df_2['returns_lagged_1'] = df_2.returns.shift(1)
  df_2['returns_lagged_2'] = df_2.returns.shift(2)
  df_2['returns_lagged_3'] = df_2.returns.shift(3)
  df_2.dropna(inplace=True)

  y = df_1['returns']
  X = df_2[['returns_lagged_1']]
  X = sm.add_constant(X)
  X, y = X.align(y, join='inner', axis=0)

  df = pd.DataFrame()
  df['returns'] = y
  df['returns_lagged_1'] = X['returns_lagged_1']
  print(df.head())
  df.plot(kind='scatter', grid=True, x='returns_lagged_1', y='returns', title='Returns/Lagged Returns correlation', alpha=0.5, figsize=(12,10))

  ols = sm.OLS(y, X).fit()
  print(ols.summary2())
  plt.show()

def returns_lagged_correlation_3(df_1, df_2):
  df_2['returns_lagged_1'] = df_2.returns.shift(1)
  df_2['returns_lagged_2'] = df_2.returns.shift(2)
  df_2['returns_lagged_3'] = df_2.returns.shift(3)
  df_2.dropna(inplace=True)

  y = df_1['returns']
  X = df_2[['returns_lagged_1', 'returns_lagged_2', 'returns_lagged_3']]
  X = sm.add_constant(X)
  X, y = X.align(y, join='inner', axis=0)

  ols = sm.OLS(y, X).fit()
  print(ols.summary2())
  plt.show()

def model_1(df_pair1, df_pair2, df_pair3, df_pair4, df_pair5):
  """
  This model combines the lagged returns of another pair (usually BTC/USD)
  as well as the lagged correlations of the pair itself
  """
  df = pd.DataFrame()
  df['returns_pair1_lag1'] = df_pair1.returns.shift(1)
  df['returns_pair1_lag2'] = df_pair1.returns.shift(2)
  df['returns_pair1_lag3'] = df_pair1.returns.shift(3)
  df['returns_pair1_lag4'] = df_pair1.returns.shift(4)

  df['returns_pair2_lag1'] = df_pair2.returns.shift(1)
  df['returns_pair2_lag2'] = df_pair2.returns.shift(2)
  df['returns_pair2_lag3'] = df_pair2.returns.shift(3)
  df['returns_pair2_lag4'] = df_pair2.returns.shift(4)

  df['returns_pair3_lag1'] = df_pair3.returns.shift(1)
  df['returns_pair3_lag2'] = df_pair3.returns.shift(2)
  df['returns_pair3_lag3'] = df_pair3.returns.shift(3)
  df['returns_pair3_lag4'] = df_pair3.returns.shift(4)

  df['returns_pair4_lag1'] = df_pair4.returns.shift(1)
  df['returns_pair4_lag2'] = df_pair4.returns.shift(2)
  df['returns_pair4_lag3'] = df_pair4.returns.shift(3)
  df['returns_pair4_lag4'] = df_pair4.returns.shift(4)

  df['returns_pair5_lag1'] = df_pair5.returns.shift(1)
  df['returns_pair5_lag2'] = df_pair5.returns.shift(2)
  df['returns_pair5_lag3'] = df_pair5.returns.shift(3)
  df['returns_pair5_lag4'] = df_pair5.returns.shift(4)
  df.dropna(inplace=True)

  X = sm.add_constant(df)
  y = df_pair1['returns']
  X, y = X.align(y, join='inner', axis=0)

  # df = pd.DataFrame()
  # df['returns'] = y
  # df['returns_lagged_1'] = X['returns_lagged_1']
  # print(df.head())
  # df.plot(kind='scatter', grid=True, x='returns_lagged_1', y='returns', title='Returns/Lagged Returns correlation', alpha=0.5, figsize=(12,10))

  ols = sm.OLS(y, X).fit()
  print(ols.summary2())
  plt.show()


regressions = {
  'ofi_model': ofi_model,
  'tfi_model': tfi_model,
  'vfi_model': vfi_model,
  'ofi_comtemporeanous_returns': ofi_comtemporeanous_returns,
  'tfi_comtemporeanous_returns': tfi_comtemporeanous_returns,
  'vfi_comtemporeanous_returns': vfi_comtemporeanous_returns,
  'flow_imbalance_1': flow_imbalance_1,
  'flow_imbalance_2': flow_imbalance_2,
  'midprice_returns_lag_1': midprice_returns_lag_1,
  'midprice_returns_lag_3': midprice_returns_lag_3,
  'winsorized_flow_imbalance_1': winsorized_flow_imbalance_1,
  'winsorized_flow_imbalance_2': winsorized_flow_imbalance_2,
  'midprice_returns_lag_3': midprice_returns_lag_3,
  'midprice_returns_lag_1': midprice_returns_lag_1,
  'returns_lag_1': returns_lag_1,
  'returns_lag_3': returns_lag_3,
  'returns_lagged_correlation_1': returns_lagged_correlation_1,
  'returns_lagged_correlation_3': returns_lagged_correlation_3,
  'model_1': model_1
}