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

def tfi_model(tf, midprice_returns, lag=1):
    df = pd.concat([midprice_returns,tf], axis = 1)
    df['next_midprice_returns'] = df.midprice_returns.shift(lag)
    df.dropna(inplace=True)
    df.plot(kind='scatter', grid=True, x='tfi', y='next_midprice_returns',
            title = 'TFI/Returns correlation', alpha=0.5, figsize=(12,10))

    tfi = sm.add_constant(df['tfi'])
    ols = sm.OLS(df.next_midprice_returns, tfi).fit()
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


def lagged_returns_1(df):
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

def lagged_returns_3(df):
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

def winsorized_lagged_returns_1(df):
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


def winsorized_lagged_returns_3(df):
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



models = {
  'ofi_model': ofi_model,
  'tfi_model': tfi_model,
  'ofi_comtemporeanous_returns': ofi_comtemporeanous_returns,
  'tri_comtemporeanous_returns': tfi_comtemporeanous_returns,
  'flow_imbalance_1': flow_imbalance_1,
  'flow_imbalance_2': flow_imbalance_2,
  'lagged_returns_1': lagged_returns_1,
  'lagged_returns_3': lagged_returns_3,
  'winsorized_flow_imbalance_1': winsorized_flow_imbalance_1,
  'winsorized_flow_imbalance_2': winsorized_flow_imbalance_2,
  'winsorized_lagged_returns_3': winsorized_lagged_returns_3,
  'winsorized_lagged_returns_1': winsorized_lagged_returns_1
}