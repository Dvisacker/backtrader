
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_graphviz
from sklearn import metrics

from yellowbrick.regressor import ResidualsPlot, PredictionError


def regression_model_1(main_pair, raw_features, options={}):
  """
  This model correlates returns with lagged returns (checks
  autocorrelations in returns)
  """
  lags = options.get("lags", 4)

  X = pd.DataFrame()

  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)

  X.dropna(inplace=True)
  X = sm.add_constant(X)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)

  ols = sm.OLS(y, X).fit()
  print(ols.summary2())
  plt.show()


def regression_model_2(main_pair, raw_features, options={}):
  """
  This model correlates returns with the lagged returns of other pairs
  """
  lags = options.get("lags", 4)

  X = pd.DataFrame()

  for pair, bars in raw_features.items():
    for i in range(1, lags + 1):
      X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)

  X.dropna(inplace=True)
  X = sm.add_constant(X)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)

  ols = sm.OLS(y, X).fit()
  print(ols.summary2())
  plt.show()

def regression_model_3(main_pair, raw_features, options={}):
  """
  This model combines the lagged returns of another pair
  as well as the lagged returns of the pair itself
  """
  lags = options.get("lags", 4)

  X = pd.DataFrame()

  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)


  X.dropna(inplace=True)
  X = sm.add_constant(X)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)

  ols = sm.OLS(y, X).fit()
  print(ols.summary2())
  plt.show()


def regression_model_4(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)

  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)


  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_train_pred = regression.predict(X_train)
  y_pred = regression.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
  print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_pred)))

  create_confusion_matrix(y_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)


def regression_model_5(main_pair, raw_features, options={}):
  '''
  This model includes
  * Lagged returns
  * Lagged volumes
  * Additional lagged returns of other pairs
  * Additional lagged volumes of other volumes
  '''
  lags = options.get("lags", 4)

  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)
    X['volume_lag_{}'.format(i)] = main_pair.volume.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
        X['{}_volume_lag_{}'.format(pair, i)] = bars.volume.shift(i)

  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_train_pred = regression.predict(X_train)
  y_pred = regression.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
  print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_pred)))

  create_confusion_matrix(y_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)


def regression_model_6(main_pair, raw_features, options={}):
    """
    StatsModel based multi linear regression
    This model uses volume weighted returns
    """
    lags = options.get("lags", 4)

    X = pd.DataFrame()

    for i in range(1, lags + 1):
      X['vol_weighted_return_{}'.format(i)] = main_pair.returns.shift(i) * main_pair.volume.shift(i)

    if raw_features:
      for pair, bars in raw_features.items():
        for i in range(1, lags + 1):
          X['{}_vol_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)

    X.dropna(inplace=True)
    X = sm.add_constant(X)
    y = main_pair['returns']
    X, y = X.align(y, join='inner', axis=0)

    ols = sm.OLS(y, X).fit()
    print(ols.summary2())
    plt.show()

def vol_weighted_returns_linear_regression_1(main_pair, raw_features, options={}):
    """
    StatsModel based multi linear regression
    This model only includes lagged volumes
    """
    lags = options.get("lags", 3)

    X = pd.DataFrame()

    for i in range(1, lags + 1):
      X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)
      X['vol_weighted_return_{}'.format(i)] = main_pair.returns.shift(i) * main_pair.volume.shift(i)

    if raw_features:
      for pair, bars in raw_features.items():
        for i in range(1, lags + 1):
          X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
          X['{}_vol_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)

    X.dropna(inplace=True)
    X = sm.add_constant(X)
    y = main_pair['returns']
    X, y = X.align(y, join='inner', axis=0)

    ols = sm.OLS(y, X).fit()
    print(ols.summary2())
    plt.show()

def vol_weighted_returns_linear_regression_2(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)

  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)
    X['vol_weighted_return_{}'.format(i)] = main_pair.returns.shift(i) * main_pair.volume.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
        X['{}_vol_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)


  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_train_pred = regression.predict(X_train)
  y_pred = regression.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
  print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_pred)))

  create_confusion_matrix(y_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)


def vol_weighted_returns_linear_regression_3(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)

  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['vol_weighted_return_{}'.format(i)] = main_pair.returns.shift(i) * main_pair.volume.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_vol_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)


  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_train_pred = regression.predict(X_train)
  y_pred = regression.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
  print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_pred)))

  create_confusion_matrix(y_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)



def residuals_plot(model, X_train, y_train, X_test, y_test):
  visualizer = ResidualsPlot(model)
  visualizer.fit(X_train, y_train)
  visualizer.score(X_test, y_test)
  visualizer.poof()

def prediction_error_plot(model, X_train, y_train, X_test, y_test):
  visualizer = PredictionError(model)
  visualizer.fit(X_train, y_train)
  visualizer.score(X_test, y_test)
  visualizer.poof()

def create_confusion_matrix(y_pred, y_true):
  df = pd.DataFrame({'pred': y_pred, 'true': y_true })
  df['pred_sign'] = np.sign(df['pred'])
  df['true_sign'] = np.sign(df['true'])
  df['correct'] = df['pred_sign'] == df['true_sign']

  df2 = df[df['true_sign'] != 0]
  print(df2['correct'].value_counts())

regressions_models = {
  'regression_model_1': regression_model_1,
  'regression_model_2': regression_model_2,
  'regression_model_3': regression_model_3,
  'regression_model_4': regression_model_4,
  'regression_model_5': regression_model_5,
  'regression_model_6': regression_model_6,
  'vol_weighted_returns_linear_regression_1': vol_weighted_returns_linear_regression_1,
  'vol_weighted_returns_linear_regression_2': vol_weighted_returns_linear_regression_2,
  'vol_weighted_returns_linear_regression_3': vol_weighted_returns_linear_regression_3
}