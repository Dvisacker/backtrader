
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_graphviz
from sklearn import metrics

from plot.models import residuals_plot, prediction_error_plot
from metrics.models import confusion_matrix, compute_scores, compute_kfold_scores
from features.default import compute_default_features, compute_differentiated_features

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
  y_test_pred = regression.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
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
  y_test_pred = regression.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
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
  y_test_pred = regression.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
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
  y_test_pred = regression.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)


def regression_model_7(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)

  X, y = compute_differentiated_features(main_pair, raw_features, options)

  num_folds = 5
  seed = 7
  kfold = KFold(n_splits=num_folds, random_state=seed)
  scores = []

  for train_index, test_index in kfold.split(X):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))


  print('Scores:', scores)
  print('Mean score: ', np.mean(scores))
  compute_kfold_scores(model, X, y, kfold)

def regression_model_8(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)

  X, y = compute_default_features(main_pair, raw_features, options)

  num_folds = 5
  seed = 7
  kfold = KFold(n_splits=num_folds, random_state=seed)

  model = LinearRegression()
  scores = []

  for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))


  print('Scores:', scores)
  print('Mean score: ', np.mean(scores))
  compute_kfold_scores(model, X, y, kfold)


def regression_model_9(main_pair, raw_features, options={}):
  lags = options.get("lags", 4)
  X, y = compute_differentiated_features(main_pair, raw_features, options)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_train_pred = regression.predict(X_train)
  y_test_pred = regression.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)


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
  y_test_pred = regression.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
  residuals_plot(regression, X_train, y_train, X_test, y_test)
  prediction_error_plot(regression, X_train, y_train, X_test, y_test)



  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  # sc = StandardScaler()
  # X_train = sc.fit_transform(X_train)
  # X_test = sc.transform(X_test)

  # regression = LinearRegression()
  # regression.fit(X_train, y_train)

  # y_train_pred = regression.predict(X_train)
  # y_test_pred = regression.predict(X_test)

  # compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  # confusion_matrix(y_test_pred, y_test)
  # residuals_plot(regression, X_train, y_train, X_test, y_test)
  # prediction_error_plot(regression, X_train, y_train, X_test, y_test)

regressions_models = {
  'regression_model_1': regression_model_1,
  'regression_model_2': regression_model_2,
  'regression_model_3': regression_model_3,
  'regression_model_4': regression_model_4,
  'regression_model_5': regression_model_5,
  'regression_model_6': regression_model_6,
  'vol_weighted_returns_linear_regression_1': vol_weighted_returns_linear_regression_1,
  'vol_weighted_returns_linear_regression_2': vol_weighted_returns_linear_regression_2,
  'vol_weighted_returns_linear_regression_3': vol_weighted_returns_linear_regression_3,
  'regression_model_7': regression_model_7,
  'regression_model_8': regression_model_8,
  'regression_model_9': regression_model_9
}