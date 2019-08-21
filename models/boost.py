import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import metrics

from yellowbrick.regressor import ResidualsPlot, PredictionError

from metrics.models import compute_scores, confusion_matrix
from plot.models import residuals_plot, prediction_error_plot

def gradient_boosting_model_1(main_pair, raw_features, options={}):
  boosting_params = { "max_depth": 2, "n_estimators": 5 }
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

  regressor = GradientBoostingRegressor(**boosting_params)
  regressor.fit(X_train, y_train)

  y_train_pred = regressor.predict(X_train)
  y_test_pred = regressor.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
  residuals_plot(regressor, X_train, y_train, X_test, y_test)
  prediction_error_plot(regressor, X_train, y_train, X_test, y_test)


def adaboost_model_1(main_pair, raw_features, options={}):
  boosting_params = { "max_depth": 2, "n_estimators": 5 }
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

  regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=boosting_params['max_depth']),
                                                      n_estimators=boosting_params['n_estimators'])
  regressor.fit(X_train, y_train)
  y_train_pred = regressor.predict(X_train)
  y_test_pred = regressor.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  confusion_matrix(y_test_pred, y_test)
  residuals_plot(regressor, X_train, y_train, X_test, y_test)
  prediction_error_plot(regressor, X_train, y_train, X_test, y_test)

boosting_models = {
  'gradient_boosting_model_1': gradient_boosting_model_1,
  'adaboost_model_1': adaboost_model_1
}


