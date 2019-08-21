import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import metrics

from yellowbrick.regressor import ResidualsPlot, PredictionError
from plot.models import residuals_plot, prediction_error_plot
from metrics.models import compute_scores

def random_forest_model_1(main_pair, raw_features, options={}):
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

  regressor = RandomForestRegressor(n_estimators=1000, random_state=0, min_weight_fraction_leaf=0.05, max_features=3)
  regressor.fit(X_train, y_train)

  y_train_pred = regressor.predict(X_train)
  y_test_pred = regressor.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  create_confusion_matrix(y_test_pred, y_test)
  residuals_plot(regressor, X_train, y_train, X_test, y_test)
  prediction_error_plot(regressor, X_train, y_train, X_test, y_test)

def random_forest_model_2(main_pair, raw_features, options={}):
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

  regressor = RandomForestRegressor(n_estimators=1000, random_state=0, min_weight_fraction_leaf=0.05, max_features=3)
  regressor.fit(X_train, y_train)

  y_train_pred = regressor.predict(X_train)
  y_test_pred = regressor.predict(X_test)

  compute_scores(y_test, y_test_pred, y_train, y_train_pred)
  create_confusion_matrix(y_test_pred, y_test)
  residuals_plot(regressor, X_train, y_train, X_test, y_test)
  prediction_error_plot(regressor, X_train, y_train, X_test, y_test)


def random_forest_model_3(main_pair, raw_features, options={}):
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
    X['vol_weighted_return_lag_{}'.format(i)] = main_pair.returns.shift(i) * main_pair.volume.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
        X['{}_volume_lag_{}'.format(pair, i)] = bars.volume.shift(i)
        X['{}_vol_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)


  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)

  params_grid = {
    'n_estimators': [100, 200, 300, 500],
    'min_weight_fraction_leaf': [0.01, 0.05, 0.1, 0.2],
    'max_features': [3, 5, 10, 20, 30, 40]
  }

  regressor = RandomForestRegressor()
  search = GridSearchCV(regressor, params_grid, iid=False, cv=5)
  search.fit(X, y)
  print('Best parameter (CV score={})'.format(search.best_score_))
  print(search.best_params_)
  print(search.best_estimator_)

def create_confusion_matrix(y_pred, y_true):
  df = pd.DataFrame({'pred': y_pred, 'true': y_true })
  df['pred_sign'] = np.sign(df['pred'])
  df['true_sign'] = np.sign(df['true'])
  df['correct'] = df['pred_sign'] == df['true_sign']

  df2 = df[df['true_sign'] != 0]
  print(df2['correct'].value_counts())



random_forest_models = {
  'random_forest_model_1': random_forest_model_1,
  'random_forest_model_2': random_forest_model_2,
  'random_forest_model_3': random_forest_model_3
}


