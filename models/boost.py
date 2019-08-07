import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import metrics

from yellowbrick.regressor import ResidualsPlot, PredictionError

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
  y_pred = regressor.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
  print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_pred)))

  create_confusion_matrix(y_pred, y_test)
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
  y_pred = regressor.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
  print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
  print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_pred)))

  create_confusion_matrix(y_pred, y_test)
  residuals_plot(regressor, X_train, y_train, X_test, y_test)
  prediction_error_plot(regressor, X_train, y_train, X_test, y_test)



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


def export(X, y, regressor):
  estimator = regressor.estimators_[5]
  export_graphviz(estimator, out_file='tree.dot',
                feature_names = X.columns,
                class_names = 'returns',
                rounded = True, proportion = False,
                precision = 2, filled = True)

  # Convert to png using system command (requires Graphviz)
  from subprocess import call
  call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


boosting_models = {
  'gradient_boosting_model_1': gradient_boosting_model_1,
  'adaboost_model_1': adaboost_model_1
}


