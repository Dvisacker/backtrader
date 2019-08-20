
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.utils import class_weight
from sklearn import metrics

from yellowbrick.regressor import ResidualsPlot, PredictionError
from indicators import bollinger_bands
from conditions import get_crossovers, get_crossunders

from .utils import add_barriers_on_buy_sell_signals, add_labels

def primary_model_1(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close

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

  # print(metrics.classification_report(y_test, y_test_pred, target_names=['no_trade', 'trade']))
  # print(pd.crosstab(y_test, y_test_pred, rownames=['Actual labels'], colnames=['Predicted labels']))

  y_pred = regression.predict(X)
  y_pred = pd.Series(y_pred, index=y.index)
  up1 = y[y_pred > y_pred.mean() + 1 * y_pred.std()]
  down1 = y[y_pred < y_pred.mean() - 1 * y_pred.std()]
  signals_up = pd.Series(1, index=up1.index)
  signals_down = pd.Series(-1, index=down1.index)
  signals = pd.concat([signals_up, signals_down]).sort_index()
  # stop_thresholds = close.ewm(30).std()
  stop_thresholds = pd.Series(2*close.std(), index=close.index)

  events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  events = add_labels(events, close)

  X2 = X
  y2 = events['label']
  X2, y2 = X2.align(y2, join='inner', axis=0)
  X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  model2 = RandomForestClassifier(max_depth=2, n_estimators=10000, criterion='entropy')
  model2.fit(X2_train, y2_train)

  y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  y2_pred = model2.predict(X2_test)
  fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)

  plt.figure(1)
  plt.plot([0,1], [0,1], 'k--')
  plt.plot(fpr, tpr, label='RF')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()

def primary_model_2(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close
  returns = main_pair.returns

  lags = options.get("lags", 4)
  X = pd.DataFrame()
  for i in range(1, lags + 1):
    X['returns_lag_{}'.format(i)] = main_pair.returns.shift(i)

  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)


  X.dropna(inplace=True)
  # We classify the return vector
  y = pd.Series(0, index=X.index)
  print(returns.std())
  y[returns > 1.5 * returns.std()] = 1
  y[returns < - 1.5 * returns.std()] = -1
  X, y = X.align(y, join='inner', axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sample_n = y_train.value_counts().idxmin()
  X_train_1 = X_train[y_train == 1].sample(sample_n)
  y_train_1 = y_train[y_train == 1].sample(sample_n)
  X_train_0 = X_train[y_train == 0].sample(sample_n)
  y_train_0 = y_train[y_train == 0].sample(sample_n)
  X_train_n1 = X_train[y_train == -1].sample(sample_n)
  y_train_n1 = y_train[y_train == -1].sample(sample_n)
  X_bal = pd.concat([X_train_1, X_train_0, X_train_n1])
  y_bal = pd.concat([y_train_1, y_train_0, y_train_n1])

  regression = LogisticRegression()
  regression.fit(X_bal, y_bal)

  y_train_pred = regression.predict(X_train)
  y_test_pred = regression.predict(X_test)

  print(metrics.classification_report(y_test, y_test_pred, target_names=['short', 'no_trade', 'long']))
  print(pd.crosstab(y_test, y_test_pred, rownames=['Actual labels'], colnames=['Predicted labels']))

  y_pred = regression.predict(X)
  y_pred = pd.Series(y_pred, index=y.index)

  signals_up = y_pred[y_pred == 1]
  signals_down = y_pred[y_pred == -1]
  signals = pd.concat([signals_up, signals_down]).sort_index()
  # stop_thresholds = close.ewm(30).std()
  stop_thresholds = pd.Series(2*close.std(), index=close.index)

  events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  events = add_labels(events, close)

  X2 = X
  y2 = events['label']
  X2, y2 = X2.align(y2, join='inner', axis=0)
  X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  model2 = RandomForestClassifier(max_depth=2, n_estimators=10000, criterion='entropy')
  model2.fit(X2_train, y2_train)

  y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  y2_pred = model2.predict(X2_test)
  fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)

  # print(metrics.classification_report(y_test, y_test_pred, target_names=['short', 'no_trade', 'long']))
  # print(pd.crosstab(y_test, y_test_pred, rownames=['Actual labels'], colnames=['Predicted labels']))

  plt.figure(1)
  plt.plot([0,1], [0,1], 'k--')
  plt.plot(fpr, tpr, label='RF')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()


def primary_model_3(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close

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
  num_folds = 10
  seed = 7
  kfold = KFold(n_splits=num_folds, random_state=seed)

  regression = LinearRegression()
  scores = []

  for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    regression.fit(X_train, y_train)
    scores.append(regression.score(X_test, y_test))


  print('Scores:', scores)
  print('Mean score: ', np.mean(scores))

  scoring = 'neg_mean_absolute_error'
  results = cross_val_score(regression, X, y, cv=kfold, scoring=scoring)
  print("MAE: {} ({})".format(results.mean(), results.std()))

  scoring = 'neg_mean_squared_error'
  results = cross_val_score(regression, X, y, cv=kfold, scoring=scoring)
  print("MSE: {} ({})".format(results.mean(), results.std()))

  scoring = 'r2'
  results = cross_val_score(regression, X, y, cv=kfold, scoring=scoring)
  print("R^2: {} ({})".format(results.mean(), results.std()))


def primary_model_4(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close

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
  num_folds = 10
  seed = 7
  kfold = KFold(n_splits=num_folds, random_state=seed)

  regressor = RandomForestRegressor(n_estimators=500, random_state=0, min_weight_fraction_leaf=0.05, max_features=3)
  scores = []

  for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor.fit(X_train, y_train)
    scores.append(regressor.score(X_test, y_test))


  print('Scores:', scores)
  print('Mean score: ', np.mean(scores))

  scoring = 'neg_mean_absolute_error'
  results = cross_val_score(regressor, X, y, cv=kfold, scoring=scoring)
  print("MAE: {} ({})".format(results.mean(), results.std()))

  scoring = 'neg_mean_squared_error'
  results = cross_val_score(regressor, X, y, cv=kfold, scoring=scoring)
  print("MSE: {} ({})".format(results.mean(), results.std()))

  scoring = 'r2'
  results = cross_val_score(regressor, X, y, cv=kfold, scoring=scoring)
  print("R^2: {} ({})".format(results.mean(), results.std()))





  # results = cross_val_score(regression, X, y, cv=kfold)

  # print(results)

  # regression.fit(X_train, y_train)

  # y_train_pred = regression.predict(X_train)
  # y_test_pred = regression.predict(X_test)

  # print(metrics.classification_report(y_test, y_test_pred, target_names=['no_trade', 'trade']))
  # print(pd.crosstab(y_test, y_test_pred, rownames=['Actual labels'], colnames=['Predicted labels']))

  # y_pred = regression.predict(X)
  # y_pred = pd.Series(y_pred, index=y.index)
  # up1 = y[y_pred > y_pred.mean() + 1 * y_pred.std()]
  # down1 = y[y_pred < y_pred.mean() - 1 * y_pred.std()]
  # signals_up = pd.Series(1, index=up1.index)
  # signals_down = pd.Series(-1, index=down1.index)
  # signals = pd.concat([signals_up, signals_down]).sort_index()
  # # stop_thresholds = close.ewm(30).std()
  # stop_thresholds = pd.Series(2*close.std(), index=close.index)

  # events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  # events = add_labels(events, close)

  # X2 = X
  # y2 = events['label']
  # X2, y2 = X2.align(y2, join='inner', axis=0)
  # X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  # model2 = RandomForestClassifier(max_depth=2, n_estimators=10000, criterion='entropy')
  # model2.fit(X2_train, y2_train)

  # y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  # y2_pred = model2.predict(X2_test)
  # fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)


  # plt.figure(1)
  # plt.plot([0,1], [0,1], 'k--')
  # plt.plot(fpr, tpr, label='RF')
  # plt.xlabel('False positive rate')
  # plt.ylabel('True positive rate')
  # plt.title('ROC curve')
  # plt.legend(loc='best')
  # plt.show()


primary_models = {
  'primary_model_1': primary_model_1,
  'primary_model_2': primary_model_2,
  'primary_model_3': primary_model_3,
  'primary_model_4': primary_model_4
}

















