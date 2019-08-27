
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

from features.default import add_default_lags, add_lagged_returns
from features.balance import balance
from features.targets import classify_target
from features.labeling import add_barriers_on_buy_sell_signals
from features.labeling import add_labels
from features.signals import compute_signals

from metrics.models import compute_kfold_scores
from metrics.models import print_classification_report
from metrics.models import print_crosstab
from plot.models import plot_roc_curve

def primary_model_1(target_data, feature_data, options={}):
  """
  Primary model: Linear Regression
  Secondary model: Random Forest Classifier

  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)
  """
  X = pd.DataFrame()
  y = target_data.returns
  close = target_data.close

  X, y = add_default_lags(X, y, feature_data)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_pred = regression.predict(X)
  y_pred = pd.Series(y_pred, index=y.index)
  signals = compute_signals(y, y_pred)
  stop_thresholds = pd.Series(close.std(), index=close.index)

  events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  events = add_labels(events, close)

  X2 = X
  y2 = events['label']
  X2, y2 = X2.align(y2, join='inner', axis=0)
  X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  model2 = RandomForestClassifier(max_depth=2, n_estimators=500, criterion='entropy')
  model2.fit(X2_train, y2_train)

  y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  y2_pred = model2.predict(X2_test)
  fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)
  plot_roc_curve(fpr, tpr)


def primary_model_2(target_data, feature_data, options={}):
  """
  Primary model: Random Forest Regressor
  Secondary model: Random Forest Classifier
  """
  X = pd.DataFrame()
  y = target_data.returns
  close = target_data.close

  X, y = add_default_lags(X, y, feature_data)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  model = RandomForestRegressor(n_estimators=500, random_state=0, min_weight_fraction_leaf=0.05, max_features=1)
  model.fit(X_train, y_train)

  y_pred = model.predict(X)
  y_pred = pd.Series(y_pred, index=y.index)
  signals = compute_signals(y, y_pred)
  stop_thresholds = pd.Series(close.std(), index=close.index)

  events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  events = add_labels(events, close)

  X2 = X
  y2 = events['label']
  X2, y2 = X2.align(y2, join='inner', axis=0)
  X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  model2 = RandomForestClassifier(max_depth=2, n_estimators=500, criterion='entropy')
  model2.fit(X2_train, y2_train)

  y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  y2_pred = model2.predict(X2_test)
  fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)
  plot_roc_curve(fpr, tpr)


def primary_model_3(target_data, feature_data, options={}):
  """
  Primary model: Logistic Regression
  Secondary model: Random Forest Classifier
  """
  X = pd.DataFrame()
  y = target_data.returns
  X, y = add_lagged_returns(X, y, feature_data)
  X, y = classify_target(X, y, options={"nb_std": 1.5})
  close = target_data.close

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  X_bal, y_bal = balance(X_train, X_test, y_train, y_test)

  regression = LogisticRegression()
  regression.fit(X_bal, y_bal)

  y_train_pred = regression.predict(X_train)
  y_test_pred = regression.predict(X_test)

  print_classification_report(y_test, y_test_pred)
  print_crosstab(y, y_test_pred)

  y_pred = regression.predict(X)
  signals = compute_signals(y, y_pred, classified=True)
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

  plot_roc_curve(fpr, tpr)


def primary_model_4(target_data, feature_data, options={}):
  """
  Primary model: Linear Regression
  Secondary model: Random Forest Classifier

  Features: Fractionally Differentiated returns
  """
  X = pd.DataFrame()
  y = target_data.returns
  close = target_data.close

  X, y = add_default_lags(X, y, feature_data)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regression = LinearRegression()
  regression.fit(X_train, y_train)

  y_pred = regression.predict(X)
  y_pred = pd.Series(y_pred, index=y.index)
  signals = compute_signals(y, y_pred)
  stop_thresholds = pd.Series(close.std(), index=close.index)

  events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  events = add_labels(events, close)

  X2 = X
  y2 = events['label']
  X2, y2 = X2.align(y2, join='inner', axis=0)
  X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  model2 = RandomForestClassifier(max_depth=2, n_estimators=500, criterion='entropy')
  model2.fit(X2_train, y2_train)

  y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  y2_pred = model2.predict(X2_test)
  fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)
  plot_roc_curve(fpr, tpr)



# def primary_model_4(target_data, feature_data, options={}):
#   """
#   Primary Model: Linear Regression
#   """
#   X = pd.DataFrame()
#   y = target_data.returns
#   X, y = add_lagged_returns(X, y, feature_data)
#   close = target_data.close

#   num_folds = 10
#   seed = 7
#   kfold = KFold(n_splits=num_folds, random_state=seed)
#   model = LinearRegression()
#   scores = []

#   for train_index, test_index in kfold.split(X):
#     X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     model.fit(X_train, y_train)
#     scores.append(model.score(X_test, y_test))

#   print('Scores:', scores)
#   print('Mean score: ', np.mean(scores))
#   compute_kfold(model, X, y, kfold)

# def primary_model_5(target_data, feature_data, options={}):
#   """
#   Primary Model: Random Forest Regressor
#   """
#   X = pd.DataFrame()
#   y = target_data.returns
#   X, y = add_lagged_returns(X, y, feature_data)
#   num_folds = 10
#   seed = 7
#   kfold = KFold(n_splits=num_folds, random_state=seed)

#   model = RandomForestRegressor(n_estimators=500, random_state=0, min_weight_fraction_leaf=0.05, max_features=3)
#   scores = []

#   for train_index, test_index in kfold.split(X):
#     X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     model.fit(X_train, y_train)
#     scores.append(model.score(X_test, y_test))

#   print('Scores:', scores)
#   print('Mean score: ', np.mean(scores))
#   compute_kfold_scores(model, X, y, kfold)


primary_models = {
  'primary_model_1': primary_model_1,
  'primary_model_2': primary_model_2,
  'primary_model_3': primary_model_3,
  'primary_model_4': primary_model_4
}

















