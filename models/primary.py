
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_graphviz
from sklearn import metrics

from yellowbrick.regressor import ResidualsPlot, PredictionError
from indicators import bollinger_bands
from conditions import get_crossovers, get_crossunders

from .utils import add_barriers_on_buy_sell_signals, add_labels

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

  y_pred = regression.predict(X)
  y_pred = pd.Series(y_pred, index=y.index)
  up1 = y[y_pred > y_pred.mean() + 1 * y_pred.std()]
  down1 = y[y_pred < y_pred.mean() - 1 * y_pred.std()]
  signals_up = pd.Series(1, index=up1.index)
  signals_down = pd.Series(-1, index=down1.index)
  signals = pd.concat([signals_up, signals_down]).sort_index()
  # stop_thresholds = close.ewm(30).std()
  stop_thresholds = pd.Series(0.0001, index=close.index)

  events = add_barriers_on_buy_sell_signals(close, signals, stop_thresholds)
  events = add_labels(events, close)

  print(events)

  X2 = X
  y2 = events['label']
  X2, y2 = X2.align(y2, join='inner', axis=0)
  X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

  model2 = RandomForestClassifier(max_depth=2, n_estimators=10000, criterion='entropy')
  model2.fit(X2_train, y2_train)

  y2_pred_probabilities = model2.predict_proba(X2_test)[:, 1]
  y2_pred = model2.predict(X2_test)
  fpr, tpr, _ = metrics.roc_curve(y2_test, y2_pred_probabilities)

  print(metrics.classification_report(y2_test, y2_pred, target_names=['no_trade', 'trade']))
  print(pd.crosstab(y2_test, y2_pred, rownames=['Actual labels'], colnames=['Predicted labels']))

  pdb.set_trace()

  # plt.figure(1)
  # plt.plot([0,1], [0,1], 'k--')
  # plt.plot(fpr, tpr, label='RF')
  # plt.xlabel('False positive rate')
  # plt.ylabel('True positive rate')
  # plt.title('ROC curve')
  # plt.legend(loc='best')
  # plt.show()


primary_models = {
  'primary_model_1': primary_model_1
}

















