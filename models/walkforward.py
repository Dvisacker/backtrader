
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

from features.default import add_default_lags, add_lagged_returns, add_returns
from features.balance import balance
from features.targets import classify_target, one_step_forward_returns
from features.labeling import add_barriers_on_buy_sell_signals
from features.labeling import add_labels
from features.signals import compute_signals

from metrics.models import compute_kfold_scores
from metrics.models import print_classification_report
from metrics.models import print_crosstab
from plot.models import plot_roc_curve

from sklearn.metrics import r2_score


def walkforward_regression_1(target_data, feature_data, options={}):
  """
  Primary model: Linear Regression
  Secondary model: Random Forest Classifier

  Features: Fractionally Differentiated returns
  """
  X = pd.DataFrame()
  y = one_step_forward_returns(target_data)
  close = target_data.close

  X, y = add_returns(X, y, feature_data)

  retrain_dates = X.resample('180T').mean().index.values[:-1]
  models = pd.Series(index=retrain_dates)

  for date in retrain_dates:
    X_train = X.loc[:date]
    y_train = y.loc[:date]
    model = LinearRegression()
    model.fit(X_train, y_train)
    models.loc[date] = model

  coefs = pd.DataFrame()

  for i, model in enumerate(models):
    model_coefs = pd.Series(model.coef_, index=X.columns)
    model_coefs.name = models.index[i]
    coefs = pd.concat([coefs, model_coefs], axis=1)

  coefs.T.plot(title='Coefficient for expanding window model')

  begin_dates = models.index
  end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))

  y_pred = pd.Series(index=X.index)

  for i,model in enumerate(models): #loop thru each models object in collection
      X_train = X[begin_dates[i]:end_dates[i]]
      if X_train.empty is not True:
        p = pd.Series(model.predict(X_train),index=X_train.index)
        y_pred.loc[X_train.index] = p


  rsq_expanding = r2_score(y_true = y,y_pred=y_pred)
  print("Expanding Window RSQ: {}".format(round(rsq_expanding,3)))


  plt.show()


def walkforward_regression_2(target_data, feature_data, options={}):
  """
  Primary model: Linear Regression
  Secondary model: Random Forest Classifier

  Features: Fractionally Differentiated returns
  """
  X = pd.DataFrame()
  y = one_step_forward_returns(target_data)
  close = target_data.close

  X, y = add_returns(X, y, feature_data)

  retrain_dates = X.resample('180T').mean().index.values[:-1]
  models = pd.Series(index=retrain_dates)

  for date in retrain_dates:
    X_train = X.loc[date-pd.Timedelta('1 day'):date]
    y_train = y.loc[date-pd.Timedelta('1 day'):date]
    model = LinearRegression()
    model.fit(X_train, y_train)
    models.loc[date] = model

  coefs = pd.DataFrame()

  for i, model in enumerate(models):
    model_coefs = pd.Series(model.coef_, index=X.columns)
    model_coefs.name = models.index[i]
    coefs = pd.concat([coefs, model_coefs], axis=1)

  coefs.T.plot(title='Coefficient for rolling window model')

  begin_dates = models.index
  end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))
  y_pred = pd.Series(index=X.index)

  for i,model in enumerate(models):
      X_train = X[begin_dates[i]:end_dates[i]]
      if X_train.empty is not True:
        p = pd.Series(model.predict(X_train),index=X_train.index)
        y_pred.loc[X_train.index] = p

  rsq_rolling = r2_score(y_true = y,y_pred=y_pred)
  print("Rolling Window RSQ: {}".format(round(rsq_rolling,3)))

  plt.show()




walkforward_models = {
  'walkforward_regression_1': walkforward_regression_1,
  'walkforward_regression_2': walkforward_regression_2
}