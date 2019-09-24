
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import pdb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.tree import export_graphviz
from sklearn.utils import class_weight
from sklearn.base import clone
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
from metrics.models import compute_custom_scores
from metrics.models import compute_custom_scores_over_time
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

def walkforward_rf_1(target_data, feature_data, options={}):
  X = pd.DataFrame()
  y = one_step_forward_returns(target_data)
  close = target_data.close

  X, y = add_returns(X, y, feature_data)

  retrain_dates = X.resample('180T').mean().index.values[:-1]
  models = pd.Series(index=retrain_dates)

  for date in retrain_dates:
    X_train = X.loc[date-pd.Timedelta('1 day'):date]
    y_train = y.loc[date-pd.Timedelta('1 day'):date]
    model = RandomForestRegressor()
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

  for i, model in enumerate(models):
    X_train = X[begin_dates[i]:end_dates[i]]
    if X_train.empty is not True:
      p = pd.Series(model.predict(X_train), index=X_train.index)
      y_pred.loc[X_train.index] = p

  rsq_rolling = r2_score(y_true=y, y_pred=y_pred)
  print("Rolling window RSQ: {}".format(rsq_rolling))

  plt.show()


def make_walkforward_model(X, y, algo=LinearRegression(), options={}):
  retrain_dates = X.resample('180T').mean().index.values[:-1]
  models = pd.Series(index=retrain_dates)

  for date in retrain_dates:
    X_train = X.loc[date-pd.Timedelta('1 day'):date]
    y_train = y.loc[date-pd.Timedelta('1 day'):date]
    model = clone(algo)
    model.fit(X_train, y_train)
    models.loc[date] = model

  begin_dates = models.index
  end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))
  y_pred = pd.Series(index=X.index)
  for i, model in enumerate(models):
    X_train = X[begin_dates[i]:end_dates[i]]
    if X_train.empty is not True:
      p = pd.Series(model.predict(X_train), index=X_train.index)
      y_pred.loc[X_train.index] = p

  return models, y_pred

def make_walkforward_model_with_coefs(X, y, algo=LinearRegression(), options={}):
  retrain_dates = X.resample('180T').mean().index.values[:-1]
  models = pd.Series(index=retrain_dates)

  for date in retrain_dates:
    X_train = X.loc[date-pd.Timedelta('1 day'):date]
    y_train = y.loc[date-pd.Timedelta('1 day'):date]
    model = clone(algo)
    model.fit(X_train, y_train)
    models.loc[date] = model

  coefs = pd.DataFrame()

  for i, model in enumerate(models):
    model_coefs = pd.Series(model.coef_, index=X.columns)
    model_coefs.name = models.index[i]
    coefs = pd.concat([coefs, model_coefs], axis=1)

  # coefs.T.plot(title='Coefficient for rolling window model')

  begin_dates = models.index
  end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))


  y_pred = pd.Series(index=X.index)
  for i, model in enumerate(models):
    X_train = X[begin_dates[i]:end_dates[i]]
    if X_train.empty is not True:
      p = pd.Series(model.predict(X_train), index=X_train.index)
      y_pred.loc[X_train.index] = p

  return models, y_pred, coefs


def stacked_model(target_data, feature_data, options={}):
  X = pd.DataFrame()
  y = one_step_forward_returns(target_data)
  X, y = add_returns(X, y, feature_data)

  linear_models, linear_preds, linear_coefs = make_walkforward_model_with_coefs(X, y, algo=LinearRegression())
  ensemble_models, ensemble_preds, ensemble_coefs = make_walkforward_model_with_coefs(X, y, algo=LassoCV(positive=True))

  # score_linear = compute_custom_scores(y_pred=linear_preds, y_true=y).rename('Linear')
  linear_scores_by_year = compute_custom_scores_over_time(y_pred=linear_preds, y_true=y)
  ensemble_scores_by_year = compute_custom_scores_over_time(y_pred=ensemble_preds, y_true=y)
  linear_coefs.T.plot(title='Coefficient for rolling window model')
  ensemble_coefs.T.plot(title='Coefficient for rolling window model')

  linear_scores_by_year.plot()
  ensemble_scores_by_year.plot()
  plt.show()

  # print()
  # print(score_linear)
  # print(scores_by_year.tail(3).T)
  # scores_by_year['edge_to_mae'].plot(title='Prediction Edge vs. MAE')
  # tree_models, tree_preds = make_walkforward_model(X, y, algo=ExtraTreesRegressor())
  # score_ens = compute_custom_scores(y_pred=ensemble_preds, y_true=y).rename('Ensemble')
  # score_tree = compute_custom_scores(y_pred=tree_preds, y_true=y).rename('Tree')
  # scores = pd.concat([score_linear, score_tree, score_ens], axis=1)
  # scores.loc['edge_to_noise'].plot.bar(color='grey', legend=True)
  # scores.loc['edge'].plot(color='green', legend=True)
  # scores.loc['noise'].plot(color='red', legend=True)







def make_ensemble_model(target_data, feature_data, options={}):
  ensemble_preds = ensemble_preds.rename('ensemble')
  print(ensemble_preds.dropna()).head()





walkforward_models = {
  'walkforward_regression_1': walkforward_regression_1,
  'walkforward_regression_2': walkforward_regression_2,
  'stacked_model': stacked_model
}