import pandas as pd

from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def compare_regressors(main_pair, raw_features, options={}):
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

  results = []
  names = []
  scoring = 'r2'

  models = []
  models.append(('LR', LinearRegression()))
  models.append(('LASSO', Lasso()))
  models.append(('EN', ElasticNet()))
  models.append(('KN', KNeighborsRegressor()))
  models.append(('DT', DecisionTreeRegressor()))
  models.append(('RT13', RandomForestRegressor(n_estimators=500, random_state=0, min_weight_fraction_leaf=0.02, max_features=13)))

  for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

  fig = pyplot.figure()
  fig.suptitle('Algorithm Comparison')
  ax = fig.add_subplot(111)
  pyplot.boxplot(results)
  ax.set_xticklabels(names)
  pyplot.show()


def compare_regressors_2(main_pair, raw_features, options={}):
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
    X['vol_lag_{}'.format(i)] = main_pair.volume.shift(i)
    X['vol_weighted_return_lag_{}'.format(i)] = main_pair.returns.shift(i) * main_pair.volume.shift(i)


  if raw_features:
    for pair, bars in raw_features.items():
      for i in range(1, lags + 1):
        X['{}_returns_lag_{}'.format(pair, i)] = bars.returns.shift(i)
        X['{}_vol_lag_{}'.format(pair, i)] = bars.volume.shift(i)
        X['{}_vol_weighted_return_lag_{}'.format(pair, i)] = bars.returns.shift(i) * bars.volume.shift(i)


  X.dropna(inplace=True)
  y = main_pair['returns']
  X, y = X.align(y, join='inner', axis=0)
  num_folds = 10
  seed = 7
  kfold = KFold(n_splits=num_folds, random_state=seed)

  results = []
  names = []
  scoring = 'r2'

  models = []
  models.append(('LR', LinearRegression()))
  models.append(('LASSO', Lasso()))
  models.append(('EN', ElasticNet()))
  models.append(('KN', KNeighborsRegressor()))
  models.append(('DT', DecisionTreeRegressor()))
  models.append(('RT13', RandomForestRegressor(n_estimators=500, random_state=0, min_weight_fraction_leaf=0.02, max_features=13)))
  models.append(('RT20', RandomForestRegressor(n_estimators=500, random_state=0, min_weight_fraction_leaf=0.02, max_features=20)))

  for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

  fig = pyplot.figure()
  fig.suptitle('Algorithm Comparison')
  ax = fig.add_subplot(111)
  pyplot.boxplot(results)
  ax.set_xticklabels(names)
  pyplot.show()


regressor_comparisons = {
  'compare_regressors': compare_regressors,
  'compare_regressors_2': compare_regressors_2
}

