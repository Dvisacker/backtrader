import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from features.default import add_default_and_diff_lags
from plot.models import feature_importance_plot

def pca_model_1(main_pair, raw_features, options={}):
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

  pca = PCA(n_components=10)
  fit = pca.fit(X)
  print("Explained variance: {}".format(fit.explained_variance_ratio_))
  print(fit.components_)


def pca_model_2(main_pair, raw_features, options={}):
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

    pca = PCA()
    linear = LinearRegression()
    pipe = Pipeline(steps=[('pca', pca), ('linear', linear)])

    param_grid = {
      'pca__n_components': [1, 3, 5, 10, 15]
    }

    search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
    search.fit(X, y)
    print('Best parameter (CV score={})'.format(search.best_score_))
    print(search.best_params_)

    pca.fit(X)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(10,10))
    ax0.plot(pca.explained_variance_, linewidth=2)
    ax0.set_ylabel('PCA explained variance')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    results = pd.DataFrame(search.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
      lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                  legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.tight_layout()
    plt.show()


def rf_feature_importance_model_1(main_pair, raw_features, options={}):
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

  model = RandomForestRegressor()
  model.fit(X,y)
  print(model.feature_importances_)

  names = X.columns.values
  ticks = [i for i in range(len(names))]
  plt.figure(figsize = (20, 15))
  plt.bar(ticks, model.feature_importances_)
  plt.xticks(ticks, names, rotation='vertical')
  plt.subplots_adjust(bottom=0.25)
  plt.show()

def rf_feature_importance_model_2(target_data, feature_data, options={}):
  X = pd.DataFrame()
  y = target_data.returns

  X, y = add_default_and_diff_lags(X, y, feature_data, options)
  model = RandomForestRegressor()
  model.fit(X, y)

  feature_importance_plot(X, model)


feature_importance_models = {
  'pca_model_1': pca_model_1,
  'pca_model_2': pca_model_2,
  'rf_feature_importance_model_1': rf_feature_importance_model_1,
  'rf_feature_importance_model_2': rf_feature_importance_model_2
}