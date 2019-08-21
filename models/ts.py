import pandas as pd
import numpy as np

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
from plot.ts import tsplot

import pdb

def get_best_model(X,maxp=1,maxd=1,maxq=1):
    best_aic = np.inf
    best_order = None
    best_model = None
    for i in range(maxp):
        for d in range(maxd):
              try:
                tmp_mdl = VARMAX(X, order=(i,d)).fit()
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d)
                    best_model = tmp_mdl
              except Exception as e:
                print(e)
    print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_model


def varmax_model_1(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close

  X = pd.DataFrame()
  X['returns'] = main_pair.returns
  if raw_features:
    for pair, bars in raw_features.items():
      X['{}_returns'.format(pair)] = bars.returns

  X.dropna(inplace=True)

  X_train = X[:int(0.8*(len(X)))]
  X_test = X[int(0.8*(len(X))):]

  model = VARMAX(X, order=(1,1)).fit()
  print('AIC: ', model.aic)
  print(model.summary())

  pdb.set_trace()
  tsplot(model.resid.returns, lags=30, title='Best GARCH Model (Residuals)')
  tsplot(model.resid.returns**2, lags=30, title='Best GARCH Model (Residuals Squared)')
  plt.show()

def varmax_model_2(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close

  X = pd.DataFrame()
  X['returns'] = main_pair.returns
  if raw_features:
    for pair, bars in raw_features.items():
      X['{}_returns'.format(pair)] = bars.returns

  X.dropna(inplace=True)
  X, y = X.align(y, join='inner', axis=0)

  kfold = KFold(n_splits=num_folds, random_state=seed)
  scores = []

  for train_index, test_index in kfold.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = VARMAX(X_train, order=(1,1)).fit()
    y_pred = model.forecast()



  X_train = X[:int(0.8*(len(X)))]
  X_test = X[int(0.8*(len(X))):]


  print('AIC: ', model.aic)
  print(model.summary())
  tsplot(model.resid.returns, lags=30, title='Best GARCH Model (Residuals)')
  tsplot(model.resid.returns**2, lags=30, title='Best GARCH Model (Residuals Squared)')



  plt.show()

def varmax_grid_search(main_pair, raw_features, options={}):
  """
  Primary model based on a bollinger bands strategy
  Different methodologies:

  1) timestamps => events => algo (decide trades + side) => (trades + side) => algo (decide sizes)
  2) timestamps => filter (decide potential trades) => events => algo (decide trades + side) => (trades + side) => algo (decides sizes)
  3) timestamps => algo (decide potential trades + side) => events + sides => algo (decide trades) => algo (decides sizes)

  For case 1) we use add_barriers_on_timestamps and
  """
  close = main_pair.close

  X = pd.DataFrame()
  X['returns'] = main_pair.returns
  if raw_features:
    for pair, bars in raw_features.items():
      X['{}_returns'.format(pair)] = bars.returns

  X.dropna(inplace=True)

  aic, order, model = get_best_model(X)
  print('Best Order: ', order)
  print('AIC: ', aic)
  print(model.summary())

  # tsplot(model.resid, lags=30, title='Best GARCH Model (Residuals). Order={}'.format(order))
  # tsplot(model.resid**2, lags=30, title='Best GARCH Model (Residuals Squared). Order={}'.format(order))
  # plt.show()

ts_models = {
  'varmax_model_1': varmax_model_1,
  'varmax_grid_search': varmax_grid_search
}