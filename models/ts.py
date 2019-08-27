import pandas as pd
import numpy as np

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt

from plot.ts import tsplot
from plot.models import residuals_ts_plot
from metrics.models import print_ts_model_metrics

from arch import arch_model

import pdb

def get_best_varmax_model(X,maxp=1,maxd=1,maxq=1):
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


def get_best_arima_model(X,maxp=2,maxd=2,maxq=2):
    best_aic = np.inf
    best_order = None
    best_model = None
    for i in range(maxp):
        for d in range(maxd):
            for j in range(maxq):
                try:
                    tmp_mdl = ARIMA(X, order=(i,d,j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_model = tmp_mdl
                except: continue
    print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_model


def varmax_model_1(main_pair, raw_features, options={}):
  """
  VARMAX Model tested without cross-validation
  """
  X = pd.DataFrame()
  X['returns'] = main_pair.returns
  if raw_features:
    for pair, bars in raw_features.items():
      X['{}_returns'.format(pair)] = bars.returns

  X.dropna(inplace=True)


  model = VARMAX(X, order=(1,1)).fit()
  print('AIC: ', model.aic)
  print(model.summary())
  residuals_ts_plot(model.resid, title='VARMAX Model')


def arima_model_1(main_pair, raw_features, options={}):
  """
  asdfasdf
  """
  returns = main_pair.returns
  aic, order, model = ARIMA(returns, order=(1,1,1)).fit(method='mle', trend='nc')
  print_ts_model_metrics(aic, order, model)
  residuals_ts_plot(model, title='Best VARMAX model: {}'.format(order))


def garch_model_1(main_pair, raw_features, options={}):
  """
  Basic GARCH model
  """
  returns = main_pair.returns
  _, order, model = get_best_arima_model(returns)
  p_, o_, q_ = order

  garch_model = arch_model(model.resid, p=p_, o=o_, q=q_, dist='StudentsT')
  garch_model = garch_model.fit(update_freq=5, disp='off')

  print(garch_model.summary())

  residuals_ts_plot(model.resid, title='Best ARIMA Model. Order={}'.format(order))
  residuals_ts_plot(garch_model.resid, title='Best GARCH Model. Order={}'.format(order))
  plt.show()

def arima_grid_search(main_pair, raw_features, options={}):
  """
  ARIMA grid search
  """
  returns = main_pair.returns

  aic, order, model = get_best_arima_model(returns, maxp=2, maxd=2, maxq=2)
  print_ts_model_metrics(aic, order, model)
  residuals_ts_plot(model, 'Best ARIMA model: {}'.format(order))

def varmax_grid_search(main_pair, raw_features, options={}):
  """
  VARMAX grid search
  """
  close = main_pair.close

  X = pd.DataFrame()
  X['returns'] = main_pair.returns
  if raw_features:
    for pair, bars in raw_features.items():
      X['{}_returns'.format(pair)] = bars.returns

  X.dropna(inplace=True)

  aic, order, model = get_best_varmax_model(X)
  print_ts_model_metrics(aic, order, model)
  residuals_ts_plot(model, title='Best VARMAX model: {}'.format(order))

ts_models = {
  'varmax_model_1': varmax_model_1,
  'varmax_grid_search': varmax_grid_search
}