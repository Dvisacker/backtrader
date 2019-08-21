import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def compute_scores(y_test, y_test_pred, y_train=None, y_train_pred=None):
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    if y_train is not None and y_train_pred is not None:
      print('In sample R2:', np.sqrt(metrics.r2_score(y_train, y_train_pred)))
    print('Out of sample R2:', np.sqrt(metrics.r2_score(y_test, y_test_pred)))

def compute_kfold_scores(model, X, y, kfold):
    results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    print("MAE: {} ({})".format(results.mean(), results.std()))
    results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    print("MSE: {} ({})".format(results.mean(), results.std()))
    results = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    print("R^2: {} ({})".format(results.mean(), results.std()))


def confusion_matrix(y_pred, y_test):
  df = pd.DataFrame({'pred': y_pred, 'true': y_test })
  df['pred_sign'] = np.sign(df['pred'])
  df['true_sign'] = np.sign(df['true'])
  df['correct'] = df['pred_sign'] == df['true_sign']

  df2 = df[df['true_sign'] != 0]
  print(df2['correct'].value_counts())



# def compute_scores(forecast, actual):
#     mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
#     me = np.mean(forecast - actual)             # ME
#     mae = np.mean(np.abs(forecast - actual))    # MAE
#     mpe = np.mean((forecast - actual)/actual)   # MPE
#     rmse = np.mean((forecast - actual)**2)**.5  # RMSE
#     corr = np.corrcoef(forecast, actual)[0,1]   # corr
#     mins = np.amin(np.hstack([forecast[:,None],
#                               actual[:,None]]), axis=1)
#     maxs = np.amax(np.hstack([forecast[:,None],
#                               actual[:,None]]), axis=1)
#     minmax = 1 - np.mean(mins/maxs)             # minmax
#     acf1 = acf(forecast-actual)[1]                      # ACF1

#     return({'mape':mape, 'me':me, 'mae': mae,
#             'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
#             'corr':corr, 'minmax':minmax})
