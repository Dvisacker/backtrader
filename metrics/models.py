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


def compute_custom_scores(y_pred, y_true):
    y_pred.name = 'y_pred'
    y_true.name = 'y_true'
    df = pd.concat([y_pred, y_true],axis=1)
    df['sign_pred'] = df.y_pred.apply(np.sign)
    df['sign_true'] = df.y_true.apply(np.sign)
    df['is_correct'] = 0
    df.loc[df.sign_pred * df.sign_true > 0, 'is_correct'] = 1
    df['is_incorrect'] = 0
    df.loc[df.sign_pred * df.sign_true < 0, 'is_incorrect'] = 1
    df['is_predicted'] = df.is_correct + df.is_incorrect
    df['result'] = df.sign_pred * df.y_true

    scorecard = pd.Series()
    scorecard.loc['accuracy'] = df.is_correct.sum() * 1. / (df.is_predicted.sum() * 1.) * 100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()

    # derived metrics
    scorecard.loc['y_true_chg'] = df.y_true.abs().mean()
    scorecard.loc['y_pred_chg'] = df.y_pred.abs().mean()
    scorecard.loc['prediction_calibration'] = scorecard.loc['y_pred_chg']/scorecard.loc['y_true_chg']
    scorecard.loc['capture_ratio'] = scorecard.loc['edge']/scorecard.loc['y_true_chg']*100

    scorecard.loc['edge_long'] = df[df.sign_pred == 1].result.mean() - df.y_true.mean()
    scorecard.loc['edge_short'] = df[df.sign_pred == -1].result.mean() - df.y_true.mean()
    scorecard.loc['edge_win'] = df[df.is_correct == 1].result.mean() - df.y_true.mean()
    scorecard.loc['edge_lose'] = df[df.is_incorrect == 1].result.mean() - df.y_true.mean()

    return scorecard


def compute_custom_scores_over_time(y_pred, y_true):
    y_pred.name = 'y_pred'
    y_true.name = 'y_true'
    df = pd.concat([y_pred, y_true], axis=1).dropna().reset_index().set_index('time')

    scores = df.resample('D').apply(lambda df: compute_custom_scores(df[y_pred.name], df[y_true.name]))
    return scores

# def compute_scorecard_by_day(df):
#   df['year'] = df.index.get_level_values('day').year
#   return df.groupby('day').apply(calc_scorecard).T

# def compute_scorecard_by_hour(df):
#   df['hour'] = df.index.get_level_values('hour').hour
#   return df.groupby('hour').apply(calc_scorecard).T






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


def compute_adfuller(series):
  adfuller_result = adfuller(series)
  print('Returns ADF statistic: {}'.format(adfuller_result[0]))
  print('P-value: {}'.format(adfuller_result[1]))

def print_ts_model_metrics(aic, order, model):
  print('Best order: ', order)
  print('AIC: ', aic)
  print(model.summary())

def print_classification_report(y, y_pred):
  print(metrics.classification_report(y, y_pred, target_names=['short', 'no_trade', 'long']))

def print_crosstab(y, y_pred):
  print(pd.crosstab(y, y_pred, rownames=['Actual Labels'], colnames=['Predicted Labels']))

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
