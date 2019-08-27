import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from yellowbrick.regressor import ResidualsPlot, PredictionError
from plot.ts import tsplot


def plot_roc_curve(fpr, tpr):
  plt.figure()
  plt.plot([0,1],[0,1], 'k--')
  plt.plot(fpr, tpr, label='RF')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC Curve')
  plt.legend(loc='best')
  plt.show()

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

def residuals_ts_plot(residuals, title=""):
  tsplot(residuals, lags=30, title="{} (Residuals)".format(title))
  tsplot(residuals**2, lags=30, title="{} (Residuals Squared)".format(title))
  plt.show()

def feature_importance_plot(features, model):
  names = features.columns.values
  ticks = [i for i in range(len(names))]
  plt.figure(figsize=(20,15))
  plt.bar(ticks, model.feature_importances_)
  plt.xticks(ticks, names, rotation='vertical')
  plt.subplots_adjust(bottom=0.25)
  plt.show()

