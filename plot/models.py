import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from yellowbrick.regressor import ResidualsPlot, PredictionError


def plot_roc_curve(fpr, tpr):
  plt.figure()
  plt.plot([0,1],[1,0], 'k--')
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
