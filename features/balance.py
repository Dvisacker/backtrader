import pandas as pd

def balance(X_train, X_test, y_train, y_test):
  sample_n = y_train.value_counts().idxmin()
  X_train_1 = X_train[y_train == 1].sample(sample_n)
  y_train_1 = y_train[y_train == 1].sample(sample_n)
  X_train_0 = X_train[y_train == 0].sample(sample_n)
  y_train_0 = y_train[y_train == 0].sample(sample_n)
  X_train_n1 = X_train[y_train == -1].sample(sample_n)
  y_train_n1 = y_train[y_train == -1].sample(sample_n)
  X = pd.concat([X_train_1, X_train_0, X_train_n1])
  y = pd.concat([y_train_1, y_train_0, y_train_n1])

  return X, y