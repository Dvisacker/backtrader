import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def random_forest_model_1(df_pair1, df_pair2, df_pair3, df_pair4, df_pair5):
  X = pd.DataFrame()
  X['returns_pair1_lag1'] = df_pair1.returns.shift(1)
  X['returns_pair1_lag2'] = df_pair1.returns.shift(2)
  X['returns_pair1_lag3'] = df_pair1.returns.shift(3)
  X['returns_pair1_lag4'] = df_pair1.returns.shift(4)

  X['returns_pair2_lag1'] = df_pair2.returns.shift(1)
  X['returns_pair2_lag2'] = df_pair2.returns.shift(2)
  X['returns_pair2_lag3'] = df_pair2.returns.shift(3)
  X['returns_pair2_lag4'] = df_pair2.returns.shift(4)

  X['returns_pair3_lag1'] = df_pair3.returns.shift(1)
  X['returns_pair3_lag2'] = df_pair3.returns.shift(2)
  X['returns_pair3_lag3'] = df_pair3.returns.shift(3)
  X['returns_pair3_lag4'] = df_pair3.returns.shift(4)

  X['returns_pair4_lag1'] = df_pair4.returns.shift(1)
  X['returns_pair4_lag2'] = df_pair4.returns.shift(2)
  X['returns_pair4_lag3'] = df_pair4.returns.shift(3)
  X['returns_pair4_lag4'] = df_pair4.returns.shift(4)

  X['returns_pair5_lag1'] = df_pair5.returns.shift(1)
  X['returns_pair5_lag2'] = df_pair5.returns.shift(2)
  X['returns_pair5_lag3'] = df_pair5.returns.shift(3)
  X['returns_pair5_lag4'] = df_pair5.returns.shift(4)
  X.dropna(inplace=True)

  y = df_pair1['returns']

  X, y = X.align(y, join='inner', axis=0)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  regressor = RandomForestRegressor(n_estimators=20, random_state=0)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)

  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


forests = {
  'random_forest_model_1': random_forest_model_1
}