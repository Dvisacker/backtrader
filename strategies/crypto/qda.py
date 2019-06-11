#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import datetime

import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from .strategy import Strategy
from event import SignalEvent, SignalEvents
from trader import CryptoBacktest
from datahandler.crypto import HistoricCSVCryptoDataHandler
from execution.crypto import SimulatedCryptoExchangeExecutionHandler
from portfolio import CryptoPortfolio
from utils import date_parse, create_lagged_crypto_series, get_data_file, get_ohlcv_window

class QDAStrategy(Strategy):
    """
    QDA strategy. Quadratic Discriminant
    Analyser predicts the returns for a subsequent time
    period and then generates long/exit signals based on the
    prediction.
    """
    def __init__(self, data, events, configuration):
        self.data = data
        self.instruments = configuration.instruments
        self.csv_dir = configuration.csv_dir
        self.exchanges = configuration.exchange_names
        self.period = get_ohlcv_window(configuration.ohlcv_window)
        self.events = events
        self.datetime_now = datetime.datetime.utcnow()

        # The traded symbol will be the first instrument on the first exchange in the provided instruments object
        self.exchange = self.exchanges[0]
        self.symbol = self.instruments[self.exchange][0]


        # NOTE This strategy was made for 1 day windows
        self.model_start_date = datetime.datetime(2017,1,1)
        self.model_end_date = datetime.datetime(2017,12,31)
        self.model_start_test_date = datetime.datetime(2017,10,1)

        self.long_market = False
        self.short_market = False
        self.bar_index = 0

        csv_file = get_data_file(self.exchange, self.symbol, self.period)

        self.model_data = pd.read_csv(
            os.path.join(self.csv_dir, csv_file),
            parse_dates=True,
            date_parser=date_parse,
            header=0,
            sep=',',
            index_col=0,
            names=['datetime', 'open', 'high', 'low', 'close', 'volume', 'id', 'date']
        )

        self.model_data.dropna(inplace=True)
        self.model_data['returns'] = self.model_data['close'].pct_change()
        self.model = self.create_symbol_forecast_model()

    def create_symbol_forecast_model(self):
        # Create a lagged series of the BTC index S&P500 US stock market index

        returns = create_lagged_crypto_series(
            self.model_data,
            self.model_start_date,
            self.model_end_date, lags=5
        )

        # Use the prior two days of returns as predictor
        # values, with direction as the response
        X = returns[["Lag1","Lag2"]]
        y = returns["Direction"]

        # Create training and test sets
        start_test = self.model_start_test_date
        X_train = X[X.index < start_test]
        X_test = X[X.index >= start_test]
        y_train = y[y.index < start_test]
        y_test = y[y.index >= start_test]

        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        return model

    def calculate_signals(self, event):
        """
        Calculate the SignalEvents based on market data.
        """
        sym = self.symbol
        dt = self.datetime_now
        ex = self.exchange
        signals = []

        if event.type == 'MARKET':
            self.bar_index += 1
            if self.bar_index > 5:
                lags = self.data.get_latest_bars_values(ex, sym, "returns", N=3)
                pred_series = pd.Series(
                    {
                        'Lag1': lags[1]*100.0,
                        'Lag2': lags[2]*100.0
                    }
                ).values.reshape(1, -1)

                pred = self.model.predict(pred_series)
                if pred > 0 and not self.long_market:
                    self.long_market = True
                    signal = SignalEvent(1, ex, sym, dt, 'LONG', 1.0)
                    self.events.put(signal)

                if pred < 0 and self.long_market:
                    self.long_market = False
                    signal = SignalEvent(1, ex, sym, dt, 'EXIT', 1.0)
                    signals.append(signal)

                if signals:
                  events = SignalEvents(signals, 1)
                  self.events.put(events)