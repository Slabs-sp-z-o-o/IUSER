"""
Forecasters using keras DNN regression.
"""
import io

import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .ml_forecasting import UpdateableMixin


def get_losses():
    # don't load keras at the top to avoid memory bloat
    from tensorflow import keras
    return {
        'mae': keras.losses.MeanAbsoluteError,
        'mse': keras.losses.MeanSquaredError,
        'mape': keras.losses.MeanAbsolutePercentageError,
    }


class BaseDNNForecaster(UpdateableMixin):
    def __init__(self, window_size=24, es_patience=50, dropout=0.2, l2_reg=0, epochs=2000,
                 batch_size=1, optimizer='adam', loss='mse', scale_output=False):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler() if scale_output else None
        self.window_size = window_size
        if self.window_size < 6:
            raise ValueError("window_size must be greater than 5")  # because of kernel_size in conv layer

        self.es_patience = es_patience
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.model = None
        self.loss = loss

    def _initialize_model(self, y, x):
        self.model = self._get_model(input_dim=x.shape[1])
        self.input_scaler.fit(x.values)
        if self.output_scaler:
            self.output_scaler.fit(y.values.reshape(-1, 1))


    def fit(self, endogenous, exogenous):
        from tensorflow import keras
        self._initialize_model(endogenous, exogenous)
        train_data = keras.preprocessing.timeseries_dataset_from_array(self._standardize_x(exogenous),
                                                                       self._standardize_y(endogenous),
                                                                       self.window_size,
                                                                       batch_size=self.batch_size)

        self.model.fit(x=train_data, epochs=self.epochs,
                       callbacks=self._get_callbacks(), verbose=2)

    def update(self, endogenous, exogenous):
        from tensorflow import keras
        if not self.model:
            self._initialize_model(endogenous, exogenous)

        train_data = keras.preprocessing.timeseries_dataset_from_array(self._standardize_x(exogenous),
                                                                       self._standardize_y(endogenous),
                                                                       self.window_size,
                                                                       batch_size=self.batch_size)
        self.model.fit(x=train_data, epochs=self.epochs,
                       callbacks=self._get_callbacks(), verbose=2)

    def predict(self, n_periods, exogenous):
        from tensorflow import keras
        assert exogenous.shape[0] >= n_periods
        exog = exogenous.iloc[:n_periods, :]

        input_data = keras.preprocessing.timeseries_dataset_from_array(exogenous, None,
                                                                       self.window_size,
                                                                       batch_size=self.batch_size)

        initial_prediction = np.repeat(np.NaN, self.window_size - 1)
        model_prediction = self.model.predict(input_data)[:, 0]
        prediction = pd.Series(np.concatenate((initial_prediction, model_prediction)),
                               index=exogenous.index,
                               name='Prediction').bfill()  # TODO, find a better way than bfill

        return self._destandardize_output(prediction)

    def _get_callbacks(self):
        from tensorflow import keras
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                           patience=self.es_patience)
        return [es, ]

    def _standardize_x(self, input_data):
        return pd.DataFrame(self.input_scaler.transform(input_data), columns=input_data.columns,
                            index=input_data.index)

    def _standardize_y(self, y):
        if self.output_scaler:
            scaled = self.output_scaler.transform(y.values.reshape(-1, 1))
            return pd.Series(scaled.reshape(-1), index=y.index, name=y.name)
        else:
            return y

    def _destandardize_output(self, prediction):
        if self.output_scaler is None:
            return prediction
        scaled_vals = self.output_scaler.inverse_transform(prediction.values.reshape(-1, 1))
        return pd.Series(scaled_vals.reshape(-1), index=prediction.index, name=prediction.name)

    def _get_model(self, input_dim):
        raise NotImplementedError()

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model:
            with io.BytesIO() as b:
                with h5py.File(b) as f:
                    self.model.save(f)
                state['model'] = b.getvalue()
        return state

    def __setstate__(self, state):
        from tensorflow import keras
        self.__dict__.update(state)
        if self.model:
            with io.BytesIO(self.model) as b:
                with h5py.File(b) as f:
                    self.model = keras.models.load_model(f)


class CNNForecaster(BaseDNNForecaster, UpdateableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, scale_output=True)

    def _get_model(self, input_dim):
        from tensorflow import keras
        shape = (self.window_size, input_dim)

        model = keras.models.Sequential()

        model.add(keras.layers.Conv1D(filters=20, kernel_size=5, activation='relu', input_shape=shape))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(50, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(self.l2_reg)))

        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(30, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(self.l2_reg)))

        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1))

        model.compile(loss=get_losses()[self.loss](), optimizer=self.optimizer)

        return model


class LSTMStatelessForecaster(BaseDNNForecaster, UpdateableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, scale_output=False)

    def _get_model(self, input_dim):
        from tensorflow import keras
        shape = (self.window_size, input_dim)
        model = keras.models.Sequential()

        model.add(keras.layers.LSTM(50, return_sequences=False, stateful=False, input_shape=shape,
                                    kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(30, activation='relu', kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(20, activation='relu', kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(10, activation='relu', kernel_regularizer=keras.regularizers.l2(self.l2_reg)))
        model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1))

        model.compile(loss=get_losses()[self.loss](), optimizer=self.optimizer)

        return model
