"""
Timeseries forecasting API.

Provides functions for creation, updating and training of forecasting models,
as well as doing inference.
"""
import datetime
import dateutil.parser
import math
import operator
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn import metrics
from sklearn.model_selection import ParameterGrid

from common import config
from common.controllers.storage_controller import StorageController
from common.logger import get_logger
from .db import enalpha
from .utils import api_endpoint, Model, ModelId, create_estimator, validate_params
from . import preprocessing as prepr
from . import ml_forecasting, dnn_forecasting
from . import debug_utils
from . import crossval
from .crossval import score_and_fit, score_and_fit_by_update, score_and_fit_autoarima, CV_FOLDS
from .exceptions import NotEnoughDataError

logger = get_logger(__name__, config.LOG_ON_GCP)


def _ml_param_schema(base_schema):
    return {
        param_name: {
            'type': 'array',
            'minItems': 1,
            'items': subschema
        } for param_name, subschema in base_schema.items()
    }


forecasting_algorithms = {
    'autoarima': {
        'estimator': pm.arima.AutoARIMA,
        'param_schema': {
        },
        'default_estimator_params': {
            'seasonal': False,
        },
        'historic_data': False,
        'cross_validation': score_and_fit_autoarima,
    },
    'autosarima': {
        'estimator': pm.arima.AutoARIMA,
        'param_schema': {
        },
        'default_estimator_params': {
            'seasonal': True,
        },
        'historic_data': False,
        'cross_validation': score_and_fit_autoarima,
    },
    'arima': {
        'estimator': pm.arima.ARIMA,
        'param_schema': _ml_param_schema({
            # 'seasonality': {'type': 'boolean'},
            'order': {
                'type': 'array',
                'minItems': 3,
                'maxItems': 3,
                'contains': {
                    'type': 'integer'
                }
            },
        }),
        'required': ['order'],
        'default_estimator_params': {
            'seasonal': False,
        },
        'historic_data': False,
        'cross_validation': score_and_fit,
    },
    'sarima': {
        'estimator': pm.arima.ARIMA,
        'param_schema': _ml_param_schema({
            'order': {
                'type': 'array',
                'minItems': 3,
                'maxItems': 3,
                'items': {
                    'type': 'integer',
                    'minimum': 0
                }
            },
            'seasonal_order': {
                'type': 'array',
                'minItems': 4,
                'maxItems': 4,
                'items': {
                    'type': 'integer',
                    'minimum': 0
                }
            },
            'trend': {'enum': ['n', 'c', 't', 'ct']}
        }),
        'required': ['order'],
        'default_estimator_params': {
            'seasonal': True,
        },
        'historic_data': False,
        'cross_validation': score_and_fit,
    },
    'decision_tree': {
        'estimator': ml_forecasting.DTForecaster,
        'param_schema': _ml_param_schema({
            'criterion': {'enum': ['mse', 'friedman_mse', 'mae', 'poisson']},
            'max_depth': {'type': 'integer', 'minimum': 1},
            'max_leaf_nodes': {'type': 'integer', 'minimum': 1},
            'splitter': {'enum': ['best', 'random']},
            'random_state': {'type': 'integer'},
            'max_features': {
                'oneOf': [{'enum': ['auto', 'sqrt', 'log2']},
                          {'type': 'number', "exclusiveMinimum": 0, "exclusiveMaximum": 1}],
            },
        }),
        'historic_data': True,
        'cross_validation': score_and_fit,
    },
    'random_forest': {
        'estimator': ml_forecasting.RFForecaster,
        'param_schema': _ml_param_schema({
            'criterion': {'enum': ['mse', 'mae']},
            'max_depth': {'type': 'integer'},
            'max_leaf_nodes': {'type': 'integer'},
            'n_estimators': {'type': 'integer', "minimum": 1},
            'random_state': {'type': 'integer'},
            'max_features': {
                'oneOf': [{'enum': ['auto', 'sqrt', 'log2']},
                          {'type': 'number', "exclusiveMinimum": 0, "exclusiveMaximum": 1}],
            },
        }),
        'historic_data': True,
        'cross_validation': score_and_fit,
    },
    'linear_regression': {
        'estimator': ml_forecasting.LinearRegressionForecaster,
        'param_schema': _ml_param_schema({
            'fit_intercept': {'type': 'boolean'},
            'normalize': {'type': 'boolean'},
        }),
        'historic_data': True,
        'cross_validation': score_and_fit,
    },
    'lasso': {
        'estimator': ml_forecasting.LassoForecaster,
        'param_schema': _ml_param_schema({
            'alpha': {'type': 'number', "minimum": 0},
            'fit_intercept': {'type': 'boolean'},
            'normalize': {'type': 'boolean'},
            'random_state': {'type': 'integer'},
        }),
        'default_estimator_params': {
            'max_iter': 1000000,
            'selection': 'random',
        },
        'historic_data': True,
        'cross_validation': score_and_fit,
    },
    'svm': {
        'estimator': ml_forecasting.SVMForecaster,
        'param_schema': _ml_param_schema({
            'C': {'type': 'number', "exclusiveMinimum": 0},
            'kernel': {
                'enum': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'gamma': {
                'oneOf': [{'enum': ['auto', 'scale']}, {'type': 'number'}],
            },
            'epsilon': {'type': 'number', "exclusiveMinimum": 0},
        }),
        'historic_data': True,
        'cross_validation': score_and_fit,
    },
    'boosted_trees': {
        'estimator': ml_forecasting.LGBMForecaster,
        'param_schema': _ml_param_schema({
            'learning_rate': {'type': 'number', "minimum": 0, 'maximum': 1},
            'num_leaves': {'type': 'number', "minimum": 1},
            'colsample_bytree': {'type': 'number', "minimum": 0, 'maximum': 1},
            'reg_lambda': {'type': 'number', "minimum": 0},
            'reg_alpha': {'type': 'number', "minimum": 0}
        }),
        'historic_data': True,
        'cross_validation': score_and_fit,
    },
    'mlp': {
        'estimator': ml_forecasting.MLPForecaster,
        'param_schema': _ml_param_schema({
            'hidden_layer_sizes': {
                'type': 'array',
                'items': {'type': 'integer', 'minimum': 1}
            },
            'alpha': {'type': 'number', "minimum": 0},
            'random_state': {'type': 'integer'},
        }),
        'historic_data': True,
        'cross_validation': score_and_fit_by_update,
    },
    'cnn': {
        'estimator': dnn_forecasting.CNNForecaster,
        'param_schema': _ml_param_schema({
            'epochs': {'type': 'integer', 'minimum': 10},
            'batch_size': {'type': 'integer', 'minimum': 1},
            'window_size': {'type': 'integer', 'minimum': 6},
            'dropout': {'type': 'number', 'minimum': 0, 'maximum': 1},
        }),
        'historic_data': True,
        'cross_validation': score_and_fit_by_update,
    },
    'lstm_stateless': {
        'estimator': dnn_forecasting.LSTMStatelessForecaster,
        'param_schema': _ml_param_schema({
            'epochs': {'type': 'integer', 'minimum': 10},
            'batch_size': {'type': 'integer', 'minimum': 1},
            'window_size': {'type': 'integer', 'minimum': 6},
            'dropout': {'type': 'number', 'minimum': 0, 'maximum': 1},
        }),
        'historic_data': True,
        'cross_validation': score_and_fit_by_update,
    },
}

forecasting_metrics = {
    'r2': {'func': metrics.r2_score, 'compare': operator.gt},
    'rmae': {'func': lambda x, y: math.sqrt(metrics.mean_absolute_error(x, y)),
             'compare': operator.lt},
    'rmse': {'func': lambda x, y: math.sqrt(metrics.mean_squared_error(x, y)),
             'compare': operator.lt},
    'mae': {'func': metrics.mean_absolute_error,
            'compare': operator.lt},
    'mse': {'func': metrics.mean_squared_error,
            'compare': operator.lt},
    'mape': {'func': metrics.mean_absolute_percentage_error,
             'compare': operator.lt},
    'me': {'func': metrics.max_error,
           'compare': operator.lt},
}


def check_model_updatable(model_id, model_store=None):
    model_store = _get_model_store(model_store)
    md = model_store.load_model_metadata(model_id)
    algorithm = md['algorithm']
    if not hasattr(forecasting_algorithms[algorithm]['estimator'], 'update'):
        msg = f'Model {model_id} of type "{algorithm}" is not updatable.'
        raise ValueError(msg)


def _calc_metrics(y_pred: pd.Series, y_data: pd.Series) -> dict:
    """ Calculate cross-validation metrics based on a dict definition """
    y_true = y_data.loc[y_pred.index]
    res = {}
    for metric_name, metric in forecasting_metrics.items():
        res[metric_name] = metric['func'](y_true, y_pred)
    return res


def _get_boundary_conditions(model_metadata: dict) -> Optional[prepr.BoundaryConditions]:
    """ Determine the range of additional records needed for timeseries transformations
    working on moving windows.
    """
    md = model_metadata
    if not md:
        return None
    preproc_md = md.get('derived_features')
    if not preproc_md:
        return None
    return prepr.determine_boundary_length(preproc_md['spec'])


def _get_model_store(model_store):
    return model_store or enalpha.EnalphaModelStore()


def _get_db_interface(db_interface, debug=False):
    return db_interface or enalpha.EnalphaDataInterface(debug_log_query_results=debug)


def _load_real_data_for_corresponding_predictions(predictions: dict, db_interface, node: int, endogenous: str,
                                                  frequency: pd.Timedelta) -> Tuple[pd.Series, pd.Series]:
    time_range_start, time_range_end = [dateutil.parser.parse(x).astimezone(datetime.timezone.utc) for x in
                                        sorted(predictions)[::len(predictions) - 1]]
    predictions = pd.Series(predictions)
    predictions.index = pd.to_datetime(predictions.index)
    training_data = {
        "node": node,
        "endogenous": endogenous,
        "time_range_start": time_range_start - frequency,
        "time_range_end": time_range_end + frequency
    }

    return load_endogenous(training_data, db_interface,
                           target_frequency=frequency), predictions


def load_endogenous(data_spec, db_interface, target_frequency,
                    boundary_condition=None) -> pd.Series:
    endog_data = db_interface.load_series(data_spec['node'], [data_spec['endogenous']],
                                          target_frequency=target_frequency,
                                          skip_anomalies=data_spec.get('skip_anomalies'),
                                          time_range_start=data_spec.get('time_range_start'),
                                          time_range_end=data_spec.get('time_range_end'),
                                          boundary_condition=boundary_condition)
    return endog_data.iloc[:, 0]


def load_exogenous_and_historic(data_spec, db_interface, target_frequency,
                                time_range_start=None, time_range_end=None,
                                boundary_condition=None, historic_horizon=None) -> pd.DataFrame:
    exog_data = None
    try:
        range_start = time_range_start or data_spec.get('time_range_start')
        range_end = time_range_end or data_spec.get('time_range_end')
        exog_data = db_interface.load_series(data_spec['node'], data_spec['exogenous'],
                                             target_frequency=target_frequency,
                                             skip_anomalies=data_spec.get('skip_anomalies'),
                                             time_range_start=range_start,
                                             time_range_end=range_end,
                                             boundary_condition=boundary_condition)
    except KeyError:
        pass

    if historic_horizon is not None:
        historic_data_start = range_start - historic_horizon if range_start else None
        historic_data_end = range_end - historic_horizon if range_end else None
        historic = db_interface.load_series(data_spec['node'], [data_spec['endogenous']],
                                            target_frequency=target_frequency,
                                            skip_anomalies=data_spec.get('skip_anomalies'),
                                            time_range_start=historic_data_start,
                                            time_range_end=historic_data_end,
                                            boundary_condition=boundary_condition)
        if not (historic_horizon / target_frequency).is_integer():
            logger.debug("historic_horizon / target_frequency is %s",
                         historic_horizon / target_frequency)
            raise ValueError("horizon should be a multiple of target frequency")
        historic.index += historic_horizon
        if exog_data is not None:
            exog_data = pd.concat([exog_data, historic], axis=1).dropna()
        else:
            exog_data = historic

    return exog_data


def _select_params(algorithm: str, model_params: dict,
                   endog_data: pd.Series, exog_data: pd.DataFrame,
                   derived_features: dict,
                   optimization_metric: str,
                   pca: Optional[dict] = None, debug: bool = False):
    best_val = None
    best_metrics = None
    best_estimator = None
    best_params = None
    y_data, train_exog_all, test_exog_all, df_metadata = crossval.derive_features_for_cv(
        endog_data, exog_data, derived_features)
    if debug:
        debug_utils.add_to_buffer(train_exog_all, 'post-transform-exog')

    for params in ParameterGrid(model_params):
        estimator = create_estimator(algorithm, params, forecasting_algorithms, pca=pca)

        cv_func = forecasting_algorithms[algorithm]['cross_validation']
        y_cv = cv_func(y_data, train_exog_all, test_exog_all, estimator)
        metric_values = _calc_metrics(y_cv, endog_data)

        compare_op = forecasting_metrics[optimization_metric]['compare']
        if best_val is None or compare_op(metric_values[optimization_metric], best_val):
            best_val = metric_values[optimization_metric]
            best_metrics = metric_values
            best_estimator = estimator
            best_params = params

    assert best_estimator is not None
    return best_estimator, best_params, best_metrics, df_metadata


def _common_subset(endog_data, exog_data):
    if exog_data is None:
        return endog_data.index, None
    new_index = endog_data.index.intersection(exog_data.index)
    return new_index


def _select_common_subset(endog_data, exog_data):
    if exog_data is None:
        return endog_data, None
    new_index = endog_data.index.intersection(exog_data.index)
    endog_data = endog_data.loc[new_index]
    exog_data = exog_data.loc[new_index]
    return endog_data, exog_data


def _debug_buf_add(debug, db_interface, postfix):
    if debug:
        debug_utils.add_to_buffer(db_interface.get_debug_query_results(), postfix)
        db_interface.clear_debug_query_results()


def _check_train_len(endog_data, typ):
    if len(endog_data) <= CV_FOLDS:
        msg = f'Not enough {typ} data. At least {CV_FOLDS + 1} samples are needed, '
        if len(endog_data) == 0:
            msg += f'but none found.'
        else:
            msg += f'but got only {len(endog_data)}. ' \
                   'Many more samples are needed to get good accuracy.'
        raise ValueError(msg)


def _load_and_validate_train_data(training_data, db_interface, target_frequency, horizon,
                                  debug) -> Tuple[pd.Series, pd.DataFrame]:
    try:
        endog_data = load_endogenous(training_data, db_interface,
                                     target_frequency=target_frequency)
    except NotEnoughDataError:
        endog_data = []
    _check_train_len(endog_data, 'endogenous')
    _debug_buf_add(debug, db_interface, 'pre-transform-endog')
    if debug:
        debug_utils.add_to_buffer(endog_data,
                                  'post-transform-endog')

    exog_data = load_exogenous_and_historic(training_data, db_interface,
                                            target_frequency=target_frequency,
                                            historic_horizon=horizon)
    if exog_data is not None:
        _debug_buf_add(debug, db_interface, 'pre-transform-exog')

        _check_train_len(exog_data, 'exogenous or historic')

        endog_data, exog_data = _select_common_subset(endog_data, exog_data)
        _check_train_len(endog_data, 'overlapping exogenous, historic and endogenous')
    return endog_data, exog_data


def _load_and_validate_update_data(db_interface, model_metadata,
                                   time_range_end, debug) -> Tuple[pd.Series, pd.DataFrame]:
    # start where we left off
    original_training_data = model_metadata['training_data']
    time_range_start = model_metadata['previous_time_range_end']
    if time_range_end and time_range_end <= time_range_start:
        raise ValueError('Cannot update model with data that it was trained on or older.')

    try:
        update_data = dict(original_training_data,
                           time_range_start=time_range_start,
                           time_range_end=time_range_end)
        endog_data = load_endogenous(update_data, db_interface, model_metadata['target_frequency'])
        _debug_buf_add(debug, db_interface, 'pre-transform-endog')
        if len(endog_data) < 3:
            raise NotEnoughDataError(f'Only {len(endog_data)} records found within specified bounds.')
        try:
            exog_data = load_exogenous_and_historic(update_data, db_interface, model_metadata['target_frequency'],
                                                    historic_horizon=model_metadata['horizon'],
                                                    boundary_condition=_get_boundary_conditions(model_metadata))
            exog_data = prepr.preprocess_pred(exog_data,
                                              model_metadata.get('derived_features'))
            endog_data, exog_data = _select_common_subset(endog_data, exog_data)
            if len(endog_data) < 3:
                raise NotEnoughDataError(f'Only {len(endog_data)} records found within specified bounds.')
            _debug_buf_add(debug, db_interface, 'pre-transform-exog')
        except ValueError:
            exog_data = None
    except NotEnoughDataError as e:
        msg = 'To update a model, training data newer than the original training data is needed.'
        msg += f' The original training data ends at {time_range_start}.'
        if time_range_end:
            msg += f' The end of new training data is specified as {time_range_end}.'
        else:
            msg += f' The end of new training data is not specified.'
        msg += f' {e}'
        raise ValueError(msg)

    assert time_range_start <= endog_data.index[0]
    return endog_data, exog_data


@api_endpoint('forecasting model training')
def train(algorithm: str, params: Dict[str, List[Any]] = None, training_data: Dict[str, Any] = None,
          target_frequency: pd.Timedelta = None, horizon: Optional[pd.Timedelta] = None,
          model_store=None, db_interface=None, optimization_metric='r2',
          derived_features=None, pca=None, debug=False) -> Dict[str, Any]:
    """
    Train a forecasting model for timeseries data.

    Arguments
    ---------
    model_id: ModelId
        The unique identifier of previously created classification
        model to be trained on new data.
    params: dict
        Parameter search space.
    training_data: dict
        Specification of training data.
    target_frequency: pd.Timedelta
    horizon: Optional[pd.Timedelta]
    model_store: object
    db_interface: object
        An object for fetching input data. Should provide following methods:
           - load_series() -> pd.DataFrame
    optimization_metric: str
        Metric to optimize for in grid search. Should be a key in the
        `forecasting_metrics` dict.
    derived_features: dict
        A specification of additional features to derive for use in training.

    Returns
    -------
    dict containing fields:
        'metrics': dict - contains mappings from metric names to values.
                          The values usually are floats, but can also be matrices,
                          as with 'confusion_matrix'.

    Raises
    ------
    ValueError
        If the input data are invalid.
    """
    params = params or {}
    if training_data is None or target_frequency is None:
        raise ValueError('training_data and target_frequency are required parameters')
    model_store = _get_model_store(model_store)
    db_interface = _get_db_interface(db_interface, debug)

    if forecasting_algorithms[algorithm]['historic_data'] and horizon is None:
        raise ValueError(f'Algorithm "{algorithm}" requires a horizon.')
    if not forecasting_algorithms[algorithm]['historic_data'] and horizon is not None:
        logger.warning(f'Horizon given, but algorithm "{algorithm}" does not support it.')
    validate_params(algorithm, params, forecasting_algorithms)

    endog_data, exog_data = _load_and_validate_train_data(training_data, db_interface,
                                                          target_frequency, horizon, debug)

    estimator, selected_params, metric_values, df_md = _select_params(algorithm, params,
                                                                      endog_data, exog_data,
                                                                      derived_features,
                                                                      optimization_metric,
                                                                      pca=pca, debug=debug)
    metadata = {
        'target_frequency': target_frequency,
        'previous_time_range_start': endog_data.index[0],
        'previous_time_range_end': endog_data.index[-1],
        'training_data': training_data,
        'derived_features': df_md,
        'horizon': horizon,
        'creation_time': datetime.datetime.now(),
        'optimization_metric': optimization_metric,
    }

    model = Model(estimator, 'forecasting',
                  algorithm, selected_params, metric_values, metadata)
    model_id = model_store.create_model(model)

    if debug:
        debug_utils.flush_buffer(model_id)
    return {
        'model_id': model_id,
        'selected_hyperparameters': selected_params,
        'metrics': model.metrics,
    }


@api_endpoint('forecasting model update')
def update(model_id: ModelId, time_range_end=None,
           model_store=None, db_interface=None, debug=False):
    """
    Update an existing, pre-trained model using new data.

    Arguments
    ---------
    model_id: ModelId
        The unique identifier of previously created classification
        model to be trained on new data.
    time_range_end: Optional[datetime]
        Cutoff for the new training data.
    model_store: object
        An object providing following methods:
           - load_model(model_id: ModelId) -> utils.Model
           - update_model(model_id: ModelId, model: utils.Model) -> None
    db_interface: object
        An object providing following methods:
           - load_data() -> pd.DataFrame - should return exogenous data
           - load_labels() -> pd.Series - should return endogenous data

    Raises
    ------
    ValueError
        If model does not exist or training data are invalid.
    """
    model_store = _get_model_store(model_store)
    db_interface = _get_db_interface(db_interface, debug=debug)
    model = model_store.load_model(model_id)
    if model.estimator is None:
        raise ValueError('Updating is only possible for pre-trained models.')

    endog_data, exog_data = _load_and_validate_update_data(db_interface, model.metadata,
                                                           time_range_end, debug)

    y_cv = crossval.update_and_score(endog_data, exog_data, model.estimator)
    metric_values = _calc_metrics(y_cv, endog_data)
    model = model._replace(metrics=metric_values)

    model.metadata['previous_time_range_end'] = endog_data.index[-1]
    model.metadata['creation_time'] = datetime.datetime.now()
    model.metadata['previous_model_id'] = model_id

    new_model_id = model_store.create_model(model)

    if debug:
        debug_utils.flush_buffer(new_model_id)
        debug_utils.log_csv(new_model_id, exog_data, 'post-transform-exog')
        debug_utils.log_csv(new_model_id, endog_data, 'post-transform-endog')

    return {
        'model_id': new_model_id,
        'metrics': model.metrics,
    }


def _get_inference_index(model_metadata, prediction_time_start, prediction_horizon):
    target_freq = model_metadata['target_frequency']
    historic_horizon = model_metadata['horizon']
    if historic_horizon:
        # models that use historic data can make predictions starting at any point
        # in the future, as long as historic data is available
        index = pd.date_range(start=prediction_time_start,
                              end=prediction_time_start + prediction_horizon,
                              freq=target_freq)
    else:
        # models that don't use historic data can only make predictions starting
        # in the period immediately following training data
        next_pred = model_metadata['previous_time_range_end'] + target_freq
        if prediction_time_start < next_pred:
            raise ValueError('ARIMA-based methods can only predict time ranges newer than training data. '
                             f'This model can predict from {next_pred}, but requested '
                             f'predictions from {prediction_time_start}.')
        index = pd.date_range(start=next_pred,
                              end=prediction_time_start + prediction_horizon,
                              freq=target_freq)
    return index


def _validate_infer_data(exogenous_data, time_diff, prediction_time_start, prediction_horizon):
    if np.any(exogenous_data.isna()):
        msg = "Not enough exogenous or historic forecast data " \
              f"for making predictions from {prediction_time_start} with horizon of " \
              f"‘{prediction_horizon}’. " \
              "No data from requested period was found."
        raise ValueError(msg)
    if time_diff < prediction_horizon:
        msg = "Not enough exogenous or historic forecast data " \
              f"for making precise predictions from {prediction_time_start} with horizon of " \
              f"‘{prediction_horizon}’. "
        if time_diff > pd.Timedelta(0):
            msg += "The biggest possible horizon with the forecasting data " \
                   f"you’ve provided is ‘{time_diff}’. "
        msg += "Providing approximate predictions by filling missing values with nearest available."
        logger.warning(msg)


def _load_and_validate_infer_data(db_interface, model_metadata, exogenous_data_source,
                                  prediction_time_start,
                                  prediction_horizon) -> Tuple[pd.Series, pd.DataFrame]:
    index = _get_inference_index(model_metadata, prediction_time_start, prediction_horizon)
    target_freq = model_metadata['target_frequency']
    historic_horizon = model_metadata['horizon']
    bc = _get_boundary_conditions(model_metadata)
    # fetch more data than required by target_freq to deal with resampling border conditions
    exogenous_data = load_exogenous_and_historic(exogenous_data_source, db_interface,
                                                 target_frequency=target_freq,
                                                 historic_horizon=historic_horizon,
                                                 time_range_start=index[0] - target_freq,
                                                 time_range_end=index[-1] + target_freq,
                                                 boundary_condition=bc)
    if exogenous_data is not None:
        exogenous_data = prepr.preprocess_pred(exogenous_data,
                                               model_metadata.get('derived_features'))
        time_diff = exogenous_data.index[-1] - prediction_time_start
        exogenous_data = exogenous_data.reindex(index=index, method='nearest')
        _validate_infer_data(exogenous_data, time_diff, prediction_time_start, prediction_horizon)
    return index, exogenous_data


def round_datetime(dt: datetime.datetime, frequency=pd.Timedelta) -> datetime.datetime:
    return pd.Timestamp(dt).floor(f'{int(frequency.total_seconds())}s').to_pydatetime()


@api_endpoint('forecasting model inference')
def infer(model_id: ModelId, prediction_time_start, prediction_horizon,
          model_store=None, db_interface=None,
          override_exogenous_data=None, debug=False, use_cache=True) -> Dict[str, Any]:
    """
    Predict future time-series values using a previously trained model.
    Predictions start at the timestamp equal to the last time point of data
    the model was trained on.

    Arguments
    ---------
    model_id: ModelId
        Identifier of a previously trained model to be used.
    prediction_time_start: datetime
    prediction_horizon: timedelta
    model_store: object
    db_interface: object
    override_exogenous_data: Optional[Dict[str, list]]
        A dictionary containing values of exogenous variables.
        The values should be lists of length ~n_periods~.
        If the model does not use exogenous variables, then it is ignored.
        The default value is None.

    Raises
    ------
    ValueError
        If the model does not exists, input data are invalid.

    Returns
    -------
    results: dict
    """
    model_store = _get_model_store(model_store)
    db_interface = _get_db_interface(db_interface, debug=debug)
    model = model_store.load_model(model_id)
    prediction_time_start = round_datetime(prediction_time_start, pd.Timedelta(model.metadata['target_frequency']))
    exogenous_data_source = {**model.metadata['training_data'], **(override_exogenous_data or {})}

    index, exogenous_data = _load_and_validate_infer_data(db_interface, model.metadata,
                                                          exogenous_data_source,
                                                          prediction_time_start, prediction_horizon)
    preds = pd.Series(model.estimator.predict(len(index),
                                              exogenous=exogenous_data),
                      index=index)[prediction_time_start:]
    ret = {
        'node': exogenous_data_source['node'],
        'model_id': model_id,
        'frequency': model.metadata['target_frequency'].isoformat(),
        'endogenous': exogenous_data_source['endogenous'],
        'predictions': {ts.isoformat(): pred for ts, pred in zip(preds.index, preds)},
        'prediction_time_start': prediction_time_start.isoformat(),
        'prediction_horizon': prediction_horizon.isoformat(),
        'forecast_creation_time': datetime.datetime.now().isoformat(),
    }
    return ret


@api_endpoint('evaluating model predictions')
def evaluate(task_id: str, db_interface=None) -> Dict[str, Any]:
    """
    Evaluate predictions using corresponding real data.

    Arguments
    ---------
    task_id: TaskId
        Identifier of task of previously calculated predictions.

    Raises
    ------
    ValueError
        If the prediction does not exists or there is not enough real data to perform evaluation

    Returns
    -------
    results: dict
    """

    task_result = StorageController.get_task_result(task_id)
    if not task_result:
        raise ValueError(f'Task of id "{task_id}" is not present in Cloud Storage!')
    task_result = task_result.get('result')
    try:
        _predictions = task_result['predictions']
    except (KeyError, TypeError):
        raise ValueError(f'Task of id: {task_id} does not contain predictions - cannot evaluate!')

    db_interface = _get_db_interface(db_interface)
    node = task_result.get('node')
    endogenous = task_result.get('endogenous')
    frequency = pd.Timedelta(task_result.get('frequency'))

    real_data, predictions = _load_real_data_for_corresponding_predictions(predictions=_predictions,
                                                                           db_interface=db_interface, node=node,
                                                                           endogenous=endogenous, frequency=frequency)
    logger.warning(f'Real data: {real_data}')
    if len(predictions) != len(real_data):
        raise ValueError('There is not enough real data in data warehouse to properly evaluate prediction!')
    metrics = _calc_metrics(y_pred=predictions, y_data=real_data)

    ret = {
        'metrics': metrics,
        'prediction_task_id': task_id,
        'real_data': {ts.isoformat(): real_data for ts, real_data in zip(real_data.index, real_data)}
    }

    return ret
