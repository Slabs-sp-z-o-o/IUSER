import copy

import numpy as np
import pandas as pd
import pmdarima as pm
from sklearn.model_selection import TimeSeriesSplit

from . import debug_utils
from . import preprocessing as prepr

CV_FOLDS = 4
TEST_FRACTION = 0.4


def _get_test_size(data_len):
    return int(TEST_FRACTION*data_len / CV_FOLDS) + 1


def _split_data(y_data, train_exog_all, test_exog_all, train_i, test_i):
    train_index = y_data.index[train_i]
    test_index = y_data.index[test_i]

    if train_exog_all is not None:
        train_exog = train_exog_all.loc[train_index]
        test_exog = test_exog_all.loc[test_index]
    else:
        train_exog, test_exog = None, None

    train_endog = y_data.loc[train_index]
    return train_endog, train_exog, test_exog, test_index


def derive_features_for_cv(endog_data, exog_data, derived_features):
    df_metadata = None
    # compute data transforms
    if exog_data is not None:
        train_exog, df_metadata = prepr.preprocess_train(
            exog_data, endog_data,
            derived_features)
        test_exog = prepr.preprocess_pred(
            exog_data,
            df_metadata)
        train_endog = endog_data[train_exog.index]
    else:
        train_exog, test_exog = None, None
        train_endog = endog_data

    assert train_exog is None or (np.all(train_endog.index == train_exog.index)
                                  and np.all(test_exog.index == train_exog.index))
    return train_endog, train_exog, test_exog, df_metadata


def score_and_fit(y_data, train_exog_all, test_exog_all, estimator):
    """ Cross-validation and final fit of the passed estimator.
    Note: y_data, train_exog_all and test_exog_all are expected to have the same index!
    """
    tscv = TimeSeriesSplit(CV_FOLDS, test_size=_get_test_size(len(y_data)))
    results = []

    for train_i, test_i in tscv.split(y_data):
        train_endog, train_exog, test_exog, test_index = _split_data(y_data,
                                                                     train_exog_all, test_exog_all,
                                                                     train_i, test_i)

        estimator.fit(train_endog, exogenous=train_exog)
        partial_res = estimator.predict(n_periods=len(test_i), exogenous=test_exog)
        results.append(pd.Series(partial_res, index=test_index))
    y_pred = pd.concat(results)

    estimator.fit(y_data, exogenous=train_exog_all)
    return y_pred


def score_and_fit_by_update(y_data, train_exog_all, test_exog_all, estimator):
    tscv = TimeSeriesSplit(CV_FOLDS, test_size=_get_test_size(len(y_data)))
    results = []
    prev_train_idx = []

    for train_i, test_i in tscv.split(y_data.index):
        train_endog, train_exog, test_exog, test_index = _split_data(y_data,
                                                                     train_exog_all, test_exog_all,
                                                                     train_i, test_i)
        diff_train_idx = train_endog.index[train_i].difference(prev_train_idx)
        estimator.update(train_endog[diff_train_idx], exogenous=train_exog.loc[diff_train_idx, :])
        partial_res = estimator.predict(n_periods=len(test_i), exogenous=test_exog)
        results.append(pd.Series(partial_res, index=test_index))
        prev_train_idx = train_endog.index[train_i]

    # complete training using the last test fold
    diff_train_idx = y_data.index.difference(prev_train_idx)
    estimator.update(y_data[diff_train_idx], exogenous=train_exog_all.loc[diff_train_idx, :])
    y_pred = pd.concat(results)
    return y_pred


def score_and_fit_autoarima(y_data, train_exog_all, test_exog_all, estimator):
    estimator.fit(y_data, exogenous=train_exog_all)

    temp_estimator = copy.deepcopy(estimator.model_)

    tscv = TimeSeriesSplit(CV_FOLDS, test_size=_get_test_size(len(y_data)))
    results = []
    for train_i, test_i in tscv.split(y_data.index):
        train_endog, train_exog, test_exog, test_index = _split_data(y_data,
                                                                     train_exog_all, test_exog_all,
                                                                     train_i, test_i)

        temp_estimator.fit(train_endog, exogenous=train_exog)
        partial_res = temp_estimator.predict(n_periods=len(test_i), exogenous=test_exog)
        results.append(pd.Series(partial_res, index=test_index))
    y_pred = pd.concat(results)
    return y_pred


def update_and_score(endog_data, exog_data, estimator):
    tscv = TimeSeriesSplit(CV_FOLDS, test_size=_get_test_size(len(endog_data)))
    results = []

    for train_i, test_i in tscv.split(endog_data.index):
        temp_estimator = copy.deepcopy(estimator)
        train_endog, train_exog, test_exog, test_index = _split_data(endog_data,
                                                                     exog_data, exog_data,
                                                                     train_i, test_i)

        temp_estimator.update(train_endog, exogenous=train_exog)
        partial_res = temp_estimator.predict(n_periods=len(test_i), exogenous=test_exog)
        results.append(pd.Series(partial_res, index=test_index))

    # complete update
    estimator.update(endog_data, exogenous=exog_data)
    y_pred = pd.concat(results)
    return y_pred
