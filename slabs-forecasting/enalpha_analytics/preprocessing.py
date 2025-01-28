""" Data preprocessing and transformation.

This file is meant for internal analytics module use only.
"""
import functools
from collections import defaultdict
from enum import Enum
from typing import List, NamedTuple, Dict, Any, Optional, Tuple, DefaultDict

import jsonschema as jsonsc
import numpy as np
import pandas as pd
from scipy import stats


FeatureSpecs = Dict[str, Dict[str, Any]]
FeatureMetadata = Dict[str, Dict[str, Any]]
BoundaryConditions = Tuple[int, pd.Timedelta]


class InvalidFeatureSpecificationError(ValueError):
    pass


class FeatureDerivationError(ValueError):
    pass


class TransformResult(NamedTuple):
    result: pd.Series
    metadata: Dict[str, Any] = {}


class InputType(Enum):
    SERIES = 1
    DATAFRAME = 2
    INDEX = 3
    Y = 4


def transform(func):
    @functools.wraps(func)
    def _wrap(data, args, metadata):
        return func(data, **args, **metadata)
    return _wrap


@transform
def rolling(data: pd.Series, window: str, window_function: str):
    w = pd.Timedelta(window)
    roll = data.rolling(w, min_periods=1)

    if window_function == 'variance':
        return TransformResult(roll.var().fillna(0.0))
    elif window_function == 'std':
        return TransformResult(roll.std().fillna(0.0))

    f = getattr(roll, window_function)
    return TransformResult(f())


@transform
def ewm(data: pd.Series, window_function:str, com=0.5):
    ewm = data.ewm(com=com)

    if window_function == 'variance':
        return TransformResult(ewm.var().fillna(0.0))
    elif window_function == 'std':
        return TransformResult(ewm.std().fillna(0.0))

    f = getattr(ewm, window_function)
    return TransformResult(f())


@transform
def row_aggregate(data: pd.DataFrame, function: str):
    fns = {'min': np.amin, 'max': np.amax, 'mean': np.mean,
           'variance': np.var, 'sum': np.sum, 'median': np.median}
    res = data.aggregate(fns[function], axis=1)
    return TransformResult(res)


@transform
def shift(data: pd.Series, periods: int):
    return TransformResult(data.shift(periods))



@transform
def diff(data: pd.Series, periods: int):
    return TransformResult(data.diff(periods))


@transform
def scale(data: pd.Series, kind: str, **kwargs):
    if kind == 'minmax':
        dmax = kwargs.get('train_max', np.max(data))
        dmin = kwargs.get('train_min', np.min(data))
        divisor = kwargs.get('divisor', (dmax - dmin) or 1.0)
        return TransformResult((data - dmin)/divisor, {
            'train_max': dmax,
            'train_min': dmin,
            'divisor': divisor,
        })
    elif kind == 'standard':
        scale = kwargs.get('scale', np.std(data) or 1.0)
        mean = kwargs.get('mean', np.mean(data))
        return TransformResult((data - mean)/scale, {
            'scale': scale,
            'mean': mean,
        })
    else:
        assert False


@transform
def boxcox(data, lmbda=None):
    if lmbda is None:
        lmbda = stats.boxcox_normmax(data.astype('float64', copy=False), method='mle')
    return TransformResult(stats.boxcox(data.values.reshape(-1), lmbda=lmbda), {'lmbda': lmbda})


@transform
def day_of_week(data: pd.Index):
    return TransformResult(data.day_of_week)


@transform
def day_of_year(data: pd.Index):
    return TransformResult(data.day_of_year)


@transform
def hour_of_day(data: pd.Index):
    return TransformResult(data.hour)


@transform
def hour_mean(x_y: Tuple[pd.DataFrame, pd.Series], hour_mean=None):
    x_df, y = x_y
    if hour_mean is None:
        assert y is not None
        hour_mean = y.groupby(y.index.hour).mean()
    else:
        hour_mean = pd.Series({int(k): v for k,v in hour_mean.items()})
    res = pd.Series(hour_mean.loc[x_df.index.hour].values, index=x_df.index)
    # convert to dict, so that it is json-serializable without issues
    return TransformResult(res, {'hour_mean': hour_mean.to_dict()})


@transform
def threshold(data: pd.Series, threshold):
    return TransformResult(data >= threshold)


@transform
def indicator(data: pd.Series, left, right):
    return TransformResult((left <= data) & (data <= right))


transforms = {
    'rolling': {
        'function': rolling,
        'input_type': InputType.SERIES,
        'args': {
            'window': {'type': 'string'},
            'window_function':  {'enum': ['min', 'max', 'mean', 'variance', 'sum', 'median', 'std']},
        }
    },
    'row_aggregate': {
        'function': row_aggregate,
        'input_type': InputType.DATAFRAME,
        'args': {
            'function':  {'enum': ['min', 'max', 'mean', 'variance', 'sum', 'median']},
        }
    },
    'shift': {
        'function': shift,
        'input_type': InputType.SERIES,
        'args': {
            'periods': {'type': 'integer'},
        }
    },
    'diff': {
        'function': diff,
        'input_type': InputType.SERIES,
        'args': {
            'periods': {'type': 'integer'},
        }
    },
    'ewm': {
        'function': ewm,
        'input_type': InputType.SERIES,
        'args': {
            'com': {'type': 'number', 'minimum': 0.0},
            'window_function':  {'enum': ['mean', 'variance', 'std']},
        }
    },
    'scale': {
        'function': scale,
        'input_type': InputType.SERIES,
        'args': {
            'kind':  {'enum': ['minmax', 'standard']},
        }
    },
    'boxcox': {
        'function': boxcox,
        'input_type': InputType.SERIES,
        'args': {
        }
    },
    'day_of_week': {
        'function': day_of_week,
        'input_type': InputType.INDEX,
        'args': {
        }
    },
    'day_of_year': {
        'function': day_of_year,
        'input_type': InputType.INDEX,
        'args': {
        }
    },
    'hour_of_day': {
        'function': hour_of_day,
        'input_type': InputType.INDEX,
        'args': {
        }
    },
    'y_hour_mean': {
        'function': hour_mean,
        'input_type': InputType.Y,
        'args': {
        }
    },
    'threshold': {
        'function': threshold,
        'input_type': InputType.SERIES,
        'args': {
            'threshold': {'type': 'number'},
        }
    },
    'range_indicator': {
        'function': indicator,
        'input_type': InputType.SERIES,
        'args': {
            'left': {'type': 'number'},
            'right': {'type': 'number'},
        }
    },
}


def _derive(fspec: Dict[str, Any], x_df: pd.DataFrame, y: Optional[pd.Series], fmetadata: Dict[str, Any]):
    t = transforms[fspec['transform']]
    if t['input_type'] == InputType.SERIES:
        in_col = x_df[fspec['input']]
        return t['function'](in_col, fspec.get('params', {}), fmetadata)
    elif t['input_type'] == InputType.DATAFRAME:
        # handling this seems identical for now
        in_cols = x_df[fspec['input']]
        return t['function'](in_cols, fspec.get('params', {}), fmetadata)
    elif t['input_type'] == InputType.INDEX:
        return t['function'](x_df.index, fspec.get('params', {}), fmetadata)
    elif t['input_type'] == InputType.Y:
        return t['function']((x_df, y), fspec.get('params', {}), fmetadata)
    else:
        assert False


def derive_features(specs: FeatureSpecs, x_df: pd.DataFrame, y: Optional[pd.Series]=None,
                    feature_metadata: FeatureMetadata=None):
    """ Compute new features based on their specifications.
    The new features are processed in the order they are specified.

    Arguments
    ---------
    specs: dict
        The dictionary containing definition of new columns to be obtained
        through transfomations. The order of entries in the dictionary is important.
    x_df: pd.DataFrame
        A dataframe containing the original features.
    feature_metadata: Optional[dict]
        Metadata containing any additional parameters that transformation may take,
        stored in a dict containing other dicts. These is meant to be used
        to pass previously learned parameter values during inference.

    Returns
    -------
    pd.DataFrame
    """
    result_df = x_df.copy(deep=False)
    if feature_metadata is None:
        feature_metadata = {}

    for name, spec in specs.items():
        try:
            res = _derive(spec, result_df, y, feature_metadata.get(name, {}))
        except Exception as e:
            raise FeatureDerivationError(f"Error deriving feature '{name}': {e}")

        if spec.get('replaces_input') and spec.get('input'):
            if isinstance(spec['input'], str):
                result_df = result_df.drop([spec['input']], axis=1)
            else:
                result_df = result_df.drop(spec['input'], axis=1)

        result_df[name] = res.result
        feature_metadata[name] = res.metadata

    return result_df.dropna(), feature_metadata


def transform_input_schema(transform_def, original_features):
    t = transform_def
    required = ['transform']
    if original_features is None:
        input = {'type': 'string'}
    else:
        input = {'enum': original_features}

    if t['input_type'] == InputType.DATAFRAME:
        input = {'type': 'array', 'items': input}
        required.append('input')
    elif t['input_type'] == InputType.SERIES:
        required.append('input')
    elif t['input_type'] == InputType.INDEX:
        input = {'type': 'null'}
    elif t['input_type'] == InputType.Y:
        input = {'type': 'null'}
    return input, required


def build_schema_for_transform(transform, original_features):
    ''' Build a json spec that checks param correctness for a specific transform. '''
    t = transforms[transform]
    input, required = transform_input_schema(t, original_features)

    schema = {
        'type': 'object',
        'properties': {
            'transform': {'enum': [transform]},
            'input': input,
            'params': {
                'type': 'object',
                'properties': t['args'],
                'required': list(t['args'].keys()),
                'additionalProperties': False
            },
            'replaces_input': {'type': 'boolean'},
        },
        'required': required,
        'additionalProperties': False
    }
    return schema


def _validate_spec(spec, original_features):
    # validate the general shape of the spec
    schema = {
        'type': 'object',
        'properties': {
            'transform': {'enum': list(transforms.keys())},
            'input': {
                'oneOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}},
                          {'type': 'null'}],
            },
            'params': {
                'type': 'object',
            },
            'replaces_input': {'type': 'boolean'},
        },
        'required': ['transform'],
        'additionalProperties': False
    }
    jsonsc.validate(instance=spec, schema=schema)
    # now validate it in detail
    detailed_schema = build_schema_for_transform(spec['transform'], original_features)
    jsonsc.validate(instance=spec, schema=detailed_schema)


def validate(specs: FeatureSpecs, original_features: Optional[List[str]]=None):
    """ Validate correctness of transform specifications.

    Arguments
    ---------
    specs: dict
        The dictionary containing definition of new columns to be obtained
        through transfomations.

    original_features: Optional[List[str]]
        The list of available features to validate against.

    Raises
    ------
    ValidationError
    """
    features = original_features
    for name, spec in specs.items():
        try:
            _validate_spec(spec, features)
            if features is not None:
                features.append(name)
        except jsonsc.ValidationError as e:
            message = f"Error validating derived feature specification '{name}': {e.message}"
            raise InvalidFeatureSpecificationError(message)


def determine_boundary_length(specs: FeatureSpecs) -> Optional[BoundaryConditions]:
    """
    Determine the number of additional rows to fetch from the database,
    so that the boundary conditions in window/timeseries transforms
    do not produce nans or other unwanted values in the desired time range.

    Arguments
    ---------
    specs: dict
        The dictionary containing definition of new columns to be obtained
        through transfomations.

    Returns
    -------
    periods: int,
        The additional number of rows needed before `date_start`.
    timedelta: pd.Timedelta,
        The additional time period needed before `date_start`.

    The meaning of the result is: one needs to fetch
    at least `periods` additional records from the time period
    immediately preceding `date_start` _and_ all records
    starting at `date_start-timedelta`
    (obviously, one of these two sets includes the other).
    """
    periods: DefaultDict[str, int] = defaultdict(int)
    timedelta: DefaultDict[str, pd.Timedelta] = defaultdict(lambda: pd.Timedelta(0))
    for n, spec in specs.items():
        inp = spec.get('input')
        if inp is None:
            continue
        try:
            periods[n] = periods[inp]
            timedelta[n] = timedelta[inp]
        except TypeError:
            # multiple inputs, inp is a list
            periods[n] = max(periods[a] for a in inp)
            timedelta[n] = max(timedelta[a] for a in inp)

        params = spec.get('params') or {}

        ps = params.get('periods')
        if ps is not None:
            periods[n] += ps

        w = params.get('window')
        if w is not None:
            timedelta[n] += pd.Timedelta(w)


    periods_max = max(periods.values(), default=0)
    td_max = max(timedelta.values(), default=pd.Timedelta(0))
    if periods_max == 0 and td_max == pd.Timedelta(0):
        return None
    else:
        return periods_max, td_max


def _check_non_timeseries(x_data: pd.DataFrame, derived_features: FeatureSpecs):
    """ Check if we are trying to do time-series transformations
        on non-timeseries data. This is problematic if we have boundary conditions
        for the transformations, because we will be unable to provide them in the future.
        If this is the case, then raise InvalidFeatureSpecificationError
    """
    try:
        p, td = determine_boundary_length(derived_features)
    except TypeError:
        return
    if not isinstance(x_data.index, pd.DatetimeIndex) and (p or td):
        raise InvalidFeatureSpecificationError(
            "Trying to do time-series only data transformations "
            "on non-time-series data.")


def preprocess_train(x_data: pd.DataFrame, y_data: pd.Series,
                     derived_features: Optional[FeatureSpecs]) -> Tuple[pd.DataFrame, Optional[dict]]:
    ''' Derive new features for training data, computing any necessary metadata. '''
    if derived_features is None:
        return x_data, None
    validate(derived_features, original_features=list(x_data.columns))
    _check_non_timeseries(x_data, derived_features)
    res, md = derive_features(derived_features, x_data, y=y_data)
    return res, {
        'spec': derived_features,
        'learned_params': md,
    }


def preprocess_pred(x_data: pd.DataFrame,
                    derived_features: Optional[Dict[str, Any]]) -> pd.DataFrame:
    ''' Derive new features for inference data. '''
    if derived_features is None:
        return x_data
    # does not need additional validation, as it was already validated
    # during training
    res, md = derive_features(derived_features['spec'], x_data,
                              feature_metadata=derived_features['learned_params'])
    return res
