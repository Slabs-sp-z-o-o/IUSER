import json
import functools
import traceback
from typing import NamedTuple, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from jsonschema import validate, ValidationError
from sklearn.base import BaseEstimator

from .ml_forecasting import PCAWrapper
from common import config
from common.logger import get_logger

ModelId = str
logger = get_logger(__name__, config.LOG_ON_GCP)


class Model(NamedTuple):
    estimator: BaseEstimator
    typ: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


def api_endpoint(endpoint_name):
    def _api(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info('Doing %s with args %s, kwargs %s',
                         endpoint_name, args, kwargs)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(traceback.format_exc())
                raise

        return wrapper
    return _api


def validate_params(model_type: str, params: Dict[str, Any], algorithms: Dict[str, Any]):
    schema = {
        'type': 'object',
        'properties': algorithms[model_type]['param_schema'],
        'required': algorithms[model_type].get('required', []),
        'additionalProperties': False
    }
    try:
        validate(instance=params, schema=schema)
    except ValidationError as e:
        raise ValueError(f'Error validating algorithm hyperparameters: {e.message} '
                         f'in {json.dumps(params)}')


def create_estimator(model_type: str, params: Dict[str, Any], algorithms: Dict[str, Any],
                     pca: Optional[dict]=None):
    try:
        estimator_class = algorithms[model_type]['estimator']
    except KeyError:
        raise ValueError(f'Unknown algorithm type "{model_type}"')

    default_estimator_params = algorithms[model_type].get('default_estimator_params', {})
    est = estimator_class(**params, **default_estimator_params)
    if pca:
        est = PCAWrapper(est, random_state=1, n_components=pca['n_components'])
    return est
