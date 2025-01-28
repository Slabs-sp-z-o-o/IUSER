import io
import os
import gzip
from typing import Optional, Tuple, List

import pandas as pd
from google.cloud import storage

from .utils import ModelId
from common import config
from common.logger import get_logger


class DebugLoggingError(Exception):
    pass


logger = get_logger(__name__, config.LOG_ON_GCP)
DEBUG_BUCKET = config.ML_MODELS_INTERMEDIATE_DATA

_buffer: List[Tuple[io.BytesIO, str]] = []


def add_to_buffer(df: Optional[pd.DataFrame], postfix: str = ''):
    if df is None:
        return

    b = io.BytesIO()  # IMPORTANT: DO NOT CLOSE THIS FILE HERE...
    with gzip.GzipFile(fileobj=b, mode='w') as gz:
        df.to_csv(gz)
    _buffer.append((b, postfix))  # ...BECAUSE THE BytesIO OBJECT GOES HERE


def _save_to_storage(model_id: ModelId, obj: io.BytesIO, postfix: str):
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(DEBUG_BUCKET)
        blob = storage.Blob(f'model_{model_id}-{postfix}.csv.gz', bucket)
        obj.seek(0)
        blob.upload_from_file(obj)
    except Exception as e:
        logger.error(f'Error logging debug data to storage: {e}')
        raise DebugLoggingError(f'Error logging debug data to storage: {e}')


def log_csv(model_id: ModelId, df: Optional[pd.DataFrame], postfix: str = ''):
    if df is None:
        return
    with io.BytesIO() as b:
        with gzip.GzipFile(fileobj=b, mode='w') as gz:
            df.to_csv(gz)
        _save_to_storage(model_id, b, postfix)


def flush_buffer(model_id):
    global _buffer
    while len(_buffer) > 0:
        df, postfix = _buffer.pop()
        _save_to_storage(model_id, df, postfix)
        del df
