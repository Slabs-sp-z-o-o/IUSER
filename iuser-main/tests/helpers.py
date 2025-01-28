import datetime
import gzip
import json
import logging
import os
import shutil
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import bigquery, ndb, storage
from google.cloud.logging import DESCENDING
from google.cloud.logging_v2.client import Client
from requests import Response

from common import config
from common.gcp.datastore.schema import MLPredictionMetadata
from common.gcp.functions.utils import dictionary_md5_hash

plt.rcParams['figure.figsize'] = (15, 6)
storage_client = storage.Client()

ELECTRICAL_MEASUREMENTS_SCHEMA_PATH = 'common/gcp/bigquery/schemas/electrical_measurements.json'
TEMPORARY_CSV_DIR = 'tests/end_to_ends/temporary_data'


def assert_cached_prediction(task_id, check_evaluation=False):
    ndb_client = ndb.Client()
    with ndb_client.context():
        prediction = MLPredictionMetadata.make_key(task_id=task_id).get()
        assert prediction
        if check_evaluation:
            assert prediction.evaluation_reference
            assert prediction.metrics


def assert_task_is_scheduled(response: Response) -> str:
    assert response.status_code == 202, f'Task not started: {response.text}'
    task_id = response.json().get('task_id')
    assert isinstance(task_id, str)
    logging.info(f'Task {task_id} created')
    return task_id


def assert_response_success_json(response: Response) -> dict:
    assert response.status_code == 200
    response_json = response.json()
    assert response_json['status'] == 'SUCCESS'
    return response_json


def assert_response_failure_json(response: Response, message: str = None) -> dict:
    response_json = response.json()
    assert response_json['status'] == 'FAILURE'
    if message:
        assert message in response_json['message']
    return response_json


def assert_training_response(response: Response) -> Tuple[str, float]:
    response_json = assert_response_success_json(response)
    assert isinstance(model_id := response_json['result']['model_id'], str)
    assert isinstance(r2 := response_json['result']['metrics']['r2'], float)
    return model_id, r2


def assert_predict_response(response: Response, provided_horizon: str) -> Tuple[str, pd.DataFrame]:
    response_json = assert_response_success_json(response)
    assert isinstance(response_json['result']['predictions'], dict)
    assert isinstance(node := response_json['result']['node'], int)

    predictions_df = PredictionsParser.create_predictions_dataframe(response.json())

    # Make sure horizon of prediction matches desired one
    assert PredictionsParser.check_if_horizon_of_prediction_matches_provided_one(
        provided_horizon=provided_horizon,
        df=predictions_df)
    return node, predictions_df


def assert_prediction_evaluation_response(response: Response, predictions_df: pd.DataFrame) -> Tuple[str, dict]:
    response_json = assert_response_success_json(response)
    assert isinstance(metrics := response_json['result']['metrics'], dict)
    assert isinstance(prediction_task_id := response_json['result']['prediction_task_id'], str)
    assert isinstance(real_data := response_json['result']['real_data'], dict)

    assert sorted(datetime.datetime.fromtimestamp(int(x) // 1e9, datetime.timezone.utc).isoformat() for x in
                  predictions_df.index.values) == sorted(real_data.keys())
    return prediction_task_id, metrics


class MLModelsStorageController:
    ml_models_bucket_name = config.ML_MODELS_BUCKET_NAME
    ml_models_metadata_bucket_name = config.ML_MODELS_METADATA_BUCKET_NAME
    ml_models_intermediate_data_bucket_name = config.ML_MODELS_INTERMEDIATE_DATA
    ml_models_metadata_format = 'model_{model_id}.json'
    ml_models_format = 'model_{model_id}'
    ml_models_intermediate_data_format = 'model_{model_id}-{moment}-transform-{target}.csv.gz'
    ml_models_intermediate_data_types = [{'moment': 'post', 'target': 'endog'}, {'moment': 'post', 'target': 'exog'},
                                         {'moment': 'pre', 'target': 'endog'}, {'moment': 'pre', 'target': 'exog'}, ]

    @staticmethod
    def _get_blob(model_id: str, file_format: str, bucket_name: str) -> Optional[storage.Blob]:
        filename = file_format.format(model_id=model_id)
        bucket = storage_client.get_bucket(bucket_name)
        return bucket.get_blob(filename)

    @classmethod
    def check_ml_model_exists_in_cloud_storage(cls, model_id: str) -> bool:
        return cls._get_blob(model_id, cls.ml_models_format, cls.ml_models_bucket_name) is not None

    @classmethod
    def get_ml_model_metadata_content(cls, model_id: str) -> str:
        return cls._get_blob(model_id, cls.ml_models_metadata_format,
                             cls.ml_models_metadata_bucket_name).download_as_text()

    @classmethod
    def check_ml_model_intermediate_data_in_cloud_storage(cls, model_id: str) -> bool:
        for type in cls.ml_models_intermediate_data_types:
            file_format = cls.get_ml_model_intermediate_data_specific_file_name(model_id, type)
            if cls._get_blob(model_id, file_format, cls.ml_models_intermediate_data_bucket_name) is None:
                return False
        return True

    @classmethod
    def get_ml_model_intermediate_data_specific_file_name(cls, model_id: str, type: dict) -> str:
        return cls.ml_models_intermediate_data_format.format(model_id=model_id, moment=type['moment'],
                                                             target=type['target'])

    @classmethod
    def download_and_decompress_ml_model_intermediate_data_specific_file(cls, file_name: str,
                                                                         file_dir: str = TEMPORARY_CSV_DIR) -> None:
        fn_compressed = os.path.join(file_dir, file_name)
        download_file_from_bucket(cls.ml_models_intermediate_data_bucket_name, file_name,
                                  destination_file_name=fn_compressed)
        with gzip.open(fn_compressed, 'rb') as f_in:
            with open(fn_compressed + '.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # delete compressed file
        os.remove(fn_compressed)

    @classmethod
    def get_ml_model_intermediate_data_to_df(cls, model_id: str, model_type: dict,
                                             delete_csv: bool = True,
                                             file_dir: str = TEMPORARY_CSV_DIR) -> pd.DataFrame:
        fn_compressed = cls.get_ml_model_intermediate_data_specific_file_name(model_id, model_type)
        cls.download_and_decompress_ml_model_intermediate_data_specific_file(fn_compressed)
        csv_filename = os.path.join(file_dir, fn_compressed + '.csv')

        result_df = pd.read_csv(csv_filename, parse_dates=['timestamp'], index_col='timestamp')
        if delete_csv:
            os.remove(csv_filename)
        return result_df


class PredictionsParser:
    @staticmethod
    def create_predictions_dataframe(predict_response_json: dict) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(predict_response_json['result']['predictions'], orient='index', columns=['y'])
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def check_if_horizon_of_prediction_matches_provided_one(provided_horizon: str, df: pd.DataFrame) -> bool:
        computed_horizon = df.index[-1] - df.index[0]
        unit = provided_horizon[-1].translate(str.maketrans('dw', 'DW'))
        assert provided_horizon[:-1].isdigit(), 'Provided horizon number of units is not a digit.'
        assert unit in 'shDW', 'Unknown unit in provided horizon.'
        horizon = pd.Timedelta(int(provided_horizon[:-1]), unit=unit)
        return computed_horizon == horizon


class CloudFunctionMonitoring:
    # TODO Take care of creating separate logging sing for CF logs, to not pay too much for logs ingestion.
    # https://cloud.google.com/stackdriver/pricing#view-usage
    # https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/logging_project_sink

    def __init__(self, payload):
        self.client = Client()
        self.execution_id = self._get_execution_id(payload)

    def _get_execution_id(self, payload: dict):
        try:
            return next(self.client.list_entries(order_by=DESCENDING,
                                                 filter_=f'textPayload:({dictionary_md5_hash(payload)})')).labels[
                'execution_id']
        except StopIteration:
            raise ValueError(f'Cannot locate execution id for payload = {payload}!')

    def _get_last_log(self):
        try:
            return next(
                self.client.list_entries(order_by=DESCENDING,
                                         filter_=f'labels.execution_id:{self.execution_id}')).payload
        except StopIteration:
            raise ValueError(f'Cannot obtain last log for execution_id = {self.execution_id}!')

    def assert_finished_ok(self):
        assert self._get_last_log().endswith("finished with status: 'ok'")

    def assert_finished_with_crash(self):
        assert self._get_last_log().endswith("finished with status: 'crash'")


def save_training_data_with_predictions(image_filename: str, df_train: pd.DataFrame, df_predict: pd.DataFrame,
                                        threshold: int = 99999) -> None:
    """

    :param image_filename: filename where to save image.
    :param df_train: Dataframe of training data.
    :param df_predict: Dataframe of predictions.
    :param threshold: Threshold to cut out outliers (by default is set to arbitrarily large number).
    :return: None

    It plots training data with predictions from model which was trained on this data.
    It saves this plot to provided file path.
    """
    df_train_delta = df_train.diff()
    df_predict_delta = df_predict
    df_train_delta = df_train_delta[(df_train_delta < threshold) & (df_train_delta > -threshold)]
    df_predict_delta = df_predict_delta[(df_predict_delta < threshold) & (df_predict_delta > -threshold)]
    plt.plot(df_train_delta)
    plt.plot(df_predict_delta)
    plt.savefig(image_filename)
    logging.info(f"Image saved to '{image_filename}'")
    plt.close()


def download_file_from_bucket(bucket_name: Optional[str], source_file_name: str, destination_file_name: str) -> None:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_file_name)
    logging.info(f'Downloading {source_file_name} to {destination_file_name}...')
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    logging.info(f'Blob {source_file_name} downloaded to {destination_file_name}.')


def load_from_df_to_bigquery(table: str, dataset: str, project: str, df: pd.DataFrame,
                             schema_path: str = ELECTRICAL_MEASUREMENTS_SCHEMA_PATH) -> None:
    client = bigquery.Client()
    table_id = f'{project}.{dataset}.{table}'
    with open(schema_path) as json_file:
        schema: list = json.load(json_file)

    job_config = bigquery.LoadJobConfig(
        schema=[col for col in schema if col['name'] in df.columns],
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field='timestamp',
            require_partition_filter=True, ),
        clustering_fields=['meter_id', 'gateway_id', 'source_type'],
        write_disposition='WRITE_APPEND')
    logging.info(f'Loading dataframe to BigQuery table: {table_id}...')
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    logging.info(f'Loaded dataframe to {table_id}')


def save_actual_training_data_with_predictions(image_filename: str, df_train: pd.DataFrame, df_predict: pd.DataFrame,
                                               df_real: pd.DataFrame = None,
                                               title: str = None, threshold: int = 99999):
    """

    It plots training data (which was actually used by ML API) with predictions from model which was trained on this
    data. It saves this plot to provided file path.

    """
    df_train = df_train[(df_train < threshold) & (df_train > -threshold)]
    if df_real is not None:
        df_real = df_real[(df_real < threshold) & (df_real > -threshold)]
        plt.plot(df_real, label='Real data')
    plt.plot(df_train, label='Training data')
    plt.plot(df_predict, label='Prediction')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(image_filename)
    logging.info(f"Image saved to '{image_filename}'")
    plt.close()
