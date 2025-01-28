import json
import logging
import os
import time
from typing import List, Union

import pytest
from google.cloud import datastore
from google.cloud import storage

from common import config
from common.gcp.datastore import schema as ds

storage_client = storage.Client()
datastore_client = datastore.Client()

ML_MODELS_BUCKET_NAME = config.ML_MODELS_BUCKET_NAME
METADATA_BUCKET_NAME = config.ML_MODELS_METADATA_BUCKET_NAME
METADATA_FORMAT = 'model_{model_id}.json'
ML_MODELS_FORMAT = 'model_{model_id}'

EXAMPLE_DATA_PATH = 'tests/cloud/temporary_data'


def _save_metadata_to_json(metadata: dict) -> None:
    """Saves provided metadata as a local JSON file."""
    with open(os.path.join(EXAMPLE_DATA_PATH, METADATA_FORMAT.format(model_id=metadata['model_id'])), 'w') as outfile:
        json.dump(metadata, outfile)


def _save_model_locally(model_id: str) -> None:
    """Saves dummy (empty file) model as a local JSON file."""
    open(os.path.join(EXAMPLE_DATA_PATH, ML_MODELS_FORMAT.format(model_id=model_id)), 'a').close()


def _put_valid_metadata_test_file_to_cloud_storage(model_id: str) -> None:
    """Uploads a metadata file to the bucket."""

    filename = METADATA_FORMAT.format(model_id=model_id)
    _put_file_to_cloud_storage(filename, bucket_name=METADATA_BUCKET_NAME)


def _put_valid_ml_model_test_file_to_cloud_storage(model_id: str) -> None:
    """Uploads a ml model file to the bucket."""

    filename = ML_MODELS_FORMAT.format(model_id=model_id)
    _put_file_to_cloud_storage(filename, bucket_name=ML_MODELS_BUCKET_NAME)


def _put_file_to_cloud_storage(filename: str, bucket_name: str) -> None:
    """Helper function for putting file to Cloud Storage."""
    source_file_name = os.path.join(EXAMPLE_DATA_PATH, filename)
    destination_blob_name = filename
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logging.info(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def _get_ml_model_datastore_matching_records(model_id: str) -> List[ds.MLModelMetadata]:
    """Returns records from MLModelMetadata Datastore entity which match given model id."""
    query = datastore_client.query(kind="MLModelMetadata")
    query.add_filter("model_id", "=", model_id)
    results = list(query.fetch())
    return results


def _delete_json_test_metadata_file(model_id: str) -> None:
    """Cleans up local directory from unnecessary temporal ML metadata JSON file."""
    logging.info('removing json file from directory')
    os.remove(os.path.join(EXAMPLE_DATA_PATH, METADATA_FORMAT.format(model_id=model_id)))


def _delete_json_test_ml_model_file(model_id: str) -> None:
    """Cleans up local directory from unnecessary temporal ML model file."""
    logging.info('removing model file from directory')
    os.remove(os.path.join(EXAMPLE_DATA_PATH, ML_MODELS_FORMAT.format(model_id=model_id)))


def _delete_test_ml_model_metadata(model_id: str) -> None:
    """Deletes a blob from the bucket."""
    blob_name = METADATA_FORMAT.format(model_id=model_id)

    bucket = storage_client.bucket(METADATA_BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.delete()

    logging.info("Blob {} deleted.".format(blob_name))


def _check_ml_model_exists_in_cloud_storage(model_id: str) -> Union[None, storage.blob.Blob]:
    """Checks if model of selected ID is present in ML Models Bucket."""
    filename = ML_MODELS_FORMAT.format(model_id=model_id)
    bucket = storage_client.get_bucket(ML_MODELS_BUCKET_NAME)
    blob = bucket.get_blob(filename)
    return blob is not None


@pytest.mark.parametrize('models_metadata', [([{'model_id': 'd0a74f8a-ba1f-11eb-8529-0242ac130003'}])],
                         indirect=['models_metadata'])
def test_ml_models_metadata_integration(models_metadata: List[dict]):
    """
    In this test we upload to Cloud Storage ML metadata file + model file,
    and we check if datastore gets updated by Cloud Function.
    """
    sleep_time = 15  # seconds
    model_id = models_metadata[0]['model_id']
    _save_metadata_to_json(models_metadata[0])  # creates a metadata json file locally
    _save_model_locally(model_id=model_id)  # creates a ml model file locally
    _put_valid_metadata_test_file_to_cloud_storage(model_id)
    _put_valid_ml_model_test_file_to_cloud_storage(model_id)
    _delete_json_test_metadata_file(model_id)
    _delete_json_test_ml_model_file(model_id)
    logging.info(f'Sleeping for {sleep_time} seconds to give time for Cloud Function to work')
    time.sleep(sleep_time)
    # Checks if automatic function added an entity
    datastore_result_after_upload = _get_ml_model_datastore_matching_records(model_id)
    cloud_storage_ml_model_state_after_upload = _check_ml_model_exists_in_cloud_storage(model_id)

    # Deletes file from storage
    _delete_test_ml_model_metadata(model_id)
    logging.info(f'Sleeping for {sleep_time} seconds to give time for Cloud Function to work')
    time.sleep(sleep_time)
    # Checks if automatic function deleted an entity
    datastore_result_after_delete = _get_ml_model_datastore_matching_records(model_id)
    cloud_storage_ml_model_state_after_delete = _check_ml_model_exists_in_cloud_storage(model_id)
    assert len(datastore_result_after_upload) == 1
    assert datastore_result_after_upload[0]['endogenous'] == 'real_energy_prod'
    assert len(datastore_result_after_delete) == 0
    assert cloud_storage_ml_model_state_after_upload is True
    assert cloud_storage_ml_model_state_after_delete is False


@pytest.mark.parametrize('models_metadata', [([{'model_id': 'd0a74f8a-ba1f-11eb-8529-0242ac130003'}])],
                         indirect=['models_metadata'])
def test_ml_models_metadata_integration_no_prior_model_exists(models_metadata):
    """
    In this test we upload to Cloud Storage ML metadata file (without file with model),
    and we check if datastore gets updated by Cloud Function.
    """
    sleep_time = 15  # seconds
    model_id = models_metadata[0]['model_id']
    _save_metadata_to_json(models_metadata[0])  # creates a metadata json file locally
    _put_valid_metadata_test_file_to_cloud_storage(model_id)
    _delete_json_test_metadata_file(model_id)
    logging.info(f'Sleeping for {sleep_time} seconds to give time for Cloud Function to work')
    time.sleep(sleep_time)
    # Checks if automatic function added an entity
    datastore_result_after_upload = _get_ml_model_datastore_matching_records(model_id)
    cloud_storage_ml_model_state_after_upload = _check_ml_model_exists_in_cloud_storage(model_id)

    # Deletes file from storage
    _delete_test_ml_model_metadata(model_id)
    logging.info(f'Sleeping for {sleep_time} seconds to give time for Cloud Function to work')
    time.sleep(sleep_time)
    # Checks if automatic function deleted an entity
    datastore_result_after_delete = _get_ml_model_datastore_matching_records(model_id)
    cloud_storage_ml_model_state_after_delete = _check_ml_model_exists_in_cloud_storage(model_id)
    assert len(datastore_result_after_upload) == 1
    assert datastore_result_after_upload[0]['endogenous'] == 'real_energy_prod'
    assert len(datastore_result_after_delete) == 0
    assert cloud_storage_ml_model_state_after_upload is False
    assert cloud_storage_ml_model_state_after_delete is False
