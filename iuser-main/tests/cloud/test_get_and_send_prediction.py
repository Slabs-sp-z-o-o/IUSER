import json
import time

import pytest

from common.logger import get_logger
from tests.end_to_ends.base import TestBase
from tests.helpers import assert_task_is_scheduled, assert_training_response, CloudFunctionMonitoring

logger = get_logger(__name__)


class TestGetAndSendPrediction(TestBase):

    def _construct_model_info(self, model_id: str) -> dict:
        return {'model_id': model_id, 'model_series': self.scenario['endogenous'],
                'prediction_time_start': self.predict_body['prediction_time_start']}

    @pytest.mark.usefixtures('nodes_with_request_body_setup')
    @pytest.mark.parametrize('scenario_name', ('scenario_1_energy_usage',))
    def test_get_and_send_prediction(self, scenario_name, get_and_send_prediction_publish):
        self.train_body['algorithm'] = 'linear_regression'
        self.train_body['horizon'] = '1d'
        response = self.ml_controller.create_and_train_model(self.train_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=15 * 60)
        model_id, r2 = assert_training_response(response)

        # Make sure model and its metadata are stored in Cloud Storage
        assert self.storage_controller.check_ml_model_exists_in_cloud_storage(model_id)
        assert self.storage_controller.get_ml_model_metadata_content(
            model_id)  # TODO check if metadata is in valid format

        model_info = self._construct_model_info(model_id)
        get_and_send_prediction_publish(json.dumps(model_info).encode('utf-8'))
        logger.info('Waiting 80s for get_and_send_prediction Cloud Function to complete...')
        time.sleep(80)
        cf_monitoring = CloudFunctionMonitoring(payload=model_info)
        cf_monitoring.assert_finished_ok()

    @pytest.mark.usefixtures('nodes_with_request_body_setup')
    @pytest.mark.parametrize('scenario_name', ('scenario_1_energy_usage',))
    def test_get_and_send_prediction_with_invalid_model_id(self, scenario_name, get_and_send_prediction_publish):
        model_id = 'some-invalid-model-id'
        model_info = self._construct_model_info(model_id)
        get_and_send_prediction_publish(json.dumps(model_info).encode('utf-8'))
        logger.info('Waiting 50s for get_and_send_prediction Cloud Function to complete...')
        time.sleep(50)
        cf_monitoring = CloudFunctionMonitoring(payload=model_info)
        cf_monitoring.assert_finished_with_crash()
