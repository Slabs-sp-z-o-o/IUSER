import time

import pytest

from tests.end_to_ends.base import TestBase
from tests.helpers import assert_cached_prediction, assert_predict_response, assert_prediction_evaluation_response, \
    assert_response_failure_json, assert_task_is_scheduled, assert_training_response, PredictionsParser, \
    save_actual_training_data_with_predictions


@pytest.mark.usefixtures('nodes_with_request_body_setup')
class TestEndToEnd(TestBase):

    @pytest.mark.parametrize('scenario_name', ('scenario_1_energy_usage',))
    def test_common_scenario(self, image_prefix: str, scenario_name: str, algorithm: str, horizon: str,
                             prediction_horizon: str):
        self.train_body['algorithm'] = algorithm
        self.train_body['horizon'] = '1d'

        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        self.predict_body = self.telemetry_data_scenario_handler.get_predict_request_body(scenario_name,
                                                                                          prediction_horizon)
        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        self.save_prediction_plot(filename=f'{image_prefix}_{r2}.png',
                                  model_id=model_id,
                                  predictions_df=predictions_df)

    @pytest.mark.parametrize('scenario_name', ('scenario_1_energy_production',))
    def test_common_scenario_with_update(self, image_prefix: str, scenario_name: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        self.save_prediction_plot(filename=f'{image_prefix}_{r2}.png',
                                  model_id=model_id,
                                  predictions_df=predictions_df)

        _, response = self.update_model(model_id=model_id)
        model_id, r2 = assert_training_response(response)

        task_id, response = self.make_predictions_using_created_model(model_id=model_id, after_update=True)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        self.save_prediction_plot(filename=f'{image_prefix}_{r2}.png',
                                  model_id=model_id,
                                  predictions_df=predictions_df)

    @pytest.mark.parametrize('scenario_name', ['scenario_derived_features_are_used_along_with_exogenous'])
    def test_derived_features_scenario(self, image_prefix: str, scenario_name: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        df_post_exog = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                               'target': 'exog'})

        expected_exog_columns_set = (set(self.train_body['training_data']['exogenous']) |
                                     set(self.train_body['derived_features']))

        assert set(df_post_exog.columns) >= expected_exog_columns_set

        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        self.save_prediction_plot(filename=f'{image_prefix}_{r2}.png',
                                  model_id=model_id,
                                  predictions_df=predictions_df)

    @pytest.mark.parametrize('scenario_name', ['scenario_low_frequency_data_is_properly_resampled'])
    def test_resampling_on_sparse_data_scenario(self, image_prefix: str, scenario_name: str, prediction_horizon: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        df_post_endo = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                               'target': 'endog'})

        low_freq_part = df_post_endo['2020-08-24':'2020-08-29'].reset_index()

        assert all(
            x in low_freq_part[low_freq_part.energy_usage != 0].index for x in
            range(len(low_freq_part)) if x % 3 == 0)  # we expect non-zero values in a period of 3

        self.predict_body = self.telemetry_data_scenario_handler.get_predict_request_body(scenario_name,
                                                                                          prediction_horizon)
        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        self.save_prediction_plot(filename=f'{image_prefix}_{r2}.png',
                                  model_id=model_id,
                                  predictions_df=predictions_df)

    @pytest.mark.parametrize('scenario_name', ['scenario_check_endogenous_pre_transform_data'])
    def test_checking_endogenous_pre_transform_data(self, scenario_name: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        df_pre_endo = self.storage_controller.get_ml_model_intermediate_data_to_df(
            model_id,
            {'moment': 'pre',
             'target': 'endog'})
        assert set(df_pre_endo.columns) == set(
            self.telemetry_data_scenario_handler.get_endogenous_pre_transform_column_names(scenario_name))

    @pytest.mark.parametrize('scenario_name', ['scenario_duplicated_data'])
    def test_duplicated_data_scenario(self, scenario_name: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

    @pytest.mark.parametrize('scenario_name', ['scenario_advanced_duplicated_data'])
    def test_different_values_on_duplicated_timestamps_scenario(self, scenario_name: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

    @pytest.mark.parametrize('scenario_name', ['scenario_evaluate_prediction'])
    def test_evaluate_prediction(self, scenario_name: str, image_prefix):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        prediction_task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        time.sleep(5)
        assert_cached_prediction(task_id=prediction_task_id)

        _, response = self.evaluate_model_predictions(prediction_task_id=prediction_task_id)
        _, metrics = assert_prediction_evaluation_response(response, predictions_df=predictions_df)

        time.sleep(5)
        assert_cached_prediction(task_id=prediction_task_id, check_evaluation=True)

    @pytest.mark.parametrize('scenario_name', ['scenario_evaluate_prediction'])
    def test_evaluate_prediction_missing_real_data(self, scenario_name: str):
        # Last measurement is at 2021-03-03 15:15:58.735 UTC
        self.train_body['training_data']['time_range_start'] = '2021-01-28T00:00:00Z'
        self.train_body['training_data']['time_range_end'] = '2021-03-03T00:00:00Z'
        self.predict_body['prediction_time_start'] = '2021-03-03T00:00:00Z'

        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        task_id, response = self.evaluate_model_predictions(prediction_task_id=task_id)
        assert_response_failure_json(response, message='There is not enough real data in data '
                                                       'warehouse to properly evaluate prediction!')

    @pytest.mark.parametrize('scenario_name', ['scenario_evaluate_prediction'])
    def test_evaluate_prediction_missing_predictions(self, scenario_name: str):
        _, response = self.evaluate_model_predictions(prediction_task_id='this-task-id-does-not-exist')
        assert_response_failure_json(response, message='Task of id \"this-task-id-does-not-exist\" '
                                                       'is not present in Cloud Storage!')

    @pytest.mark.parametrize('scenario_name', ['scenario_evaluate_prediction'])
    def test_evaluate_prediction_task_id_does_not_reference_predictions(self, scenario_name: str):
        training_task_id, _ = self.create_and_train_model()
        _, response = self.evaluate_model_predictions(prediction_task_id=training_task_id)
        assert_response_failure_json(response, message=f'Task of id: {training_task_id} '
                                                       'does not contain predictions - cannot evaluate!')

    @pytest.mark.parametrize('scenario_name', ('scenario_prediciton_cache',))
    def test_prediction_cache(self, scenario_name: str):
        _, response = self.create_and_train_model()
        model_id, r2 = assert_training_response(response)

        first_task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        # Make second identical prediction but set 'use_cache' to False
        self.predict_body['use_cache'] = False
        second_task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        # Make sure task ids are different
        assert first_task_id != second_task_id

        # Make third identical prediction with 'use_cache' as true
        self.predict_body['use_cache'] = True
        third_task_id, response = self.make_predictions_using_created_model(model_id=model_id)
        _, predictions_df = assert_predict_response(response, provided_horizon=self.predict_body['prediction_horizon'])

        # Make sure task ids are the same
        assert first_task_id == third_task_id
