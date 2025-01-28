import pytest

from tests.end_to_ends.base import TestBase
from tests.helpers import assert_task_is_scheduled, assert_training_response, PredictionsParser, \
    save_actual_training_data_with_predictions


@pytest.mark.usefixtures('nodes_with_request_body_setup')
class TestEndToEndHardcoded(TestBase):

    @pytest.mark.parametrize('scenario_name', ['scenario_presentation'])
    def test_presentation(self, image_prefix: str, scenario_name: str):
        response = self.ml_controller.create_and_train_model(self.train_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=15 * 60)
        model_id, r2 = assert_training_response(response)

        df_post_endog = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                               'target': 'endog'})

        hardcoded_model_id_with_real_data = 'a2bb81e9-1e8c-4423-b75d-ca123e729097'
        real_data_df = self.storage_controller.get_ml_model_intermediate_data_to_df(hardcoded_model_id_with_real_data, {'moment': 'post',
                                                                                                'target': 'endog'})


        response = self.ml_controller.predict(model_id, self.predict_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=5 * 60)
        predictions_df = PredictionsParser.create_predictions_dataframe(response.json())

        save_actual_training_data_with_predictions(image_filename=f'{image_prefix}_{self.train_body["algorithm"]}.png',
                                                   df_train=df_post_endog[self.scenario['endogenous']],
                                                   df_real=real_data_df,
                                                   df_predict=predictions_df,
                                                   title=f'Predictions for algorithm {self.train_body["algorithm"]} '
                                                         f'for {self.scenario["endogenous"]}.')

    @pytest.mark.skip(reason='Skipping tests with hardcoded values.')
    @pytest.mark.parametrize('scenario_name', ['scenario_demo_day_train'])
    def test_training_demoday(self, image_prefix: str, scenario_name: str):
        response = self.ml_controller.create_and_train_model(self.train_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=15 * 60)
        model_id, r2 = assert_training_response(response)

        hardcoded_model_id_with_real_data = '0eff8e32-c9e0-40b8-992e-6e4b178aa7e0'
        df_post_exog = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                               'target': 'endog'})

        real_data_df = self.storage_controller.get_ml_model_intermediate_data_to_df(
            hardcoded_model_id_with_real_data,
            {'moment': 'post',
             'target': 'endog'})[
            self.scenario['endogenous']]



        response = self.ml_controller.predict(model_id, self.predict_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=5 * 60)
        predictions_df = PredictionsParser.create_predictions_dataframe(response.json())

        _, response = self.evaluate_model_predictions(prediction_task_id=task_id)

        selected_metric = 'rmse'

        metric_value = response.json()['result']['metrics'][selected_metric]
        # Save plots of predictions
        save_actual_training_data_with_predictions(image_filename=f'{image_prefix}_{self.train_body["algorithm"]}_{selected_metric}_{metric_value}.png',
                                                   df_train=df_post_exog[self.scenario['endogenous']],
                                                   df_predict=predictions_df,
                                                   df_real=real_data_df,
                                                   title=f'Predictions for algorithm {self.train_body["algorithm"]} '
                                                         f'for {self.scenario["endogenous"]}. {selected_metric.upper()} = {metric_value}')



    @pytest.mark.skip(reason='Skipping tests with hardcoded values.')
    @pytest.mark.parametrize('scenario_name', ['scenario_demo_day_train_with_update'])
    def test_training_with_update_demoday(self, image_prefix: str, scenario_name: str):
        response = self.ml_controller.create_and_train_model(self.train_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=15 * 60)
        model_id, r2 = assert_training_response(response)

        hardcoded_model_id_with_real_data = '0eff8e32-c9e0-40b8-992e-6e4b178aa7e0'
        df_post_exog = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                               'target': 'endog'})

        real_data_df = self.storage_controller.get_ml_model_intermediate_data_to_df(
            hardcoded_model_id_with_real_data,
            {'moment': 'post',
             'target': 'endog'})[
            self.scenario['endogenous']]

        response = self.ml_controller.predict(model_id, self.predict_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=5 * 60)
        predictions_df = PredictionsParser.create_predictions_dataframe(response.json())

        # Save plots of predictions
        save_actual_training_data_with_predictions(image_filename=f'{image_prefix}_{r2}_pre_update.png',
                                                   df_train=df_post_exog[self.scenario['endogenous']],
                                                   df_predict=predictions_df,
                                                   df_real=real_data_df,
                                                   title=f'Predictions for algorithm {self.train_body["algorithm"]} '
                                                         f'for {self.scenario["endogenous"]} before update. '
                                                         f'R2 = {r2}')

        response = self.ml_controller.update_model(model_id, self.update_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=15 * 60)
        model_id_after_update, r2_after_update = assert_training_response(response)

        df_post_exog_after_update = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id_after_update,
                                                                                                 {'moment': 'post',
                                                                                                  'target': 'endog'})

        response = self.ml_controller.predict(model_id_after_update, self.predict_body_after_update)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=5 * 60)
        predictions_df_after_update = PredictionsParser.create_predictions_dataframe(response.json())

        # Save plots of predictions after update
        save_actual_training_data_with_predictions(image_filename=f'{image_prefix}_{r2_after_update}_after_update.png',
                                                   df_train=df_post_exog_after_update[self.scenario['endogenous']],
                                                   df_predict=predictions_df_after_update,
                                                   df_real=real_data_df,
                                                   title=f'Predictions for algorithm {self.train_body["algorithm"]} '
                                                         f'for {self.scenario["endogenous"]} after update. '
                                                         f'R2 = {r2_after_update}')

    @pytest.mark.skip(reason='Skipping tests with hardcoded values.')
    @pytest.mark.parametrize('scenario_name', ['scenario_demo_day_train_on_long_period'])
    def test_training_on_long_period_demoday(self, image_prefix: str, scenario_name: str):
        model_id = 'cdd2522c-975c-475f-921c-5d0a219524ee'
        hardcoded_model_id_with_real_data = '71b9dfa7-e1d1-46f5-a292-9510ea037c00'

        df_post_exog = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                               'target': 'endog'})

        real_data_df = self.storage_controller.get_ml_model_intermediate_data_to_df(
            hardcoded_model_id_with_real_data,
            {'moment': 'post',
             'target': 'endog'}, delete_csv=False)[
            self.scenario['endogenous']]

        predict_body = self.telemetry_data_scenario_handler.get_predict_request_body(
            scenario_name,
            overrided_node_id=self.current_node_id)
        response = self.ml_controller.predict(model_id, predict_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=5 * 60)
        predictions_df = PredictionsParser.create_predictions_dataframe(response.json())

        # Save plots of predictions
        save_actual_training_data_with_predictions(image_filename=f'{image_prefix}.png',
                                                   df_train=df_post_exog[self.scenario['endogenous']],
                                                   df_predict=predictions_df,
                                                   df_real=real_data_df,
                                                   title=f'Predictions for algorithm {self.train_body["algorithm"]} '
                                                         f'for {self.scenario["endogenous"]}', threshold=5)

        # Save plots of predictions
        save_actual_training_data_with_predictions(image_filename=f'{image_prefix}_zoomed.png',
                                                   df_train=df_post_exog[self.scenario['endogenous']]['2020-10-10':],
                                                   df_predict=predictions_df,
                                                   df_real=real_data_df['2020-10-10':],
                                                   title=f'Predictions for algorithm {self.train_body["algorithm"]} '
                                                         f'for {self.scenario["endogenous"]}', threshold=5)
