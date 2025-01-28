import logging
from typing import Tuple, Callable

import pytest
import requests

import pandas as pd
from common.controllers.ml_controller import MLController
from common.controllers.nodes_controller import NodesController
from tests.conftest import Node, Meter
from tests.end_to_ends import nodes_setup
from tests.end_to_ends.scenarios import SCENARIOS
from tests.end_to_ends.scenarios_handlers import TelemetryDataScenarioHandler, WeatherForecastsDataScenarioHandler
from tests.end_to_ends.telemetry_data_scenarios import TELEMETRY_DATA_SCENARIOS
from tests.end_to_ends.weather_forecasts_data_scenarios import WEATHER_FORECASTS_DATA_SCENARIOS
from tests.helpers import assert_task_is_scheduled, assert_training_response, \
    MLModelsStorageController, PredictionsParser, save_actual_training_data_with_predictions


class TestBase:
    storage_controller = MLModelsStorageController
    telemetry_data_scenario_handler = TelemetryDataScenarioHandler(scenarios=SCENARIOS,
                                                                   data_scenarios=TELEMETRY_DATA_SCENARIOS)
    weather_measurements_data_scenario_handler = WeatherForecastsDataScenarioHandler(
        scenarios=SCENARIOS,
        data_scenarios=WEATHER_FORECASTS_DATA_SCENARIOS)

    def setup_method(self, method):

        """Executed before each scenario. It setup data in BigQuery and SQL (raw SQL scripts - soon to be replaced
        with REST API calls) for the given scenario."""

        logging.info(f'########## SETUP METHOD START ({method.__name__}) ##########')

        self.ml_controller = MLController()

        try:
            self.scenario_name = self._item.callspec.getparam('scenario_name')
        except (ValueError, AttributeError):
            raise AttributeError('You need to parameterize your test with "scenario_name" variable.')
        logging.info(f'########## SCENARIO: {self.scenario_name} ##########')
        self.scenario = self.telemetry_data_scenario_handler.get_loaded_scenario(self.scenario_name)
        self.telemetry_data_scenario_handler.setup_bigquery_for_test_case(self.scenario_name)
        self.weather_measurements_data_scenario_handler.setup_bigquery_for_test_case(self.scenario_name)
        logging.info(f'########## SETUP METHOD END ({method.__name__}) ##########')

    @pytest.fixture(scope='function')
    def nodes_setup(self, set_node: Tuple[Callable[[dict], Node], Callable[[dict], Meter]], pa_api: requests.Session):

        """Executed before each scenario. It loads nodes setup for the given scenario to SQL database
        (using REST API requests). It also creates train body."""
        nodes_setup_scenario = getattr(nodes_setup, self.scenario['nodes_setup_scenario'])()
        nodes_controller = NodesController()

        if (location_id := nodes_setup_scenario.get('node', {}).get('location', {}).get('id')) and (
            location_node_json := pa_api.get(f'{nodes_controller.url}/nodes?location={location_id}').json()):
            n = Node(**location_node_json[0])
            m = [Meter(**x) for x in pa_api.get(f'{nodes_controller.url}/nodes/{n.id}/meters').json()]
            logging.warning(f'Using previously loaded node {n.id} for location {location_id}.')
        else:
            n = set_node[0](nodes_setup_scenario['node'])
            m = [set_node[1](x) for x in nodes_setup_scenario['meters']]
        logging.info(f'Testing with {n} and {m} meters')
        self.current_node_id = n.id

    @pytest.fixture(scope='function')
    def nodes_with_request_body_setup(self, nodes_setup):
        self.train_body = self.telemetry_data_scenario_handler.get_create_and_train_request_body(
            self.scenario_name,
            node_id=self.current_node_id,
            debug=True)
        self.predict_body = self.telemetry_data_scenario_handler.get_predict_request_body(
            self.scenario_name)
        if 'update_request_body' in self.scenario:
            self.update_body = self.telemetry_data_scenario_handler.get_update_request_body(self.scenario_name,
                                                                                            debug=True)
            self.predict_body_after_update = self.telemetry_data_scenario_handler.get_predict_request_body_after_update(
                self.scenario_name)

    def create_and_train_model(self, timeout=15 * 60, refresh_time=5) -> Tuple[str, float]:
        response = self.ml_controller.create_and_train_model(self.train_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=timeout, refresh_time=refresh_time)
        return task_id, response

    def update_model(self, model_id:str, timeout=15 * 60, refresh_time=5) -> Tuple[str, float]:
        response = self.ml_controller.update_model(model_id=model_id, request_body=self.update_body)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=timeout, refresh_time=refresh_time)
        return task_id, response

    def make_predictions_using_created_model(self, model_id: str, after_update=False) -> Tuple[str, requests.Response]:
        response = self.ml_controller.predict(model_id,
                                              self.predict_body if not after_update else self.predict_body_after_update)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=5 * 60, refresh_time=5)
        return task_id, response

    def evaluate_model_predictions(self, prediction_task_id: str,
                                   timeout=15 * 60, refresh_time=5) -> Tuple[str, requests.Response]:
        response = self.ml_controller.evaluate_model_predictions(prediction_task_id)
        task_id = assert_task_is_scheduled(response)
        response = self.ml_controller.wait_for_task(task_id, timeout=timeout, refresh_time=refresh_time)
        return task_id, response

    def save_prediction_plot(self, filename: str, model_id: str, predictions_df: pd.DataFrame,
                             real_data_df: pd.DataFrame = None, title: str = None):
        df_post_endog = self.storage_controller.get_ml_model_intermediate_data_to_df(model_id, {'moment': 'post',
                                                                                                'target': 'endog'})
        if not title:
            title = f'Predictions for algorithm {self.train_body["algorithm"]} for {self.scenario["endogenous"]}'
        save_actual_training_data_with_predictions(image_filename=filename,
                                                   df_train=df_post_endog[self.scenario['endogenous']],
                                                   df_predict=predictions_df,
                                                   df_real=real_data_df,
                                                   title=title)
