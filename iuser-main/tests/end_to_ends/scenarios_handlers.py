import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from google.cloud import bigquery
from common import config
from tests.helpers import TEMPORARY_CSV_DIR, download_file_from_bucket, load_from_df_to_bigquery

ENDOGENOUS_COLUMN_FORMAT = '{endogenous}-{dataset}.{table}.{column}-meter={meter_id}-gw={gateway_id}'


class BaseDataScenarioHandler:
    data_scenario_key = None

    def __init__(self, scenarios: Dict[str, Dict[str, Any]], data_scenarios: Dict[str, Dict[str, Any]]):
        self._load_scenarios(scenarios)
        self._load_data_scenarios(data_scenarios)
        self.bigquery_client = bigquery.Client()

    def setup_bigquery_for_test_case(self, scenario_name: str) -> None:
        raise NotImplementedError('This function can be used in derived class only!')

    def get_loaded_scenario(self, scenario_name: str) -> Dict[str, Any]:
        try:
            scenario = self.loaded_scenarios[scenario_name]
        except KeyError:
            raise AttributeError(f'Scenario {scenario_name} is not in your scenarios list!.'
                                 f' Please add this scenario to scenarios.py file.')
        return scenario

    def _load_data_scenarios(self, data_scenarios: Dict[str, Dict[str, Any]]) -> None:
        self.loaded_data_scenarios = data_scenarios

    def _load_scenarios(self, scenarios: Dict[str, Dict[str, Any]]) -> None:
        self.loaded_scenarios = scenarios

    def _get_loaded_data_scenario(self, data_scenario_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if data_scenario_name is None:
            return
        try:
            data_scenario = self.loaded_data_scenarios[data_scenario_name]
        except KeyError:
            raise AttributeError(f'Data scenario {data_scenario_name} is not in your data scenarios list!'
                                 f' Please add to {self.data_scenario_key} file.')
        return data_scenario

    def _get_loaded_data_scenario_from_scenario_name(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        if self.data_scenario_key:
            try:
                data_scenario_name = self.get_loaded_scenario(scenario_name)[self.data_scenario_key]
                return self._get_loaded_data_scenario(data_scenario_name)
            except KeyError:
                raise AttributeError(f'You have not provided any data scenario for scenario: {scenario_name}.'
                                     f' Expected {self.data_scenario_key} key in scenario.')
        else:
            raise NotImplementedError('This function can be used in derived class only!')

    def _get_table_id(self, scenario_name: str) -> str:
        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        return f'{data_scenario["project_id"]}.{data_scenario["dataset"]}.{data_scenario["table"]}'

    def _execute_bq_query(self, query):
        logging.info(f'Executing BigQuery query: {query}.')
        job = self.bigquery_client.query(query)
        job.result()
        logging.info('Successfully executed query.')

    @staticmethod
    def _get_destination_filename(filename):
        return os.path.join(TEMPORARY_CSV_DIR, filename)

    def _append_df_to_bq_table(self, df: pd.DataFrame, scenario_name: str):
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
        )

        table_id = self._get_table_id(scenario_name)
        logging.info(f'Loading data to BigQuery table: {table_id}')
        job = self.bigquery_client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        logging.info('Data successfully loaded.')


class WeatherForecastsDataScenarioHandler(BaseDataScenarioHandler):
    data_scenario_key = 'weather_forecasts_data_scenario'

    def __init__(self, scenarios: dict, data_scenarios: dict):
        super().__init__(scenarios, data_scenarios)

    def setup_bigquery_for_test_case(self, scenario_name: str) -> None:
        if not self._get_loaded_data_scenario_from_scenario_name(scenario_name):
            logging.warning(f'You have not provided any need for weather forecasts for scenario: {scenario_name}. '
                            'Skipping loading data.')
            return
        self._delete_current_scenario_data_from_bigquery(scenario_name)
        self._load_current_scenario_data_to_bigquery(scenario_name)

    def _delete_current_scenario_data_from_bigquery(self, scenario_name: str):
        table_id = self._get_table_id(scenario_name)
        location = self._get_location(scenario_name)
        query = f'DELETE FROM {table_id} WHERE location = "{location}"'
        self._execute_bq_query(query)

    def _load_current_scenario_data_to_bigquery(self, scenario_name: str):
        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        source_file_name = data_scenario['source_csv']
        destination_file_name = self._get_destination_filename(source_file_name)
        download_file_from_bucket(bucket_name=data_scenario['source_bucket'],
                                  source_file_name=source_file_name,
                                  destination_file_name=destination_file_name)
        df = pd.read_csv(destination_file_name)
        df['period_end'] = pd.to_datetime(df['period_end'])
        df['day_from'] = pd.to_datetime(df['day_from'])
        self._append_df_to_bq_table(df, scenario_name)

    def _get_location(self, scenario_name: str) -> str:
        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        return data_scenario['location']


class TelemetryDataScenarioHandler(BaseDataScenarioHandler):
    data_scenario_key = 'telemetry_data_scenario'
    fetched_data_scenarios = {}

    def __init__(self, scenarios: dict, data_scenarios: dict):
        super().__init__(scenarios, data_scenarios)

    def setup_bigquery_for_test_case(self, scenario_name: str) -> None:
        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        if self._check_if_bigquery_table_has_test_data(scenario_name):
            logging.info(
                f'Data is already loaded into BigQuery table: {data_scenario["dataset"]}.{data_scenario["table"]}')
            return
        logging.info('Data is not present in BigQuery')
        self._download_change_and_load_data_to_bigquery(scenario_name)

    def get_endogenous_pre_transform_column_names(self, scenario_name: str) -> List[str]:
        scenario = self.get_loaded_scenario(scenario_name)
        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        gateway_id = next(iter(data_scenario['gateway_id'].values()))
        scenario_config = scenario['config']
        result = []
        for operation, device_config in scenario_config.items():
            for meter_id, columns in device_config.items():
                for column in columns:
                    df_pre_transform_column_name = ENDOGENOUS_COLUMN_FORMAT.format(
                        endogenous=scenario['endogenous'],
                        dataset=config.BIGQUERY_DATASET_NAME,
                        table=config.BIGQUERY_TABLE_ELECTRICAL_MEASUREMENTS,
                        column=column, meter_id=meter_id,
                        gateway_id=gateway_id)
                    result.append(df_pre_transform_column_name)
        return result

    def get_create_and_train_request_body(self, scenario_name: str, node_id: int, algorithm: str = None,
                                          horizon: str = None, debug: bool = False) -> dict:
        scenario = self.get_loaded_scenario(scenario_name)
        result = scenario['create_and_train_request_body']
        if horizon is not None:
            result['horizon'] = horizon
        if algorithm is not None:
            result['algorithm'] = algorithm
        result['debug'] = debug
        result['training_data']['endogenous'] = scenario['endogenous']
        result['training_data']['node'] = node_id
        return result

    def get_predict_request_body(self, scenario_name: str, prediction_horizon: str = None,
                                 overrided_node_id: str = None) -> dict:
        scenario = self.get_loaded_scenario(scenario_name)
        result = scenario['predict_request_body']
        if result['prediction_horizon'] is None:
            result['prediction_horizon'] = prediction_horizon
        if overrided_node_id:
            result['override_exogenous_data'] = {
                'node': overrided_node_id
            }
        return result

    def get_update_request_body(self, scenario_name: str, debug: bool = False) -> dict:
        scenario = self.get_loaded_scenario(scenario_name)
        result = scenario['update_request_body']
        result['debug'] = debug
        return result

    def get_predict_request_body_after_update(self, scenario_name: str, prediction_horizon: str = None) -> dict:
        scenario = self.get_loaded_scenario(scenario_name)
        result = self.get_predict_request_body(scenario_name, prediction_horizon).copy()
        result['prediction_time_start'] = scenario['update_request_body']['time_range_end']
        return result

    def _check_if_bigquery_table_has_test_data(self, scenario_name) -> bool:

        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        table = data_scenario['table']
        dataset = data_scenario['dataset']
        project = data_scenario['project_id']
        validation_data = data_scenario['validation_data']
        logging.info(f'Checking if validation data {validation_data} is already loaded into BigQuery...')
        client = bigquery.Client()
        for data in validation_data:
            query = f'''SELECT count(*) FROM {project}.{dataset}.{table}
                        WHERE DATE(timestamp) = "{data["date"]}"
                        AND meter_id = "{data["meter_id"]}" AND gateway_id = "{data["gateway_id"]}"'''
            count_rows = int(next(client.query(query).result().to_dataframe_iterable()).values[0][0])
            if count_rows != data['number_of_rows']:
                if count_rows:
                    raise ValueError('Something is wrong, mismatch of BigQuery records, stopped loading data.')
                return False
        return True

    def _download_change_and_load_data_to_bigquery(self, scenario_name: str) -> None:

        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        source_filename = data_scenario['source_csv']
        source_bucket = data_scenario['source_bucket']
        table = data_scenario['table']
        dataset = data_scenario['dataset']
        project = data_scenario['project_id']

        saved_csv_file_path = self._get_destination_filename(source_filename)

        if not os.path.exists(source_filename):
            download_file_from_bucket(source_bucket, source_filename, saved_csv_file_path)
        else:
            logging.info('File already downloaded')

        df = pd.read_csv(saved_csv_file_path)
        self._change_df_columns_for_test_scenario(df, scenario_name)
        load_from_df_to_bigquery(table, dataset, project, df)

    def _change_df_columns_for_test_scenario(self, df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
        data_scenario = self._get_loaded_data_scenario_from_scenario_name(scenario_name)
        df.dropna(subset=['timestamp', 'meter_id', 'gateway_id', 'source_type'], inplace=True)
        for bq_column_name, scenario_column_name in data_scenario['columns_to_change'].items():
            df[bq_column_name].replace(data_scenario[scenario_column_name], inplace=True)
            logging.info(f'Changing {bq_column_name} values: {data_scenario[scenario_column_name]}')

        # change values of column 'source_type' to 'test_scenario'
        logging.info(f'Changing source_type values to test_scenario')
        df['source_type'] = 'test_scenario'
        logging.info(f'Changing timestamp values to datetime type')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
