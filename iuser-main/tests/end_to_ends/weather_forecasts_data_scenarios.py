from common import config

WEATHER_FORECASTS_DATA_SCENARIOS = {
    'weather_scenario_1': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_WEATHER_FORECASTS,
        'location': '10-001',
        'source_csv': 'zero_forecast_2020_08_13.csv',
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
    },
    'weather_demo_day_scenario': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_WEATHER_FORECASTS,
        'location': '10-002',
        'source_csv': 'mock_meteo_mogilany.csv',
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
    },
    'presentation_weather_data': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_WEATHER_FORECASTS,
        'location': '10-003',
        'source_csv': 'mock_meteo_mogilany.csv',
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
    }
}
