from tests.end_to_ends.tests_inputs.train_request_bodies import SUMMER_MODEL

SCENARIOS = {
    'scenario_1_energy_usage':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_1',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': None,
            'config':
                {
                    'plus':
                        {
                            'meter_1001': ['ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
                            'meter_1002': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3']
                        },
                    'minus':
                        {
                            'meter_1001': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3']
                        }

                },
            'create_and_train_request_body':
                {'algorithm': None,
                 'optimization_metric': 'r2',
                 'horizon': None,
                 'training_data': {
                     'skip_anomalies': True,
                     'time_range_start': '2020-09-01T00:00:00Z',
                     'time_range_end': '2020-09-03T00:00:00Z'},
                 'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': None}
        },
    'scenario_1_energy_production':
        {
            'endogenous': 'real_energy_prod',
            'nodes_setup_scenario': 'scenario_1',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': None,
            'config':
                {
                    'plus':
                        {
                            'meter_1002': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3']
                        }
                },
            'create_and_train_request_body':
                {'algorithm': 'boosted_trees',
                 'optimization_metric': 'r2',
                 'horizon': '1d',
                 'training_data': {
                     'skip_anomalies': True,
                     'time_range_start': '2020-09-02T23:00:00Z',
                     'time_range_end': '2020-09-03T00:00:00Z'},
                 'target_frequency': '60s'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': '60s'},
            'update_request_body': {'time_range_end': '2020-09-03T01:00:00Z'}
        },
    'scenario_derived_features_are_used_along_with_exogenous':
        {
            'endogenous': 'real_energy_prod',
            'nodes_setup_scenario': 'scenario_2',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': 'weather_scenario_1',
            'config':
                {
                    'plus':
                        {
                            'meter_1002': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3']
                        },
                },
            'create_and_train_request_body':
                {
                    'algorithm': 'linear_regression',
                    'optimization_metric': 'r2',
                    'horizon': '1w',
                    'training_data': {
                        "exogenous": [
                            "ghi",
                            "dni"
                        ],
                        'skip_anomalies': True,
                        'time_range_start': '2020-08-13T00:00:00Z',
                        'time_range_end': '2020-09-03T00:00:00Z'},
                    'target_frequency': '1h',
                    "derived_features": {
                        "ghi_rolling_mean": {
                            "transform": "rolling",
                            "input": "ghi",
                            "params": {
                                "window_function": "mean",
                                "window": "30m"
                            }
                        }
                    }
                },
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': '1w'}
        },
    'scenario_low_frequency_data_is_properly_resampled':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_3',
            'telemetry_data_scenario': 'telemetry_scenario_low_frequency',
            'weather_forecasts_data_scenario': None,
            'config':
                {
                    'plus':
                        {
                            'meter_1003': ['ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
                        }

                },
            'create_and_train_request_body':
                {
                    'algorithm': 'linear_regression',
                    'optimization_metric': 'r2',
                    'horizon': '1w',
                    'training_data': {
                        'skip_anomalies': True,
                        'time_range_start': '2020-08-13T00:00:00Z',
                        'time_range_end': '2020-09-03T00:00:00Z'},
                    'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': None}
        },
    'scenario_check_endogenous_pre_transform_data':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_1',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': None,
            'config':
                {
                    'plus':
                        {
                            'meter_1001': ['ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
                            'meter_1002': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3']
                        },
                    'minus':
                        {
                            'meter_1001': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3']
                        }

                },
            'create_and_train_request_body':
                {
                    'algorithm': 'linear_regression',
                    'optimization_metric': 'r2',
                    'horizon': '1w',
                    'training_data': {
                        'skip_anomalies': True,
                        'time_range_start': '2020-08-13T00:00:00Z',
                        'time_range_end': '2020-09-03T00:00:00Z'},
                    'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': '1w'}
        },

    'scenario_demo_day_train':
        {
            'endogenous': 'real_energy_prod',
            'nodes_setup_scenario': 'scenario_demo_day',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': 'weather_demo_day_scenario',
            'create_and_train_request_body':
                SUMMER_MODEL,
            'predict_request_body': {'prediction_time_start': SUMMER_MODEL['training_data']['time_range_end'],
                                     'prediction_horizon': '1w'},
        },
    'scenario_demo_day_train_with_update':
        {
            'endogenous': 'real_energy_prod',
            'nodes_setup_scenario': 'scenario_demo_day',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': 'weather_demo_day_scenario',
            'create_and_train_request_body': {
                'algorithm': 'boosted_trees',
                'optimization_metric': 'r2',
                'horizon': '1w',
                'training_data': {
                    "exogenous": [
                        "ghi",
                        "dni"
                    ],
                    'skip_anomalies': True,
                    'time_range_start': '2020-07-01T00:00:00Z',
                    'time_range_end': '2020-08-23T00:00:00Z'},
                'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-08-23T00:00:00Z', 'prediction_horizon': '1w'},
            'update_request_body': {'time_range_end': '2020-09-01T00:00:00Z'}
        },
    'scenario_demo_day_train_on_long_period':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_demo_day',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': 'weather_demo_day_scenario',
            'create_and_train_request_body': {
                'algorithm': 'boosted_trees',
                'optimization_metric': 'r2',
                'horizon': '1w',
                'training_data': {
                    "exogenous": [
                        "air_temp",
                    ],
                    'skip_anomalies': True,
                    'time_range_start': '2020-08-01T00:00:00Z',
                    'time_range_end': '2020-10-23T00:00:00Z'},
                'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-10-23T00:00:00Z', 'prediction_horizon': '1w'},
        },
    'behave_setup_scenario_1':
        {
            'nodes_setup_scenario': 'behave_nodes_setup_1',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': None,
        },
    'scenario_duplicated_data':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_duplicates',
            'telemetry_data_scenario': 'telemetry_scenario_duplicates',
            'weather_forecasts_data_scenario': None,
            'create_and_train_request_body':
                {'algorithm': 'boosted_trees',
                 'optimization_metric': 'r2',
                 'horizon': '1d',
                 'training_data': {
                     'skip_anomalies': True,
                     'time_range_start': '2020-09-01T00:00:00Z',
                     'time_range_end': '2020-09-03T00:00:00Z'},
                 'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': '1d'}
        },
    'scenario_advanced_duplicated_data':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_advanced_duplicates',
            'telemetry_data_scenario': 'telemetry_scenario_advanced_duplicates',
            'weather_forecasts_data_scenario': None,
            'create_and_train_request_body':
                {'algorithm': 'boosted_trees',
                 'optimization_metric': 'r2',
                 'horizon': '1d',
                 'training_data': {
                     'skip_anomalies': True,
                     'time_range_start': '2020-10-22T00:00:00Z',
                     'time_range_end': '2020-10-26T00:00:00Z'},
                 'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-10-26T00:00:00Z', 'prediction_horizon': '1d'}
        },
    'scenario_evaluate_prediction':
        {
            'endogenous': 'real_energy_prod',
            'nodes_setup_scenario': 'scenario_1',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': None,
            'create_and_train_request_body':
                {'algorithm': 'boosted_trees',
                 'optimization_metric': 'r2',
                 'horizon': '1d',
                 'training_data': {
                     'skip_anomalies': True,
                     'time_range_start': '2020-09-01T00:00:00Z',
                     'time_range_end': '2020-09-03T00:00:00Z'},
                 'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z',
                                     'prediction_horizon': '1d'}
        },
    'scenario_prediciton_cache':
        {
            'endogenous': 'energy_usage',
            'nodes_setup_scenario': 'scenario_1',
            'telemetry_data_scenario': 'telemetry_scenario_1',
            'weather_forecasts_data_scenario': None,
            'create_and_train_request_body':
                {'algorithm': 'linear_regression',
                 'optimization_metric': 'r2',
                 'horizon': '1d',
                 'training_data': {
                     'skip_anomalies': True,
                     'time_range_start': '2020-09-01T00:00:00Z',
                     'time_range_end': '2020-09-03T00:00:00Z'},
                 'target_frequency': '1h'},
            'predict_request_body': {'prediction_time_start': '2020-09-03T00:00:00Z', 'prediction_horizon': '7d',
                                     'use_cache': True}
        },

    'scenario_presentation': {
        'endogenous': 'energy_usage',
        'telemetry_data_scenario': 'presentation_data',
        'weather_forecasts_data_scenario': 'presentation_weather_data',
        'nodes_setup_scenario': 'presentation_nodes_setup',
        'create_and_train_request_body': {
            'algorithm': 'linear_regression',
            'optimization_metric': 'r2',
            'horizon': '1d',
            'training_data': {
                'skip_anomalies': True,
                'time_range_start': '2020-08-26T00:00:00Z',
                'time_range_end': '2020-09-05T00:00:00Z'},
            'target_frequency': '1h'},
        'predict_request_body': {'prediction_time_start': '2020-09-05T00:00:00Z', 'prediction_horizon': '1d'}
    }
}
