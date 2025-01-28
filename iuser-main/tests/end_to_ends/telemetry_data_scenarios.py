from common import config

TELEMETRY_DATA_SCENARIOS = {
    'telemetry_scenario_1': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_ELECTRICAL_MEASUREMENTS,
        'columns': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3', 'ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
        'columns_to_change': {'meter_id': 'meter_ids', 'gateway_id': 'gateway_id'},
        'meter_ids': {'met13': 'meter_1001', 'met14': 'meter_1002'},
        'gateway_id': {'GW2103040011': 'GWTMP1000001'},
        'date_from': '2020-08-13T00:00:00Z',
        'date_to': '2020-03-01T00:00:00Z',
        'source_csv': 'node_38_august_march.csv',
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
        'validation_data': [{'date': '2020-09-07', 'meter_id': 'meter_1001', 'gateway_id': 'GWTMP1000001',
                             'number_of_rows': 32630},
                            {'date': '2021-02-07', 'meter_id': 'meter_1002', 'gateway_id': 'GWTMP1000001',
                             'number_of_rows': 7392}
                            ]
    },
    'telemetry_scenario_low_frequency': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_ELECTRICAL_MEASUREMENTS,
        'columns': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3', 'ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
        'columns_to_change': {'meter_id': 'meter_ids', 'gateway_id': 'gateway_id'},
        'meter_ids': {'dom_adam': 'meter_1003'},
        'gateway_id': {'adam': 'GWTMP1000002'},
        'date_from': '2020-08-13T00:00:00Z',
        'date_to': '2020-09-03T00:00:00Z',
        'source_csv': 'influx_data_2020-08-13_low_freq_part.csv',  # part from 2020/08/23 - 2020/08-29 is with freq 3H.
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
        'validation_data': [{'date': '2020-09-07', 'meter_id': 'meter_1003', 'gateway_id': 'GWTMP1000002',
                             'number_of_rows': 32630},
                            {'date': '2020-08-26', 'meter_id': 'meter_1003', 'gateway_id': 'GWTMP1000002',
                             'number_of_rows': 8}]
    },
    'telemetry_scenario_duplicates': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_ELECTRICAL_MEASUREMENTS,
        'columns': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3', 'ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
        'columns_to_change': {'meter_id': 'meter_ids', 'gateway_id': 'gateway_id'},
        'meter_ids': {'met13': 'meter_1004', 'met14': 'meter_1005'},
        'gateway_id': {'GW2103040011': 'GWTMP1000003'},
        'date_from': '2020-08-31T15:15:36Z',
        'date_to': '2020-09-04T15:15:34Z',
        'source_csv': 'duplicated_data_august_31_to_september_04.csv',
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
        'validation_data': [{'date': '2020-09-02', 'meter_id': 'meter_1004', 'gateway_id': 'GWTMP1000003',
                             'number_of_rows': 66252},
                            {'date': '2020-08-31', 'meter_id': 'meter_1005', 'gateway_id': 'GWTMP1000003',
                             'number_of_rows': 24326}
                            ]
    },
    'telemetry_scenario_advanced_duplicates': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_ELECTRICAL_MEASUREMENTS,
        'columns': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3', 'ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
        'columns_to_change': {'meter_id': 'meter_ids', 'gateway_id': 'gateway_id'},
        'meter_ids': {'dom_adam': 'meter_1006', 'inverter_adam': 'meter_1007'},
        'gateway_id': {'GW2006010007': 'GWTMP1000004'},
        'date_from': '2020-07-31T15:15:36Z',
        'date_to': '2021-11-03T15:15:34Z',
        'source_csv': 'gateway_7_advanced_duplicates.csv',
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
        'validation_data': [{'date': '2021-09-02', 'meter_id': 'meter_1006', 'gateway_id': 'GWTMP1000004',
                             'number_of_rows': 175},
                            {'date': '2021-08-31', 'meter_id': 'meter_1007', 'gateway_id': 'GWTMP1000004',
                             'number_of_rows': 174}
                            ]
    },

    'presentation_data': {
        'project_id': config.PROJECT_ID,
        'dataset': config.BIGQUERY_DATASET_NAME,
        'table': config.BIGQUERY_TABLE_ELECTRICAL_MEASUREMENTS,
        'columns': ['ea_rev_1', 'ea_rev_2', 'ea_rev_3', 'ea_fwd_1', 'ea_fwd_2', 'ea_fwd_3'],
        'columns_to_change': {'meter_id': 'meter_ids', 'gateway_id': 'gateway_id'},
        'meter_ids': {'dom_adam': 'meter_julo_1'},
        'gateway_id': {'adam': 'GWTMP1000005'},
        'source_csv': 'influx_data_2020-08-13_low_freq_part.csv',  # part from 2020/08/23 - 2020/08-29 is with freq 3H.
        'source_bucket': config.TEST_DATA_BUCKET_NAME,
        'validation_data': [{'date': '2020-09-07', 'meter_id': 'meter_julo_1', 'gateway_id': 'GWTMP1000005',
                             'number_of_rows': 32630},
                            {'date': '2020-08-26', 'meter_id': 'meter_julo_1', 'gateway_id': 'GWTMP1000005',
                             'number_of_rows': 8}]
    }
}
