import logging
import os

import pytest

from tests.end_to_ends.telemetry_data_scenarios import TELEMETRY_DATA_SCENARIOS


@pytest.fixture(scope='module', params=['energy_usage', 'real_energy_prod'])
def endogenous(request):
    return request.param


@pytest.fixture(scope='module', params=[
    # 'arima',  # 'order' is required
    # 'autoarima',
    # 'autosarima',
    # 'boosted_trees',
    # 'cnn', # 'order' is required
    #  'decision_tree',
    # 'lasso',
    'linear_regression',
    # 'lstm_stateful' # 'order' is required
    # 'lstm_stateless', # 'order' is required
    # 'mlp',
    # 'random_forest',
    # 'sarima', # 'order' is required
    # 'svm',
])
def algorithm(request):
    return request.param


@pytest.fixture
def horizon():
    return '1w'


@pytest.fixture
def prediction_horizon():
    return '1w'


@pytest.fixture
def image_prefix(request):
    directory = 'tests/end_to_ends/tests_results'
    os.makedirs(directory, exist_ok=True)
    logging.info(f'Using {directory} for generated images')
    return os.path.join(directory, request.node.name)


@pytest.fixture(scope='session', autouse=True)
def _validate_telemetry_data_scenarios():
    """
    Fixture for checking if telemetry data scenarios are written in a proper way - which means that meters & gateways
    pairs doesn't collide with each other between different data scenarios.
    """

    logging.info('Validating telemetry data scenarios...')
    used_configurations = set()
    for name, data_scenario in TELEMETRY_DATA_SCENARIOS.items():
        gateway_id = next(iter(data_scenario['gateway_id'].values()))
        configuration = {(meter_id, gateway_id) for meter_id in data_scenario.get('meter_ids').values()}
        assert not used_configurations & configuration, f'You have provided already such configuration. Change your ' \
                                                        f'meters/gateway in data scenario: "{name}".'
        validation_configuration = {(x['meter_id'], x['gateway_id']) for x in data_scenario['validation_data']}
        assert configuration >= validation_configuration, f'Configuration & validation configuration mismatch in ' \
                                                          f'data scenario: "{name}".'
        used_configurations |= configuration
    logging.info('Telemetry data scenarios validated!')
