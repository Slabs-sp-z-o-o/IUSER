from tests.config import BEHAVE_NODE_LOCATION_ID


def scenario_1():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'post_code': '10-001'}},
        'meters': [{'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1001', 'role': 'input_home'},
                   {'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1002', 'role': 'inverter'}],
    }


def scenario_2():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'post_code': '10-001'}},
        'meters': [{'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1002', 'role': 'inverter'}],
    }


def scenario_3():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'post_code': '10-001'}},
        'meters': [{'gateway_id': 'GWTMP1000002', 'meter_id': 'meter_1003', 'role': 'input_home'}],
    }


def scenario_demo_day():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'post_code': '10-002'}},
        'meters': [{'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1001', 'role': 'input_home'},
                   {'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1002', 'role': 'inverter'}],
    }


def behave_nodes_setup_1():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'id': BEHAVE_NODE_LOCATION_ID, 'post_code': '10-001'}},
        'meters': [{'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1001', 'role': 'input_home'},
                   {'gateway_id': 'GWTMP1000001', 'meter_id': 'meter_1002', 'role': 'inverter'}],
    }


def scenario_duplicates():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'post_code': '10-001'}},
        'meters': [{'gateway_id': 'GWTMP1000003', 'meter_id': 'meter_1004', 'role': 'input_home'},
                   {'gateway_id': 'GWTMP1000003', 'meter_id': 'meter_1005', 'role': 'inverter'}],
    }


def scenario_advanced_duplicates():
    return {
        'node': {'active_from': '2020-01-01T00:00', 'location': {'post_code': '10-001'}},
        'meters': [{'gateway_id': 'GWTMP1000004', 'meter_id': 'meter_1006', 'role': 'input_home'},
                   {'gateway_id': 'GWTMP1000004', 'meter_id': 'meter_1007', 'role': 'inverter'}],
    }


def presentation_nodes_setup():
    return {
        'node': {'active_from': '2020-08-13T00:00', 'location': {'post_code': '10-003'}},
        'meters': [{'gateway_id': 'GWTMP1000005', 'meter_id': 'meter_julo_1', 'role': 'input_home'}]
    }
