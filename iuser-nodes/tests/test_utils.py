import pytest
from webtest import TestApp

from conftest import check_problem


def test_ping(testapp: TestApp) -> None:
    pong = testapp.get('/v1/ping')
    assert pong.status_int == 204
    assert pong.content_length is None


def test_data_series(testapp: TestApp, meteo_csv: list, meter_csv: list) -> None:
    expected = {row['serie'] for row in meteo_csv + meter_csv}
    resp = testapp.get('/v1/data_series')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, list)
    assert expected == set(resp.json)


@pytest.mark.parametrize('prefix, expected',
                         ((' ', ''), ('x_x_x', ''), ('?', ''), ('%', ''), ('dh%', ''), ('dh_', ''),
                          ('dhi', 'dhi'), ('ghi', 'ghi ghi10 ghi90'),
                          ('real_energy_prod', 'real_energy_prod real_energy_prod_1 real_energy_prod_2 real_energy_prod_3'),
                          ('real_energy_prod_', 'real_energy_prod_1 real_energy_prod_2 real_energy_prod_3'),
                          ('real_power', 'real_power_prod'),
                          ('energy', 'energy_usage energy_usage_1 energy_usage_2 energy_usage_3')))
def test_data_series_prefix(testapp: TestApp, meteo_csv: list, meter_csv: list, prefix: str, expected: str) -> None:
    loaded = {row['serie'] for row in meteo_csv + meter_csv if row['serie'].startswith(prefix)}
    resp = testapp.get(f'/v1/data_series?prefix={prefix}')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, list)
    assert set(expected.split()) == loaded
    assert set(expected.split()) == set(resp.json)


@pytest.mark.parametrize('query', (' ', 'pref', 'pref=', 'pref=a', 'prefix', 'prefix='))
def test_data_series_wrong_prefix(testapp: TestApp, query: str) -> None:
    resp = testapp.get(f'/v1/data_series?{query}', status=400)
    check_problem(resp, 400)


@pytest.mark.parametrize('auth', [None, ('Basic', ('foo', 'b_r'))])
@pytest.mark.parametrize('path', ['/..', '/nodes', '/nooodes'])
def test_paths_no_auth(testapp: TestApp, path: str, auth: tuple) -> None:
    testapp.authorization = auth
    resp = testapp.get(f'/v1{path}', status=401)
    check_problem(resp, 401)


@pytest.mark.parametrize('path', ['/', '/xdf', '/v1/', '/v1'])
def test_root(testapp: TestApp, path: str) -> None:
    resp = testapp.get(path, status=404)
    check_problem(resp, 404)
