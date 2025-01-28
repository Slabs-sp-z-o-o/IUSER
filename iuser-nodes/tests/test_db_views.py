import urllib

import pytest
from flask_sqlalchemy import SQLAlchemy
from webtest import TestApp

from models import (Location, MeteoConfig, MeterAnomalies, MeterConfig, Node,
                    NodeConfig, NodeMeteo, Serie)
from conftest import check_problem, sql_query_to_set_of_sets


def test_meter_anomalies(testapp: TestApp) -> None:
    expected = (0, 1, 0, 7)
    q = MeterAnomalies.query

    for n, rows in enumerate(expected, start=1):
        assert q.filter_by(node_id=n).count() == rows

    assert q.count() == sum(expected)


def test_nodes_locations(db: SQLAlchemy) -> None:
    src = db.session.query(Node.id, Location.post_code).outerjoin(Node.location)
    view = NodeMeteo.query
    assert {(x.node_id, x.bq_identifier) for x in view} == set(src)


def test_meteo_config(meteo_csv: list) -> None:
    expected = {frozenset(row.items()) for row in meteo_csv}
    resp = sql_query_to_set_of_sets(MeteoConfig.query)
    assert resp == expected


def test_meter_config(meter_csv: list) -> None:
    expected = {frozenset(row.items()) for row in meter_csv}
    resp = sql_query_to_set_of_sets(MeterConfig.query)
    assert resp == expected


def test_series(meteo_csv: list, meter_csv: list) -> None:
    expected = {frozenset((('name', row['serie']), ('fetching_logic', 'meteo'))) for row in meteo_csv}
    expected |= {frozenset((('name', row['serie']), ('fetching_logic', 'telemetry'))) for row in meter_csv}
    resp = sql_query_to_set_of_sets(Serie.query)
    assert resp == expected


def test_telemetry_info(testapp: TestApp) -> None:
    expected = (0, 15, 0, 126)
    """Number of rows returned for nodes ID: 1, 2, 3 and 4."""
    q = NodeConfig.query

    for node, rows in enumerate(expected, start=1):
        resp = testapp.get(f'/v1/nodes/{node}/data_view')
        assert q.filter_by(node_id=node).count() == rows
        assert len(resp.json) == rows


@pytest.mark.parametrize('serie, rows_exp', (('energy_usage', (0, 3, 0, 42)),
                                             ('real_energy_import', (0, 0, 0, 21)),
                                             ('real_energy_export', (0, 0, 0, 21)),
                                             ('real_power_prod', (0, 3, 0, 0)),
                                             ('real_energy_prod', (0, 3, 0, 0))))
def test_telemetry_info_serie(testapp: TestApp, serie: str, rows_exp: tuple) -> None:
    q = NodeConfig.query
    for node, rows in enumerate(rows_exp, start=1):
        resp = testapp.get(f'/v1/nodes/{node}/data_view/{serie}')
        assert q.filter_by(node_id=node, output_series=serie).count() == rows
        assert len(resp.json) == rows


@pytest.mark.parametrize('serie', ('', ' ', 'skdjvhbaiuebqvwie', urllib.parse.quote('żółœπ…əµę©ß')))
def test_telemetry_info_wrong_serie(testapp: TestApp, serie: str) -> None:
    for node in range(1, 5):
        resp = testapp.get(f'/v1/nodes/{node}/data_view/{serie}', status=404)
        check_problem(resp, 404)


@pytest.mark.parametrize('query', (' ', 'serie', 'serie=', 'serie=a', 'serie_prefix', 'serie_prefix='))
def test_telemetry_info_wrong_prefix(testapp: TestApp, query: str) -> None:
    for node in range(1, 5):
        resp = testapp.get(f'/v1/nodes/{node}/data_view?{query}', status=400)
        check_problem(resp, 400)


@pytest.mark.parametrize('prefix, rows_exp', ((' ', (0, 0, 0, 0)),
                                              ('x_x_x', (0, 0, 0, 0)),
                                              ('?', (0, 0, 0, 0)),
                                              ('%', (0, 0, 0, 0)),
                                              ('_', (0, 0, 0, 0)),
                                              ('real', (0, 9, 0, 42)),
                                              ('real_energy_', (0, 6, 0, 42)),
                                              ('real_power', (0, 3, 0, 0)),
                                              ('energy', (0, 6, 0, 84)),
                                              ('energy_usage_', (0, 3, 0, 42))))
def test_telemetry_info_prefix(testapp: TestApp, prefix: str, rows_exp: tuple) -> None:
    q = NodeConfig.query
    for node, rows in enumerate(rows_exp, start=1):
        resp = testapp.get(f'/v1/nodes/{node}/data_view?serie_prefix={prefix}')
        n = q.filter_by(node_id=node).filter(NodeConfig.output_series.startswith(prefix, autoescape=True)).count()
        assert n == rows
        assert len(resp.json) == rows
