import logging
import random
import time
from datetime import datetime, timedelta

import pytest
from webtest import TestApp
from mysql.connector.cursor import MySQLCursorNamedTuple


@pytest.mark.parametrize('nodes', (200,))
def test_node_creation_time(testapp: TestApp, nodes: int) -> None:
    times = []

    for _ in range(nodes):
        request = {'active_from': '2020-01-01T00:00:00Z',
                   'location': {'id': random.randint(1, 2**63), 'country': 'Poland', 'post_code': '00-950'}}
        start = time.time()
        resp = testapp.post_json('/v1/nodes', request)
        times.append(time.time() - start)
        assert resp.status_int == 201
    avg = sum(times) / nodes
    maximum = max(times)
    if True:  # avg > 0.030 or maximum > 0.050:
        logging.warning(f'WSGI node creation time = avg {avg*1000:.2f} / max {maximum*1000:.2f} ms')
    assert maximum < 0.200
    assert avg < 0.040


@pytest.mark.parametrize('meters', (200,))
def test_meter_creation_time(testapp: TestApp, meters: int) -> None:
    start_date = datetime(2020, 1, 1)
    request = {'active_from': start_date.isoformat(sep='T'),
               'location': {'id': random.randint(1, 2**63), 'country': 'Poland', 'post_code': '00-950'}}
    resp = testapp.post_json('/v1/nodes', request)
    assert resp.status_int == 201
    node = resp.json

    times = []

    for i in range(meters):
        request = {
            'active_from': (start_date + timedelta(days=i)).isoformat(sep='T'),
            'active_to': (start_date + timedelta(days=i, hours=3)).isoformat(sep='T'),
            'model': 'fif',
            'role': 'input_home',
            'gateway_id': 'GWTMP0123456',
            'meter_id': 'm1',
        }
        start = time.time()
        resp = testapp.post_json(f'/v1/nodes/{node}/meters', request)
        times.append(time.time() - start)
        assert resp.status_int == 201
    avg = sum(times) / meters
    maximum = max(times)
    if True:  # avg > 0.060 or maximum > 0.100:
        logging.warning(f'WSGI meter creation time = avg {avg*1000:.2f} / max {maximum*1000:.2f} ms')
    assert maximum < 0.200
    assert avg < 0.090


@pytest.mark.parametrize('anomalies', (3, 5))
# @pytest.mark.parametrize('meters', (1, 10))  # tests duration >10 minutes
@pytest.mark.parametrize('meters', (1,))
def test_telemetry_info_time(testapp: TestApp, meters: int, anomalies: int, pa_sql: MySQLCursorNamedTuple) -> None:
    nodes = []

    # create 1, 10, 100 node batches
    for pkg in (1, 9, 90):
        for _ in range(pkg):
            start_date = datetime(2020, 1, 1)
            request = {'active_from': start_date.isoformat(sep='T'),
                       'location': {'id': random.randint(1, 2**63), 'country': 'Poland', 'post_code': '00-950'}}
            resp = testapp.post_json('/v1/nodes', request)
            assert resp.status_int == 201
            node = resp.json
            nodes.append(node)

            # add meters
            for i in range(meters):
                request = {
                    'active_from': (start_date + timedelta(days=i)).isoformat(sep='T'),
                    'active_to':   (start_date + timedelta(days=i, hours=3)).isoformat(sep='T'),
                    'model': 'fif',
                    'role': 'input_home',
                    'gateway_id': 'GWTMP0123456',
                    'meter_id': 'm1',
                }
                resp = testapp.post_json(f'/v1/nodes/{node}/meters', request)
                assert resp.status_int == 201
                meter = resp.json

                # add anomalies
                for j in range(anomalies):
                    request = {
                        'begin': (start_date + timedelta(days=i, minutes=5+j*2)).isoformat(sep='T'),
                        'end':   (start_date + timedelta(days=i, minutes=6+j*2)).isoformat(sep='T'),
                    }
                    resp = testapp.post_json(f'/v1/nodes/{node}/meters/{meter}/anomalies', request)
                    assert resp.status_int == 201

        def sql_get_(node: int = None) -> list:
            pa_sql.execute('SELECT * FROM telemetry_info' + (f' WHERE node_id = {node}' if node else ''))
            return pa_sql.fetchall()

        def rest_get_(node: int = None) -> list:
            assert node
            return testapp.get(f'/v1/nodes/{node}/data_view').json

        checks = {'SQL': sql_get_, 'WSGI': rest_get_}

        for check in checks:
            if check == 'SQL':
                # retrieve all nodes at once
                start = time.time()
                res = checks[check]()
                timer = time.time() - start
                assert res
                if True:
                    logging.warning(f'{check} {len(nodes)}/{len(nodes)*meters}/{len(nodes)*meters*anomalies}'
                                    f' nodes/meters/anomalies all data_view = {timer:.3f} s')
                assert timer < 0.300

            times = []

            # retrieve all nodes one by one
            for node in nodes:
                start = time.time()
                res = checks[check](node)
                end = time.time()
                assert res
                times.append({'time': end - start, 'rows': len(res)})
            avg = sum(t['time'] for t in times) / len(nodes)
            avg_per_row = avg * len(nodes) / sum(r['rows'] for r in times)
            maximum = max(times, key=lambda x: x['time'])
            max_per_row = maximum["time"] / maximum["rows"]
            if True:
                desc = f'{check} single node with {meters}/{anomalies} meters/anom data_view'
                if len(nodes) == 1:
                    logging.warning(f'{desc} = {avg*1000:.3f} ms')
                    logging.warning(f'{desc} = {avg_per_row*1000:.3f} ms per row ({times[0]["rows"]})')
                else:
                    logging.warning(f'{desc} = avg {avg*1000:.3f} / max {maximum["time"]*1000:.3f} ms')
                    logging.warning(f'{desc} = avg {avg_per_row*1000:.3f} / max {max_per_row*1000:.3f} ms per row ({times[0]["rows"]})')
