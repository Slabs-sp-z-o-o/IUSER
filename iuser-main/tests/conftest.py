import functools
import logging
import os
import random
from collections import namedtuple
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, List
from urllib.parse import urlparse

import mysql.connector
import pytest
import requests

from mysql.connector.cursor import MySQLCursorNamedTuple

from common import config
from common.controllers.nodes_controller import NodesController
from tests.config import BEHAVE_NODE_LOCATION_ID

Node = namedtuple('Node', 'id active_from active_to location', defaults=4 * [None])
Meter = namedtuple('Meter', 'id active_from active_to gateway_id meter_id model role', defaults=7 * [None])


def _stat_db(pa_sql: MySQLCursorNamedTuple):
    """Log row number for selected tables from db."""
    tables = 'node/location/meteo_locations/meter/anomaly/telemetry_info'
    cnt = []
    for x in tables.split('/'):
        pa_sql.execute(f'SELECT COUNT(*) AS cnt FROM {x}')
        cnt.append(str(pa_sql.fetchone().cnt))
    logging.info(f'{tables} = {"/".join(cnt)}')


@pytest.fixture
def pa_sql() -> Generator[MySQLCursorNamedTuple, None, None]:
    """Prepare MySQL cursor connected to nodes db."""
    conn = urlparse(os.environ['SQLALCHEMY_DATABASE_URI'])
    logging.info(f'Connecting to {conn.hostname} MySQL instance...')
    with mysql.connector.connect(host=conn.hostname, port=conn.port,
                                 user=conn.username, password=conn.password,
                                 database=conn.path[1:], autocommit=True,
                                 charset='utf8mb4') as db:
        logging.info('Connected successfully')
        with db.cursor(named_tuple=True) as sql:
            yield sql
            _stat_db(sql)


@pytest.fixture
def pa_api() -> Generator[requests.Session, None, None]:
    """Prepare HTTP session connected to PA REST API server."""
    with requests.Session() as sess:
        sess.auth = (config.NODES_API_USERNAME, config.NODES_API_PASSWORD)
        yield sess


@pytest.fixture
def node(request, set_node: Tuple[Callable[[dict], Node], Callable[[dict], Meter]]) -> Node:
    """Create node with default values or specified as dict parameter.

    All objects related to the node are deleted from db after test.
    """
    return set_node[0](getattr(request, 'param', None))


@pytest.fixture
def meters(request, set_node: Tuple[Callable[[dict], Node], Callable[[dict], Meter]]) -> Set[Meter]:
    """Add one or more meters with default config or specified as list of dict param."""
    assert hasattr(request, 'param') and isinstance(request.param, list), 'meters fixture must be parametrized'
    meters = set()
    add_meter = set_node[1]
    for m in request.param:
        if not isinstance(m, dict):
            assert m is None, 'unknown parameter type - should be dict'
        meters.add(add_meter(m))
    return meters


@pytest.fixture
def set_node(pa_api: requests.Session, pa_sql: MySQLCursorNamedTuple, fixed_node_locations: List[str]) -> Generator[
    Tuple[Callable[[Dict[str, Any]], Node],
          Callable[[Dict[str, Any]], Meter]], None, None]:
    _node: Optional[Node] = None

    def _create_object(endpoint: str, request: dict) -> dict:
        nodes_controller = NodesController()
        resp = pa_api.post(f'{nodes_controller.url}/{endpoint}', json=request)
        assert resp.ok, f'creation error {resp.reason}: {resp.text}'
        assert resp.status_code == 201
        assert resp.headers['location']
        resp = pa_api.get(resp.headers['location'])
        assert resp.ok, f'get error {resp.reason}: {resp.text}'
        assert resp.status_code == 200
        return resp.json()

    def _set_node(params: dict) -> Node:
        nonlocal _node
        assert _node is None, 'only 1 node can be used in test'
        DEFAULTS: Dict[str, Any] = {
            'active_from': datetime.utcnow().isoformat(),
            'location': {
                'id': random.randint(1, 2 ** 63),
                'country': 'Poland',
                'post_code': '00-950'
            }
        }
        node_params = DEFAULTS.copy()
        if params:
            node_params.update((k, v) for k, v in params.items() if k != 'location')
            if 'location' in params:
                node_params['location'].update(params['location'])
        _node = Node(**_create_object('nodes', node_params))
        logging.info(f'created node: {_node}')
        return _node

    def _add_meter(params: dict) -> Meter:
        nonlocal _node
        assert _node, 'node must be created first'
        DEFAULTS: Dict[str, Any] = {
            'active_from': _node.active_from,
            'model': 'fif',
            'role': 'input_home',
            'gateway_id': 'GWTMP0123456',
            'meter_id': 'm1',
        }
        meter_params = DEFAULTS.copy()
        if params:
            meter_params.update(params)
        meter = Meter(**_create_object(f'nodes/{_node.id}/meters', meter_params))
        logging.info(f'created meter: {meter}')
        return meter

    yield _set_node, _add_meter

    if _node is not None:
        if (_node_location := _node.location['id']) in fixed_node_locations:
            logging.warning(f'Not removing node {_node.id} in location {_node_location} '
                            f'because this location is defined in fixed node locations')
            return
        logging.info(f'removing node {_node.id}')
        pa_sql.execute(f'DELETE FROM location WHERE id = {_node.location["id"]}')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Hook for allowing to obtain parametrized value inside test setup method."""
    if item.cls:
        item.cls._item = item
    yield


@pytest.fixture
def get_and_send_prediction_publish():
    from google.cloud import pubsub_v1

    publisher = pubsub_v1.PublisherClient()
    return functools.partial(publisher.publish, publisher.topic_path(config.PROJECT_ID, config.DAILY_PREDICTION_TOPIC))


@pytest.fixture
def fixed_node_locations():
    return [BEHAVE_NODE_LOCATION_ID]
