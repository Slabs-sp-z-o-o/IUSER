import urllib

import pytest
from webtest import TestApp

from conftest import check_problem


def check_node_structure(obj, meters: bool = None) -> None:
    assert isinstance(obj, dict)
    assert {'id', 'active_from', 'location'} <= set(obj.keys())
    assert {'id', 'active_from', 'active_to', 'location', 'meters'} >= set(obj.keys())
    assert isinstance(obj['location'], dict)
    assert {'id', 'post_code'} <= set(obj['location'].keys())
    assert {'id', 'lat', 'lon', 'country', 'city', 'post_code', 'street', 'building', 'flat'} >= set(
        obj['location'].keys())
    assert (meters is None) == ('meters' not in obj)
    assert (meters is not None) == isinstance(obj.get('meters'), list)
    assert meters is None or meters == bool(obj['meters'])


def test_search(testapp: TestApp) -> None:
    resp = testapp.get('/v1/nodes')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, list)
    assert len(resp.json) == 4
    map(check_node_structure, resp.json)
    assert {n['id'] for n in resp.json} == {1, 2, 3, 4}


@pytest.mark.parametrize('query', (' ', 'xyx', 'wer=', '324/234',
                                   'active/', 'active=sd34', 'active=', 'active',
                                   'active=1', 'active=tru', 'Active=true', 'acTIVe=false',
                                   'location+', 'location=', 'location', 'locATIon=12',
                                   'location=7abc40a5-928f-4ab1-b624-7fcfdaf7302b',
                                   urllib.parse.quote('żółœπ…ę©ß')))
def test_search_wrong_param(testapp: TestApp, query: str) -> None:
    resp = testapp.get(f'/v1/nodes?{query}', status=400)
    check_problem(resp, 400)


@pytest.mark.parametrize('act, ids', (('true', {1, 2, 4}), ('True', {1, 2, 4}),
                                      ('false', {3}), ('False', {3})))
def test_search_active_param(testapp: TestApp, act: str, ids: set) -> None:
    resp = testapp.get(f'/v1/nodes?active={act}')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, list)
    assert len(resp.json) == len(ids)
    map(check_node_structure, resp.json)
    assert {n['id'] for n in resp.json} == ids


def test_search_loc_param(testapp: TestApp) -> None:
    nodes = {x['location']['id']: x['id'] for x in testapp.get('/v1/nodes').json}
    for loc, id in nodes.items():
        resp = testapp.get(f'/v1/nodes?location={loc}')
        assert resp.status_int == 200
        assert resp.content_type == 'application/json'
        assert isinstance(resp.json, list)
        assert len(resp.json) == 1
        map(check_node_structure, resp.json)
        assert resp.json[0]['id'] == id


@pytest.mark.parametrize('loc', (8888888888888888888, -8888888888888888888))
def test_search_unknown_loc_param(testapp: TestApp, loc: int) -> None:
    resp = testapp.get(f'/v1/nodes?location={loc}')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, list)
    assert len(resp.json) == 0


@pytest.mark.parametrize('id', range(1, 5))
def test_get(testapp: TestApp, id: int) -> None:
    resp = testapp.get(f'/v1/nodes/{id}')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    check_node_structure(resp.json)
    assert resp.json['id'] == id


@pytest.mark.parametrize('id', ('aaa', -15, 0, 95, '', ' ', 2 * 68))
@pytest.mark.parametrize('full', ('', '?full=true'))
def test_get_wrong(testapp: TestApp, id, full: str) -> None:
    resp = testapp.get(f'/v1/nodes/{id}{full}', status='40*')
    assert resp.status_int in {400, 404}
    check_problem(resp)


@pytest.mark.parametrize('id, meters', ((1, False), (2, True), (3, False), (4, True)))
@pytest.mark.parametrize('query', ('full=true', 'full=True', 'full=trUe'))
def test_get_full(testapp: TestApp, id: int, query: str, meters: bool) -> None:
    resp = testapp.get(f'/v1/nodes/{id}?{query}')
    assert resp.status_int == 200
    assert resp.content_type == 'application/json'
    check_node_structure(resp.json, meters)
    assert resp.json['id'] == id


@pytest.mark.parametrize('id', (0, 1))
@pytest.mark.parametrize('query', (' ', 'xyx', 'wer=', '324/234',
                                   'full/', 'full=sd34', 'full=', 'full', 'full=yes'
                                                                          'full=1', 'full=tru', 'Full=true',
                                   'fUlL=false',
                                   'full+', 'full=7abc40a5-928f-4ab1-b624-7fcfdaf7302b',
                                   urllib.parse.quote('żółœπ…ę©ß'), urllib.parse.quote('full=żółœπ…ę©ß')))
def test_get_wrong_full(testapp: TestApp, id: int, query: str) -> None:
    resp = testapp.get(f'/v1/nodes/{id}?{query}', status=400)
    check_problem(resp, 400)


@pytest.mark.parametrize('diff', ({},
                                  {'location': {'id': 9223372036854775807, 'country': 'Poland', 'post_code': '00-950'}},
                                  {'location': {'id': -9223372036854775808, 'country': 'Poland',
                                                'post_code': '00-950'}},
                                  {'active_to': '2020-01-01T00:00:00'},
                                  ))
def test_post(testapp: TestApp, diff: dict) -> None:
    request = {'active_from': '2020-01-01T00:00:00',
               'location': {'id': 992, 'country': 'Poland', 'post_code': '00-950'}}
    request.update(diff)
    resp = testapp.post_json('/v1/nodes', request)
    assert resp.status_int == 201
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, int)
    assert 'location' in resp.headers
    assert resp.location.rpartition('/')[2] == str(resp.json)

    check = testapp.get(resp.location)
    assert check.status_int == 200
    assert check.content_type == 'application/json'
    assert isinstance(check.json, dict)
    check_node_structure(check.json)
    assert check.json == {**request, 'id': resp.json}


@pytest.mark.parametrize('diff', ({'location': {'id': 1}},
                                  {'location': {'country': 'Poland'}},
                                  {'id': 4535},
                                  {'id': 'fddfsdf'},
                                  {'active_from': None},
                                  {'active_from': 'xxx'},
                                  {'active_from': '2020-01-01'},
                                  {'active_from': '2020-01-01T'},
                                  {'active_from': '2020-01-01T00'},
                                  {'active_from': '-120-01-01T00:00:00Z'},
                                  {'active_from': '2020-02-31T00:00:00Z'},
                                  {'active_from': None, 'active_to': '2010-01-31T00:00:00Z'},
                                  {'active_to': 'xxx'},
                                  {'active_to': '2010-01-31T00:00:00Z'},
                                  {'active_to': '2019-12-31T23:59:59Z'},
                                  {'xxx': 'yyy'},
                                  ))
def test_post_incorrect(testapp: TestApp, diff: dict) -> None:
    request = {'active_from': '2020-01-01T00:00',
               'location': {'id': 992, 'country': 'Poland', 'post_code': '00-950'}}
    request.update(diff)
    resp = testapp.post_json('/v1/nodes', request, status='4*')
    assert resp.status_int in {400, 422}
    check_problem(resp)


def test_post_duplicated_loc_id(testapp: TestApp) -> None:
    request = {'active_from': '2020-01-01T00:00:00',
               'location': {'id': 992, 'country': 'Poland', 'post_code': '00-950'}}
    resp = testapp.post_json('/v1/nodes', request)
    assert resp.status_int == 201
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, int)
    assert 'location' in resp.headers
    assert resp.location.rpartition('/')[2] == str(resp.json)
    check = testapp.get(resp.location)
    assert check.status_int == 200
    assert check.content_type == 'application/json'
    assert isinstance(check.json, dict)
    check_node_structure(check.json)
    assert check.json == {**request, 'id': resp.json}
    resp = testapp.post_json('/v1/nodes', request, status=409)
    check_problem(resp, 409)


from mysql.connector.cursor import MySQLCursorNamedTuple
from flask_sqlalchemy import SQLAlchemy


def test_post_duplicated_broken_loc_id(testapp: TestApp, db: SQLAlchemy, pa_sql: MySQLCursorNamedTuple) -> None:
    request = {'active_from': '2020-01-01T00:00:00',

               'location': {'id': 992, 'country': 'Poland', 'post_code': '00-950'}}

    resp = testapp.post_json('/v1/nodes', request)
    assert resp.status_int == 201
    assert resp.content_type == 'application/json'
    assert isinstance(resp.json, int)
    assert 'location' in resp.headers
    assert resp.location.rpartition('/')[2] == str(resp.json)

    # broke DB relations - remove node with orphaned locations
    pa_sql.execute(f'DELETE FROM node WHERE id = {resp.json}')
    # force SQLAlchemy to reload cache
    db.session.expire_all()
    db.session.commit()
    resp = testapp.post_json('/v1/nodes', request, status=409)
    check_problem(resp, 409)
