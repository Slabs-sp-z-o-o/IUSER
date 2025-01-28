import csv
import os
from distutils.util import strtobool
from typing import Any, FrozenSet, Generator, List, Set, Tuple
from urllib.parse import urlparse

import flask
import mysql.connector
import pytest
from webtest import TestApp, TestResponse
from flask_sqlalchemy import BaseQuery, SQLAlchemy
from mysql.connector.cursor import MySQLCursorNamedTuple

from common import config
from config import create_app, db as _db
from init_db import load_config, replace_schema, setup_test_data

# don't try to collect WebTest classes as test cases
TestApp.__test__ = False


@pytest.fixture(scope='session')
def meteo_csv() -> List[dict]:
    """Parsed meteo_config.csv file content."""
    with open('meteo_config.csv', 'r', newline='') as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope='session')
def meter_csv() -> List[dict]:
    """Parsed meter_config.csv file content."""
    with open('meter_config.csv', 'r', newline='') as f:
        data = [{k: v if v != '' else None for k, v in row.items()} for row in csv.DictReader(f)]
    for row in data:
        row['cumulative'] = True if strtobool(row['cumulative']) else False
    return data


@pytest.fixture(scope='session')
def app() -> Generator[flask.Flask, None, None]:
    """A Flask application used in tests."""
    _app = create_app()

    ctx = _app.app.test_request_context()
    ctx.push()

    yield _app.app

    ctx.pop()


@pytest.fixture(scope='session')
def testapp(app: flask.Flask) -> TestApp:
    """A Webtest application for tests."""
    t = TestApp(app)
    t.authorization = ('Basic', (config.NODES_API_USERNAME, config.NODES_API_PASSWORD))
    return t


@pytest.fixture(autouse=True)
def db(app: flask.Flask) -> Generator[SQLAlchemy, None, None]:
    """A database for the tests."""
    _db.app = app
    with app.app_context():
        replace_schema(_db)
        load_config(_db)
        setup_test_data(_db)

    yield _db

    _db.session.close()


@pytest.fixture
def pa_sql() -> Generator[MySQLCursorNamedTuple, None, None]:
    """Prepare MySQL cursor connected to nodes db."""
    conn = urlparse(os.environ['SQLALCHEMY_DATABASE_URI'])
    with mysql.connector.connect(host=conn.hostname, port=conn.port,
                                 user=conn.username, password=conn.password,
                                 database=conn.path[1:], autocommit=True,
                                 charset='utf8mb4') as db:
        with db.cursor(named_tuple=True) as sql:
            yield sql


def check_problem(resp: TestResponse, code: int = None) -> None:
    assert code is None or resp.status_int == code
    assert resp.content_type == 'application/problem+json'
    data = resp.json
    assert all(map(lambda x: x in data, 'title type detail status'.split()))
    assert data['status'] == resp.status_int


def sql_query_to_set_of_sets(query: BaseQuery) -> Set[FrozenSet[Tuple[str, Any]]]:
    return {frozenset({f.key: getattr(row, f.key) for f in row.__mapper__.iterate_properties}.items()) for row in query}
