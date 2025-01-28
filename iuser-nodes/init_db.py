#!/usr/bin/env python3
import csv
import logging
import os
from datetime import datetime
from distutils.util import strtobool
from time import sleep, time
from typing import Any, Dict

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import DatabaseError

import models as m


def replace_schema(db: SQLAlchemy) -> None:
    timer = time() + int(os.getenv('CREATE_DB_TIMEOUT', 20))
    while time() < timer:
        try:
            db.session.execute('SELECT 1')
            break
        except DatabaseError as e:
            logging.info(f'waiting for db... ({e.orig.args})')
            sleep(3)
    db.drop_all()
    logging.info('old db schema removed')
    db.create_all()
    logging.info('new db schema created')


def load_config(db: SQLAlchemy) -> None:
    name = 'meteo_config'
    logging.info(f'loading {name}.csv info {name} table...')
    with open(name + '.csv', 'r', newline='') as f:
        db.session.add_all([m.MeteoConfig(**row) for row in csv.DictReader(f)])
    db.session.commit()
    logging.info(f'initial {name} loaded into db')

    name = 'meter_config'
    logging.info(f'loading {name}.csv info {name} table...')
    with open(name + '.csv', 'r', newline='') as f:
        for row in csv.DictReader(f):
            fixed: Dict[str, Any] = {k: v if v != '' else None for k, v in row.items()}
            fixed['cumulative'] = strtobool(fixed['cumulative'])
            db.session.add(m.MeterConfig(**fixed))
    db.session.commit()
    logging.info(f'initial {name} loaded into db')


def setup_test_data(db: SQLAlchemy) -> None:
    loc = {'country': 'Poland', 'city': 950463, 'street': 3839,
           'post_code': '00-950', 'building': '7a', 'lat': 39.72, 'lon': 54.77}

    def dt(s: str) -> datetime:
        return datetime.strptime(s, '%Y-%m-%d')

    db.session.add_all([
        m.Node(location=m.Location(**dict(loc, id=1, post_code='00-950'))),
        m.Node(location=m.Location(**dict(loc, id=2, post_code='31-154')),
               meters=[m.Meter(gateway_id='GWTMP1234567', meter_id='m2',
                               model='fif', role='inverter', active_from=dt('2022-01-07'))]),
        m.Node(location=m.Location(**dict(loc, id=3, post_code='31-134')),
               active_from=dt('2019-01-01'), active_to=dt('2020-12-31')),
        m.Node(location=m.Location(**dict(loc, id=4, post_code='31-100')), active_from=dt('2020-01-01'),
               meters=[m.Meter(gateway_id='GWTMP1234568', meter_id='wewe',
                               model='fif', role='input_home', active_from=dt('2020-01-07'),
                               anomalies=[
                                    m.Anomaly(begin=dt('2020-02-05'), end=dt('2020-02-15')),
                                    m.Anomaly(begin=dt('2020-03-08'), end=dt('2020-08-08')),
                                    m.Anomaly(begin=dt('2021-01-02'), end=dt('2021-02-23'))])])
    ])
    db.session.add_all([
        m.MeteoLocations(location_id=a, meteo_id=b) for a, b in db.session.query(m.Location.id, m.Location.post_code)
    ])
    db.session.commit()
    logging.info('test data loaded into db')


if __name__ == '__main__':
    from config import create_app, db

    app = create_app()
    with app.app.app_context():
        replace_schema(db)
        load_config(db)
        if os.getenv('CREATE_DB_ADD_EXAMPLES', False):
            setup_test_data(db)
