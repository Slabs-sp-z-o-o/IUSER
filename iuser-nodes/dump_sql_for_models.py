#!/usr/bin/env python3
import os

from sqlalchemy import create_engine

from config import db


def dump_sql(sql, *multiparams, **params) -> None:
    print(str(sql.compile(dialect=engine.dialect)).strip() + ';\n')


dialect = os.getenv('SQLALCHEMY_DATABASE_URI', 'mysql+mysqlconnector://').split('://')[0]
engine = create_engine(f'{dialect}://', strategy='mock', executor=dump_sql)
db.metadata.create_all(engine, checkfirst=False)
