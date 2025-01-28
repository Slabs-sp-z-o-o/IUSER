import logging
import os
from distutils.util import strtobool
from json import dumps

from connexion import FlaskApp, RestyResolver
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from swagger_ui_bundle import swagger_ui_3_path
from werkzeug.wrappers import Response
from werkzeug.exceptions import Unauthorized

db = SQLAlchemy()
ma = Marshmallow()


def Unauthorized_handler(exception: Unauthorized) -> Response:
    return Response(dumps({'title': exception.name, 'status': exception.code,
                           'type': 'about:blank', 'detail': exception.description},
                          indent=2, sort_keys=False),
                    exception.code,
                    {'WWW-Authenticate': 'Basic realm="Login Required"'},
                    'application/problem+json')


def create_app() -> FlaskApp:
    logging.basicConfig(level='INFO')
    connex_app = FlaskApp('pa-api-server', options={'swagger_path': swagger_ui_3_path})
    connex_app.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    connex_app.app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
    connex_app.app.config['SQLALCHEMY_ECHO'] = strtobool(os.getenv('SQLALCHEMY_ECHO', 'False'))
    connex_app.add_error_handler(401, Unauthorized_handler)

    db.init_app(connex_app.app)
    ma.init_app(connex_app.app)

    with connex_app.app.app_context():
        connex_app.add_api('pa_openapi.yaml',
                           strict_validation=True,
                           validate_responses=True,
                           auth_all_paths=True,
                           resolver=RestyResolver('api'),
                           resolver_error=501)

    return connex_app
