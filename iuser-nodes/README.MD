# Analytical platform REST API

## Description

This is API definition and implementation for EnAlpha analytical platform used for
compute nodes manipulation.

[API first] design approach is used.

## API definition

[API definition] is stored in [OpenAPI Document] (aka swagger) file [`pa_openapi.yaml`].
Version is stored in this file in info.version line.

## Architecture and components

![components diagram][comp diagram]

### SQL database

#### Schema

Tables and views are defined in [`models.py`] file using [Flask-SQLAlchemy] with
[SQLAlchemy ORM] features.

![database schema diagram][db diagram]

#### RDBMS

MySQL Server v8 is used with [MySQL Connector/Python] library
as a python driver.

#### Database

Data is stored in `pa_api` database, inialized by [`init_db.py`] script.

### Flask app

REST API server is build as Flask application using [Connexion] tool.
[`app.py`] and [`config.py`] are used to configure and run application.
[`auth.py`] implements authentication for API services.
[`models.py`] defines all ORM models used by database and application.
Views used by application are defined in [`views.py`] using [flask-marshmallow]
package with [marshmallow-sqlalchemy] extension.
All bussiness logic (controler) implementations are stored in `api` sub-directory.
[RestyResolver] is used to map *operationId* from swagger file
to module/function names.

## Submodules

Create submodules using:
`git submodule update --remote --force --init`

## Tests

[`run_tests.sh`] script can be used to run unit tests using dockerized test
environment with dedicated database instance (`pa_api_test`).
Tests are using [pytest] framework with [WebTest] module for WSGI-based
applications.

## Deployment

### Local

Docker-compose is used to build and deploy:

* MySQL server
* REST API server
* testing environment

API is available at `http://localhost:PORT/v1/` after deployment and uses `foo`
user with `bar` password for secured endpoints. Port number can be checked using
`docker-compose ps` command or `docker-compose port rest 5000` command.

To flush and initialize local database use `docker-compose run init_db` script.
You can also decide to add example data by setting CREATE_DB_ADD_EXAMPLES environment variable.

[Swagger UI] tool is available at `http://localhost:PORT/v1/ui/` path on
the server.

### Cloud

Cloud Run is used to build and deploy (`./deploy_api_to_cloud_run.sh`):

* REST API server


This API communicates with Cloud SQL instance which is created on GCP.

CAREFUL!
Script `./flush_and_init_gcp.sh` will flush the Cloud SQL database and create
a new one, with the newest configuration!!!

[API definition]: https://swagger.io/resources/articles/difference-between-api-documentation-specification/
[OpenAPI Document]: https://github.com/OAI/OpenAPI-Specification
[API first]: https://swagger.io/resources/articles/adopting-an-api-first-approach/
[Flask-SQLAlchemy]: https://flask-sqlalchemy.palletsprojects.com/
[SQLAlchemy ORM]: https://docs.sqlalchemy.org/en/13/orm/tutorial.html
[MySQL Connector/Python]: https://dev.mysql.com/doc/connector-python
[Connexion]: https://connexion.readthedocs.io/
[flask-marshmallow]: https://flask-marshmallow.readthedocs.io/
[marshmallow-sqlalchemy]: https://marshmallow-sqlalchemy.readthedocs.io/
[RestyResolver]: https://connexion.readthedocs.io/en/latest/routing.html#automatic-routing
[Swagger UI]:https://swagger.io/tools/swagger-ui/
[pytest]: https://docs.pytest.org/
[WebTest]: https://docs.pylonsproject.org/projects/webtest/
[comp diagram]: doc/REST_API.png
[db diagram]: doc/db_schema.png
[`pa_openapi.yaml`]: pa_openapi.yaml
[`models.py`]: models.py
[`init_db.py`]: init_db.py
[`app.py`]: app.py
[`config.py`]: config.py
[`auth.py`]: auth.py
[`models.py`]: models.py
[`views.py`]: views.py
[`run_tests.sh`]: run_tests.sh
