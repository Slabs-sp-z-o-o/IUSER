FROM python:3.8-slim AS rest_server
LABEL project=pwrability

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV PYTHONPATH /app

COPY api api/
COPY app.py auth.py config.py models.py pa_openapi.yaml views.py ./
COPY common common/
CMD [ "./app.py" ]
HEALTHCHECK --timeout=2s CMD bash -c 'cat /dev/null >/dev/tcp/localhost/5000'

FROM rest_server AS init_db

COPY init_db.py *_config.csv ./
CMD [ "./init_db.py" ]
HEALTHCHECK NONE


FROM init_db AS tests

COPY tests tests/
COPY .coveragerc ./
CMD pytest -ra --log-level=info --log-cli-level=warning --cov
