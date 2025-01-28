FROM python:3.8-slim as pytest_tests

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV PYTHONPATH /app

COPY . .
CMD ./run_tests_on_dev_no_docker.sh

FROM pytest_tests AS behave_tests

WORKDIR /app/tests/behave
CMD behave --logging-level=debug --no-logcapture
