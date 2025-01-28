# syntax=docker/dockerfile:1

FROM python:3.8-slim as autopep8
LABEL project=pwrability
LABEL image=flake8

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /prj
VOLUME [ "/prj" ]
ENTRYPOINT [ "autopep8" ]
CMD [ "--help" ]

FROM autopep8 as unify
ENTRYPOINT [ "unify" ]

FROM autopep8 as isort
ENTRYPOINT [ "isort" ]

FROM autopep8 as flake8
ENTRYPOINT [ "flake8" ]
