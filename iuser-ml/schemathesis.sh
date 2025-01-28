#!/bin/sh

# Uruchomienie tego skryptu nie wymaga instalacji żadnych narzędzi, poza dockerem.
# Uruchamiamy go z bieżącego katalogu podając jako parametr nazwę pliku swaggera
# do przetestowania.
#
# Wyniki testów
# - prezentowane są na ekranie wraz z podsumowaniem
# - zapisywane w raporcie JUnit XML w pliku junit_report.xml (dowolna nazwa pliku
#   podana niżej)
# - wszystkie zrealizowane wywołania API w trakcie testów zapisywane są
#   ze szczegółami w pliku run_log.yaml (dowolna nazwa pliku podana niżej)
#   Uwaga - body/response są zapisywane jako bas64 żeby nie psuć YAML!
#
# Dokumentacja: https://schemathesis.readthedocs.io/

API='http://172.17.0.1:8000/api'

if [ ! -r "$1" ]; then
    echo 'podaj jako parametr nazwę pliku YAML swaggera do przetestowania'
    exit 1
fi

docker run --rm -v $PWD:$PWD -w $PWD schemathesis/schemathesis \
    run --checks=all --target=all --stateful=links --workers=1 \
    --store-network-log=run_log.yaml --junit-xml=junit_report.xml \
    --hypothesis-max-examples=1000 --hypothesis-deadline=10000 \
    --hypothesis-report-multiple-bugs=true --max-response-time=30000 \
    --show-errors-tracebacks --base-url "$API" "$1"
