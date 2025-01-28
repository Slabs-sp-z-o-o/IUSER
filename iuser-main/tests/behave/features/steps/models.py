import logging
import uuid
import datetime
from pprint import pformat
from typing import Dict

from behave import then, use_step_matcher, when

use_step_matcher('re')


model_defaults: Dict[str, dict] = {
    '_ALL': {
        'params': {},
        'optimization_metric': 'r2',
        'horizon': '1d',
        'training_data': {
            'time_range_start': '2020-12-12T00:00:00Z',
            'time_range_end': '2020-12-30T00:00:00Z'
        },
        'target_frequency': '1h'
    },
    'autoarima': {
        'algorithm': 'autoarima',
    },
    'autosarima': {
        'algorithm': 'autosarima',
    }
}


@when(r'zlecę utworzenie modelu (?P<algo>\w+) dla (?P<output>zużycia|produkcji)')
def train_model(context, algo: str, output: str):
    endogenous = {'zużycia': 'energy_usage', 'produkcji': 'real_energy_prod'}
    assert context.node_id, 'node ID not set'
    request = model_defaults['_ALL']
    request.update(model_defaults[algo])
    request['training_data']['node'] = context.node_id
    request['training_data']['endogenous'] = endogenous[output]
    logging.debug(f'request:\n{pformat(request)}')
    resp = context.ml.post_json('/forecasting/', request)
    logging.debug(f'POST /forecasting code: {resp.status_int}, resp: {resp.text}')
    task = resp.json['task_id']
    context.tasks.append({'task_id': task})
    context.models.append({'task_id': task})
    logging.info(f'task #{len(context.tasks)} {task} created')


@then(r'(?P<number>[1-9][0-9]* |)model (?P<neg>nie |)będzie gotowy')
def check_if_model_created(context, number: str, neg: str):
    idx = int(number or len(context.models))
    try:
        m = context.models[idx-1]
        logging.debug(f'model #{idx} = {m}')
        logging.info(f'model #{idx} {m["model_id"]} R2 = {m["metrics"]["r2"]}')
        assert not neg, f'model #{idx} exists'
    except IndexError:
        assert neg, f'model #{idx} not requested'
    except KeyError:
        if 'model_id' in m:
            assert neg, f'model #{idx} incomplete - missing R2'
        else:
            assert neg, f'model #{idx} not ready'


@when('zlecę aktualizację nieistniejącego modelu')
def update_nonexisting_model(context):
    model = uuid.uuid1()
    resp = context.ml.post_json(f'/forecasting/{model}/update', {'time_range_end': '2020-12-31T00:00:00Z'})
    logging.debug(f'POST /forecasting/{model}/update code: {resp.status_int}, resp: {resp.text}')
    task = resp.json['task_id']
    context.tasks.append({'task_id': task})
    context.models.append({'task_id': task})
    logging.info(f'task #{len(context.tasks)} {task} created')


@when(r'zlecę aktualizację ostatniego modelu do daty (?P<date>\d{4}-\d{2}-\d{2})(?P<time> \d{2}:\d{2}|)')
def update_model(context, date: str, time: str):
    dt = datetime.datetime.fromisoformat(f'{date}T{time.strip() or "00:00"}:00')
    assert context.models, 'model does not exist'
    model = context.models[-1].get('model_id')
    assert model, 'model not ready'

    request = {'time_range_end': dt.isoformat() + 'Z'}
    logging.debug(f'request: {pformat(request)}')
    resp = context.ml.post_json(f'/forecasting/{model}/update', request)
    logging.debug(f'POST /forecasting/{model}/update code: {resp.status_int}, resp: {resp.text}')
    task = resp.json['task_id']
    context.tasks.append({'task_id': task})
    context.models.append({'task_id': task})
    logging.info(f'task #{len(context.tasks)} {task} created')
