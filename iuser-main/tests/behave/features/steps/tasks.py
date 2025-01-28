import logging
import time

from behave import given, then, use_step_matcher, when

use_step_matcher('re')

FINAL_STATES = {'SUCCESS', 'FAILURE', 'CANCELLED'}
QUEUED_STATES = {'PENDING', 'STARTED'}
STATES = FINAL_STATES | QUEUED_STATES

@given(r'nie więcej niż (?P<length>[0-9]+) zadań aktualnie w kolejce')
def validate_queue_length(context, length: str):
    resp = context.ml.get('/task/')
    logging.debug(f'got status list code: {resp.status_int}, resp: {resp.text[:50]}…')
    active_tasks = sum(1 for t in resp.json['tasks'] if t['status'] in QUEUED_STATES)
    logging.info(f'queue length = {active_tasks}')
    if active_tasks > int(length):
        context.scenario.skip(reason=f'there are {active_tasks} (more then {length}) tasks queued')


@when(r'poczekam na (?P<expected>uruchomienie|zakończenie) ostatniego zadania'
      r' nie dłużej niż (?P<timeout>\d+) (?P<time_unit>sekund.?|minut.?)')
def wait_for_task(context, expected: str, timeout: str, time_unit: str):
    states = {'zakończenie': FINAL_STATES}
    states['uruchomienie'] = states['zakończenie'] | {'STARTED'}

    assert context.tasks, 'no tasks started in this scenario'
    task = context.tasks[-1]
    logging.debug(f'looking for #{len(context.tasks)} task {task}')
    if task.get('status', 'PENDING') not in states[expected]:
        _timeout = int(timeout) * (60 if time_unit.startswith('minut') else 1)
        max_time = time.time() + _timeout
        while time.time() < max_time:
            time.sleep(5)
            update_last_task_status(context)
            if task.get('status') in states[expected]:
                break
        else:
            raise TimeoutError(f'{task["task_id"]} not in {states[expected]} in {_timeout} sec')
    else:
        logging.warning(f'not waiting for task {task["task_id"]} state {states[expected]}')


@when(r'anuluję (?P<mode>ostatnie|wszystkie) zadani[ea]')
def cancel_task(context, mode: str):
    assert context.tasks, 'no tasks started in this scenario'
    if mode == 'ostatnie':
        to_cancel = context.tasks[-1:]
        start = len(context.tasks)
    else:
        to_cancel = context.tasks
        start = 1
    for n, task in enumerate(to_cancel, start):
        logging.debug(f'last #{n} task {task}')
        resp = context.ml.post(f'/task/{task["task_id"]}/cancel')
        logging.debug(f'cancel task #{n} code: {resp.status_int}, resp: {resp.text}')
        assert resp.status_int in {202, 204}, f'unknown http code: {resp.status_int}'
        msg = 'already canceled' if resp.status_int == 204 else 'cancellation requested'
        logging.info(f'task #{n} {msg}')


def update_last_task_status(context):
    update_task_status(context, -1)


def update_all_tasks_statuses(context):
    for n, _ in enumerate(context.tasks, 1):
        update_task_status(context, n)


def update_task_status(context, task_number: int):
    """Update task data at context.tasks list.

    Tasks numbered as 1, 2, 3…, N or -N…, -2, -1.
    """
    assert context.tasks, 'no tasks started in this scenario'
    if task_number < 0:
        task_number += len(context.tasks) + 1
    assert 0 < task_number <= len(context.tasks), f'incorrect task numer {task_number}: max {len(context.tasks)}'
    id = context.tasks[task_number-1].get('task_id')
    assert id, f'unknown #{task_number} task ID'
    resp = context.ml.get(f'/task/{id}')
    logging.debug(f'got status for task #{task_number} code: {resp.status_int}, resp: {resp.text}')
    if context.tasks[task_number-1].get('status') != resp.json['status']:
        logging.info(f'task #{task_number} {id} new state {resp.json["status"]}' +
                     (f' {resp.json["message"]:.40}…' if resp.json['status'] == 'FAILURE' else ''))
    context.tasks[task_number-1].update(resp.json)

    # store result in models/predictions list
    if resp.json.get('status') == 'SUCCESS':
        if 'model_id' in resp.json['result']:
            idx = next(i for i, m in enumerate(context.models) if m.get('task_id') == id)
            context.models[idx].update(resp.json['result'])
        elif 'forecast_creation_time' in resp.json['result']:
            idx = next(i for i, p in enumerate(context.predictions) if p.get('task_id') == id)
            context.predictions[idx].update(resp.json['result'])


@then(r'komunikat błędu będzie zawierał "(?P<txt>.+)"')
def check_task_message(context, txt: str):
    assert context.tasks, 'no tasks started in this scenario'
    update_last_task_status(context)
    last = context.tasks[-1]
    assert 'message' in last, f'no error message in task #{len(context.tasks)}'
    msg = last['message']
    assert txt in msg, f'{txt} not found in #{len(context.tasks)} task error message: {msg}'
    logging.info(f'#{len(context.tasks)} task error message: {msg}')


@then(r'status ostatniego zadania będzie równy (?P<status>[-A-Z/]+)')
def check_task_status(context, status: str):
    assert context.tasks, 'no tasks started in this scenario'
    expected = status.split('/')
    assert all([st in STATES for st in expected]), 'unknown status'
    update_last_task_status(context)
    last = context.tasks[-1]
    assert 'status' in last, f'no status in task #{len(context.tasks)}'
    current = last['status']
    logging.info(f'#{len(context.tasks)} task status: {current}')
    assert current in expected, f'current #{len(context.tasks)} task status {current} is not {status}'


@then(r'statusy wszystkich zadań będą równe (?P<status>[-A-Z/]+)')
def check_all_tasks_status(context, status: str):
    assert context.tasks, 'no tasks started in this scenario'
    expected = status.split('/')
    assert all([st in STATES for st in expected]), 'unknown status'
    update_all_tasks_statuses(context)
    statuses = []
    for n, task in enumerate(context.tasks, 1):
        statuses.append(task.get('status'))
        logging.info(f'#{n} task status: {statuses[-1]}')
    assert all(s in expected for s in statuses), f'not all task statuses {statuses} are {status}'
