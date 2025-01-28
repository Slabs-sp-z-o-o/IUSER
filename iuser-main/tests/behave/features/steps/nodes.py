import logging
import random

from behave import given, use_step_matcher

use_step_matcher('re')


@given(r'(?P<undefined>nie|)zdefiniowany węzeł w lokacji o numerze (?P<location_id>[0-9]+)')
def node_definition(context, undefined: str, location_id: str):
    resp = context.pa.get(f'/nodes?location={location_id}')
    assert resp.status_int in {200}
    if resp.json:
        context.node_id = int(resp.json[0]['id'])
    else:
        context.node_id = random.randint(1, 2 ** 63)  # Some node id which doesn't exist (with probability ~1)
    resp = context.pa.get(f'/nodes/{context.node_id}', expect_errors=True)
    logging.debug(f'got node {context.node_id}, code: {resp.status_int}, resp: {resp.text}')
    assert resp.status_int in {200, 404}
    assert resp.content_type in {'application/json', 'application/problem+json'} and resp.has_body and isinstance(
        resp.json, object)
    assert bool(undefined) != (resp.status_int == 200)
    logging.info(f'node {context.node_id} is {"not " if undefined else ""}active')
