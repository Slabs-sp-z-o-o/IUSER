import logging

from webtest import TestApp
from common import config
from common.controllers.ml_controller import MLController
from common.controllers.nodes_controller import NodesController


def before_all(context):
    logging.basicConfig(level='DEBUG')

    nodes_controller = NodesController()
    ml_controller = MLController()

    context.pa = TestApp(nodes_controller.url)
    """gotowy klient do wywołań API platformy analitycznej"""
    context.pa.authorization = ('Basic', (config.NODES_API_USERNAME, config.NODES_API_PASSWORD))

    context.ml = TestApp(ml_controller.url)
    """gotowy klient do wywołań API modułu ML"""


def before_scenario(context, scenario):
    context.node_id = None
    """ustawiany przez @given numer węzła na którym działa scenariusz"""
    context.tasks = []
    """lista statusów wszystkich zadań zlecanych przez scenariusz, jako dict(uuid: dict)"""
    context.models = []
    """lista parametrów wszystkich modeli utworzonych przez scenariusz"""
    context.predictions = []


def after_step(context, step):
    # workaround for behave bug
    if context.config.color and (context.config.junit or not context.config.stdout_capture):
        print()
