import logging
import time

from behave import use_step_matcher, when

use_step_matcher('re')


@when(r'poczekam (?P<delay>[1-9][0-9]*) (?P<time_unit>sekund.?|minut.?)')
def wait(context, delay: str, time_unit: str):
    period = int(delay) * (60 if time_unit.startswith('minut') else 1)
    logging.debug(f'sleeping {period} seconds…')
    time.sleep(period)
