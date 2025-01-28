import pytest
from tests.end_to_ends.base import TestBase


@pytest.mark.usefixtures('nodes_setup')
class TestBehaveSetup(TestBase):
    """
    This is special class for setting up BigQuery and nodes database for Behave tests.
    It doesn't test anything in reality, it just set up things.
    """

    @pytest.mark.behave_setup
    @pytest.mark.parametrize('scenario_name', ['behave_setup_scenario_1'])
    def test_behave_setup(self, scenario_name: str):
        pass
