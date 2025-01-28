import os

import pytest
import json
from common.helpers import get_metadata_file_content


@pytest.fixture
def models_metadata(request):
    results = []
    for param in request.param:
        model_id = param['model_id']
        result = json.loads(get_metadata_file_content(model_id='c8ef82a2-ba16-11eb-8529-0242ac130003'))
        result['model_id'] = model_id
        results.append(result)
    return results


@pytest.fixture(autouse=True)
def temporary_data_path():
    directory = 'tests/cloud/temporary_data'
    os.makedirs(directory, exist_ok=True)
