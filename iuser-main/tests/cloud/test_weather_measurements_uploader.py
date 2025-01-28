import json
import time
from datetime import datetime
from google.cloud import bigquery
from google.cloud import pubsub_v1
import pytest

from common import config

table_id = f'{config.PROJECT_ID}.{config.BIGQUERY_DATASET_NAME}.{config.BIGQUERY_TABLE_WEATHER_FORECASTS}'

client = bigquery.Client()
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(config.PROJECT_ID, config.WEATHER_MEASUREMENTS_UPLOADER_TOPIC)


@pytest.mark.parametrize('postal_code, lat, lon, date',
                         [('99-999', 22.22, 22.22, datetime.now().date())])
def test_gcp_weather_measurements_uploader(postal_code, lat, lon, date):
    QUERY_SPECIFIC_LOCATION = f'''Select * From {table_id}
    WHERE day_from = "{date}"
    AND location = "{postal_code}"'''
    data_to_send = {'postal_code': postal_code, 'lat': lat, 'lon': lon}
    data = json.dumps(data_to_send).encode("utf-8")
    future = publisher.publish(topic_path, data)
    future.result()

    # wait 30s for function to complete
    time.sleep(30)
    df = client.query(QUERY_SPECIFIC_LOCATION).result().to_dataframe()

    assert not df.shape[0] % 336 and df.shape[0] > 0
    # number of minimum records for 7days * 48 predictions(can be more due to launching this test few times)
