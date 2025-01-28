import pickle
import datetime
import zipfile
import json
import uuid
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Tuple, Iterable
from common.logger import get_logger
from common import config

import pytz
import pandas as pd
import sqlalchemy
import zstandard as zstd
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from google.cloud import storage
from google.cloud import exceptions as gexceptions
from pybigquery.sqlalchemy_bigquery import BigQueryDialect
from tenacity import retry, wait_exponential, stop_after_attempt

from ..exceptions import DatabaseError, NotEnoughDataError, ModelNotFoundError
from ..utils import Model, ModelId
from ..preprocessing import BoundaryConditions

logger = get_logger(__name__, config.LOG_ON_GCP)


def fix_timezone(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if len(df.index) == 0:
            raise NotEnoughDataError('No data in specified time bounds.')
        else:
            raise RuntimeError(f'Internal error: Data index is not a Datetime Index, but {df.index.__class__}.')

    if not df.index.tzinfo:
        df.index = df.index.tz_localize(datetime.timezone.utc)
    else:
        df.index = df.index.tz_convert(datetime.timezone.utc)
    return df


TIME_COL_NAME = 'timestamp'


class EnalphaDataInterface:
    """
    An interface for retrieving training/inference timeseries data
    using PostgreSQL.

    Arguments
    ---------
    database_name: str
        Name of postgresql database containing the data tables.
    table_name: str
        Name of the table in database `database_name` that contains feature data.
    columns: List[str]
        Names of columns in the table `table_name` that contain feature data.
    label_table_name: str
        Name of the table in database `database_name` that contains label data for training,
        i.e. a sample's class or regression target.
    label_column: str
        Name of the column in the table `table_name` that contains the label data.
    index_column: str
        Name of the column in the table `table_name` that contains the label data.
    user: str
        Database credentials. Defaults to None.
    passwd: str
        Database credentials. Defaults to None.
    host: str
        Database server hostname.
    port: str
        Database server port number.
    start_index: any
        The index of the first record to fetch.
    end_index: any
        The index of the last record to fetch.
    start_time: datetime
        The timestamp of the first record to fetch.
        In reality an alias for start_index.
    end_time: datetime
        The timestamp of the last record to fetch.
        In reality an alias for end_index.
    concurrency: int
        Determines the number of concurrent threads to be used for fetching data.
        Higher value means lower latency, but can raise memory use significantly.
    """

    def __init__(self, project=None, node_db=None, debug_log_query_results=False,
                 concurrency=8):
        node_db = node_db or {}
        if node_db.get('socket_dir') and node_db.get('connection_name'):
            sql_url = sqlalchemy.engine.url.URL('mysql+pymysql',
                                                username=node_db.get('user'),
                                                password=node_db.get('pass'),
                                                query={
                                                    "unix_socket": "{}/{}".format(
                                                        node_db['socket_dir'],
                                                        node_db["connection_name"])
                                                },
                                                database=node_db.get('database'))
        else:
            sql_url = sqlalchemy.engine.url.URL('mysql+pymysql',
                                                username=node_db.get('user'),
                                                password=node_db.get('pass'),
                                                host=node_db.get('host'),
                                                port=node_db.get('port'),
                                                database=node_db.get('database'))
        self.sql_engine = sqlalchemy.engine.create_engine(sql_url)
        self.project = project
        self.bq_engine = sqlalchemy.engine.create_engine(f'bigquery://{project or ""}')
        self.index_col = TIME_COL_NAME
        self.telemetry_executor = ThreadPoolExecutor(max_workers=concurrency)
        self.series_executor = ThreadPoolExecutor(max_workers=concurrency)
        self.debug_log_query_results = debug_log_query_results
        if self.debug_log_query_results:
            self.debug_lock = threading.Lock()
            self.debug_query_result_log = pd.DataFrame()

    def _get_boundary_length(self, boundary_condition: Tuple[Any, Any],
                             resample_freq: pd.Timedelta):
        periods, timedelta = boundary_condition
        if periods != 0:
            # we do resampling later anyway, so no complicated queries to fetch
            # exactly `periods` elements are needed.
            timedelta = max(timedelta, periods * resample_freq)
        return timedelta

    def _get_series_type(self, series_name: str):
        tbl = sqlalchemy.Table('OutputSeries',
                               sqlalchemy.MetaData(),
                               autoload=True,
                               autoload_with=self.sql_engine)
        query = sqlalchemy.select([tbl.c.fetching_logic]).select_from(tbl).where(tbl.c.name == series_name)
        with self.sql_engine.connect() as connection:
            result = connection.execute(query).fetchone()
        if not result:
            raise ValueError(f'Invalid series name: {series_name}')
        return result[0]

    def _get_node_location(self, node_id: int):
        info_tbl = sqlalchemy.Table('nodes_locations',
                                    sqlalchemy.MetaData(),
                                    autoload=True,
                                    autoload_with=self.sql_engine)
        q = sqlalchemy.select([info_tbl.c.bq_identifier]) \
            .select_from(info_tbl) \
            .where(info_tbl.c.node_id == node_id)
        with self.sql_engine.connect() as connection:
            result = connection.execute(q).fetchone()
        if not result:
            raise ValueError(f'Invalid node ID: {node_id}')
        return result[0]

    def _execute_info_query(self, query):
        with self.sql_engine.connect() as connection:
            return connection.execute(query).fetchall()

    def get_debug_query_results(self):
        if self.debug_log_query_results:
            return self.debug_query_result_log
        return None

    def clear_debug_query_results(self):
        if self.debug_log_query_results:
            self.debug_query_result_log = pd.DataFrame()

    def _execute_data_query(self, query) -> pd.DataFrame:
        query = str(query.compile(dialect=BigQueryDialect(), compile_kwargs={"literal_binds": True}))
        logger.debug("executing query %s", query)
        ret = pd.read_gbq(query, index_col=self.index_col)
        return ret

    def _debug_log_result(self, result: pd.Series, series_info: dict):
        if not self.debug_log_query_results:
            return
        with self.debug_lock:
            name = f"{series_info['output_series']}" \
                   f"-{series_info['bq_dataset']}.{series_info['bq_table']}.{series_info['bq_column']}"
            try:
                name += f"-meter={series_info['meter_id']}-gw={series_info['gateway_id']}"
            except KeyError:  # meteo does not have these fields
                pass
            if name not in self.debug_query_result_log:
                self.debug_query_result_log[name] = result
            else:
                self.debug_query_result_log[name] = self.debug_query_result_log[name] \
                    .combine_first(result)

    def _get_mapping_table(self, table_name: str):
        return sqlalchemy.Table(table_name,
                                sqlalchemy.MetaData(),
                                autoload=True,
                                autoload_with=self.sql_engine)

    def _get_data_table(self, table_name: str):
        return sqlalchemy.Table(table_name,
                                sqlalchemy.MetaData(),
                                autoload=True,
                                autoload_with=self.bq_engine)

    def _get_telemetry_info(self, node_id: int, series_name: str, skip_anomalies: bool):
        info_tbl = self._get_mapping_table('telemetry_info')
        cols = [
            'meter_id',
            'gateway_id',
            'bq_dataset',
            'bq_table',
            'bq_column',
            'date_from',
            'date_to',
            'operation',
            'is_cumulative',
        ]
        q = sqlalchemy.select([info_tbl.c[col] for col in cols]) \
            .select_from(info_tbl) \
            .where(info_tbl.c.node_id == node_id) \
            .where(info_tbl.c.output_series == series_name)
        if skip_anomalies:
            q = q.where(info_tbl.c.is_anomaly == False)

        result = self._execute_info_query(q)

        if not result:
            raise ValueError(f'Invalid telemetry series name: {series_name}')
        ret = [{c: v for c, v in zip(cols, row)} for row in result]
        for t in ret:
            t['output_series'] = series_name
            t['date_from'] = t['date_from'].replace(tzinfo=pytz.timezone('UTC'))
            if t['date_to']:
                t['date_to'] = t['date_to'].replace(tzinfo=pytz.timezone('UTC'))
        return ret

    def _add_time_constraints(self, query, start_index: pd.Timestamp,
                              end_index: pd.Timestamp, col=None):
        if col is None:
            col = query.c[self.index_col]
        timestamp = col
        if start_index:
            query = query.where(timestamp >= start_index.isoformat())
        if end_index:
            query = query.where(timestamp <= end_index.isoformat())
        return query.order_by(timestamp)

    def _make_telemetry_query(self, telemetry: dict,
                              start_index: pd.Timestamp, end_index: pd.Timestamp,
                              target_freq: pd.Timedelta):
        tbl_name = f'{telemetry["bq_dataset"]}.{telemetry["bq_table"]}'
        tbl = self._get_data_table(tbl_name)
        col = tbl.c[telemetry['bq_column']]
        logger.debug('Constructing telemetry query from table %s, col %s, range: %s to %s',
                      tbl_name, col, start_index, end_index)

        timestamp = tbl.c[self.index_col]
        res_cols = [timestamp.label('timestamp'), func.min(col).label('values')]
        query = sqlalchemy.select(res_cols) \
            .select_from(tbl) \
            .where(timestamp >= telemetry['date_from'].isoformat()) \
            .where(tbl.c.meter_id == telemetry['meter_id']) \
            .where(tbl.c.gateway_id == telemetry['gateway_id']).group_by(timestamp)
        if telemetry['date_to']:
            query = query.where(timestamp < telemetry['date_to'].isoformat())

        if start_index and telemetry['is_cumulative']:
            start_index -= target_freq
        query = self._add_time_constraints(query, start_index, end_index, col=timestamp)
        return query

    def _handle_resample_error(self, e, series_index, telemetry):
        msg = f"Error while resampling telemetry data {telemetry}."
        if series_index.has_duplicates:
            dups = series_index[series_index.duplicated()].values
            msg += f" Duplicated timestamps exist at: {', '.join(map(str, dups[:20]))}"
            if len(dups) > 20:
                msg += '...'
        else:
            msg += str(e)
        raise ValueError(msg)

    def _get_single_telemetry(self, telemetry: dict,
                              start_index: pd.Timestamp, end_index: pd.Timestamp,
                              target_freq: pd.Timedelta):

        # don't waste time if there is no overlap between the this telemetry and requested period.
        if end_index and end_index < telemetry['date_from']:
            return telemetry, pd.Series()
        if start_index and telemetry['date_to'] and telemetry['date_to'] < start_index:
            return telemetry, pd.Series()

        query = self._make_telemetry_query(telemetry, start_index, end_index, target_freq)
        series = self._execute_data_query(query)['values']

        self._debug_log_result(series, telemetry)

        try:
            series = self.resample(series, target_freq)
        except ValueError as e:
            self._handle_resample_error(e, series.index, telemetry)

        if telemetry['is_cumulative']:
            series = series.diff(1).dropna()

        return telemetry, series

    def _reconstruct_telemetry_series(self, telemetry_value_pairs: Iterable[Tuple[dict, pd.Series]]):
        result = pd.Series()

        for telemetry, value in telemetry_value_pairs:
            op = telemetry['operation']
            if op == 'plus':
                result = result.add(value, fill_value=0.0)
            elif op == 'minus':
                result = result.sub(value, fill_value=0.0)
            else:
                raise ValueError(f'Unrecognized operation in telemetry info: "{op}" in {telemetry}')
        return result

    def _get_telemetry_series(self, node_id: int, series_name: str,
                              start_index: pd.Timestamp, end_index: pd.Timestamp,
                              skip_anomalies: bool, target_freq: pd.Timedelta):
        telemetry_info = self._get_telemetry_info(node_id, series_name, skip_anomalies)

        # BigQuery has huge overhead for small requests, so parallelize them.
        telemetry_value_pairs = (f.result() for f in as_completed(
            [self.telemetry_executor.submit(self._get_single_telemetry, telemetry,
                                            start_index, end_index,
                                            target_freq)
             for telemetry in telemetry_info]))

        return self._reconstruct_telemetry_series(telemetry_value_pairs)

    def _get_meteo_info(self, series_name: str):
        info_tbl = self._get_mapping_table('meteo_output_series')
        cols = [
            'output_series',
            'bq_dataset',
            'bq_table',
            'bq_column',
        ]
        q = sqlalchemy.select([info_tbl.c[col] for col in cols]) \
            .select_from(info_tbl) \
            .where(info_tbl.c.output_series == series_name)

        result = self._execute_info_query(q)

        if not result:
            raise ValueError(f'Invalid meteo series name ID: {series_name}')
        return {c: v for c, v in zip(cols, result[0])}

    def _make_meteo_query(self, meteo_info: dict,
                          start_index: pd.Timestamp, end_index: pd.Timestamp,
                          location):
        tbl_name = f'{meteo_info["bq_dataset"]}.{meteo_info["bq_table"]}'
        logger.debug('Fetching from table %s, range: %s to %s', tbl_name, start_index, end_index)
        tbl = self._get_data_table(tbl_name)
        latest = sqlalchemy.select([func.max(tbl.c.day_from).label('latest'),
                                    tbl.c.period_end]) \
            .select_from(tbl) \
            .where(tbl.c.location == location) \
            .group_by(tbl.c.period_end)
        latest = self._add_time_constraints(latest, start_index, end_index,
                                            col=tbl.c.period_end).alias("latest")
        curr_loc = tbl.select().where(tbl.c.location == location).alias('curr_loc')
        q = sqlalchemy.join(curr_loc, latest,
                            sqlalchemy.and_(latest.c.latest == curr_loc.c.day_from,
                                            curr_loc.c.period_end == latest.c.period_end))
        q = sqlalchemy.select([curr_loc.c['period_end'].label('timestamp'),
                               curr_loc.c[meteo_info['bq_column']].label(meteo_info['output_series'])]) \
            .select_from(q)
        q = self._add_time_constraints(q, start_index, end_index, col=q.c['timestamp'])
        return q

    def _get_meteo_raw_values(self, meteo_info: dict,
                              start_index: pd.Timestamp, end_index: pd.Timestamp,
                              location):
        q = self._make_meteo_query(meteo_info, start_index, end_index, location)
        raw_series = self._execute_data_query(q)[meteo_info['bq_column']]
        self._debug_log_result(raw_series, meteo_info)

        return raw_series

    def _get_meteo_series(self, series_name: str, start_index: pd.Timestamp, end_index: pd.Timestamp,
                          location, target_freq: pd.Timedelta):
        meteo_info = self._get_meteo_info(series_name)
        vals = self._get_meteo_raw_values(meteo_info, start_index, end_index, location)

        if not vals.index.is_unique:
            logger.warning("Meteo series %s contains non-unique forecasts, using only one of the values",
                            series_name)
            vals = vals.groupby(level=0).first()

        return self.resample(vals, target_freq)

    def _fetch(self, node_id: int, output_series: str, target_freq: pd.Timedelta,
               skip_anomalies: bool, boundary_condition: Tuple[Any, Any] = None,
               start_index: pd.Timestamp = None, end_index: pd.Timestamp = None):

        # we need more data if time-series transforms are done
        if boundary_condition is not None and start_index is not None:
            start_index -= self._get_boundary_length(boundary_condition, target_freq)

        df = pd.DataFrame()
        location = self._get_node_location(node_id)
        futures = []
        for s in output_series:
            typ = self._get_series_type(s)
            if typ == 'telemetry':
                futures.append((s, self.series_executor.submit(self._get_telemetry_series,
                                                               node_id, s, start_index, end_index,
                                                               skip_anomalies, target_freq)))
            elif typ == 'meteo':
                futures.append((s, self.series_executor.submit(self._get_meteo_series, s,
                                                               start_index, end_index, location,
                                                               target_freq)))
            else:
                raise ValueError(f'Unrecognized series type: {typ}')

        for name, f in futures:
            values = f.result()
            df[name] = values

        return fix_timezone(df)

    def resample(self, values, target_freq: pd.Timedelta):
        if len(values) > 0:
            values = values.resample(target_freq, origin='epoch').ffill()
            values.dropna(inplace=True)
        return values

    def load_series(self, node: int, series: str, target_frequency: pd.Timedelta,
                    skip_anomalies: bool = False,
                    time_range_start: pd.Timestamp = None, time_range_end: pd.Timestamp = None,
                    boundary_condition: Optional[BoundaryConditions] = None) -> pd.DataFrame:
        """ Load data from SQL and return it as a pandas DataFrame.

        Arguments
        ---------
        columns: Optional[List[str]]
            List of column names to fetch that can override the list provided in constructor.
        start_index: Optional[Any]
            The index of the first record to fetch, excluding the boundary conditions.
            If not given, the value of start_index provided in the constructor is used.
        end_index: Optional[Any]
            The index of the last record to fetch.
            If not given, the value of end_index provided in the constructor is used.
        boundary_conditions: Optional[BoundaryConditions]
            The elements of this tuple indicate, respectively, the number of additional
            records and the time period before start_time that should be fetched
            in order to satisfy boundary conditions for data preprocessing transforms.

        Returns
        -------
        pd.DataFrame
        """
        try:
            ret = self._fetch(node, series, target_frequency, skip_anomalies, boundary_condition,
                              start_index=time_range_start,
                              end_index=time_range_end)
            return ret
        except (SQLAlchemyError, gexceptions.GoogleCloudError) as e:
            raise DatabaseError(e)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp) or isinstance(obj, datetime.datetime):
            return {
                '__timestamp__': True,
                'value': obj.isoformat()
            }
        if isinstance(obj, pd.Timedelta):
            return {
                '__timedelta__': True,
                'value': obj.isoformat()
            }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def as_datetime(obj):
    if obj.get('__timedelta__'):
        return pd.Timedelta(obj['value'])
    if obj.get('__timestamp__'):
        return datetime.datetime.fromisoformat(obj['value'])
    return obj


class EnalphaModelStore:
    """
    A class for retrieving and storing models using PostgreSQL.

    Arguments
    ---------
    database_name: str
        Name of postgresql database containing the data tables.
    user: str
    passwd: str
        Database credentials. Defaults to None.
    host: str
    port: str
        Database server hostname and port number.
    """

    def __init__(self, model_bucket, metadata_bucket):
        self.storage_client = storage.Client()
        self.model_bucket_name = model_bucket
        self.metadata_bucket_name = metadata_bucket

    def _get_bucket(self, bucket_name):
        try:
            return self.storage_client.get_bucket(bucket_name)
        except gexceptions.NotFound:
            raise DatabaseError(f'The Cloud Storage bucket {self.metadata_bucket_name} not found.')

    def _download_model_blob(self, model_id: ModelId):
        model_bucket = self._get_bucket(self.model_bucket_name)
        model_blob = storage.Blob(f'model_{model_id}', model_bucket)
        try:
            return model_blob.download_as_bytes()
        except gexceptions.NotFound:
            raise ModelNotFoundError(
                'Model data not found. Either the metadata bucket is desynchronized with '
                'the model bucket or a model delete operation happened concurrently.')

    def load_model_metadata(self, model_id: ModelId):
        metadata_bucket = self._get_bucket(self.metadata_bucket_name)
        try:
            metadata_blob = storage.Blob(f'model_{model_id}.json', metadata_bucket)
            return json.loads(metadata_blob.download_as_string(), object_hook=as_datetime)
        except gexceptions.NotFound:
            raise ModelNotFoundError(
                'Model metadata not found. Either the metadata bucket is desynchronized with '
                'the model bucket or a model delete operation happened concurrently.')

    def _decode_model(self, est_blob):
        dctx = zstd.ZstdDecompressor()
        try:
            est_blob = dctx.decompress(est_blob)
        except zstd.ZstdError:
            pass  # probably uncompressed
        return pickle.loads(est_blob)

    def model_exists(self, model_id: ModelId):
        metadata_bucket = self._get_bucket(self.metadata_bucket_name)
        metadata_blob = storage.Blob(f'model_{model_id}.json', metadata_bucket)
        return metadata_blob.exists()

    def load_model(self, model_id: ModelId):
        """ Fetch and deserialize a model from the database. """
        estimator = self._decode_model(self._download_model_blob(model_id))
        metadata = self.load_model_metadata(model_id)
        return Model(estimator=estimator, **metadata)

    def create_model(self, model: Model) -> ModelId:
        """ Create a new model in the database. """
        model_id = uuid.uuid4()
        return self.update_model(str(model_id), model)

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4))
    def update_model(self, model_id: ModelId, model: Model):
        """ Update an existing model in GCS.
        The stuff we are trying to save may be a result of an expensive calculation,
        so retry multiple times if uploading fails.
        """
        try:
            metadata_bucket = self.storage_client.get_bucket(self.metadata_bucket_name)

            # upload metadata first, so that it does not get lost if uploading model fails
            metadata_blob = storage.Blob(f'model_{model_id}.json', metadata_bucket)
            metadata_blob.upload_from_string(json.dumps({
                'typ': model.typ,
                'algorithm': model.algorithm,
                'hyperparameters': model.hyperparameters,
                'metrics': model.metrics,
                'metadata': model.metadata,
            }, cls=DateTimeEncoder))

            model_bucket = self.storage_client.get_bucket(self.model_bucket_name)
            model_blob = storage.Blob(f'model_{model_id}', model_bucket)
            cctx = zstd.ZstdCompressor()
            model_blob.upload_from_string(cctx.compress(pickle.dumps(model.estimator,
                                                                     protocol=5)))
        except gexceptions.GoogleCloudError as e:
            raise DatabaseError(f"Failed uploading model to storage: {e}")
        return str(model_id)

    def delete_model(self, model_id: ModelId):
        """ Update an existing model in the database. """
        metadata_bucket = self.storage_client.get_bucket(self.metadata_bucket_name)

        try:
            # delete model first, so that metadata does not get lost if the delete call fails
            model_bucket = self.storage_client.get_bucket(self.model_bucket_name)
            model_blob = storage.Blob(f'model_{model_id}', model_bucket)
            model_blob.delete()

            metadata_blob = storage.Blob(f'model_{model_id}.json', metadata_bucket)
            metadata_blob.delete()
        except gexceptions.NotFound:
            raise ValueError('Invalid model ID')
        except gexceptions.GoogleCloudError as e:
            raise DatabaseError(f"Failed deleting model: {e}")
