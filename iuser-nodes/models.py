from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy_utils.view import create_view

from config import db

# values used in meter configurations
ROLES = {'input_home', 'inverter', 'meter'}
OPERATIONS = {'plus', 'minus'}

DateTime = DATETIME(fsp=6)


class MeteoConfig(db.Model):
    __tablename__ = 'meteo_output_series'
    serie = db.Column('output_series', db.String(45), primary_key=True)
    database = db.Column('bq_dataset', db.String(45), db.CheckConstraint("bq_dataset > ''"), nullable=False)
    table = db.Column('bq_table', db.String(45), db.CheckConstraint("bq_table > ''"), nullable=False)
    column = db.Column('bq_column', db.String(45), db.CheckConstraint("bq_column > ''"), nullable=False)


class Location(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    country = db.Column(db.Enum('Poland'), nullable=False)
    city = db.Column(db.BigInteger)
    post_code = db.Column(db.String(6), db.CheckConstraint("post_code > ''"), nullable=False)
    street = db.Column(db.BigInteger)
    building = db.Column(db.String(15))
    flat = db.Column(db.String(15))
    lat = db.Column(db.Float, db.CheckConstraint('lat BETWEEN -90 AND 90'))
    lon = db.Column(db.Float, db.CheckConstraint('lon BETWEEN -180 AND 180'))

    __table_args__ = (db.CheckConstraint('(lat IS NULL) = (lon IS NULL)'),)


class MeteoLocations(db.Model):
    location_id = db.Column(db.BigInteger, db.ForeignKey(Location.id, ondelete='CASCADE'), primary_key=True)
    meteo_id = db.Column(db.String(6), db.CheckConstraint("meteo_id > ''"), primary_key=True)

    location = db.relationship(Location, lazy='joined', innerjoin=True, uselist=False, single_parent=True,
                               backref=db.backref('meteo', lazy="joined", innerjoin=True, uselist=False, single_parent=True))


class Node(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    active_from = db.Column(DateTime, nullable=False, server_default=db.text('(utc_timestamp)'))
    active_to = db.Column(DateTime, server_default=None)
    location_id = db.Column(db.BigInteger, db.ForeignKey(Location.id, ondelete='CASCADE'), unique=True, nullable=False)

    __table_args__ = (db.CheckConstraint('active_from <= active_to'),)

    location = db.relationship(Location, lazy='joined', innerjoin=True, uselist=False, single_parent=True,
                               backref=db.backref('node', lazy='joined', innerjoin=True, uselist=False, single_parent=True))


class MeterConfig(db.Model):
    model = db.Column(db.String(45), db.CheckConstraint("model > ''"), primary_key=True)
    role = db.Column(db.Enum(*ROLES), primary_key=True)
    serie = db.Column(db.String(45), primary_key=True)
    database = db.Column(db.String(45), db.CheckConstraint("`database` > ''"), primary_key=True)
    table = db.Column(db.String(45), db.CheckConstraint("`table` > ''"), primary_key=True)
    column = db.Column(db.String(45), db.CheckConstraint("`column` > ''"), primary_key=True)
    description = db.Column(db.String(45))
    operation = db.Column(db.Enum(*OPERATIONS), nullable=False, server_default='plus')
    cumulative = db.Column(db.Boolean, nullable=False, server_default=db.text('True'))


class Meter(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    node_id = db.Column(db.BigInteger, db.ForeignKey(Node.id, ondelete='CASCADE'), nullable=False)
    gateway_id = db.Column(db.String(45), db.CheckConstraint("gateway_id REGEXP '^GW([0-9]{3}|TMP|DEV)[0-9]{7}$'"), nullable=False)
    meter_id = db.Column(db.String(45), db.CheckConstraint("meter_id > ''"), nullable=False)
    active_from = db.Column(DateTime, nullable=False, server_default=db.text('(utc_timestamp)'))
    active_to = db.Column(DateTime)
    model = db.Column(db.String(45), nullable=False)
    role = db.Column(db.Enum(*ROLES), nullable=False)

    db.ForeignKeyConstraint([model, role], [MeterConfig.model, MeterConfig.role])
    node = db.relationship(Node, lazy='joined', innerjoin=True,
                           backref=db.backref('meters', lazy='joined', uselist=True))
    anomalies = db.relationship('Anomaly', lazy='joined', uselist=True, back_populates='meter')

    # real PK for this table
    db.UniqueConstraint(active_from, node_id, gateway_id, meter_id, name='must have different dates')

    __table_args__ = (db.CheckConstraint('active_from <= active_to'),)


class Anomaly(db.Model):
    id = db.Column(db.BigInteger, primary_key=True)
    meter_id = db.Column(db.BigInteger, db.ForeignKey(Meter.id, ondelete='CASCADE'), nullable=False)
    begin = db.Column(DateTime, nullable=False)
    end = db.Column(DateTime, nullable=False)
    db.UniqueConstraint(begin, meter_id)

    __table_args__ = (db.CheckConstraint('begin < end'),)

    meter = db.relationship(Meter, lazy='joined', innerjoin=True, back_populates='anomalies')

# ----------------------- CREATE VIEW... -------------------------------


class MeterAnomalies(db.Model):
    M_COLS = (Meter.id.label('id'), Meter.node_id.label('node_id'),
              Meter.gateway_id.label('gateway_id'), Meter.meter_id.label('meter_id'),
              Meter.model.label('model'), Meter.role.label('role'))

    _meters_without_anomalies = (db.Query([*M_COLS,
                                           Meter.active_from.label('active_from'),
                                           Meter.active_to.label('active_to'),
                                           db.literal(False).label('is_anomaly')])
                                   .select_from(Meter)
                                   .outerjoin(Meter.anomalies)
                                   .filter(Anomaly.meter_id.is_(None)))

    _meters_anomalies = (db.Query([*M_COLS,
                                   Anomaly.begin,
                                   Anomaly.end,
                                   db.literal(True)])
                           .select_from(Meter)
                           .join(Meter.anomalies))

    _first_anom_per_meter = (db.Query([Anomaly.meter_id,
                                       db.func.min(Anomaly.begin).label('begin')])
                               .group_by(Anomaly.meter_id)
                               .subquery())

    _before_anomaly = (db.Query([*M_COLS,
                                 Meter.active_from,
                                 _first_anom_per_meter.c.begin,
                                 db.literal(False)])
                         .select_from(Meter)
                         .join(_first_anom_per_meter, _first_anom_per_meter.c.meter_id == Meter.id)
                         .filter(_first_anom_per_meter.c.begin > Meter.active_from))

    _anom = db.aliased(Anomaly, name='anom')
    _next = db.aliased(Anomaly, name='next')
    _after_from = _anom.end
    _after_to = db.func.ifnull(db.func.min(_next.begin), Meter.active_to)
    _after_anomalies = (db.Query([*M_COLS,
                                  _after_from,
                                  _after_to,
                                  db.literal(False)])
                          .select_from(Meter)
                          .join(_anom, Meter.anomalies)
                          .filter(db.or_(Meter.active_to.is_(None), Meter.active_to > _anom.end))
                          .outerjoin(_next, db.and_(_next.meter_id == _anom.meter_id, _next.begin > _anom.end))
                          .group_by(_anom.id, Meter.active_to)
                          .having(db.or_(_after_to.is_(None), _after_from < _after_to)))

    _PK = ('node_id', 'gateway_id', 'meter_id', 'active_from')
    __table__ = create_view(
        name='meter_anomalies',
        selectable=db.union(_meters_without_anomalies,
                            _meters_anomalies,
                            _before_anomaly,
                            _after_anomalies)
                     .order_by(*_PK),
        metadata=db.metadata)
    for c in _PK:
        __table__.c[c].primary_key = True
    db.ForeignKeyConstraint([__table__.c.node_id], [Node.id])
    db.ForeignKeyConstraint([__table__.c.model, __table__.c.role],
                            [MeterConfig.model, MeterConfig.role])
    config = db.relationship('MeterConfig', lazy='joined', innerjoin=True, uselist=True)
    # config = db.relationship('MeterConfig', lazy='joined', innerjoin=True, uselist=True,
    #                          backref=db.backref('meters_anomalies', lazy='joined', uselist=True))
    # node = db.relationship('Node', lazy='joined', innerjoin=True,
    #                        backref=db.backref('meters_anomalies', lazy='joined', uselist=True))


class NodeConfig(db.Model):
    _telemetry_info = (db.Query([db.literal(1).label('user_id'),
                                 MeterAnomalies.node_id.label('node_id'),
                                 db.literal(True).label('is_active'),
                                 MeterConfig.serie.label('output_series'),
                                 MeterAnomalies.gateway_id.label('gateway_id'),
                                 MeterAnomalies.meter_id.label('meter_id'),
                                 MeterConfig.database.label('bq_dataset'),
                                 MeterConfig.table.label('bq_table'),
                                 MeterConfig.column.label('bq_column'),
                                 MeterConfig.description.label('description'),
                                 MeterConfig.operation.label('operation'),
                                 MeterConfig.cumulative.label('is_cumulative'),
                                 MeterAnomalies.active_from.label('date_from'),
                                 MeterAnomalies.active_to.label('date_to'),
                                 MeterAnomalies.is_anomaly.label('is_anomaly')])
                         .select_from(MeterAnomalies)
                         .join(MeterAnomalies.config))

    __table__ = create_view(name='telemetry_info',
                            selectable=_telemetry_info.selectable,
                            metadata=db.metadata)
    __table__.c.date_from.primary_key = True


class NodeMeteo(db.Model):
    __table__ = create_view(name='nodes_locations',
                            selectable=db.Query([Node.id.label('node_id'),
                                                 MeteoLocations.meteo_id.label('bq_identifier'),
                                                 db.literal(True).label('is_active')])
                                         .select_from(MeteoLocations)
                                         .outerjoin(MeteoLocations.location)
                                         .outerjoin(Location.node)
                                         .selectable,
                            metadata=db.metadata)


class Serie(db.Model):
    _meteo_series = (db.select([MeteoConfig.serie.label('name'),
                                db.literal('meteo').label('fetching_logic')])
                       .select_from(MeteoConfig))
    _telemetry_series = (db.select([MeterConfig.serie.label('name'),
                                    db.literal('telemetry').label('fetching_logic')])
                           .select_from(MeterConfig))
    __table__ = create_view(name='OutputSeries',
                            selectable=(db.union(_meteo_series,
                                                 _telemetry_series)
                                          .order_by('name', 'fetching_logic')),
                            metadata=db.metadata)
