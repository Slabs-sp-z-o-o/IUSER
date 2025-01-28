import logging
from datetime import datetime

from connexion import request, ProblemException
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import StatementError, IntegrityError
from marshmallow.exceptions import ValidationError

from config import db
from models import Meter, Anomaly
from views import MeterSchema
from .nodes import _get_node

_schema = MeterSchema()


def _get_meter(node: int = None, meter: int = None) -> Meter:
    try:
        return Meter.query.filter_by(id=meter, node_id=node).one()
    except (NoResultFound, StatementError):
        raise ProblemException(404, 'Specified resource not found',
                               'Meter instance with specified ID and node ID does not exist')


def _validate(m: Meter) -> None:
    def _overlapping(x: Meter) -> bool:
        return min(m.active_to or datetime.max, x.active_to or datetime.max) > max(m.active_from, x.active_from)

    def _in_range(a: Anomaly) -> bool:
        return m.active_from <= a.begin <= a.end <= (m.active_to or datetime.max)

    if m.active_from > (m.active_to or datetime.max):
        raise ProblemException(422, 'Invalid object definition', 'Meter end date must be later then start date')
    if m.active_from < m.node.active_from:
        raise ProblemException(409, 'Conflict with a parent object', "Meter start date is earlier then node's date")
    if (m.active_to or datetime.max) > (m.node.active_to or datetime.max):
        raise ProblemException(409, 'Conflict with a parent object', "Meter end date is later then node's date")
    try:
        if not all(map(_in_range, Anomaly.query.with_parent(m))):
            raise ProblemException(409, 'Conflict with a child object', "There is an anomaly beyond the meter's lifetime")
        if any(map(_overlapping, [x for x in Meter.query.with_parent(m.node).filter_by(gateway_id=m.gateway_id, meter_id=m.meter_id)
                                          if x != m and x.id != m.id])):
            raise ProblemException(409, 'Conflict with a sibling object', 'Meter instance overlaps with an existing one')
    except IntegrityError as e:
        raise ProblemException(422, 'Invalid object definition', e.orig.msg)


def search(node: int = None, active: bool = None, role: str = None, meter_id: str = None, gateway_id: str = None) -> list:
    q = Meter.query.with_parent(_get_node(node))
    if role:
        q = q.filter_by(role=role)
    if meter_id:
        q = q.filter_by(meter_id=meter_id)
    if gateway_id:
        q = q.filter_by(gateway_id=gateway_id)
    if role:
        q = q.filter_by(role=role)
    if active is not None:
        now = datetime.utcnow()
        clause = db.and_(Meter.active_from <= now,
                         db.or_(Meter.active_to == None,
                                Meter.active_to >= now))
        q = q.filter(clause if active else db.not_(clause))
    return _schema.dump(q, many=True)


def get(node: int = None, meter: int = None) -> object:
    return _schema.dump(_get_meter(node, meter))


def post(node: int = None, body=None) -> tuple:
    _get_node(node)
    try:
        m = _schema.load(dict(**body, node_id=node, node=node))
    except ValidationError as e:
        msg = [f'{f!r} ' + ', '.join(m) for f, m in e.normalized_messages().items()]
        raise ProblemException(422, 'Invalid object definition', f"Invalid params for meter: {'; '.join(msg)}")
    _validate(m)
    db.session.add(m)
    db.session.commit()
    logging.info(f'Meter {m.id} created')
    return m.id, 201, {'Location': f'{request.base_url}/{m.id}'}


def patch(node: int = None, meter: int = None, body=None) -> object:
    m = _get_meter(node, meter)
    try:
        m = _schema.load(dict(**body, node_id=node, node=node), instance=m)
    except ValidationError as e:
        msg = [f'{f!r} ' + ', '.join(m) for f, m in e.normalized_messages().items()]
        raise ProblemException(422, 'Invalid object definition', f"Invalid params for meter: {'; '.join(msg)}")
    _validate(m)
    db.session.add(m)
    db.session.commit()
    logging.info(f'Meter {m.id} updated')
    return _schema.dump(m)
