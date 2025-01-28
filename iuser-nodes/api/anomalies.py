import logging

from connexion import request, NoContent, ProblemException
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import StatementError, IntegrityError
from marshmallow.exceptions import ValidationError

from config import db
from models import Anomaly, Meter
from views import AnomalySchema
from .meters import _get_meter

_schema = AnomalySchema()


def _get_anomaly(node: int = None, meter: int = None, anomaly: int = None) -> Anomaly:
    try:
        return (Anomaly.query.filter_by(id=anomaly, meter_id=meter)
                       .join(Meter).filter(Meter.node_id == node)
                       .one())
    except (NoResultFound, StatementError):
        raise ProblemException(404, 'Specified resource not found',
                               'Anomaly with specified ID and node/meter ID does not exist')


def _validate(a: Anomaly) -> None:
    def _overlapping(x: Anomaly) -> bool:
        return min(a.end, x.end) > max(a.begin, x.begin)

    if a.begin >= a.end:
        raise ProblemException(422, 'Invalid object definition', 'Anomaly end date must be later then start date')
    if a.begin < a.meter.active_from:
        raise ProblemException(409, 'Conflict with a parent object', "Anomaly start date is earlier then meter's date")
    if a.meter.active_to is not None and a.end > a.meter.active_to:
        raise ProblemException(409, 'Conflict with a parent object', "Anomaly end date is later then meter's date")
    try:
        if any(map(_overlapping, [x for x in Anomaly.query.with_parent(a.meter) if x != a and x.id != a.id])):
            raise ProblemException(409, 'Conflict with a sibling object', 'Anomaly overlaps with an existing one')
    except IntegrityError:
        raise ProblemException(409, 'Conflict with a sibling object', 'Anomaly with the same stard date exists')


def search(node: int = None, meter: int = None) -> list:
    q = Anomaly.query.with_parent(_get_meter(node, meter))
    return _schema.dump(q, many=True)


def get(node: int = None, meter: int = None, anomaly: int = None) -> object:
    return _schema.dump(_get_anomaly(node, meter, anomaly))


def post(node: int = None, meter: int = None, body=None) -> tuple:
    _get_meter(node, meter)
    try:
        a = _schema.load(dict(**body, meter_id=meter, meter=meter))
    except ValidationError as e:
        msg = [f'{f!r} ' + ', '.join(m) for f, m in e.normalized_messages().items()]
        raise ProblemException(422, 'Invalid object definition', f"Invalid params for anomaly: {'; '.join(msg)}")
    _validate(a)
    db.session.add(a)
    db.session.commit()
    logging.info(f'Anomaly {a.id} created')
    return a.id, 201, {'Location': f'{request.base_url}/{a.id}'}


def put(node: int = None, meter: int = None, anomaly: int = None, body=None) -> tuple:
    a = _get_anomaly(node, meter, anomaly)
    try:
        a = _schema.load(dict(**body, meter_id=meter), instance=a)
    except ValidationError as e:
        msg = [f'{f!r} ' + ', '.join(m) for f, m in e.normalized_messages().items()]
        raise ProblemException(422, 'Invalid object definition', f"Invalid params for anomaly: {'; '.join(msg)}")
    _validate(a)
    db.session.add(a)
    db.session.commit()
    logging.info(f'Anomaly {a.id} updated')
    return NoContent, 204


def delete(node: int = None, meter: int = None, anomaly: int = None) -> tuple:
    db.session.delete(_get_anomaly(node, meter, anomaly))
    db.session.commit()
    logging.info(f'Anomaly {anomaly} deleted')
    return NoContent, 204
