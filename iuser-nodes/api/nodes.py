import logging
from datetime import datetime

from connexion import request, ProblemException
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import StatementError, IntegrityError, InvalidRequestError
from marshmallow import ValidationError

from config import db
from models import Node, NodeConfig, Meter, MeteoLocations, Serie
from views import NodeSchema, NodeNestedSchema, NodeConfigSchema

_schema = NodeSchema()
_schema_nested = NodeNestedSchema()
_schema_cfg = NodeConfigSchema(many=True)


def _get_node(node: int = None) -> Node:
    try:
        q = Node.query.options(db.noload('meters'))
        n = q.get(node)
        if n is not None:
            return n
    except (NoResultFound, StatementError):
        pass
    raise ProblemException(404, 'Specified resource not found',
                           'Node with specified ID does not exist')


def _get_serie(serie: str = None) -> Serie:
    try:
        s = Serie.query.get(serie)
        if s is not None:
            return s
    except (NoResultFound, StatementError):
        pass
    raise ProblemException(404, 'Specified resource not found',
                           'Data serie with specified name does not exist')


def _validate(n: Node) -> None:
    def _in_range(m: Meter) -> bool:
        return n.active_from <= m.active_from <= (m.active_to or datetime.max) <= (n.active_to or datetime.max)

    if n.active_to is not None and n.active_from > n.active_to:
        raise ProblemException(422, 'Invalid object definition', 'Node end date must be later then start date')
    if not all(map(_in_range, n.meters)):
        raise ProblemException(409, 'Conflict with a child object', "There is a meter beyond the node's lifetime")
    try:
        Node.query.with_parent(n.location).options(db.noload('location'), db.noload('meters')).one_or_none()
    except IntegrityError:
        raise ProblemException(409, 'Conflict with a sibiling object', 'There is a node with this location already')


def search(active: bool = None, location: int = None) -> list:
    q = Node.query.options(db.noload('meters'))
    if location is not None:
        q = q.filter_by(location_id=location)
    if active is not None:
        now = datetime.utcnow()
        clause = db.and_(Node.active_from <= now,
                         db.or_(Node.active_to == None,
                                Node.active_to >= now))
        q = q.filter(clause if active else db.not_(clause))
    return _schema.dump(q, many=True)


def get(node: int = None, full: bool = None) -> object:
    n = _get_node(node)
    if not full:
        return _schema.dump(n)
    q = Node.query.filter_by(id=node).outerjoin(Node.meters).outerjoin(Meter.anomalies) \
            .options(db.contains_eager(Node.meters, Meter.anomalies)).populate_existing()
    return _schema_nested.dump(q.one())


def post(body=None) -> tuple:
    try:
        node = _schema.load(body)
    except ValidationError as e:
        msg = [f'{f!r} ' + ', '.join(m) for f, m in e.normalized_messages().items()]
        raise ProblemException(422, 'Invalid object definition', f"Invalid params for node: {'; '.join(msg)}")
    except InvalidRequestError:
        raise ProblemException(400, 'Bad Request', 'Node cannot be added to the database')
    _validate(node)
    db.session.add(node)
    try:
        db.session.commit()
    except IntegrityError as e:
        raise ProblemException(409, 'Conflict with a sibling object', e.orig.msg)
    db.session.add(MeteoLocations(location_id=node.location_id,
                                  meteo_id=body['location']['post_code']))
    db.session.commit()
    logging.info(f'Node {node.id} created')
    return node.id, 201, {'Location': f'{request.base_url}/{node.id}'}


def patch(node: int = None, body=None) -> object:
    n = _get_node(node)
    try:
        n = _schema.load(body, instance=n)
    except ValidationError as e:
        msg = [f'{f!r} ' + ', '.join(m) for f, m in e.normalized_messages().items()]
        raise ProblemException(422, 'Invalid object definition', f"Invalid params for node: {'; '.join(msg)}")
    _validate(n)
    db.session.add(n)
    db.session.commit()
    n.location.meteo.location_id = n.location_id  # TODO: gdy zmieniony location_id!
    n.location.meteo.meteo_id = n.location.post_code
    db.session.add(n)
    db.session.commit()
    logging.info(f'loc = {n.location}')
    logging.info(f'loc.met = {n.location.meteo}')
    logging.info(f'Node {n.id} updated')
    return _schema.dump(n)


def get_cfg(node: int = None, serie: str = None, serie_prefix: str = None) -> list:
    _get_node(node)
    q = NodeConfig.query.filter_by(node_id=node)
    if serie is None:
        if serie_prefix is not None:
            q = q.filter(NodeConfig.output_series.startswith(serie_prefix, autoescape=True))
        return _schema_cfg.dump(q)
    _get_serie(serie)
    assert serie_prefix is None, 'serie_prefix ignored when serie name specified'
    return _schema_cfg.dump(q.filter_by(output_series=serie))


get_cfg_one_serie = get_cfg
