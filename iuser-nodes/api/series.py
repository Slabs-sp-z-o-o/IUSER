from models import Serie
from views import SerieNamesSchema

_schema = SerieNamesSchema(many=True)

def search_clone(prefix: str = None) -> list:
    q = Serie.query
    if prefix is not None:
        q = q.filter(Serie.name.startswith(prefix, autoescape=True))
    return _schema.dump(q)

def search(prefix: str = None) -> list:
    q = Serie.query
    if prefix is not None:
        q = q.filter(Serie.name.startswith(prefix, autoescape=True))
    return _schema.dump(q)
