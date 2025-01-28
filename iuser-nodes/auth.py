from connexion import ProblemException
from common import config


def basic_auth(username: str, password: str, required_scopes=None) -> dict:
    if username == config.NODES_API_USERNAME and password == config.NODES_API_PASSWORD:
        return {'sub': 'user1', 'scope': 'api'}
    headers = {'WWW-Authenticate': 'Basic realm="Login Required"'}
    raise ProblemException(401, 'Unauthorized', 'Authorization refused',
                           headers=headers)
