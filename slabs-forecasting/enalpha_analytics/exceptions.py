class AnalyticsError(Exception):
    pass


class InvalidFeatureSpecificationError(AnalyticsError):
    pass


class DatabaseError(Exception):
    pass


class NotEnoughDataError(ValueError):
    pass


class ModelNotFoundError(ValueError):
    pass
