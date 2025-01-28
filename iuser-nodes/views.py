from datetime import timezone
from marshmallow import post_dump

from config import ma
from models import Anomaly, Location, Meter, Node, NodeConfig, Serie


class MySchemaWithoutNulls(ma.SQLAlchemyAutoSchema):
    @post_dump
    def remove_none_values(self, data, **kwargs):
        return {key: value for key, value in data.items() if value is not None}

    class Meta:
        load_instance = True


class LocationSchema(MySchemaWithoutNulls):
    class Meta(MySchemaWithoutNulls.Meta):
        model = Location


class NodeSchema(MySchemaWithoutNulls):
    class Meta(MySchemaWithoutNulls.Meta):
        model = Node

    location = ma.Nested(LocationSchema, many=False)
    active_from = ma.NaiveDateTime(timezone=timezone.utc)
    active_to = ma.NaiveDateTime(timezone=timezone.utc, allow_none=True)


class MeterSchema(MySchemaWithoutNulls):
    class Meta(MySchemaWithoutNulls.Meta):
        model = Meter
        include_fk = True
        include_relationships = True
        load_only = ('node_id', 'node', 'anomalies', 'config')

    active_from = ma.NaiveDateTime(timezone=timezone.utc)
    active_to = ma.NaiveDateTime(timezone=timezone.utc, allow_none=True)


class AnomalySchema(MySchemaWithoutNulls):
    class Meta(MySchemaWithoutNulls.Meta):
        model = Anomaly
        include_fk = True
        include_relationships = True
        load_only = ('meter_id', 'meter')

    begin = ma.NaiveDateTime(timezone=timezone.utc)
    end = ma.NaiveDateTime(timezone=timezone.utc)


class MeterNestedSchema(MeterSchema):
    class Meta(MeterSchema.Meta):
        load_only = ('node_id', 'node', 'config')

    anomalies = ma.Nested(AnomalySchema, many=True)


class NodeNestedSchema(NodeSchema):
    class Meta(MySchemaWithoutNulls.Meta):
        model = Node
        include_relationships = True

    meters = ma.Nested(MeterNestedSchema, many=True)


class NodeConfigSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = NodeConfig
        exclude = ('bq_dataset', 'bq_table', 'bq_column', 'is_cumulative',
                   'is_anomaly', 'user_id', 'is_active')

    database = ma.auto_field('bq_dataset')
    table = ma.auto_field('bq_table')
    column = ma.auto_field('bq_column')
    cumulative = ma.auto_field('is_cumulative')
    anomaly = ma.auto_field('is_anomaly')


class SerieNamesSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Serie
        fields = ('name',)

    @post_dump
    def flat_str_array(self, data: dict, **kwargs) -> str:
        return data['name']
