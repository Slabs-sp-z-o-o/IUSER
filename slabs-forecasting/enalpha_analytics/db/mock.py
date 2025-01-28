import pickle
from typing import Optional, List, Dict, Any

import pandas as pd

from ..utils import Model
from ..preprocessing import BoundaryConditions

class MockDBInterface:
    """ Mock database interface that returns a dataframe provided in constructor.
    For testing purposes.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def load_series(self, node, series, target_frequency, skip_anomalies=False,
                    time_range_start=None, time_range_end=None,
                    boundary_condition: Optional[BoundaryConditions]=None) -> pd.DataFrame:
        res = self.df
        res = res[series]
        if time_range_start is not None:
            if boundary_condition:
                periods, timedelta = boundary_condition
                ss = len(res.loc[:time_range_start, :].index)
                ss_idx = res.index[ss - periods] if ss >= periods else res.index[0]
                time_range_start = min(ss_idx, time_range_start - timedelta)
            res = res.loc[time_range_start:, :]
        if time_range_end is not None:
            res = res.loc[:time_range_end, :]
        return res.resample(target_frequency, origin='epoch').interpolate()

    # def load_exogenous(self, columns: Optional[List[str]]=None,
    #                    boundary_condition: Optional[BoundaryConditions]=None,
    #                    start_index: Optional[Any]=None, end_index: Optional[Any]=None):
    #     res = self.df
    #     if columns is not None:
    #         res = res[columns]
    #     if start_index is not None:
    #         if boundary_condition:
    #             periods, timedelta = boundary_condition
    #             ss = len(res.loc[:start_index, :].index)
    #             ss_idx = res.index[ss - periods] if ss >= periods else res.index[0]
    #             start_index = min(ss_idx, start_index - timedelta)
    #         res = res.loc[start_index:, :]
    #     if end_index is not None:
    #         res = res.loc[:end_index, :]
    #     return res

    # def load_endogenous(self, boundary_condition: Optional[BoundaryConditions]=None,
    #                     start_index: Optional[Any]=None, end_index: Optional[Any]=None):
    #     assert self.y is not None
    #     return pd.DataFrame([self.y])


class MockModelStore:
    """ Mock model store.
    For testing purposes.
    """

    def __init__(self):
        self.model_ctr = 0
        self.models: Dict[int, bytes] = {}

    def create_model(self, model: Model):
        model_id = self.model_ctr
        self.model_ctr += 1
        self.models[model_id] = pickle.dumps(model)
        return model_id

    def update_model(self, model_id: int, model: Model):
        self.models[model_id] = pickle.dumps(model)

    def load_model(self, model_id: int) -> Model:
        return pickle.loads(self.models[model_id])
