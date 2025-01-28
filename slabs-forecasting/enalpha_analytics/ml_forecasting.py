import numpy as np
import pandas as pd

from sklearn import linear_model, svm, tree, ensemble, neural_network
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor


class MLForecaster:
    def __init__(self, est_class, **kwargs):
        self.est_params = kwargs
        self.estimator = est_class(**self.est_params)

    def fit(self, endogenous, exogenous):
        """ Assuming `exogenous` and `y` are aligned to epoch.
        """
        self.estimator.fit(exogenous, endogenous)
        try:
            self.exogenous_cols = list(exogenous.columns)
        except AttributeError:
            pass

    def predict(self, n_periods, exogenous):
        """ Assuming `exogenous` and `historic` are aligned to epoch.
        """
        assert exogenous.shape[0] >= n_periods
        res = self.estimator.predict(exogenous.iloc[:n_periods, :])
        return pd.Series(res, index=exogenous.index)


class UpdateableMixin:
    def update(self, y, exogenous):
        raise NotImplementedError()


class LGBMForecaster(MLForecaster, UpdateableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(LGBMRegressor, *args, **kwargs)

    def update(self, y, exogenous):
        self.estimator.fit(exogenous, y, init_model=self.estimator)


class SVMForecaster(MLForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(svm.SVR, *args, **kwargs)


class LinearRegressionForecaster(MLForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(linear_model.LinearRegression, *args, **kwargs)


class LassoForecaster(MLForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(linear_model.Lasso, *args, **kwargs)


class DTForecaster(MLForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(tree.DecisionTreeRegressor, *args, **kwargs)


class RFForecaster(MLForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(ensemble.RandomForestRegressor, *args, **kwargs)


class MLPForecaster(MLForecaster, UpdateableMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(neural_network.MLPRegressor, *args, **kwargs)

    def update(self, y, exogenous):
        self.estimator.partial_fit(exogenous, y)


class PCAWrapper:
    def __init__(self, est, *args, **kwargs):
        self.pca = PCA(*args, **kwargs)
        self.est = est

    def fit(self, endogenous, exogenous):
        xnew = pd.DataFrame(self.pca.fit_transform(exogenous), index=exogenous.index)
        return self.est.fit(endogenous, xnew)

    def predict(self, n_periods, exogenous):
        xnew = pd.DataFrame(self.pca.transform(exogenous), index=exogenous.index)
        return self.est.predict(n_periods, xnew)

    def update(self, y, exogenous):
        xnew = pd.DataFrame(self.pca.fit_transform(exogenous), index=exogenous.index)
        return self.est.update(y, xnew)
