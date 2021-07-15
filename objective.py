from benchopt import BaseObjective
import numpy as np


class Objective(BaseObjective):
    name = "Total Variation"

    parameters = {
        'fit_intercept': [False],
        'reg': [.1, .5]
    }

    def __init__(self, reg, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.reg = reg

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()

    def compute(self, beta):
        diff = self.y - self.X.dot(beta)
        diff_beta = np.diff(beta)
        l1 = sum(abs(diff_beta))
        return .5 * diff.dot(diff) + self.lmbd * l1

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)
