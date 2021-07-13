from benchopt import BaseObjective


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

    def _beta_diff(beta):
        return abs([b-a for a, b in zip(beta[:-1], beta[1:])])

    def compute(self, beta):
        diff = self.y - self.X.dot(beta)
        diff_beta = [b-a for a, b in zip(beta[:-1], beta[1:])]
        return .5 * diff.dot(diff) + self.lmbd * sum(diff_beta)

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)
