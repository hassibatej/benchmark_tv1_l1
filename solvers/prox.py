from benchopt import BaseSolver
import numpy as np
import prox_tv as ptv


class Solver(BaseSolver):
    name = 'prox_tv'
    requirements = ['prox_tv']

    # Any parameter defined here is accessible as an attribute of the solver.
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    # Main function of the solver, which compute a solution estimate.

    def run(self, n_iter):
        L = np.linalg.norm(self.X, ord=2) ** 2

        n_features = self.X.shape[1]

        w = np.zeros(n_features)

        for _ in range(n_iter):
            w -= self.X.T.dot(self.X.dot(w) - self.y)/L
            w = ptv.tv1_1d(w, self.lmbd)
        self.w = w

    # Return the solution estimate computed.
    def get_result(self):
        return self.w
