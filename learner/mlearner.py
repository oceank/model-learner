from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np


class MLearner:

    def __init__(self, budget, ndim, power_model):
        self.budget = budget
        self.degree = ndim
        self.model = power_model

    def discover(self):
        # performance models has interaction degree of two, based on our study
        model = Pipeline([("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)),
                               ("linear", LinearRegression(fit_intercept=True))])

        # take some ran dom samples
        # this should be replaced with pair wise sampling
        X = np.random.randint(2, size=(self.budget, self.degree))
        y = self.model.evaluateModelFast(X)

        # fit the polynomial model regression
        pmodel = model.fit(X, y)

        return pmodel

    def get_pareto_frontier(self, Xs, Ys, maxX=True, maxY=True):
        # Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        idx_sorted = sorted(range(len(Xs)), key=lambda k: Xs[k])
        # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]
        i = 0
        pareto_idx = [idx_sorted[i]]
        # Loop through the sorted list
        for pair in myList[1:]:
            i += 1
            if maxY:
                if pair[1] >= p_front[-1][1]:
                    p_front.append(pair)
                    pareto_idx.append(idx_sorted[i])
            else:
                if pair[1] <= p_front[-1][1]:
                    p_front.append(pair)
                    pareto_idx.append(idx_sorted[i])
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
        return pareto_idx, p_frontX, p_frontY


