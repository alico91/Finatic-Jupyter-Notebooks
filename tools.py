from scipy.optimize import minimize
import numpy as np


class Tools:

    def __init__(self, tickers, returns, risk_free):
        self.tickers = tickers
        self.returns = returns
        self.covariance = returns.cov()
        self.risk_free = risk_free
        self.annualized_returns = (returns + 1).prod() ** (252 / returns.shape[0]) - 1
        self.annualized_volatility = returns.std() * (252 ** 0.5)

    def portfolio_return(self, weights, returns=None):
        if returns is None:
            returns = self.annualized_returns
            # return weights.T @ returns
        return weights.T @ returns

    def portfolio_vol(self, weights, cov_mat=None):
        if cov_mat is None:
            cov_mat = self.covariance
            # return (weights.T @ cov_mat @ weights) ** 0.5
        return (weights.T @ cov_mat @ weights) ** 0.5

    def minimize_vol(self, target_return, returns=None, cov=None):
        if returns is None:
            returns = self.annualized_returns
        n = returns.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # Range of allowed weights

        # Defining the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights_: np.sum(weights_) - 1
                            }
        return_is_target = {'type': 'eq',
                            'args': (returns,),
                            'fun': lambda weights_, returns_: target_return - self.portfolio_return(weights_, returns_)
                            }
        weights = minimize(self.portfolio_vol, init_guess,
                           args=(cov,), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1, return_is_target),
                           bounds=bounds)
        return weights.x

    def calc_optimal_weights(self, n_points, returns=None, cov=None):
        if returns is None:
            returns = self.annualized_returns

        if cov is None:
            cov = self.covariance

        target_rs = np.linspace(returns.min(), returns.max(), n_points)
        weights = [self.minimize_vol(target_return, returns, cov) for target_return in target_rs]
        return weights

    def max_sharp_ratio(self, returns=None, cov=None, risk_free=None):

        if returns is None:
            returns = self.annualized_returns

        if cov is None:
            cov = self.covariance

        if risk_free is None:
            risk_free = self.risk_free

        n = returns.shape[0]
        init_guess = np.repeat(1 / n, n)
        bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
        # construct the constraints
        weights_sum_to_1 = {'type': 'eq',
                            'fun': lambda weights_: np.sum(weights_) - 1
                            }

        def neg_sharpe(weights_, risk_free_, returns_, cov_):
            r = self.portfolio_return(weights_, returns_)
            vol = self.portfolio_vol(weights_, cov_)
            return -(r - risk_free_) / vol

        weights = minimize(neg_sharpe,
                           init_guess,
                           args=(risk_free, returns, cov), method='SLSQP',
                           options={'disp': False},
                           constraints=(weights_sum_to_1,),
                           bounds=bounds)
        return weights.x
