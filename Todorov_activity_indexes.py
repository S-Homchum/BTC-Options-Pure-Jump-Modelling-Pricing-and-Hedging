import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gamma
from scipy import integrate


# Activity index estimator
def Beta_hat(p,X):
    V_delta=np.sum(np.abs(X['log_price'].diff())**p)
    # k=2
    V_delta_II=np.sum(np.abs(X['log_price'].iloc[::2].diff())**p)
    beta_hat= (np.log(2)*p)/(np.log(2)+np.log(V_delta_II)-np.log(V_delta))
    return beta_hat   


# test pure jump vs jump diffusion

class PJvsJDTest:
    def __init__(self, X, p, delta_n):
        self.X = X
        self.p = p
        self.delta_n = delta_n
        self.mu_p = self._compute_mu_p()
        self.mu_2p = self._compute_mu_2p()
        self.mu_p_k = self._compute_mu_p_k()

    def _compute_mu_p(self):
        return (2 ** (self.p / 2)) * gamma((self.p + 1) / 2) / np.sqrt(np.pi)

    def _compute_mu_2p(self):
        return (2 ** self.p) * gamma((2 * self.p + 1) / 2) / np.sqrt(np.pi)

    def _compute_mu_p_k(self):
        p = self.p
        f_u     = lambda u_2, u_1: np.exp(-(u_1**2 + u_2**2) / 2) / (2 * np.pi)
        pos_g_u = lambda u_2, u_1: np.sign(u_1)*np.abs((u_1**p)) * np.sign(u_1+u_2)*np.abs(((u_1 + u_2)**p))
        neg_g_u = lambda u_2, u_1: -np.sign(u_1)*np.abs((u_1**p)) * np.sign(u_1+u_2)*np.abs(((u_1 + u_2)**p))

        def integrand_pos(u_2, u_1):
            return pos_g_u(u_2, u_1) * f_u(u_2, u_1)

        def integrand_neg(u_2, u_1):
            return neg_g_u(u_2, u_1) * f_u(u_2, u_1)

        I1, _ = integrate.dblquad(integrand_pos, 0, np.inf, lambda u_1: -u_1, np.inf)
        I2, _ = integrate.dblquad(integrand_pos, -np.inf, 0, -np.inf, lambda u_1: -u_1)
        I3, _ = integrate.dblquad(integrand_neg, 0, np.inf, -np.inf, lambda u_1: -u_1)
        I4, _ = integrate.dblquad(integrand_neg, -np.inf, 0, lambda u_1: -u_1, np.inf)

        return I1 + I2 + I3 + I4

    def _abs_series(self):
        return self.X.abs() ** (self.p / 2)

    def compute_K1(self):
        return 2 / (np.log(2) * self.mu_p * self.p * np.sqrt(self.delta_n))

    def compute_K2(self):
        abs_s = self._abs_series()
        numerator   = np.sqrt(abs_s.rolling(window=4).apply(np.prod).dropna().sum())
        denominator = abs_s.rolling(window=2).apply(np.prod).dropna().sum()
        return numerator / denominator

    def compute_K3(self):
        return np.sqrt(
            3 * self.mu_2p
            - 2 * (2 ** (1 - self.p / 2)) * self.mu_p_k
            + self.mu_p ** 2)

    def compute_K_hat(self):
        return self.compute_K1() * self.compute_K2() * self.compute_K3()
    
    def test(self,beta_hat):
        K_hat = self.compute_K_hat()
        dist = norm(loc=0, scale=K_hat)   # N(0, K_hat^2)
        z= (np.log(beta_hat)-np.log(2))/np.sqrt(self.delta_n)
        p_value = dist.cdf(z)
        reject= p_value < 0.05  # one-sided 95% confidence interval
        return {"p_value": p_value,
                "z":z,
            "reject H_0": reject}
    

