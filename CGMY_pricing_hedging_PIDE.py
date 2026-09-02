"""
WORK IN PROGRESS — this module is being actively developed.
 
Status
------
Working / drafted:
  - Jump_measure    : the CGMY Levy density k(y)
  - g_1_func, g_2_func : incomplete-gamma helpers used by the jump integral
  - integral_part   : the four-piece quadrature of the integral operator
                      (piecewise-constant plus linear correction, both tails)
 
To be completed:
  - differntial_part : assembles the tridiagonal operator but does not yet
                       return it, and the diagonal terms need checking against
                       the derivation.
  - The time-stepping loop and terminal/boundary conditions.
  - Delta and vega extraction for the hedging leg.
 
The solver takes risk-neutral parameters from the calibration module, so it is
blocked on that module being finalised.
"""
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.special import gammaincc
from scipy.sparse import diags
import scipy.linalg  # use to solve matrix equation


# Finite Difference Scheme
class PIDE_Pricer:
    """
    A = 5*empr_std
    X_max = A
    X_min = -A
    delta_X = 2A/N

    """
    def __init__(self, df, r, X_max, X_min, lambda_n, lambda_p, delta_t, N, K, Y, nu): # X=log(S)
        self.df   = df
        self.X_max = X_max
        self.X_min = X_min
        self.N    = N
        self.K    = K
        self.Y    = Y
        self.lp   = lambda_p
        self.ln_  = lambda_n
        self.dt   = delta_t
        self.nu   = nu # 1/C in CGMY
        self.dx   = (X_max - X_min) / N
        self.H    = self.generate_H()   # N*1 matrix
        self.g1   = self.g_1_func()
        self.g2   = self.g_2_func()
        self.R_i  = self.integral_part() #j=0
        self.k_y  = self.Jump_measure()  #k(y)
        self.r=r #forward rate
# Jump Measure :
    def Jump_measure(self, y):
        I_p = int(bool(np.abs(y)>0))
        I_n = int(bool(np.abs(y))<0)
        return (np.exp(-self.lp*np.abs(y))*I_p/(self.nu*(y**(1+self.Y))))+(np.exp(-self.ln_*np.abs(y))*I_n/(self.nu*(np.abs(y)**(1+self.Y))))

# Gamma function 1&2
    def g_1_func(x,alpha): #alpha>0
        return gammaincc(1-alpha,x)
    def g_2_func(x,alpha): #alpha>0
        return (np.exp(-x)*(x**(-alpha))/alpha)-(gammaincc(1-alpha,x)/alpha)

# Matrix Operation
    def generate_H(self):
        X = np.arange(self.X_min, self.X_max + self.dx, self.dx).reshape(len(self.df),1)
        H = X-np.full(len(self.df,1),self.K) # Option Payoff (price at maturity)
        return H

    def integral_part(self,i): # R_{i,j}; j=0
    # I: negative jump, within the grid bound
        I = sum(
        self.ln_**self.Y * (self.H[i-k] - self.H[i] - k*(self.H[i-k-1] - self.H[i-k])) *
        (self.g2(k*self.dx*self.ln_, self.Y) - self.g2((k+1)*self.dx*self.ln_, self.Y))
        for k in range(1, i))
    # II: negative jump, beyond the grid bound
        II = sum(
        (self.H[i-k-1] - self.H[i-k]) / (self.ln_**(1-self.Y) * self.dx) *
        (self.g1(k*self.dx*self.ln_, self.Y) - self.g1((k+1)*self.dx*self.ln_, self.Y))
        for k in range(1, i))
    # III: positive jump, within the grid bound
        III = sum(
        self.lp**self.Y * (self.H[i+k] - self.H[i] - k*(self.H[i+k+1] - self.H[i+k])) *
        (self.g2(k*self.dx*self.lp, self.Y) - self.g2((k+1)*self.dx*self.lp, self.Y))
        for k in range(1, self.N - i))
    # IV: positive jump, beyond the grid bound
        IV = sum(
        (self.H[i+k+1] - self.H[i+k]) / (self.lp**(1-self.Y) * self.dx) *
        (self.g1(k*self.dx*self.lp, self.Y) - self.g1((k+1)*self.dx*self.lp, self.Y))
        for k in range(1, self.N - i))

        return I + II + III + IV

    def differntial_part(self, q=0):
        g1 = PIDE_Pricer.g_1_func
        g2 = PIDE_Pricer.g_2_func
        g1_0 = gamma(1-self.Y)

        # omega(eps)
        omega_func = (
            (self.lp**self.Y / self.nu)          * g2(self.lp        * self.dx, self.Y)
            - ((self.lp - 1)**self.Y / self.nu)  * g2((self.lp - 1)  * self.dx, self.Y)
            + (self.ln_**self.Y / self.nu)        * g2(self.ln_       * self.dx, self.Y)
            - ((self.ln_ + 1)**self.Y / self.nu)  * g2((self.ln_ + 1) * self.dx, self.Y))

        # sigma(eps)
        pos = (self.lp**(self.Y - 2) / self.nu) * (
            -(self.lp * self.dx)**(1 - self.Y) * np.exp(-self.lp * self.dx)
            + (1 - self.Y) * (g1_0 - g1(self.lp * self.dx, self.Y)))
        neg = (self.ln_**(self.Y - 2) / self.nu) * (
            -(self.ln_ * self.dx)**(1 - self.Y) * np.exp(-self.ln_ * self.dx)
            + (1 - self.Y) * (g1_0 - g1(self.ln_ * self.dx, self.Y)))
        
        sig_func = pos + neg

        B_l = (sig_func * self.dt / (2 * self.dx**2)
               - (self.r - q + omega_func - 0.5 * sig_func) * self.dt / (2 * self.dx))
        B_u = (sig_func * self.dt / (2 * self.dx**2)
               + (self.r - q + omega_func - 0.5 * sig_func) * self.dt / (2 * self.dx))
        
        # coeffs in tridiag matrix
        # l_i and u_i must be vectors
        l_i=np.full(self.N,-B_l)
        u_i=np.full(self.N,-B_u)
        gen_d_I=np.arange(self.N)
        gen_d_II=np.full(self.N,self.N)
        gen_d=gen_d_II-gen_d_I
        d_g2_I=PIDE_Pricer.g_2_func(gen_d_I*self.dx*self.ln_,self.Y)
        d_g2_II=PIDE_Pricer.g_2_func(gen_d*self.dx*self.lp,self.Y)
        d_i= 1+self.r+B_l+B_u+(self.dt*((self.ln_**self.Y)*d_g2_I+(self.lp**self.Y)*d_g2_II)/self.nu)
        tri_diag=diags([l_i,d_i,u_i],offsets=[-1, 0, 1], format='csc')
        
        return 


# ref code: how to construct tridiagonal matrix
def tridiagonal(lower, main, upper):
    n = len(main)
    A = np.zeros((n, n))
    np.fill_diagonal(A, main)
    np.fill_diagonal(A[1:], lower)
    np.fill_diagonal(A[:, 1:], upper)
    return A



def tridiagonal_sparse(lower, main, upper):
    return diags([lower, main, upper], offsets=[-1, 0, 1], format='csc')
