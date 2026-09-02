"""
WORK IN PROGRESS — this module is being actively developed.
 
Status
------
Working:
  - Jump_Models.phi_CGMY  : CGMY characteristic function
  - Jump_Models.FT_pricer : Carr-Madan Fourier pricer for inverse options,
                            vectorised across strikes
  - relative_entropy      : e(Q|P) between two CGMY Levy measures
  - calibrate_entropy     : vega-weighted least squares, with or without the
                            entropy penalty (SLSQP)
  - regularised_factor_bisection : discrepancy-principle search for the
                            regularisation weight
 
To be completed:
  - FFT_pricer  : the multi-strike FFT grid version. FT_pricer is the one
                  currently used; FFT_pricer is not yet correct. 
  - Levy_Gradient / Gradient_aux_func : analytic gradients of the objective,
  - BFGS       : hand-rolled optimiser, awaiting the gradients above.
  - DE_Global  : Differential Evoluition, a global optimisation method, used to scope the plausible parmeters space
                before using the BFGS. 

"""
import numpy as np
from scipy.special import gamma
from scipy.stats import norm

class Jump_Models:  #ν(z) 
    def __init__(self,X,Σ,μ,m,K,t): 
        self.X=X # log(F) df : Termed futures price
        self.Σ=Σ #cont part in Levy triplet
        self.μ=μ #drif part in Levy triplet
        self.m=m #log(moneyness)
        self.K=K #K --> strike (array)
        self.k=np.log(K) # log of strike prices
        self.t=t #ttm

    def φ_CGMY(self, u, C, β, λ_n, λ_p, t):     # t passed in, not read from self # matter !!! 
        return np.exp(t * C * gamma(-β) * ((λ_p - u*1j)**β - λ_p**β+ (λ_n + u*1j)**β - λ_n**β))

# Fourier Transform pricer with single strike
# Price the BTC-ETH inverse option
    def FT_pricer(self, params, N, η, α, r): # α must be negative (Inverse Call = Put)
        C, β, λ_n, λ_p = params
        k = np.atleast_1d(np.asarray(self.k, dtype=float))          # (M,)
        M = k.shape[0]
        def col(x):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            if x.size == 1:
                x = np.full(M, x.item())
            return x.reshape(-1, 1)
        
        t = col(self.t)                                             # (M,1)
        X_inverse = col(-self.X)        #log(1/F)                   # (M,1)
        r = col(r)                                                  # (M,1)

        i = np.arange(N); v = η * i; u = v - (α + 1)*1j             # (N,)

        ω = -np.log(self.φ_CGMY(-1j, C, β, λ_n, λ_p, t)).real / t   # (M,1)
        ψ = (np.exp(-r * t)
            * self.φ_CGMY(u, C, β, λ_n, λ_p, t)
            * np.exp(1j * u * (X_inverse + t * (r + ω)))
            / (α**2 + α - v**2 + 1j*(2*α + 1)*v))                  # (M,N)

        m = np.arange(N); w = (3 + (-1.0)**(m + 1)) * η / 3; w[0] = η / 3
        integrand = np.exp(-1j * (k.reshape(-1, 1) * v)) * ψ        # (M,N)
        return np.exp(-α * k) / np.pi * (integrand @ w).real        # (M,)

# ε(ℚ|ℙ) 
# Also use Trapezoidal rule in entropy integration
### Check the integration in negative side
from scipy.special import xlogy
def relative_entropy(old_params, params, t, N, η):
    C_old, β_old, λ_n_old, λ_p_old = old_params
    C, β, λ_n, λ_p = params
    i = np.arange(N)
    v = η * (i + 1)   
    dQ_by_dP_pos = (C/C_old)*np.exp((λ_p_old-λ_p)*v)*(λ_p*v+1)/((v**(β-β_old))*(λ_p_old*v+1))
    dQ_by_dP_neg = (C/C_old)*np.exp((λ_n_old-λ_n)*v)*(λ_n*v+1)/((v**(β-β_old))*(λ_n_old*v+1))

    m = np.arange(N)
    w = (η/3) * (3 + (-1.0)**(m + 1))
    w[0] = η/3

    P_pos = (C_old*np.exp(-λ_p_old))/(v**(1+β_old))
    P_neg = (C_old*np.exp(-λ_n_old))/(v**(1+β_old))

    integrand_pos = xlogy(dQ_by_dP_pos, dQ_by_dP_pos) + 1 - dQ_by_dP_pos
    integrand_neg = xlogy(dQ_by_dP_neg, dQ_by_dP_neg) + 1 - dQ_by_dP_neg

    entropy = t * ((integrand_pos @ (P_pos*w)) + (integrand_neg @ (P_neg*w)))
    return entropy
# Auxiliatary Function (compute only once) 
### put this into the Levy Gradient
def Gradient_aux_func(): #vec func 
    pass
# f(x)= C_T(k)
def Levy_Gradient(params):#vec func
    pass

###need checking and (may-be) upgrading
def BFGS(func,step,N_iter,tol,H_0_inv,grad,x_0):
    def obj_func(x):   # must be a vector function
        return func(x,*args,**kwargs)
    def grad_k(x): # must be a vector function
        return grad(x,*args,**kwargs)
    def factor_H_k(d_grad,d_x,H_k): #inverse Hessian
        d_x=np.asarray(d_x).reshape(-1,1)
        d_grad=np.asarray(d_grad).reshape(-1,1)
        H_k=np.asarray(H_k)
        γ=d_x.T@d_grad
        a=H_k@d_grad # (n,n) @ (n,1)= (n,1) 
        return ((γ + d_grad.T @ a) / γ**2) * (d_x @ d_x.T)- (1 / γ) * (a @ d_x.T)- (1 / γ) * (d_x @ a.T)
    #loop
    i=0
    x_input=x_0 #parameters 
    H_input=H_0_inv #initial Hessian
    while i<N_iter:
        obj=obj_func(x_input)
        if np.abs(obj)<=tol:
            break
        grad_lag=grad_k(x_input)
        x_lag=x_input
        H_lag=H_input
        x_input=x_input-step*(H_input@grad_lag)
        H_input=H_input+factor_H_k(grad_k(x_input)-grad_lag,x_input-x_lag,H_lag)
        i=i+1

    return x_input

def DE_global():
    pass
# Finding regularised factor
#--- Use Bisection Search--- #
def regularised_factor_bisection(e_0, market_params, K_market, C_market, vega, X, Σ, μ, t,
                       r, δ, N, η, α_damp,
                       α_lo, α_hi, max_eval, tol=1e-3):
    """
    Discrepancy principle: find α such that  e(α) := Σ w_i (C_model,i − C_mkt,i)² ≈ δ·e_0.
    e(α) is nondecreasing, so bisect on log α.
    """
    pricer   = Jump_Models(X, Σ, μ, r, K_market, t)      # K, not log K
    w        = 1.0 / np.asarray(vega, dtype=float)
    C_market = np.asarray(C_market, dtype=float)
    target   = δ * e_0
    cache    = {}

    def e(α_, x0=None):
        if α_ not in cache:
            θ, _ = calibrate_entropy(market_params, K_market, C_market, vega,
                                     X, Σ, μ, t, N=N, η=η, α_damp=α_damp, r=r,
                                     α_ent=α_, x0=x0, ε=1e-6, ent=True)
            diff = pricer.FT_pricer(θ, N, η, α_damp, r) - C_market
            cache[α_] = (float(w @ diff**2), θ)
        return cache[α_]

    e_lo, θ_lo = e(α_lo)
    if e_lo >= target:                       # even α→0 overshoots: δ too small
        return α_lo, θ_lo, {'status': 'lower_bracket', 'err': e_lo, 'target': target}

    e_hi, θ_hi = e(α_hi, x0=θ_lo)
    if e_hi <= target:                       # penalty never bites: widen α_hi
        return α_hi, θ_hi, {'status': 'upper_bracket', 'err': e_hi, 'target': target}

    n = 2
    while n < max_eval and np.log(α_hi / α_lo) > 1e-6:
        α_mid = np.sqrt(α_lo * α_hi)                     # geometric midpoint
        e_mid, θ_mid = e(α_mid, x0=θ_lo)                 # warm start from better fit
        n += 1
        if abs(e_mid - target) <= tol * target:
            return α_mid, θ_mid, {'status': 'converged', 'err': e_mid, 'n_eval': n}
        if e_mid < target:
            α_lo, e_lo, θ_lo = α_mid, e_mid, θ_mid
        else:
            α_hi, e_hi, θ_hi = α_mid, e_mid, θ_mid

    return α_lo, θ_lo, {'status': 'max_eval', 'err': e_lo,
                        'bracket': (α_lo, α_hi), 'n_eval': n}



### not being used here 
### not finished yet
### may be useful in the future      
### need further modification
    def FFT_pricer(self,params,N,η,α,φ,r=0):
        C, β, λ_n, λ_p = params # use 'plumbing' 
         #log(S) discretization
        i = np.arange(N) 
        v = η * i
        ω = -np.log(self.φ_CGMY(-1j,C,β,λ_n,λ_p))/self.t # Convexity correction term
        #log(K) discretization
        λ=(2*np.pi)/(N*η)

        #### not finished yet
        k_FFT=np.arange(self.k-np.pi/η, self.k+(N-1))
        ψ_FFT = np.exp(-r * self.t) * self.φ_CGMY(v - (α + 1) * 1j,C,β,λ_n,λ_p)*np.exp(1j*(v-(α+1)*1j)*(self.X+self.t*(r+ω)))/(
                    α**2 + α - v**2 + 1j * (2 * α+ 1) * v)

        w= np.empty(N)
        for m in range(N):
            w[m] = (3 + (-1)**(m + 1) - (1 if m == 0 else 0)) * η/ 3 #Simpson Rule

        x = np.exp(1j * b * v) * ψ_FFT * w
        y = np.fft.fft(x)

        C_price = np.exp(-α * k) / np.pi * np.real(y)
        K = np.exp(k)

        if strikes is None:
            return K, C
        return np.interp(np.log(strikes), k, C_price)

