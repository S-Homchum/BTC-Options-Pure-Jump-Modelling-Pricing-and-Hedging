#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import root
from numba import njit, vectorize, float64


# ============================================================
# BLACK-SCHOLES FUNCTIONS
# ============================================================

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol_scalar(S, K, T, r, C, x0):
    root_fn=lambda x: black_scholes_call(S, K, T, r, x)-C
    result= root(root_fn, x0**2)['x'][0]
    return np.sqrt(result)


implied_vol_scipy = np.vectorize(implied_vol_scalar)


def I(S, K, T, r):
    """Inflexion Point (x0)"""
    m = S / (K * np.exp(-r * T))
    return np.sqrt(2 * np.abs(np.log(m)) / T)


# ============================================================
# NUMBA-OPTIMIZED FUNCTIONS
# ============================================================

@njit
def brentq_numba(f, a, b, args, xtol=1e-12, rtol=1e-12, maxiter=100):
    """
    Numba-compatible Brent's method implementation.
    """
    fa = f(a, *args)
    fb = f(b, *args)
    
    if fa * fb > 0:
        return np.nan
    
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    mflag = True
    d = 0.0
    
    for _ in range(maxiter):
        if abs(b - a) < xtol or abs(fb) < xtol:
            return b
        
        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc)) +
                 b * fa * fc / ((fb - fa) * (fb - fc)) +
                 c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)
        
        tmp2 = (3 * a + b) / 4
        if not ((s > tmp2 and s < b) or (s < tmp2 and s > b)):
            mflag = True
        elif mflag and abs(s - b) >= abs(b - c) / 2:
            mflag = True
        elif not mflag and abs(s - b) >= abs(c - d) / 2:
            mflag = True
        elif mflag and abs(b - c) < xtol:
            mflag = True
        elif not mflag and abs(c - d) < xtol:
            mflag = True
        else:
            mflag = False
        
        if mflag:
            s = (a + b) / 2
            mflag = True
        
        fs = f(s, *args)
        d = c
        c = b
        fc = fb
        
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
    
    return b


@njit
def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit
def norm_pdf(x):
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@njit
def black_scholes_call_numba(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


@njit
def black_scholes_vega_numba(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    
    return S * norm_pdf(d1) * sqrt_T


@njit
def obj_black_scholes(sigma, S, K, T, r, C):
    """Objective function for root finding: BS_price - market_price"""
    return black_scholes_call_numba(S, K, T, r, sigma) - C


@njit
def implied_vol_brentq_numba(S, K, T, r, C, sigma_min=1e-6, sigma_max=5.0, tol=1e-12):
    """Numba-optimized implied volatility calculation using Brent's method."""
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    
    if C < intrinsic * 0.999 or C > S:
        return np.nan
    
    if abs(C - intrinsic) < 1e-10:
        return sigma_min
    
    f_min = black_scholes_call_numba(S, K, T, r, sigma_min) - C
    f_max = black_scholes_call_numba(S, K, T, r, sigma_max) - C
    
    if f_min * f_max > 0:
        if f_min > 0:
            return np.nan
        else:
            sigma_max = 10.0
            f_max = black_scholes_call_numba(S, K, T, r, sigma_max) - C
            if f_min * f_max > 0:
                return np.nan
    
    sigma_implied = brentq_numba(
        obj_black_scholes,
        sigma_min,
        sigma_max,
        (S, K, T, r, C),
        xtol=tol,
        rtol=tol,
        maxiter=100
    )
    
    return sigma_implied


@vectorize([float64(float64, float64, float64, float64, float64)], nopython=True)
def implied_vol_vectorized(S, K, T, r, C):
    """Vectorized implied volatility calculation."""
    return implied_vol_brentq_numba(S, K, T, r, C, 1e-6, 5.0, 1e-12)

@vectorize([float64(float64, float64, float64, float64, float64)], nopython=True)
def black_scholes_vega_vectorized(S, K, T, r, sigma):
    return black_scholes_vega_numba(S, K, T, r, sigma)

def implied_vol(S, K, T, r, C):
    """
    Vectorized implied volatility calculation.
    Handles both scalar and array inputs efficiently.
    """
    S = np.atleast_1d(S)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    r = np.atleast_1d(r)
    C = np.atleast_1d(C)
    
    result = implied_vol_vectorized(S, K, T, r, C)
    
    if result.size == 1:
        return float(result[0])
    return result


def verify_round_trip(S, K, T, r, sigma_true, C_original, tol=1e-12):
    """Test round-trip accuracy: sigma -> price -> implied_vol -> price"""
    sigma_implied = implied_vol_brentq_numba(S, K, T, r, C_original, tol=tol)
    C_recovered = black_scholes_call_numba(S, K, T, r, sigma_implied)
    
    price_error = abs(C_recovered - C_original)
    relative_price_error = price_error / C_original if C_original > 0 else 0
    
    results = {
        'sigma_true': sigma_true,
        'sigma_implied': sigma_implied,
        'C_original': C_original,
        'C_recovered': C_recovered,
        'price_error': price_error,
        'relative_price_error': relative_price_error,
        'passes_roundtrip': price_error < 1e-8
    }
    
    return results
