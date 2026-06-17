
import numpy as np

def white_noise(n=4096, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)*sigma
    return x

def brownian_motion(n=4096, dt=1.0, sigma=1.0, seed=0):
    """Discrete-time Brownian motion with Gaussian increments N(0, sigma^2 dt)."""
    rng = np.random.default_rng(seed)
    dx = rng.standard_normal(n)*sigma*np.sqrt(dt)
    x = np.cumsum(dx)
    return x

def ar1_markov(n=4096, phi=0.9, sigma=1.0, seed=0):
    """Stationary AR(1): X_t = phi X_{t-1} + eps_t, eps_t~N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)*sigma
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi*x[t-1] + eps[t]
    return x

def poisson_counts(n=4096, lam=5.0, dt=0.01, seed=0):
    """Homogeneous Poisson process counts with rate lam (events per unit time), sampled each dt."""
    rng = np.random.default_rng(seed)
    rate = lam*dt
    counts = rng.poisson(rate, size=n)
    N = np.cumsum(counts)
    return N, counts

def random_sinusoid(n=4096, fs=100.0, A=1.0, f0=5.0, df=0.0, seed=0):
    """Random-phase sinusoid; if df>0, jitter frequency uniformly in [f0-df, f0+df]."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/fs
    f = f0 + (2*rng.random()-1.0)*df
    phase = 2*np.pi*rng.random()
    x = A*np.sin(2*np.pi*f*t + phase)
    return t, x, f, phase

def psd_fft(x, fs=1.0):
    n = len(x)
    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, d=1.0/fs)
    S = (np.abs(X)**2) * (2.0/(n**2)) * (1.0/fs)
    S[0] = 0.0
    return f, S

def autocorr(x, maxlag=None):
    x = x - np.mean(x)
    n = len(x)
    if maxlag is None:
        maxlag = n//4
    R = np.correlate(x, x, mode='full')[n-1:n+maxlag]
    lags = np.arange(0, maxlag+1)
    R = R/(n - lags)
    R0 = R[0] if R[0]!=0 else 1.0
    return lags, R/R0
