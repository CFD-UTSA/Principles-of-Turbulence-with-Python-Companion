import numpy as np

def gen_isotropic_1d(n=2**14, L=10.0, seed=1, spec_slope=-5/3, k0=2*np.pi/10.0, k1=2*np.pi/0.5, smooth=0.2):
    rng = np.random.default_rng(seed)
    dx = L/n
    k = np.fft.rfftfreq(n, d=dx)*2*np.pi
    E = np.zeros_like(k)
    kk = np.where(k==0, 1e-12, k)
    band = (k>=k0)*(k<=k1)
    E[band] = kk[band]**(spec_slope)
    if smooth>0:
        roll_lo = 1/(1+np.exp(-(k-k0)/ (smooth*k0+1e-12)))
        roll_hi = 1/(1+np.exp((k-k1)/ (smooth*k1+1e-12)))
        E *= roll_lo*roll_hi
    amp = np.sqrt(np.maximum(E,0))
    phi = rng.uniform(0, 2*np.pi, size=amp.shape)
    Uhat = amp * np.exp(1j*phi)
    Uhat[0] = 0.0
    u = np.fft.irfft(Uhat, n=n)
    u = (u - np.mean(u))
    if np.std(u)>0: u = u/np.std(u)
    x = np.linspace(0, L, n, endpoint=False)
    return x, u, k, E

def fft_spectrum(u, L):
    n = len(u); dx = L/n
    k = np.fft.rfftfreq(n, d=dx)*2*np.pi
    Uhat = np.fft.rfft(u)
    E = (np.abs(Uhat)**2) * (2.0/(n**2)) * (dx/2.0)
    E[0] = 0.0
    return k, E

def two_point_corr(u, maxlag=None):
    u = np.asarray(u) - np.mean(u)
    n = len(u)
    if maxlag is None: maxlag = n//2
    R = np.correlate(u, u, mode='full')[n-1:n+maxlag]
    lags = np.arange(0, maxlag+1)
    R = R/(n - lags)
    R0 = R[0] if R[0]!=0 else 1.0
    return lags, R/R0

def structure_functions(u, maxsep=2048, order=2):
    u = np.asarray(u)
    n = len(u); maxsep = min(maxsep, n-1)
    seps = np.arange(1, maxsep+1)
    Sp = np.zeros_like(seps, dtype=float)
    for i, r in enumerate(seps):
        diff = u[r:] - u[:-r]
        if order==3:
            Sp[i] = np.mean(diff**3)
        else:
            Sp[i] = np.mean(np.abs(diff)**order)
    return seps, Sp

def taylor_microscale_from_corr(x, R):
    m = min(10, len(x))
    xx = x[:m]; RR = R[:m]
    A = np.vstack([np.ones_like(xx), xx**2]).T
    coef, *_ = np.linalg.lstsq(A, RR, rcond=None)
    a = coef[1]
    if a >= 0: return np.inf
    lam = np.sqrt(-0.5/a)
    return lam
