"""
ch6_statsa.py — Stochastic processes lab: generators, utilities, and an interactive widget.

Usage in a notebook:
    from ch6_statsa import launch_stoch_explorer
    launch_stoch_explorer()

If you already have `ch6_stoch.py` in the same folder and want to use its generators,
this module will automatically import it; otherwise it falls back to built-ins.
"""

import numpy as np
import pandas as pd

# --------- Try to import user's generators if available ---------
try:
    import ch6_stoch as sp  # user's module with compatible API
except Exception:
    class _SP:
        @staticmethod
        def white_noise(n=4096, sigma=1.0, seed=0):
            rng = np.random.default_rng(seed); return rng.standard_normal(n)*sigma
        @staticmethod
        def brownian_motion(n=4096, dt=1.0, sigma=1.0, seed=0):
            rng = np.random.default_rng(seed); dx = rng.standard_normal(n)*sigma*np.sqrt(dt); return np.cumsum(dx)
        @staticmethod
        def ar1_markov(n=4096, phi=0.9, sigma=1.0, seed=0):
            rng = np.random.default_rng(seed); eps = rng.standard_normal(n)*sigma; x = np.zeros(n)
            for t in range(1,n): x[t] = phi*x[t-1] + eps[t]
            return x
        @staticmethod
        def poisson_counts(n=4096, lam=5.0, dt=0.01, seed=0):
            rng = np.random.default_rng(seed); rate = lam*dt; counts = rng.poisson(rate, size=n); N = np.cumsum(counts); return N, counts
        @staticmethod
        def random_sinusoid(n=4096, fs=100.0, A=1.0, f0=5.0, df=0.0, seed=0):
            rng = np.random.default_rng(seed); t = np.arange(n)/fs; f = f0 + (2*rng.random()-1.0)*df; phase = 2*np.pi*rng.random()
            x = A*np.sin(2*np.pi*f*t + phase); return t, x, f, phase
        @staticmethod
        def psd_fft(x, fs=1.0):
            n = len(x); X = np.fft.rfft(x - np.mean(x)); f = np.fft.rfftfreq(n, d=1.0/fs)
            S = (np.abs(X)**2) * (2.0/(n**2)) * (1.0/fs); S[0] = 0.0; return f, S
        @staticmethod
        def autocorr(x, maxlag=None):
            x = x - np.mean(x); n = len(x); 
            if maxlag is None: maxlag = n//4
            R = np.correlate(x, x, mode='full')[n-1:n+maxlag]
            lags = np.arange(0, maxlag+1); R = R/(n - lags); R0 = R[0] if R[0]!=0 else 1.0
            return lags, R/R0
    sp = _SP()

# --------- Statistics helpers (exported) ---------
def skewness(x):
    x = np.asarray(x); mu = np.mean(x); s = np.std(x)
    return 0.0 if s==0 else float(np.mean(((x-mu)/s)**3))

def corr_time_from_acf(lags, R, dt):
    """Approximate correlation time = area under normalized ACF until first zero (seconds)."""
    idx = int(np.argmax(R<=0)) if np.any(R<=0) else len(R)-1
    return float(np.trapz(R[:idx+1], lags[:idx+1]) * dt)

def psd_midband_slope(x, fs, fmin_frac=0.05, fmax_frac=0.5):
    """Fit slope of log10(PSD) vs log10(f) over a midband fraction of [0, f_Nyq]."""
    f, S = sp.psd_fft(x, fs=fs)
    if len(f) < 5: return np.nan
    fnyq = f[-1]
    mask = (f > fnyq*fmin_frac) & (f < fnyq*fmax_frac) & (S > 0)
    if np.count_nonzero(mask) < 5:
        return np.nan
    X = np.log10(f[mask]); Y = np.log10(S[mask])
    A = np.vstack([X, np.ones_like(X)]).T
    slope, _ = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(slope)

def summarize_process(kind, n=8192, fs=100.0, dt=0.01, sigma=1.0, phi=0.9, lam=5.0, A=1.0, f0=6.0, df=1.0, seed=42):
    """Generate a realization for 'kind' and return (t, x, df_stats, (tau, R), (f, S))."""
    if kind=='White noise':
        x = sp.white_noise(n=n, sigma=sigma, seed=seed); t = np.arange(n)/fs; xstat = x; fs_eff = fs
    elif kind=='Brownian motion':
        x_level = sp.brownian_motion(n=n, dt=dt, sigma=sigma, seed=seed); x = np.diff(x_level); t = np.arange(len(x))*dt; xstat = x; fs_eff = 1.0/dt
    elif kind=='Gaussian Markov (AR1)':
        x = sp.ar1_markov(n=n, phi=phi, sigma=sigma, seed=seed); t = np.arange(n)/fs; xstat = x; fs_eff = fs
    elif kind=='Poisson counts':
        N, counts = sp.poisson_counts(n=n, lam=lam, dt=dt, seed=seed); x = counts.astype(float); t = np.arange(n)*dt; xstat = x; fs_eff = 1.0/dt
    elif kind=='Random sinusoid':
        t, x, f_eff, phase = sp.random_sinusoid(n=n, fs=fs, A=A, f0=f0, df=df, seed=seed); xstat = x; fs_eff = fs
    else:
        raise ValueError("Unknown kind")

    # ACF and PSD
    lags, R = sp.autocorr(xstat, maxlag=min(len(xstat)//8, 8000))
    dstep = (t[1]-t[0]) if len(t)>1 else 1.0
    tau = lags * dstep
    f, S = sp.psd_fft(xstat, fs=fs_eff)

    # Stats table
    def lag1_autocorr(xx):
        xx = np.asarray(xx) - np.mean(xx)
        return np.nan if len(xx)<2 else float(np.corrcoef(xx[:-1], xx[1:])[0,1])
    stats = dict(
        mean=float(np.mean(xstat)), std=float(np.std(xstat)), var=float(np.var(xstat)),
        skew=skewness(xstat),
        acf_lag1=lag1_autocorr(xstat),
        corr_time=corr_time_from_acf(lags, R, dstep),
        psd_slope=(np.nan if kind=='Random sinusoid' else psd_midband_slope(xstat, fs_eff)),
    )
    df_stats = pd.DataFrame([stats], index=[kind])
    return t, x, df_stats, (tau, R), (f, S)

# --------- Interactive explorer (exported) ---------
def launch_stoch_explorer():
    """Render an interactive explorer with time series, ACF, PSD, and a stats table."""
    import matplotlib.pyplot as plt
    from ipywidgets import (Dropdown, FloatSlider, IntSlider, VBox, interactive_output, HTML, Layout)
    import warnings
    warnings.filterwarnings("ignore")

    w_kind = Dropdown(options=['White noise','Brownian motion','Gaussian Markov (AR1)','Poisson counts','Random sinusoid'],
                      value='White noise', description='Process', layout=Layout(width='280px'))
    w_n   = IntSlider(value=8192, min=1024, max=65536, step=1024, description='N', continuous_update=False)
    w_fs  = FloatSlider(value=100.0, min=10.0, max=2000.0, step=10.0, description='fs [Hz]', continuous_update=False)
    w_dt  = FloatSlider(value=0.01, min=0.001, max=0.2, step=0.001, description='dt', continuous_update=False)
    w_sigma = FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='sigma', continuous_update=False)
    w_phi   = FloatSlider(value=0.9, min=-0.99, max=0.99, step=0.01, description='phi (AR1)', continuous_update=False)
    w_lam   = FloatSlider(value=5.0, min=0.1, max=50.0, step=0.1, description='λ (Poisson)', continuous_update=False)
    w_A     = FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='A (sin)', continuous_update=False)
    w_f0    = FloatSlider(value=6.0, min=0.1, max=40.0, step=0.1, description='f0 (Hz)', continuous_update=False)
    w_df    = FloatSlider(value=1.0, min=0.0, max=10.0, step=0.1, description='Δf (Hz)', continuous_update=False)
    w_seed  = IntSlider(value=42, min=0, max=99999, step=1, description='seed', continuous_update=False)

    help_box = HTML("<em>Tip:</em> For Brownian motion and Poisson, we analyze <b>increments</b> (counts per Δt) for stationarity; for sinusoid, PSD slope isn't meaningful—check the peak.")

    def _ui_for(kind):
        common = [w_kind, w_seed, w_n]
        if kind in ['White noise','Gaussian Markov (AR1)','Random sinusoid']:
            common += [w_fs]
        if kind in ['Brownian motion','Poisson counts']:
            common += [w_dt]
        if kind in ['White noise','Brownian motion','Gaussian Markov (AR1)']:
            common += [w_sigma]
        if kind=='Gaussian Markov (AR1)':
            common += [w_phi]
        if kind=='Poisson counts':
            common += [w_lam]
        if kind=='Random sinusoid':
            common += [w_A, w_f0, w_df]
        return VBox(common + [help_box])

    ui_box = _ui_for(w_kind.value)

    def driver(kind, n, fs, dt, sigma, phi, lam, A, f0, df, seed):
        t, x, df_stats, (tau, R), (f, S) = summarize_process(kind, n=n, fs=fs, dt=dt, sigma=sigma,
                                                             phi=phi, lam=lam, A=A, f0=f0, df=df, seed=seed)
        # Plots (each in its own figure, per matplotlib rule)
        plt.figure(); plt.plot(t, x); plt.xlabel("t"); plt.ylabel("x(t)"); plt.title(f"{kind}: time series"); plt.grid(True); plt.show()
        plt.figure(); plt.plot(tau, R); plt.xlabel("lag"); plt.ylabel("R(lag)"); plt.title(f"{kind}: normalized autocorrelation"); plt.grid(True); plt.show()
        plt.figure()
        if len(f)>1:
            plt.loglog(f[1:], np.maximum(S[1:],1e-16))
        else:
            plt.plot(f, S)
        plt.xlabel("frequency"); plt.ylabel("PSD"); plt.title(f"{kind}: power spectral density"); plt.grid(True, which='both'); plt.show()
        # Table
        from IPython.display import display
        display(df_stats.round(4))

    out = interactive_output(driver, {
        "kind": w_kind, "n": w_n, "fs": w_fs, "dt": w_dt, "sigma": w_sigma,
        "phi": w_phi, "lam": w_lam, "A": w_A, "f0": w_f0, "df": w_df, "seed": w_seed
    })

    def on_kind_change(change):
        nonlocal ui_box
        ui_box = _ui_for(change['new'])
        container.children = (ui_box, out)
    w_kind.observe(on_kind_change, names='value')

    from ipywidgets import VBox
    container = VBox([ui_box, out])
    display(container)


