"""Diagnostics for the gaze VAE: reconstruction fidelity, preserved properties, latent health.

These are the numbers checkpoints get compared on. Reconstruction error alone is
misleading here: a decoder that smooths tremor away scores well on MSE while lowering
the noise floor the microsaccade detector thresholds against, which silently changes
how many events it finds. So `property_stats` and `latent_stats` carry as much weight
as `reconstruction_metrics`.

    from src.diagnostics import collect, report
    report(*collect(model, val_loader, dev))
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import welch

try:  # works both as `src.diagnostics` and as a plain script
    from .util.microsaccade import microsaccade_extraction
except ImportError:  # pragma: no cover
    from util.microsaccade import microsaccade_extraction

RATE = 1000
V_CLIP = 200.0  # the clip used by GazeBaseWindows.preprocess_sin


def deg(v_sin: np.ndarray) -> np.ndarray:
    """Undo the sinusoidal squash: back to deg/s."""
    return (2 * V_CLIP / np.pi) * np.arcsin(np.clip(v_sin, -1, 1))


def noise_floor(arr: np.ndarray) -> float:
    """Median-based SD of velocity, i.e. what `microsacc` sets its threshold from."""
    v = deg(arr)
    return float(np.median(np.sqrt(np.median((v - np.median(v, -1, keepdims=True)) ** 2, -1))))


@torch.no_grad()
def collect(model, loader, device: str, n_batches: int | None = None):
    """One pass over a loader: returns (x, x_hat, mu, sigma) as (N, C, T) numpy arrays.

    Reconstructions are deterministic (decode mu), which is what reconstruction
    quality should be judged on; sampling is the generative model's concern.
    """
    xs, xh, mus, sds = [], [], [], []
    for i, (v, *_) in enumerate(loader):
        if n_batches is not None and i >= n_batches:
            break
        x = v.float().to(device)
        mu, logvar = model.encode(x)
        xs.append(x.cpu().numpy())
        xh.append(model.decode(mu).cpu().numpy())
        mus.append(mu.cpu().numpy())
        sds.append((0.5 * logvar).exp().cpu().numpy())
    return tuple(np.concatenate(a) for a in (xs, xh, mus, sds))


def reconstruction_metrics(x: np.ndarray, xh: np.ndarray, f_split: float = 60.0) -> dict:
    """Error, and the two ways this decoder is known to fail: lost highs, lower floor."""
    err = xh - x
    f, p_x = welch(deg(x[:, 0]), fs=RATE, nperseg=1024, axis=-1)
    _, p_h = welch(deg(xh[:, 0]), fs=RATE, nperseg=1024, axis=-1)
    hi = f >= f_split
    return {
        "mse": float((err ** 2).mean()),
        "mae": float(np.abs(err).mean()),
        "var_explained": float(1 - err.var() / x.var()),
        "corr_vx": float(np.corrcoef(x[:, 0].ravel(), xh[:, 0].ravel())[0, 1]),
        "corr_vy": float(np.corrcoef(x[:, 1].ravel(), xh[:, 1].ravel())[0, 1]),
        # Fraction of >60 Hz power the reconstruction keeps; 1.0 is perfect.
        #
        # USE THIS ONLY AS A SEED-AVERAGED NUMBER. Three repeats of one identical
        # config gave 0.22, 0.17, 0.12, 0.10 -- a 2x spread from training randomness
        # alone, so it cannot rank single runs. That is a real property of the model,
        # not of the metric: a log/geometric version is worse still, because it is
        # then dominated by bins where the reconstruction has almost no power.
        "hf_retained": float(p_h.mean(0)[hi].mean() / p_x.mean(0)[hi].mean()),
        "noise_floor_ratio": noise_floor(xh) / noise_floor(x),
    }


def microsaccades(v_sin: np.ndarray, vfac: float = 5, mindur: int = 6) -> np.ndarray:
    """Detect in one (2, T) sin-velocity window. Columns: 3 = peak velocity, 6 = amplitude."""
    v = deg(v_sin).T                     # (T, 2) deg/s
    pos = np.cumsum(v, axis=0) / RATE    # integrate, so both sides are treated identically
    return np.asarray(microsaccade_extraction(v, pos, RATE, VFAC=vfac, MINDUR=mindur))


def property_stats(arr: np.ndarray, n_windows: int = 300) -> dict:
    """Microsaccade statistics -- the properties the latent actually has to preserve."""
    arr = arr[:n_windows]
    sacs = [s for s in (microsaccades(w) for w in arr) if len(s)]
    if not sacs:
        return {"rate": 0.0, "amp": np.nan, "peak_vel": np.nan, "main_seq_slope": np.nan}
    s = np.concatenate(sacs)
    amp, vpk = s[:, 6], s[:, 3]
    ok = (amp > 0) & (vpk > 0)
    return {
        "rate": len(s) / (len(arr) * arr.shape[-1] / RATE),
        "amp": float(np.median(amp)),
        "peak_vel": float(np.median(vpk)),
        # the main sequence is a power law; its exponent is the shape-preserving check
        "main_seq_slope": float(np.polyfit(np.log(amp[ok]), np.log(vpk[ok]), 1)[0]),
        "noise_floor": noise_floor(arr),
    }


def latent_stats(mu: np.ndarray, sd: np.ndarray, active_thresh: float = 0.01,
                 snr_thresh: float = 1.0) -> dict:
    """Is the latent using its capacity, and what does its time structure look like?

    `hi_lo` is only meaningful on a channel that carries signal. On a near-unused
    channel the KL drives mu to 0 and sigma to 1, and the leftover residue is
    spectrally white, so a high ratio there just measures that noise -- which is why
    `hf_channels` is gated on SNR, not on KL. `snr > 1` means the posterior mean
    varies more across windows than the posterior noise within one.
    """
    kl_pc = (0.5 * (mu ** 2 + sd ** 2 - 2 * np.log(sd) - 1)).mean(axis=(0, 2))
    snr = mu.std(axis=(0, 2)) / sd.mean(axis=(0, 2))

    z = (mu - mu.mean(axis=(0, 2), keepdims=True)) / mu.std(axis=(0, 2), keepdims=True)
    lag1 = (z[:, :, :-1] * z[:, :, 1:]).mean(axis=(0, 2))
    # power along the *time* axis: how fast each latent channel varies
    p = (np.abs(np.fft.rfft(z, axis=-1)) ** 2).mean(0)
    q = p.shape[1] // 4
    hi_lo = p[:, -q:].mean(1) / p[:, 1:q].mean(1)

    informative = snr > snr_thresh
    return {
        "active_channels": int((kl_pc > active_thresh).sum()),
        "informative_channels": int(informative.sum()),
        "n_channels": mu.shape[1],
        # Descriptive, NOT a defect count: informative channels whose latent series is
        # high-frequency dominated. With stft_weight > 0 the decoder must reproduce
        # high-frequency detail, so the encoder has to carry it, and a channel holding
        # it necessarily varies fast. Reading a high count as "aliasing" is the error
        # that produced the retraction in VAE_REPORT.md.
        "hf_channels": int(((hi_lo > 2) & informative).sum()),
        "kurtosis": float((((mu - mu.mean()) / mu.std()) ** 4).mean()),
        "scale_factor": float(1 / mu.std()),
        "kl_per_dim": kl_pc,
        "snr": snr,
        "lag1": lag1,
        "hi_lo": hi_lo,
    }


def report(x: np.ndarray, xh: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    """Print every diagnostic and return them flat, so a sweep can tabulate the same call."""
    rec = reconstruction_metrics(x, xh)
    real, fake = property_stats(x), property_stats(xh)
    lat = latent_stats(mu, sd)

    print(f"reconstruction ({len(x)} windows)")
    for k, v in rec.items():
        print(f"  {k:20s} {v:9.4f}")
    print("properties               real   reconstruction")
    for k in ("rate", "amp", "peak_vel", "main_seq_slope", "noise_floor"):
        print(f"  {k:20s} {real[k]:8.3f} {fake[k]:12.3f}")
    print(f"latent  {lat['active_channels']}/{lat['n_channels']} active, "
          f"{lat['informative_channels']} informative (SNR>1), "
          f"{lat['hf_channels']} high-frequency, kurtosis {lat['kurtosis']:.1f}, "
          f"scale_factor {lat['scale_factor']:.3f}")
    print("  ch   KL/dim     SNR    lag1   hi/lo")
    for c in range(lat["n_channels"]):
        print(f"  {c:2d} {lat['kl_per_dim'][c]:8.3f} {lat['snr'][c]:7.2f} "
              f"{lat['lag1'][c]:7.2f} {lat['hi_lo'][c]:7.2f}")

    return {**rec, **{f"real_{k}": v for k, v in real.items()},
            **{f"recon_{k}": v for k, v in fake.items()},
            **{k: v for k, v in lat.items() if np.isscalar(v) or isinstance(v, (int, float))}}
