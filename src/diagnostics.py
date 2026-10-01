"""Diagnostics for the gaze VAE: reconstruction fidelity, preserved properties, latent health.

These are the numbers checkpoints get compared on. Reconstruction error alone is
misleading here: a decoder that smooths tremor away scores well on MSE while lowering
the noise floor the microsaccade detector thresholds against, which silently changes
how many events it finds. So `property_stats` and `latent_stats` carry as much weight
as `reconstruction_metrics`.

The model works in model space (whatever `--features` selected). Everything physical
(velocity, PSDs, microsaccades) is computed after `to_deg`, the single path that real,
reconstructed and generated windows all go through.

    from src.diagnostics import collect, report
    report(*collect(model, val_loader, dev), denorm=val_ds.denorm)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from scipy.signal import welch

try:  # works both as `src.diagnostics` and as a plain script
    from .util.microsaccade import microsaccade_extraction
    from .util.prepare import smooth
except ImportError:  # pragma: no cover
    from util.microsaccade import microsaccade_extraction
    from util.prepare import smooth

RATE = 1000
# Applied identically to every side before velocity is taken. Set to None to see
# raw model output -- the shared filter otherwise hides high-frequency junk.
SMOOTH: dict | None = dict(window_length=23, window="bartlett")


def to_deg(x: np.ndarray, denorm: Callable, smooth_cfg: dict | None = SMOOTH):
    """Model-space (N, C, T) -> (position in deg, velocity in deg/s).

    `denorm` is the dataset's `denorm`, so this works for any feature set. Position
    is used when the model has it (velocity is then re-derived from it); otherwise it
    is integrated from velocity, with an arbitrary offset nothing downstream uses.
    """
    f = denorm(x)
    pos = f["pos"] if "pos" in f else np.cumsum(f["vel"], axis=-1) / RATE
    if smooth_cfg:
        pos = smooth(pos, axis=-1, **smooth_cfg)
    return pos, np.gradient(pos, 1 / RATE, axis=-1)


def noise_floor(v: np.ndarray) -> float:
    """Median-based SD of velocity (deg/s), i.e. what `microsacc` sets its threshold from."""
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


def reconstruction_metrics(x: np.ndarray, xh: np.ndarray, v: np.ndarray, vh: np.ndarray,
                           f_split: float = 60.0) -> dict:
    """Error, and the two ways this decoder is known to fail: lost highs, lower floor.

    x, xh: model-space position (what the loss sees). v, vh: velocity in deg/s.
    Correlation and variance explained are on velocity: on position they sit near 1
    for any decoder that gets the window offset right, and tell you nothing.
    """
    err, verr = xh - x, vh - v
    f, p_x = welch(v[:, 0], fs=RATE, nperseg=1024, axis=-1)
    _, p_h = welch(vh[:, 0], fs=RATE, nperseg=1024, axis=-1)
    hi = f >= f_split
    return {
        "mse": float((err ** 2).mean()),
        "mae": float(np.abs(err).mean()),
        "vel_rmse": float(np.sqrt((verr ** 2).mean())),  # deg/s
        "var_explained": float(1 - verr.var() / v.var()),
        "corr_vx": float(np.corrcoef(v[:, 0].ravel(), vh[:, 0].ravel())[0, 1]),
        "corr_vy": float(np.corrcoef(v[:, 1].ravel(), vh[:, 1].ravel())[0, 1]),
        # Fraction of >60 Hz power the reconstruction keeps; 1.0 is perfect.
        #
        # USE THIS ONLY AS A SEED-AVERAGED NUMBER. Three repeats of one identical
        # config gave 0.22, 0.17, 0.12, 0.10 -- a 2x spread from training randomness
        # alone, so it cannot rank single runs. That is a real property of the model,
        # not of the metric: a log/geometric version is worse still, because it is
        # then dominated by bins where the reconstruction has almost no power.
        # (Those numbers were measured on the old sin-velocity pipeline.)
        "hf_retained": float(p_h.mean(0)[hi].mean() / p_x.mean(0)[hi].mean()),
        "noise_floor_ratio": noise_floor(vh) / noise_floor(v),
    }


def microsaccades(v: np.ndarray, pos: np.ndarray, vfac: float = 5, mindur: int = 6) -> np.ndarray:
    """Detect in one window. v: (2, T) deg/s, pos: (2, T) deg. Columns: 3 = peak velocity, 6 = amplitude."""
    return np.asarray(microsaccade_extraction(v.T, pos.T, RATE, VFAC=vfac, MINDUR=mindur))


def property_stats(pos: np.ndarray, v: np.ndarray, n_windows: int = 300) -> dict:
    """Microsaccade statistics -- the properties the latent actually has to preserve."""
    pos, v = pos[:n_windows], v[:n_windows]
    floor = noise_floor(v)
    sacs = [s for s in map(microsaccades, v, pos) if len(s)]
    if not sacs:
        return {"rate": 0.0, "amp": np.nan, "peak_vel": np.nan, "main_seq_slope": np.nan,
                "noise_floor": floor}
    s = np.concatenate(sacs)
    amp, vpk = s[:, 6], s[:, 3]
    ok = (amp > 0) & (vpk > 0)
    return {
        "rate": len(s) / (len(v) * v.shape[-1] / RATE),
        "amp": float(np.median(amp)),
        "peak_vel": float(np.median(vpk)),
        # the main sequence is a power law; its exponent is the shape-preserving check
        "main_seq_slope": float(np.polyfit(np.log(amp[ok]), np.log(vpk[ok]), 1)[0]),
        "noise_floor": floor,
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


def report(x: np.ndarray, xh: np.ndarray, mu: np.ndarray, sd: np.ndarray,
           denorm: Callable) -> dict:
    """Print every diagnostic and return them flat, so a sweep can tabulate the same call."""
    pos, v = to_deg(x, denorm)
    pos_h, vh = to_deg(xh, denorm)

    rec = reconstruction_metrics(x, xh, v, vh)
    real, fake = property_stats(pos, v), property_stats(pos_h, vh)
    lat = latent_stats(mu, sd)

    print(f"reconstruction ({len(x)} windows)")
    for k, val in rec.items():
        print(f"  {k:20s} {val:9.4f}")
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

    return {**rec, **{f"real_{k}": val for k, val in real.items()},
            **{f"recon_{k}": val for k, val in fake.items()},
            **{k: val for k, val in lat.items() if np.isscalar(val) or isinstance(val, (int, float))}}