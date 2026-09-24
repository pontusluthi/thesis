"""Rectified-flow (flow matching) model for fixational gaze windows, DiffWave backbone.

Works directly on the (2, T) sinusoidally squashed velocity -- no latent -- as an
alternative to the VAE + latent DDPM route. Linear path between noise x0 ~ N(0, I)
and data x1, x_t = (1 - t) x0 + t x1; the network predicts the velocity x1 - x0 and
samples are drawn by integrating that ODE from t = 0 to t = 1.

Data is divided by its per-channel std before training. The sine-squashed velocities
have std ~0.045, so without it the signal is buried under unit-variance noise until
t > ~0.95 and almost every training step is spent where there is nothing to learn.

    python -m src.flow --raw data/GazeBase_v2_0 --prepared data/prepared
    python -m src.flow --load checkpoints/flow.pt --sample 256 --sample-out gen.npy
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # works both as `src.flow` and as a plain script
    from .vae import loaders
except ImportError:  # pragma: no cover
    from vae import loaders


@dataclass
class FlowConfig:
    """Everything a checkpoint needs to rebuild the model and undo the data scaling."""

    in_channels: int = 2
    res_channels: int = 64
    n_layers: int = 30
    dilation_cycle: int = 10  # dilations 1..512, three cycles -> ~6k-sample receptive field
    t_emb_dim: int = 128
    data_std: tuple[float, ...] = (1.0, 1.0)  # fitted on the train split, see fit_data_std

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(2 ** (i % self.dilation_cycle) for i in range(self.n_layers))


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of t in [0, 1].

    Scaled by 1000 first: the standard frequencies run from 1 down to 1e-4, which
    suits integer diffusion steps. Fed t in [0, 1] directly, sin(t * f) barely moves
    for all but the first few dimensions and the network can hardly tell t apart.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
    args = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class DilatedConv1d(nn.Conv1d):
    """Kernel-3 dilated conv computed as three shifted copies and one 1x1 conv.

    Same weights and output as nn.Conv1d(padding=d, dilation=d). On ROCm/MIOpen the
    native dilated kernel is ~15x slower at large dilations (72 vs 5 ms fwd+bwd at
    d=256 on 16x64x5000), which made the full model ~2 s/step.
    """

    def __init__(self, cin: int, cout: int, dilation: int):
        super().__init__(cin, cout, 3)
        self.d = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d, T = self.d, x.shape[-1]
        xp = F.pad(x, (d, d))
        shifted = torch.cat([xp[..., :T], xp[..., d:d + T], xp[..., 2 * d:]], dim=1)
        w = self.weight.permute(0, 2, 1).reshape(self.out_channels, -1, 1)
        return F.conv1d(shifted, w, self.bias)


class ResidualBlock(nn.Module):
    """DiffWave block: dilated conv, gated tanh * sigmoid, split into residual and skip."""

    def __init__(self, channels: int, dilation: int, t_emb_dim: int):
        super().__init__()
        self.t_proj = nn.Linear(t_emb_dim, channels)  # per layer, as in DiffWave
        self.dilated = DilatedConv1d(channels, 2 * channels, dilation)
        self.out = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, filt = self.dilated(x + self.t_proj(t_emb)[:, :, None]).chunk(2, dim=1)
        residual, skip = self.out(torch.sigmoid(gate) * torch.tanh(filt)).chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class DiffWave(nn.Module):
    """Predicts the flow velocity v(x_t, t), same shape as x_t."""

    def __init__(self, cfg: FlowConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or FlowConfig()
        self.inp = nn.Sequential(nn.Conv1d(cfg.in_channels, cfg.res_channels, 1), nn.ReLU())
        self.t_mlp = nn.Sequential(
            nn.Linear(cfg.t_emb_dim, 4 * cfg.t_emb_dim), nn.SiLU(),
            nn.Linear(4 * cfg.t_emb_dim, cfg.t_emb_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(cfg.res_channels, 2 ** (i % cfg.dilation_cycle), cfg.t_emb_dim)
            for i in range(cfg.n_layers)
        )
        self.head = nn.Sequential(
            nn.Conv1d(cfg.res_channels, cfg.res_channels, 1), nn.ReLU(),
            nn.Conv1d(cfg.res_channels, cfg.in_channels, 1),
        )
        nn.init.zeros_(self.head[-1].weight)  # start as v = 0, as DiffWave does
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        t_emb = self.t_mlp(time_embedding(t, self.cfg.t_emb_dim))
        skip = 0
        for block in self.blocks:
            h, s = block(h, t_emb)
            skip = skip + s
        return self.head(skip / math.sqrt(len(self.blocks)))

    # --- data scaling: the model only ever sees unit-variance data ---
    def _std(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_tensor(self.cfg.data_std)[None, :, None]

    def to_model(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._std(x)

    def from_model(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._std(x)).clamp(-1.0, 1.0)  # back to the sine range

    def loss(self, x: torch.Tensor, t: torch.Tensor | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
        x1 = self.to_model(x)
        x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
        if t is None:
            t = sample_t(len(x1), x1.device)
        tb = t[:, None, None]
        return F.mse_loss(self((1 - tb) * x0 + tb * x1, t), x1 - x0)


def sample_t(n: int, device, sigma: float = 1.0, p_uniform: float = 0.1) -> torch.Tensor:
    """Logit-normal t (dense mid-path, as in SD3), mixed with some uniform for the ends."""
    t = torch.sigmoid(sigma * torch.randn(n, device=device))
    u = torch.rand(n, device=device)
    return torch.where(torch.rand(n, device=device) < p_uniform, u, t)


@torch.no_grad()
def sample(model: DiffWave, n: int, length: int, steps: int = 50, method: str = "heun",
           batch_size: int = 32, seed: int | None = None) -> np.ndarray:
    """Integrate the ODE from noise to data. Returns (n, C, length) in sine units.

    Uniform grid. Heun costs two evaluations per step, so `steps` Heun steps are 2x
    the cost of `steps` Euler steps.
    """
    model.eval()
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None
    ts = torch.linspace(0.0, 1.0, steps + 1, device=dev)
    out = []
    for i in range(0, n, batch_size):
        b = min(batch_size, n - i)
        x = torch.randn(b, model.cfg.in_channels, length, device=dev, generator=g)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            h = t1 - t0
            v0 = model(x, t0.expand(b))
            if method == "euler":
                x = x + h * v0
            else:
                v1 = model(x + h * v0, t1.expand(b))
                x = x + 0.5 * h * (v0 + v1)
        out.append(model.from_model(x).float().cpu().numpy())
    return np.concatenate(out)


def fit_data_std(loader: DataLoader, n_batches: int = 50) -> tuple[float, ...]:
    xs = [v for _, (v, *_) in zip(range(n_batches), loader)]
    return tuple(float(s) for s in torch.cat(xs).float().std(dim=(0, 2)))


@torch.no_grad()
def evaluate(model: DiffWave, loader: DataLoader, device: str) -> float:
    """Flow loss on a fixed t grid and fixed noise, so epochs are comparable.

    The training loss is dominated by the draw of t and x0; with both fixed, a
    change here reflects the model.
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    total, n = 0.0, 0
    for v, *_ in loader:
        x = v.float().to(device)
        t = torch.linspace(0.05, 0.95, len(x), device=device)
        total += float(model.loss(x, t=t, generator=g)) * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float, step: int) -> None:
    decay = min(decay, (1 + step) / (10 + step))  # warm-up: early EMA is not stuck at init
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1.0 - decay)


def save(path: str, model: DiffWave, ema: DiffWave, window: int, epoch: int) -> None:
    torch.save({"cfg": asdict(model.cfg), "window": window, "epoch": epoch,
                "state_dict": model.state_dict(), "ema": ema.state_dict()}, path)


def load(path: str, device: str, use_ema: bool = True) -> tuple[DiffWave, int]:
    """Model from a checkpoint; the EMA weights by default, which are the ones to sample."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DiffWave(FlowConfig(**{**cfg, "data_std": tuple(cfg["data_std"])})).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "state_dict"])
    return model, ckpt["window"]


def train(args: argparse.Namespace) -> DiffWave:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = loaders(args.raw, args.prepared, args.window,
                                       args.batch_size, args.workers)
    cfg = FlowConfig(res_channels=args.res_channels, n_layers=args.n_layers,
                     data_std=fit_data_std(train_loader))
    model = DiffWave(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(total, 1),
                                                pct_start=0.02, anneal_strategy="cos")
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device == "cuda")
    print(f"{cfg}\nreceptive field {cfg.receptive_field} for window {args.window}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {device}")

    for epoch in range(args.epochs):
        model.train()
        for step, (v, *_) in enumerate(train_loader):
            with amp:
                loss = model.loss(v.float().to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema_update(ema, model, args.ema_decay, epoch * len(train_loader) + step)
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step:5d}  loss {float(loss):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}")

        with amp:
            print(f"epoch {epoch} val  loss {evaluate(model, val_loader, device):.4f}  "
                  f"ema {evaluate(ema, val_loader, device):.4f}")
        save(args.out, model, ema, args.window, epoch)
    return ema


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="data/GazeBase_v2_0")
    p.add_argument("--prepared", default="data/prepared")
    p.add_argument("--out", default="checkpoints/flow.pt")
    p.add_argument("--window", type=int, default=5000)
    p.add_argument("--res-channels", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--load", default="", help="skip training and sample from this checkpoint")
    p.add_argument("--sample", type=int, default=0, help="windows to generate after training")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--method", choices=["heun", "euler"], default="heun")
    p.add_argument("--sample-out", default="checkpoints/flow_samples.npy")
    args = p.parse_args()

    if args.load:
        model, window = load(args.load, "cuda" if torch.cuda.is_available() else "cpu")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        model, window = train(args), args.window
    if args.sample:
        x = sample(model, args.sample, window, steps=args.steps, method=args.method)
        np.save(args.sample_out, x)
        print(f"wrote {x.shape} to {args.sample_out}")


if __name__ == "__main__":
    main()
