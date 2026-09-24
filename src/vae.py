"""A small 1D convolutional VAE for fixational gaze windows.

The encoder maps a (2, T) window of sinusoidally squashed gaze velocity to a
*sequence* of latents, (latent_channels, T / f), rather than to a single vector:
a four second window at 1000 Hz does not survive a global bottleneck, and a
latent that keeps its time axis is what the DDPM will diffuse over later.

Train on the prepared memmap:

    python -m src.vae --raw data/GazeBase_v2_0 --prepared data/prepared
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # works both as `src.vae` and as a plain script
    from .util.dataset import GazeBaseDataset
except ImportError:  # pragma: no cover
    from util.dataset import GazeBaseDataset


@dataclass
class VAEConfig:
    """Everything that changes the weights' shapes, so a checkpoint carries it."""

    in_channels: int = 2
    base_channels: int = 32
    channel_mults: tuple[int, ...] = (1, 2, 4, 4, 4)  # one /2 per step between levels
    latent_channels: int = 8
    kl_weight: float = 1e-3  # per-window sums, see VAE.loss
    stft_weight: float = 0.0  # 0 disables the spectral term, see stft_loss

    @property
    def downsample(self) -> int:
        return 2 ** (len(self.channel_mults) - 1)


def norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, c), c)


# class BlurPool1d(nn.Module):
#     """Anti-aliased /2 downsample: fixed binomial low-pass, then stride-2 subsample.

#     A bare stride-2 conv decimates without filtering, so content above the new Nyquist
#     folds back rather than being discarded. Kept because it is the correct way to
#     downsample and costs nothing, but note it was added to fix a latent "aliasing"
#     problem that turned out not to exist -- see the retraction in VAE_REPORT.md. It
#     did not measurably change any reconstruction or property metric.
#     """

#     def __init__(self, c: int):
#         super().__init__()
#         k = torch.tensor([1.0, 3.0, 3.0, 1.0])
#         self.register_buffer("kernel", (k / k.sum()).view(1, 1, -1).repeat(c, 1, 1))
#         self.c = c

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return F.conv1d(F.pad(x, (1, 2), mode="replicate"), self.kernel, stride=2, groups=self.c)


def stft_loss(x_hat: torch.Tensor, x: torch.Tensor, sizes=(128, 512, 2048),
              eps: float = 1e-5) -> torch.Tensor:
    """Multi-resolution log-magnitude distance.

    Time-domain MSE weights errors by absolute magnitude, so tremor at a few deg/s
    counts for almost nothing against microsaccades at 200 deg/s, and the decoder
    correctly learns to smooth it away. This term scores the spectrum in the log
    domain instead, where a quiet band counts as much as a loud one.

    `eps` matters: these magnitudes have a median around 1e-4, so log1p would be
    linear over the whole range (log1p(u) ~ u for u << 1) and would just reproduce
    MSE's weighting. log(|S| + 1e-5) spans ~15 nats and actually compresses.
    """
    total = x.new_zeros(())
    for n in sizes:
        w = torch.hann_window(n, device=x.device)
        spec = lambda a: torch.stft(a.flatten(0, 1), n, n // 4, window=w,
                                    return_complex=True).abs()
        total = total + F.l1_loss(torch.log(spec(x_hat) + eps), torch.log(spec(x) + eps))
    return total / len(sizes)


class ResBlock(nn.Module):
    """Pre-activation residual block. Kernel 5: ~5 ms of context per conv at 1 kHz."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            norm(cin), nn.SiLU(), nn.Conv1d(cin, cout, 5, padding=2),
            norm(cout), nn.SiLU(), nn.Conv1d(cout, cout, 5, padding=2),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        chans = [cfg.base_channels * m for m in cfg.channel_mults]
        layers: list[nn.Module] = [nn.Conv1d(cfg.in_channels, chans[0], 5, padding=2)]
        for cin, cout in zip(chans, chans[1:]):
            layers += [ResBlock(cin, cout), nn.Conv1d(cout, cout, 3, padding=1)]
        layers += [
            ResBlock(chans[-1], chans[-1]),
            norm(chans[-1]), nn.SiLU(),
            nn.Conv1d(chans[-1], 2 * cfg.latent_channels, 3, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x).chunk(2, dim=1)  # mu, logvar


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        chans = [cfg.base_channels * m for m in cfg.channel_mults][::-1]
        layers: list[nn.Module] = [
            nn.Conv1d(cfg.latent_channels, chans[0], 3, padding=1),
            ResBlock(chans[0], chans[0]),
        ]
        for cin, cout in zip(chans, chans[1:]):
            # interpolate + conv rather than ConvTranspose: no checkerboard artefacts
            layers += [
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(cin, cin, 3, padding=1),
                ResBlock(cin, cout),
            ]
        layers += [
            norm(chans[-1]), nn.SiLU(),
            nn.Conv1d(chans[-1], cfg.in_channels, 5, padding=2),
            nn.Tanh(),  # the sinusoidal normalization already bounds the data to [-1, 1]
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAEConfig()
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        return mu, logvar.clamp(-30.0, 20.0)  # keeps exp() finite early in training

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        return self.decode(self.sample(mu, logvar)), mu, logvar

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Both terms are summed over a window and averaged over the batch.

        `kl_per_dim` is the number to watch: if it drifts towards 0 the posterior
        has collapsed and the latent carries nothing for the DDPM to model, so
        lower `kl_weight`; raise it if latents look too spiky to diffuse.
        """
        x_hat, mu, logvar = self(x)
        recon = F.mse_loss(x_hat, x, reduction="none").flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).flatten(1).sum(1).mean()
        spec = stft_loss(x_hat, x) if self.cfg.stft_weight else x.new_zeros(())
        return {
            "loss": recon + self.cfg.kl_weight * kl + self.cfg.stft_weight * spec,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "stft": spec.detach(),
            "kl_per_dim": kl.detach() / mu[0].numel(),
        }


def loaders(raw: str, prepared: str, window: int, batch_size: int, workers: int):
    ds = GazeBaseDataset(raw)
    kw = dict(normalization="sinusoidal", window=window, stride=window)
    splits = [ds.windows(prepared, s, **kw) for s in ("train", "test")]
    return [
        DataLoader(d, batch_size=batch_size, shuffle=train, num_workers=workers,
                   pin_memory=True, drop_last=train)
        for d, train in zip(splits, (True, False))
    ]


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    totals, n = {}, 0
    for v, *_ in loader:
        out = model.loss(v.float().to(device))
        for k, t in out.items():
            totals[k] = totals.get(k, 0.0) + float(t) * len(v)
        n += len(v)
    return {k: t / n for k, t in totals.items()}


def train(args: argparse.Namespace) -> None:
    cfg = VAEConfig(kl_weight=args.kl_weight, latent_channels=args.latent_channels,
                    stft_weight=args.stft_weight)
    if args.window % cfg.downsample:
        raise ValueError(f"window must be a multiple of {cfg.downsample}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = loaders(
        args.raw, args.prepared, args.window, args.batch_size, args.workers
    )
    model = VAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"{model.cfg}\nlatent {cfg.latent_channels}x{args.window // cfg.downsample} "
          f"on {device}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    for epoch in range(args.epochs):
        model.train()
        for step, (v, *_) in enumerate(train_loader):
            out = model.loss(v.float().to(device))
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step:5d}  " + "  ".join(
                    f"{k} {float(t):.4f}" for k, t in out.items()))

        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val  " + "  ".join(f"{k} {t:.4f}" for k, t in val.items()))
        torch.save({"cfg": asdict(cfg), "window": args.window,
                    "state_dict": model.state_dict()}, args.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", default="data/GazeBase_v2_0")
    p.add_argument("--prepared", default="data/prepared")
    p.add_argument("--out", default="checkpoints/vae.pt")
    p.add_argument("--window", type=int, default=4992) # has to be divisible by 16 to fit latent space
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--kl-weight", type=float, default=1e-3)
    p.add_argument("--stft-weight", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
