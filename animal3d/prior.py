"""A learned pose prior: a species-conditioned VAE over SMAL joint rotations, trained on the fits.

Poses are the 34 local joint rotations (global orientation excluded), as 6D vectors: the first two columns of each
rotation matrix, which is continuous, unlike axis-angle. The decoder turns 6D back into rotations by Gram-Schmidt.
The prior is used to smooth motion: a track's poses are replaced by latent codes, one per frame, kept close in time.
"""
from __future__ import annotations

import torch
from torch import nn

SPECIES_CLASSES = ("giraffe", "cattle", "rhino", "camel", "lion", "donkey", "other")
N_JOINTS = 34


def species_class(name: str | None) -> int:
    """Index in SPECIES_CLASSES from a species or label name (e.g. 'white rhinoceros', 'Ankle cow', 'graffe')."""
    name = (name or "").lower()
    for key, words in (("giraffe", ("giraffe", "graffe")), ("cattle", ("cattle", "cow", "ankole", "watusi", "bull")),
                       ("rhino", ("rhino",)), ("camel", ("camel", "dromedary")), ("lion", ("lion",)),
                       ("donkey", ("donkey",))):
        if any(w in name for w in words):
            return SPECIES_CLASSES.index(key)
    return SPECIES_CLASSES.index("other")


def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): the first two columns."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def six_d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt."""
    a, b = x[..., :3], x[..., 3:]
    c1 = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = b - (c1 * b).sum(-1, keepdim=True) * c1
    c2 = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c3 = torch.linalg.cross(c1, c2)
    return torch.stack([c1, c2, c3], dim=-1)


class PoseVAE(nn.Module):
    def __init__(self, latent: int = 32, hidden: int = 512, n_species: int = len(SPECIES_CLASSES), dropout: float = 0.1):
        super().__init__()
        self.latent = latent
        self.species = nn.Embedding(n_species, 16)
        d_in = N_JOINTS * 6
        block = lambda i, o: [nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*block(d_in + 16, hidden), *block(hidden, hidden), nn.Linear(hidden, 2 * latent))
        self.decoder = nn.Sequential(*block(latent + 16, hidden), *block(hidden, hidden), nn.Linear(hidden, d_in))

    def encode(self, pose: torch.Tensor, species: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pose (N, 34, 3, 3) -> mean and log-variance (N, latent)."""
        x = torch.cat([matrix_to_6d(pose).flatten(1), self.species(species)], dim=1)
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 5)

    def decode(self, z: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        """z (N, latent) -> pose (N, 34, 3, 3)."""
        out = self.decoder(torch.cat([z, self.species(species)], dim=1))
        return six_d_to_matrix(out.view(-1, N_JOINTS, 6))

    def forward(self, pose, species):
        mu, logvar = self.encode(pose, species)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decode(z, species), mu, logvar


def rotation_error_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotations (..., 3, 3), in degrees."""
    trace = (a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)))


def vae_loss(model: PoseVAE, pose, species, beta: float) -> tuple[torch.Tensor, dict]:
    recon, mu, logvar = model(pose, species)
    rec = (recon - pose).pow(2).sum(dim=(-1, -2)).mean()  # squared Frobenius distance per joint, ~ geodesic for small angles
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return rec + beta * kl, {"rec": float(rec), "kl": float(kl)}
