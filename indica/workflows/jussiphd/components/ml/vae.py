"""cVAE model and training utilities extracted from the surrogate notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from indica.workflows.jussiphd.components.preprocessing.dataset_creation import (
    create_dataset_and_dataloaders,
)


class CVAENetwork(nn.Module):
    """Notebook-style conditional VAE for emissivity from bolometry."""

    def __init__(self, b_dim: int, e_dim: int, latent_dim: int = 4):
        super().__init__()
        input_dim = b_dim + e_dim
        self.b_dim = b_dim
        self.e_dim = e_dim
        self.latent_dim = latent_dim

        self.fc1m = nn.Linear(input_dim, 64)
        self.fc1s = nn.Linear(input_dim, 64)
        self.fc2m = nn.Linear(64, 32)
        self.fc2s = nn.Linear(64, 32)
        self.fc3m = nn.Linear(32, latent_dim)
        self.fc3s = nn.Linear(32, latent_dim)

        self.fc_dec1 = nn.Linear(latent_dim + b_dim, 32)
        self.fc_dec2 = nn.Linear(32, 64)
        self.fc_dec3 = nn.Linear(64, e_dim)

    def encode(self, emissivity: torch.Tensor, bolom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((emissivity, bolom), dim=1)

        m = torch.relu(self.fc1m(x))
        logvar = torch.relu(self.fc1s(x))

        m = torch.relu(self.fc2m(m))
        logvar = torch.relu(self.fc2s(logvar))

        m = self.fc3m(m)
        logvar = self.fc3s(logvar)
        return m, logvar

    def decode(self, bolom: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat((bolom, z), dim=1)
        x = torch.relu(self.fc_dec1(x))
        x = torch.relu(self.fc_dec2(x))
        return self.fc_dec3(x)

    @staticmethod
    def reparametrize(m: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return m + eps * std

    @staticmethod
    def kl_divergence(m: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - m.pow(2) - logvar.exp(), dim=1)

    def total_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        m: torch.Tensor,
        logvar: torch.Tensor,
        kl_beta: float = 0.0,
        return_components: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        kl_loss = self.kl_divergence(m, logvar).sum()
        total = (recon_loss + kl_beta * kl_loss) / x.shape[0]
        if return_components:
            return total, recon_loss / x.shape[0], kl_loss * kl_beta / x.shape[0]
        return total

    def forward(self, emissivity: torch.Tensor, bolom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m, logvar = self.encode(emissivity, bolom)
        z = self.reparametrize(m, logvar)
        e_hat = self.decode(bolom, z)
        return e_hat, m, logvar, z


def train_vae_from_csv(
    b_path: str,
    eps_path: str,
    meta_path: str | None = None,
    latent_dim: int = 4,
    n_epochs: int = 25,
    lr: float = 1e-3,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 0,
    output_dir: str = ".",
    model_filename: str = "vae_model.pt",
) -> dict[str, Any]:
    """Train cVAE and save checkpoint for later evaluation."""
    torch.manual_seed(seed)

    bundle = create_dataset_and_dataloaders(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=meta_path,
        train_fraction=train_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    dataset = bundle["dataset"]
    train_loader = bundle["train_loader"]

    b_dim = int(dataset.b_slices.shape[1])
    e_dim = int(dataset.eps_slices.shape[1])

    model = CVAENetwork(b_dim=b_dim, e_dim=e_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_epoch_loss = None
    last_recon_loss = None
    last_kl_loss = None

    for epoch in range(1, n_epochs):
        model.train()
        train_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        kl_epoch_beta = (epoch / n_epochs) * 0.5

        for emissivity, bolom in train_loader:
            optimizer.zero_grad()
            recon_emis, m, logvar, _ = model(emissivity, bolom)
            loss, rec, kl = model.total_loss(
                recon_emis,
                emissivity,
                m,
                logvar,
                return_components=True,
                kl_beta=kl_epoch_beta,
            )
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            recon_loss_total += float(rec.item())
            kl_loss_total += float(kl.item())

        denom = max(1, len(train_loader))
        last_epoch_loss = train_loss / denom
        last_recon_loss = recon_loss_total / denom
        last_kl_loss = kl_loss_total / denom

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / model_filename
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "b_dim": b_dim,
            "e_dim": e_dim,
            "n_epochs": n_epochs,
            "lr": lr,
        },
        model_path,
    )

    return {
        "model_path": str(model_path),
        "latent_dim": latent_dim,
        "b_dim": b_dim,
        "e_dim": e_dim,
        "n_epochs": n_epochs,
        "lr": lr,
        "last_epoch_loss": last_epoch_loss,
        "last_recon_loss": last_recon_loss,
        "last_kl_loss": last_kl_loss,
    }
