"""
NORMALIZED SPECIES-CONDITIONED BIRDSONG VAE

Input:
    Normalized Mel spectrogram [B, 1, 128, 501]

Value range:
    0.0 ... 1.0

0.0 = -80 dB
1.0 =   0 dB

No LSTM is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONDITIONAL VAE
# ============================================================

class BirdsongConditionalVAE(nn.Module):

    def __init__(
        self,
        num_classes=10,
        latent_dim=64,
        species_embedding_dim=16
    ):

        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.species_embedding_dim = (
            species_embedding_dim
        )

        # ----------------------------------------------------
        # Species conditioning
        # ----------------------------------------------------

        self.species_embedding = nn.Embedding(
            num_classes,
            species_embedding_dim
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                (8, 32)
            )
        )

        self.encoder_flatten_dim = (
            256 * 8 * 32
        )

        # ----------------------------------------------------
        # Latent
        # ----------------------------------------------------

        combined_dim = (
            self.encoder_flatten_dim
            + species_embedding_dim
        )

        self.fc_mu = nn.Linear(
            combined_dim,
            latent_dim
        )

        self.fc_logvar = nn.Linear(
            combined_dim,
            latent_dim
        )

        # ----------------------------------------------------
        # Decoder input
        # ----------------------------------------------------

        decoder_input_dim = (
            latent_dim
            + species_embedding_dim
        )

        self.decoder_input = nn.Sequential(

            nn.Linear(
                decoder_input_dim,
                self.encoder_flatten_dim
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            ),

            # CRITICAL:
            # normalized spectrogram must be 0...1
            nn.Sigmoid()
        )

    # ========================================================
    # REPARAMETERIZATION
    # ========================================================

    def reparameterize(
        self,
        mu,
        logvar
    ):

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(
            std
        )

        return (
            mu
            + epsilon * std
        )

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        x,
        species
    ):

        features = self.encoder(
            x
        )

        features = features.flatten(
            start_dim=1
        )

        species_features = (
            self.species_embedding(
                species
            )
        )

        combined = torch.cat(
            [
                features,
                species_features
            ],
            dim=1
        )

        mu = self.fc_mu(
            combined
        )

        logvar = self.fc_logvar(
            combined
        )

        return mu, logvar

    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        z,
        species
    ):

        species_features = (
            self.species_embedding(
                species
            )
        )

        combined = torch.cat(
            [
                z,
                species_features
            ],
            dim=1
        )

        x = self.decoder_input(
            combined
        )

        x = x.view(
            -1,
            256,
            8,
            32
        )

        x = self.decoder(
            x
        )

        # Exact target shape.
        x = F.interpolate(
            x,
            size=(128, 501),
            mode="bilinear",
            align_corners=False
        )

        # Ensure valid range.
        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        return x

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        species
    ):

        mu, logvar = self.encode(
            x,
            species
        )

        z = self.reparameterize(
            mu,
            logvar
        )

        reconstruction = self.decode(
            z,
            species
        )

        return (
            reconstruction,
            mu,
            logvar,
            z
        )


# ============================================================
# VAE LOSS
# ============================================================

def vae_loss(
    reconstruction,
    target,
    mu,
    logvar,
    beta=0.0005
):

    # MSE captures spectral intensity.
    mse_loss = F.mse_loss(
        reconstruction,
        target,
        reduction="mean"
    )

    # L1 encourages sharper structure.
    l1_loss = F.l1_loss(
        reconstruction,
        target,
        reduction="mean"
    )

    # Combined reconstruction objective.
    reconstruction_loss = (
        0.7 * mse_loss
        +
        0.3 * l1_loss
    )

    # KL divergence.
    kl_divergence = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = (
        reconstruction_loss
        + beta * kl_divergence
    )

    return (
        total_loss,
        reconstruction_loss,
        kl_divergence
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# TEST
# ============================================================

def test_vae():

    print("=" * 70)
    print("NORMALIZED BIRDSONG CONDITIONAL VAE TEST")
    print("=" * 70)

    model = BirdsongConditionalVAE(
        num_classes=10,
        latent_dim=64,
        species_embedding_dim=16
    )

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    batch_size = 4

    dummy_input = torch.rand(
        batch_size,
        1,
        128,
        501
    )

    dummy_species = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long
    )

    model.eval()

    with torch.no_grad():

        reconstruction, mu, logvar, z = (
            model(
                dummy_input,
                dummy_species
            )
        )

    print(
        f"\nInput shape:"
        f"\n  {dummy_input.shape}"
    )

    print(
        f"\nReconstruction shape:"
        f"\n  {reconstruction.shape}"
    )

    print(
        f"\nLatent mean shape:"
        f"\n  {mu.shape}"
    )

    print(
        f"\nLatent log variance shape:"
        f"\n  {logvar.shape}"
    )

    print(
        f"\nLatent vector shape:"
        f"\n  {z.shape}"
    )

    print(
        f"\nOutput minimum:"
        f" {reconstruction.min().item():.6f}"
    )

    print(
        f"Output maximum:"
        f" {reconstruction.max().item():.6f}"
    )

    assert reconstruction.shape == (
        batch_size,
        1,
        128,
        501
    )

    assert mu.shape == (
        batch_size,
        64
    )

    assert logvar.shape == (
        batch_size,
        64
    )

    assert z.shape == (
        batch_size,
        64
    )

    assert reconstruction.min() >= 0.0
    assert reconstruction.max() <= 1.0

    loss, reconstruction_loss, kl = vae_loss(
        reconstruction,
        dummy_input,
        mu,
        logvar
    )

    print(
        f"\nTotal VAE loss:"
        f" {loss.item():.6f}"
    )

    print(
        f"Reconstruction loss:"
        f" {reconstruction_loss.item():.6f}"
    )

    print(
        f"KL divergence:"
        f" {kl.item():.6f}"
    )

    print(
        "\nVAE model test successful!"
    )

    print("=" * 70)


if __name__ == "__main__":
    test_vae()