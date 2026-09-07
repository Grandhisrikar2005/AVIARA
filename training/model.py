import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN frontend for extracting rich time-frequency
    patterns from bird-song Mel spectrograms.
    """

    def __init__(self, input_channels=1, feature_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(

            # ==================================================
            # BLOCK 1
            # ==================================================

            nn.Conv2d(
                input_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Dropout2d(0.10),

            # ==================================================
            # BLOCK 2
            # ==================================================

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Dropout2d(0.10),

            # ==================================================
            # BLOCK 3
            # ==================================================

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            nn.Dropout2d(0.15),

            # ==================================================
            # BLOCK 4
            # ==================================================

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),

            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),

            nn.Dropout2d(0.15)
        )

    def forward(self, x):
        return self.cnn(x)


class AttentionPooling(nn.Module):
    """
    Attention-based temporal pooling.

    Learns which parts of a bird-song segment
    contain the most informative acoustic patterns.
    """

    def __init__(self, feature_dim):
        super().__init__()

        hidden_dim = feature_dim // 2

        self.attention = nn.Sequential(
            nn.Linear(
                feature_dim,
                hidden_dim
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(self, x):

        # x:
        # [batch, time, feature_dim]

        attention_scores = self.attention(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled = torch.sum(
            x * attention_weights,
            dim=1
        )

        return pooled


class BirdsongCNNTransformer(nn.Module):
    """
    V2 CNN + Transformer architecture.

    Input:
        [batch, 1, 128, time]

    Output:
        logits:
            [batch, num_classes]

        embedding:
            [batch, embedding_dim]
    """

    def __init__(
        self,
        num_classes=10,
        cnn_dim=256,
        transformer_dim=256,
        num_heads=4,
        num_layers=3,
        embedding_dim=128,
        dropout=0.30
    ):
        super().__init__()

        # ==================================================
        # CNN FEATURE EXTRACTOR
        # ==================================================

        self.cnn = CNNFeatureExtractor(
            input_channels=1,
            feature_dim=cnn_dim
        )

        # ==================================================
        # PROJECT CNN FEATURES
        # ==================================================

        self.feature_projection = nn.Sequential(
            nn.Linear(
                cnn_dim,
                transformer_dim
            ),
            nn.LayerNorm(
                transformer_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ==================================================
        # TRANSFORMER
        # ==================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,

            # Pre-normalization is disabled here.
            # This also removes the nested-tensor warning.
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ==================================================
        # FINAL TRANSFORMER NORMALIZATION
        # ==================================================

        self.transformer_norm = nn.LayerNorm(
            transformer_dim
        )

        # ==================================================
        # ATTENTION POOLING
        # ==================================================

        self.attention_pooling = AttentionPooling(
            transformer_dim
        )

        # ==================================================
        # ACOUSTIC EMBEDDING
        # ==================================================

        self.embedding_layer = nn.Sequential(
            nn.Linear(
                transformer_dim,
                embedding_dim
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            )
        )

        # ==================================================
        # CLASSIFIER
        # ==================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        # ==================================================
        # 1. CNN
        # ==================================================

        x = self.cnn(x)

        # Shape:
        # [B, 256, frequency, time]

        # ==================================================
        # 2. PRESERVE TIME-FREQUENCY INFORMATION
        # ==================================================

        # Average only over frequency.
        #
        # The CNN has already extracted rich
        # frequency-local features.

        x = x.mean(dim=2)

        # [B, 256, time]

        # ==================================================
        # 3. TRANSPOSE FOR TRANSFORMER
        # ==================================================

        x = x.transpose(1, 2)

        # [B, time, 256]

        # ==================================================
        # 4. FEATURE PROJECTION
        # ==================================================

        x = self.feature_projection(x)

        # [B, time, transformer_dim]

        # ==================================================
        # 5. TRANSFORMER
        # ==================================================

        x = self.transformer(x)

        x = self.transformer_norm(x)

        # [B, time, transformer_dim]

        # ==================================================
        # 6. ATTENTION POOLING
        # ==================================================

        x = self.attention_pooling(x)

        # [B, transformer_dim]

        # ==================================================
        # 7. ACOUSTIC EMBEDDING
        # ==================================================

        embedding = self.embedding_layer(x)

        # [B, 128]

        # ==================================================
        # 8. SPECIES CLASSIFICATION
        # ==================================================

        logits = self.classifier(
            embedding
        )

        # [B, num_classes]

        return logits, embedding


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_model(batch_size=1):

    print("=" * 70)
    print("BIRDSONG CNN + TRANSFORMER V2 MODEL TEST")
    print("=" * 70)

    # ======================================================
    # CREATE MODEL
    # ======================================================

    model = BirdsongCNNTransformer(
        num_classes=10,
        cnn_dim=256,
        transformer_dim=256,
        num_heads=4,
        num_layers=3,
        embedding_dim=128,
        dropout=0.30
    )

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ======================================================
    # TRAINING MODE TEST
    # ======================================================

    model.train()

    dummy_input = torch.randn(
        batch_size,
        1,
        128,
        501
    )

    logits, embedding = model(
        dummy_input
    )

    print("\nInput:")
    print(
        f"  Shape: "
        f"{dummy_input.shape}"
    )

    print("\nClassifier output:")
    print(
        f"  Shape: "
        f"{logits.shape}"
    )

    print("\nBirdsong embedding:")
    print(
        f"  Shape: "
        f"{embedding.shape}"
    )

    # ======================================================
    # VERIFY TRAINING OUTPUTS
    # ======================================================

    assert logits.shape == (
        batch_size,
        10
    )

    assert embedding.shape == (
        batch_size,
        128
    )

    print(
        "\nSingle-sample training test successful!"
    )

    # ======================================================
    # EVALUATION MODE TEST
    # ======================================================

    model.eval()

    with torch.no_grad():

        logits_eval, embedding_eval = model(
            dummy_input
        )

    assert logits_eval.shape == (
        batch_size,
        10
    )

    assert embedding_eval.shape == (
        batch_size,
        128
    )

    print(
        "Evaluation test successful!"
    )

    print(
        "\nV2 model test successful!"
    )

    print("=" * 70)


if __name__ == "__main__":

    test_model(batch_size=1)