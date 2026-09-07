from pathlib import Path

import torch
import torch.nn as nn

from model import BirdsongCNNTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SPECIES_CHECKPOINT = (
    BASE_DIR
    / "models"
    / "checkpoints"
    / "v2_best_model.pth"
)


# ============================================================
# COMMUNICATION LABELS
# ============================================================

COMMUNICATION_TO_IDX = {
    "alarm_warning": 0,
    "begging_distress": 1,
    "contact_social": 2,
    "song": 3,
}

IDX_TO_COMMUNICATION = {
    value: key
    for key, value in COMMUNICATION_TO_IDX.items()
}


# ============================================================
# COMMUNICATION HEAD
# ============================================================

class CommunicationHead(nn.Module):
    """
    Classification head operating on the existing
    128-dimensional V2 birdsong embedding.
    """

    def __init__(
        self,
        embedding_dim=128,
        num_classes=4,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(
                embedding_dim,
                128,
            ),

            nn.LayerNorm(128),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(64),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(self, embedding):

        return self.classifier(
            embedding
        )


# ============================================================
# COMPLETE COMMUNICATION MODEL
# ============================================================

class CommunicationModel(nn.Module):
    """
    Existing V2 CNN + Transformer
                ↓
          128-D embedding
                ↓
       Communication head
                ↓
          4 classes
    """

    def __init__(
        self,
        num_classes=4,
    ):
        super().__init__()

        # ----------------------------------------------------
        # EXACT EXISTING V2 MODEL
        # ----------------------------------------------------

        self.backbone = BirdsongCNNTransformer(
            num_classes=10,
            cnn_dim=256,
            transformer_dim=256,
            num_heads=4,
            num_layers=3,
            embedding_dim=128,
            dropout=0.30,
        )

        # ----------------------------------------------------
        # NEW COMMUNICATION CLASSIFIER
        # ----------------------------------------------------

        self.communication_head = CommunicationHead(
            embedding_dim=128,
            num_classes=num_classes,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        return_embedding=False,
    ):

        # Existing V2 model already returns:
        #
        # logits, embedding
        #
        _, embedding = self.backbone(x)

        communication_logits = (
            self.communication_head(
                embedding
            )
        )

        if return_embedding:
            return (
                communication_logits,
                embedding,
            )

        return communication_logits


# ============================================================
# LOAD EXACT V2 CHECKPOINT
# ============================================================

def load_pretrained_v2(
    model,
    checkpoint_path=SPECIES_CHECKPOINT,
    device="cpu",
):
    """
    Load the exact V2 species-classification checkpoint
    into the existing BirdsongCNNTransformer backbone.

    The original 10-class species classifier remains loaded,
    but will NOT be used by the communication head.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Resolve checkpoint state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Some training checkpoints may contain prefixes.
    # Remove common DataParallel prefix if present.
    # --------------------------------------------------------

    cleaned_state = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state[key] = value

    # --------------------------------------------------------
    # Load into exact V2 backbone
    # --------------------------------------------------------

    missing, unexpected = (
        model.backbone.load_state_dict(
            cleaned_state,
            strict=True,
        )
    )

    # strict=True should produce no mismatch.
    # The return values are retained for completeness.

    print()
    print(
        "Exact V2 checkpoint loaded successfully."
    )

    print(
        f"Missing keys    : {len(missing)}"
    )

    print(
        f"Unexpected keys : {len(unexpected)}"
    )

    return model


# ============================================================
# FREEZE / UNFREEZE HELPERS
# ============================================================

def freeze_backbone(model):
    """
    Freeze the pretrained V2 acoustic encoder.
    """

    for parameter in model.backbone.parameters():
        parameter.requires_grad = False


def unfreeze_embedding_layer(model):
    """
    Allow fine-tuning of the final V2 embedding layer
    while keeping the earlier feature extractor frozen.
    """

    for parameter in (
        model.backbone.embedding_layer.parameters()
    ):
        parameter.requires_grad = True


def trainable_parameter_count(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def total_parameter_count(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# MODEL TEST
# ============================================================

def main():

    print("=" * 70)
    print("COMMUNICATION MODEL TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CommunicationModel(
        num_classes=4
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameter_count(model):,}"
    )

    # --------------------------------------------------------
    # Load exact V2 checkpoint
    # --------------------------------------------------------

    load_pretrained_v2(
        model,
        device=device,
    )

    # --------------------------------------------------------
    # Freeze backbone for initial communication training
    # --------------------------------------------------------

    freeze_backbone(
        model
    )

    print()
    print(
        f"Trainable parameters after freezing V2: "
        f"{trainable_parameter_count(model):,}"
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        1,
        128,
        501,
        device=device,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        logits, embedding = model(
            x,
            return_embedding=True,
        )

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    print()
    print("MODEL SHAPES")
    print("-" * 70)

    print(
        f"Input shape     : "
        f"{tuple(x.shape)}"
    )

    print(
        f"Embedding shape : "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Logits shape    : "
        f"{tuple(logits.shape)}"
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    print(
        f"Probability shape: "
        f"{tuple(probabilities.shape)}"
    )

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    print()
    print("COMMUNICATION MAPPING")
    print("-" * 70)

    for idx, label in sorted(
        IDX_TO_COMMUNICATION.items()
    ):

        print(
            f"{idx}: {label}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_index = (
        probabilities[0]
        .argmax()
        .item()
    )

    confidence = (
        probabilities[0]
        .max()
        .item()
    )

    print()
    print("TEST PREDICTION")
    print("-" * 70)

    print(
        f"Predicted class : "
        f"{predicted_index}"
    )

    print(
        f"Predicted type  : "
        f"{IDX_TO_COMMUNICATION[predicted_index]}"
    )

    print(
        f"Confidence      : "
        f"{confidence:.4f}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert embedding.shape == (
        2,
        128,
    )

    assert logits.shape == (
        2,
        4,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMMUNICATION MODEL TEST SUCCESSFUL")
    print("=" * 70)


if __name__ == "__main__":
    main()