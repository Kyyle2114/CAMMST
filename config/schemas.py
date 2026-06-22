from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    """Base configuration class with common functionality.
    
    Provides dictionary-style access and strict validation settings.
    All config sections should inherit from this class.
    """
    
    model_config = ConfigDict(
        extra='forbid',  # raise error on unknown fields
        validate_default=True,  # validate default values
        str_strip_whitespace=True,  # strip whitespace from strings
    )
    
    def __getitem__(self, key: str):
        """Enable dictionary-style access (e.g., config['seed'])."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value) -> None:
        """Enable dictionary-style assignment (e.g., config['seed'] = 42)."""
        setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get attribute with optional default value."""
        return getattr(self, key, default)


class GeneralConfig(ConfigBase):
    """General configuration settings."""
    
    seed: int = Field(
        default=3927,
        description="Random seed for reproducibility (same as MERGE)",
        ge=0
    )
    output_dir: str = Field(
        default="./output_dir",
        description="Path where to save checkpoints and logs"
    )


class DataConfig(ConfigBase):
    """Dataset configuration settings."""
    
    dataset_path: str = Field(
        default="./data/stnet",
        description="Path to the dataset root directory"
    )
    feature_path: str = Field(
        default="./uni_features",
        description="Path to the UNI features root directory"
    )
    batch_size: int = Field(
        default=1,
        description="Batch size per GPU",
        ge=1
    )
    num_workers: int = Field(
        default=4,
        description="Number of subprocesses for data loading",
        ge=0
    )
    folds: int = Field(
        default=8,
        description="Number of folds for cross-validation (same as MERGE)",
        ge=2
    )


class TrainingConfig(ConfigBase):
    """Training configuration settings."""
    
    epoch: int = Field(
        default=1000,
        description="Total number of training epochs",
        ge=1
    )
    patience: int = Field(
        default=200,
        description="Patience for early stopping (epochs)",
        ge=1
    )
    lr: float = Field(
        default=2e-4,
        description="Base learning rate",
        gt=0
    )
    weight_decay: float = Field(
        default=5e-2,
        description="Weight decay for optimizer",
        ge=0
    )
    grad_accum_steps: int = Field(
        default=1,
        description="Gradient accumulation steps to increase effective batch size",
        ge=1
    )
    warmup_epochs: int = Field(
        default=100,
        description="Number of warmup epochs for learning rate scheduler",
        ge=0
    )
    clip_grad: float | None = Field(
        default=None,
        description="Gradient clipping norm (None for no clipping)",
        gt=0
    )
    mixed_precision: Literal["no", "fp16"] = Field(
        default="no",
        description="Mixed precision training ('no' for fp32, 'fp16' for mixed precision)"
    )


class ModelConfig(ConfigBase):
    """CAMMST model configuration settings."""

    embed_dim: int = Field(
        default=768,
        description="Token embedding dimension",
        ge=64
    )
    num_genes: int = Field(
        default=250,
        description="Number of genes in expression profiles",
        ge=1
    )
    input_dim: int = Field(
        default=1536,
        description="Input feature dimension (1536 for UNI2-h)",
        ge=64
    )
    visible_ratio: float = Field(
        default=0.1,
        description="Fraction of tokens to keep visible (0 < r <= 1)",
        gt=0,
        le=1
    )
    joint_depth: int = Field(
        default=3,
        description="Number of Transformer blocks for the fusion path",
        ge=1
    )
    num_heads: int = Field(
        default=8,
        description="Number of attention heads",
        ge=1
    )
    mlp_ratio: float = Field(
        default=2.0,
        description="Expansion ratio for feedforward networks",
        gt=0
    )
    attn_dropout: float = Field(
        default=0.1,
        description="Dropout for attention probabilities",
        ge=0,
        le=1
    )
    proj_dropout: float = Field(
        default=0.1,
        description="Dropout for projection layers",
        ge=0,
        le=1
    )
    ffn_activation: Literal["gelu", "swiglu"] = Field(
        default="swiglu",
        description="FFN activation function {'gelu', 'swiglu'}"
    )
    contrastive_dim: int | None = Field(
        default=384,
        description="Projection dimension for contrastive head (defaults to embed_dim)",
        ge=64
    )
    decoder_depth: int = Field(
        default=2,
        description="Number of decoder Transformer blocks",
        ge=1
    )
    norm_type: Literal["rms", "layer"] = Field(
        default="rms",
        description="Normalization type {'rms', 'layer'}"
    )
    region_based_sampling: bool = Field(
        default=True,
        description="Use region-based sampling instead of individual spot sampling"
    )
    num_regions: int | None = Field(
        default=3,
        description="Number of regions for region-based sampling",
        ge=1
    )
    sampler_type: Literal["adaptive", "random"] = Field(
        default="adaptive",
        description="Sampler type {'adaptive', 'random'}"
    )


class LossConfig(ConfigBase):
    """Loss function configuration settings."""

    recon_weight: float = Field(
        default=1.0,
        description="Weight for reconstruction loss",
        ge=0
    )
    pcc_weight: float = Field(
        default=0.5,
        description="Weight for PCC loss (λ in paper)",
        ge=0,
        le=1.0
    )
    sample_weight: float = Field(
        default=1e-3,
        description="Weight for sampling loss",
        ge=0
    )
    contrast_weight: float = Field(
        default=0.05,
        description="Weight for contrastive loss",
        ge=0
    )
    contrast_temp: float = Field(
        default=1.0,
        description="Temperature for contrastive loss",
        gt=0
    )
    contrastive_type: Literal["hard", "soft"] = Field(
        default="soft",
        description="Type of contrastive loss: 'hard' (InfoNCE with one-hot targets) or 'soft' (BLEEP-style with similarity-based soft targets)"
    )
    
    # --- Bio-Salience Guided Sampling Loss ---
    bio_salience_method: Literal["regression", "scale_aware_ranking"] = Field(
        default="scale_aware_ranking",
        description="Bio-salience loss method"
    )
    bio_salience_beta: float = Field(
        default=1.5,
        description="Beta exponent for scale-aware ranking weighting",
        ge=0
    )


class WandBConfig(ConfigBase):
    """Weights & Biases configuration settings."""

    project_name: str | None = Field(
        default=None,
        description="WandB project name (None for no logging)"
    )
    run_name: str | None = Field(
        default="CAMMST",
        description="WandB run name"
    )


class Config(ConfigBase):
    """
    Overall configuration structure for model training.
    
    This class aggregates all configuration sections into a single
    validated configuration object.
    """
    
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandBConfig = Field(default_factory=WandBConfig)