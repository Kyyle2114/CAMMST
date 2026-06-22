from __future__ import annotations

import timm

import torch
from torch import nn


__all__ = ["UNI"]


class UNI(nn.Module):
    """Wrapper around the UNI2-h foundation model to produce patch embeddings."""

    def __init__(
        self,
        model_name: str = "hf-hub:MahmoodLab/UNI2-h",
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            model_name: timm model identifier (default: UNI2-h from HF hub).
            pretrained: Whether to load pretrained weights.
        """
        super().__init__()

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        self.model = timm.create_model(model_name, pretrained=pretrained, **timm_kwargs)
        self.model.eval()
        self.embed_dim: int = timm_kwargs["embed_dim"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode batched images to embeddings.

        Args:
            images: Tensor of shape (B, C, H, W) or (B, N, C, H, W) if flattened before call.

        Returns:
            Tensor of shape (B, embed_dim) for 4D input or (B, N, embed_dim) for 5D input.
        """
        if images.dim() == 5:
            bsz, num_patches, c, h, w = images.shape
            flat = images.view(bsz * num_patches, c, h, w)
            feats = self.model(flat)
            feats = feats.view(bsz, num_patches, -1)
        elif images.dim() == 4:
            feats = self.model(images)
        else:
            raise ValueError(f"images must have shape (B, C, H, W) or (B, N, C, H, W), got {images.shape}")
        return feats