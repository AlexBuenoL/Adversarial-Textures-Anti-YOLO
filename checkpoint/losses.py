"""
losses.py
---------
All loss components used during adversarial training.

Three terms are combined into the final scalar:

    L = L_det + λ_recon * L_recon + λ_tv * L_tv

L_det   — detection suppression loss.
            Minimise the top-k person-class confidence scores from YOLO's
            raw output. Driving this toward 0 suppresses person detections.

L_recon — reconstruction loss (MSE between adversarial and original image).
            Keeps the perturbation visually imperceptible.

L_tv    — total variation loss on the perturbation.
            Encourages spatially smooth noise (no salt-and-pepper artefacts).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from config import Config, cfg


# ------------------------------------------------------------------ #
# Individual loss components
# ------------------------------------------------------------------ #

def detection_loss(
    raw_yolo_output: torch.Tensor,
    target_class_id: int,
    topk: int,
) -> torch.Tensor:
    """
    Suppress person detections by minimising the top-k class confidence
    scores across all anchor proposals.

    Parameters
    ----------
    raw_yolo_output : torch.Tensor
        Shape (1, 84, 8400) — raw tensor from the YOLO model forward pass
        before any NMS.  Channels 4..83 are class logits/scores.
    target_class_id : int
        COCO class index to suppress (0 = person).
    topk : int
        Number of highest-scoring proposals to include in the loss.
        Larger k is more thorough but slower on CPU.

    Returns
    -------
    torch.Tensor
        Scalar loss.  Minimising this drives person scores toward zero.
    """
    # raw_yolo_output: (1, 84, 8400)
    # Channel layout: [cx, cy, w, h, cls_0, cls_1, ..., cls_79]
    person_scores = raw_yolo_output[:, 4 + target_class_id, :]  # (1, 8400)

    # Take the top-k and average them — a soft approximation of "max detection"
    k = min(topk, person_scores.numel())
    topk_scores = torch.topk(person_scores.flatten(), k=k).values

    return topk_scores.mean()


def reconstruction_loss(
    adv_image: torch.Tensor,
    orig_image: torch.Tensor,
) -> torch.Tensor:
    """
    Pixel-wise MSE between the adversarial image and the original.
    Drives the network to produce imperceptible perturbations.
    """
    return F.mse_loss(adv_image, orig_image)


def total_variation_loss(perturbation: torch.Tensor) -> torch.Tensor:
    """
    Total variation of the perturbation tensor.
    Penalises abrupt pixel-to-pixel changes, yielding smoother noise.

    Parameters
    ----------
    perturbation : torch.Tensor
        Shape (1, 3, H, W).
    """
    diff_h = torch.abs(perturbation[:, :, :-1, :] - perturbation[:, :, 1:, :])
    diff_w = torch.abs(perturbation[:, :, :, :-1] - perturbation[:, :, :, 1:])
    return diff_h.mean() + diff_w.mean()


# ------------------------------------------------------------------ #
# YOLO input preprocessing
# ------------------------------------------------------------------ #

def preprocess_for_yolo(
    image_tensor: torch.Tensor,
    yolo_size: int,
) -> torch.Tensor:
    """
    Resize a (1, 3, H, W) tensor to the size YOLO expects.
    Kept differentiable so gradients flow back through the resize.
    """
    return F.interpolate(
        image_tensor,
        size=(yolo_size, yolo_size),
        mode="bilinear",
        align_corners=False,
    )


# ------------------------------------------------------------------ #
# Combined loss
# ------------------------------------------------------------------ #

class AdversarialLoss:
    """
    Aggregates the three loss components and returns a named dict
    for clean logging.

    Supports adaptive lambda scheduling: when detection loss gets small,
    the model automatically focuses on reconstruction and smoothness.

    Parameters
    ----------
    config : Config
        Provides lambda weights, topk, and class id.
    """

    def __init__(self, config: Config = cfg):
        self.cfg = config

    def _compute_adaptive_lambdas(self, l_det: float) -> tuple[float, float]:
        """
        Compute adaptive lambda weights based on current detection loss.
        
        When det_loss < threshold, gradually increase lambda_recon and lambda_tv
        to focus on reconstruction quality instead of further suppression.
        
        Parameters
        ----------
        l_det : float
            Current detection loss value.
        
        Returns
        -------
        lambda_recon_adaptive, lambda_tv_adaptive : tuple[float, float]
        """
        if not self.cfg.use_adaptive_lambdas or l_det >= self.cfg.det_loss_threshold:
            return self.cfg.lambda_recon, self.cfg.lambda_tv
        
        # When det_loss is below threshold, increase recon/tv weight
        # Smooth interpolation: 0 → 1 as det_loss goes from threshold → 0
        progress = 1.0 - (l_det / self.cfg.det_loss_threshold)
        progress = min(1.0, max(0.0, progress))  # clamp to [0, 1]
        
        # Exponential interpolation for smoother transition
        progress = progress ** 2  # squared for smoother curve
        
        lambda_recon_adaptive = self.cfg.lambda_recon + (self.cfg.lambda_recon_max - self.cfg.lambda_recon) * progress
        lambda_tv_adaptive = self.cfg.lambda_tv + (self.cfg.lambda_tv_max - self.cfg.lambda_tv) * progress
        
        return lambda_recon_adaptive, lambda_tv_adaptive

    def __call__(
        self,
        raw_yolo_output: torch.Tensor,
        adv_image: torch.Tensor,
        orig_image: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute and return the total loss plus a loggable breakdown.

        Returns
        -------
        total : torch.Tensor
            Scalar loss for .backward().
        breakdown : dict[str, float]
            {'det': ..., 'recon': ..., 'tv': ..., 'total': ..., 'lambda_recon': ..., 'lambda_tv': ...}
        """
        l_det = detection_loss(
            raw_yolo_output,
            self.cfg.target_class_id,
            self.cfg.topk_detections,
        )
        l_recon = reconstruction_loss(adv_image, orig_image)
        l_tv    = total_variation_loss(perturbation)
        
        # Compute adaptive lambdas
        lambda_recon_adaptive, lambda_tv_adaptive = self._compute_adaptive_lambdas(l_det.item())

        total = (
            l_det
            + lambda_recon_adaptive * l_recon
            + lambda_tv_adaptive    * l_tv
        )

        breakdown = {
            "det":   l_det.item(),
            "recon": l_recon.item(),
            "tv":    l_tv.item(),
            "total": total.item(),
            "lambda_recon": lambda_recon_adaptive,
            "lambda_tv":    lambda_tv_adaptive,
        }

        return total, breakdown