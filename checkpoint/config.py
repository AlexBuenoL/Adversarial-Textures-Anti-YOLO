"""
config.py
---------
Single source of truth for every hyperparameter and path used across the
project. Change values here; nothing else needs editing for basic experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    sample_dir: Path = Path("outputs/samples")

    # ------------------------------------------------------------------ #
    # Device (CPU or GPU)
    # ------------------------------------------------------------------ #
    device: str = "cpu"            # 'cpu' or 'cuda' (or 'cuda:0', 'cuda:1', etc.)
                                    # Note: CUDA is NVIDIA-only. For Intel GPU, use CPU.

    # ------------------------------------------------------------------ #
    # Dataset  (HuggingFace streaming)
    # ------------------------------------------------------------------ #
    hf_dataset_name: str = "bitmind/MS-COCO-unique-256_training_faces"
    hf_config_name: str = "base_transforms"  # 'base_transforms' or 'random_aug_transforms'
    hf_split: str = "train"
    hf_train_split_ratio: float = 0.8       # 80% train, 20% validation (manual split)
    dataset_size: int = 7180      # estimated total dataset size (adjust if needed)
    image_size: int = 256          # resize every image to this square size

    # ------------------------------------------------------------------ #
    # YOLO target
    # ------------------------------------------------------------------ #
    yolo_weights: str = "yolov8n.pt"
    yolo_input_size: int = 640     # YOLO expects 640x640
    target_class_id: int = 0       # COCO class 0 = person

    # Top-k detection scores to suppress per image.
    # Lowering k speeds up the loss computation on CPU.
    topk_detections: int = 10

    # ------------------------------------------------------------------ #
    # Perturbation network (UNet)
    # ------------------------------------------------------------------ #
    unet_base_channels: int = 8    # kept small for CPU training

    # ------------------------------------------------------------------ #
    # Adversarial constraint
    # ------------------------------------------------------------------ #
    # Maximum L-inf perturbation in [0, 1] pixel space.
    # 8/255 ≈ 0.031 — imperceptible to humans.
    epsilon: float = 8 / 255.0

    # ------------------------------------------------------------------ #
    # Loss weights
    # ------------------------------------------------------------------ #
    lambda_recon: float = 50.0     # reconstruction fidelity
    lambda_tv: float = 5         # total variation (smooth perturbation)
    
    # Adaptive lambda scheduling
    use_adaptive_lambdas: bool = True         # dynamically adjust lambdas based on detection loss
    det_loss_threshold: float = 0.05           # when det_loss < threshold, start boosting recon/tv
    lambda_recon_max: float = 250.0           # maximum reconstruction weight (when det_loss is very low)
    lambda_tv_max: float = 25.0                # maximum TV weight (when det_loss is very low)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    epochs: int = 1                # number of complete passes through training data
    steps_per_epoch: int | None = None  # calculated as: (dataset_size * train_ratio) // epochs
    learning_rate: float = 1e-3
    log_every: int = 50            # print loss every N steps
    save_every: int = 250          # save checkpoint every N steps
    sample_every: int = 250        # save visual sample every N steps
    num_sample_images: int = 4     # images in the sample grid

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    eval_steps: int | None = None  # calculated as: dataset_size / eval_partition

    def __post_init__(self):
        """Create output directories and calculate training/eval steps."""
        for d in (self.output_dir, self.checkpoint_dir, self.sample_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        
        # Calculate steps_per_epoch to cover all training data
        # steps_per_epoch = (total_dataset * train_ratio) / epochs
        if self.steps_per_epoch is None:
            train_size = int(self.dataset_size * self.hf_train_split_ratio)
            self.steps_per_epoch = train_size // self.epochs
        
        # Calculate eval_steps to process all eval data
        if self.eval_steps is None:
            eval_partition_ratio = 1.0 - self.hf_train_split_ratio
            self.eval_steps = int(self.dataset_size * eval_partition_ratio)


# Module-level default instance — import this everywhere.
cfg = Config()