from __future__ import annotations

from typing import Iterator

import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from PIL import Image

from config import Config, cfg

def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """
    Convert a PIL image to a (1, 3, size, size) float32 tensor in [0, 1].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = TF.resize(image, [size, size])
    tensor = TF.to_tensor(image)          # (3, H, W), [0, 1]
    return tensor.unsqueeze(0)            # (1, 3, H, W)


def _iter_hf_images(
    dataset_name: str,
    config_name: str,
    split: str,
    image_size: int,
    split_type: str = "train",
    split_ratio: float = 0.8,
    image_key: str = "image",
) -> Iterator[torch.Tensor]:
    """
    Generator over a HuggingFace streaming dataset,
     so training can request any number of steps.
    """
    split_threshold = int(1.0 / (1.0 - split_ratio)) if split_ratio < 1.0 else 1
    
    while True:
        stream = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=True,
        )
        idx = 0
        for sample in stream:
            if split_type == "train":
                use_sample = (idx % split_threshold) < split_threshold - 1
            else:  # eval
                use_sample = (idx % split_threshold) == (split_threshold - 1)
            
            idx += 1
            
            if not use_sample:
                continue
            
            raw = sample.get(image_key)

            if raw is None:
                for v in sample.values():
                    if isinstance(v, Image.Image):
                        raw = v
                        break

            if raw is None:
                continue 

            yield _to_tensor(raw, image_size)

def build_stream(
    config: Config = cfg,
    split: str | None = None,
    split_type: str = "train",
) -> Iterator[torch.Tensor]:
    """
    Return an infinite iterator of (1, 3, H, W) image tensors.
    """
    return _iter_hf_images(
        dataset_name=config.hf_dataset_name,
        config_name=config.hf_config_name,
        split=split or config.hf_split,
        image_size=config.image_size,
        split_type=split_type,
        split_ratio=config.hf_train_split_ratio,
    )