from __future__ import annotations

import itertools
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from config import Config, cfg
from dataset import build_stream
from losses import AdversarialLoss, preprocess_for_yolo
from model import PerturbationUNet, build_perturbation_net

# -- helpers --

def _load_yolo_backbone(weights: str, device: torch.device) -> torch.nn.Module:
    """
    Load YOLOv8, freeze all parameters, and set it to eval mode.
    """
    yolo = YOLO(weights)
    model = yolo.model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[trainer] YOLO backbone loaded | params: {total_params:,} (frozen)")
    return model

def _save_checkpoint(
    net: PerturbationUNet,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: Path,
) -> None:
    path = checkpoint_dir / f"ckpt_ep{epoch:03d}_s{step:06d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"[trainer] Checkpoint saved → {path}")

def _max_person_confidence(results) -> float:
    """Extract maximum person (class 0) confidence from YOLO results."""
    if results.boxes is None or len(results.boxes) == 0:
        return 0.0
    
    person_boxes = results.boxes[results.boxes.cls == 0]
    if len(person_boxes) == 0:
        return 0.0
    
    return float(person_boxes.conf.max().item())

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (3, H, W) torch tensor [0,1] to PIL Image."""
    img_np = (tensor.cpu().detach().permute(1, 2, 0).numpy() * 255).astype('uint8')
    return Image.fromarray(img_np)

def _draw_prediction_text(
    pil_img: Image.Image, 
    confidence: float,
    position: str = "top"
) -> Image.Image:
    """
    Draw YOLO prediction text on image.
    """
    draw = ImageDraw.Draw(pil_img)
    text = f"Conf: {confidence:.3f}"
    
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # position text
    if position == "top":
        x, y = 5, 5
    else:
        x, y = 5, pil_img.height - text_height - 5
    
    # draw background rectangle
    padding = 2
    draw.rectangle(
        [(x - padding, y - padding), 
         (x + text_width + padding, y + text_height + padding)],
        fill=(0, 0, 0)
    )
    
    # draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return pil_img

def _save_samples(
    net: PerturbationUNet,
    sample_images: list[torch.Tensor],
    yolo: YOLO,
    step: int,
    sample_dir: Path,
) -> None:
    """
    Run the network on a fixed set of sample images and save a grid showing:
        [original (w/ conf) | adversarial (w/ conf) | perturbation]
    
    Also displays YOLO person detection confidence on original and adversarial.
    """
    net.eval()
    grid_images = []

    with torch.no_grad():
        for img in sample_images:
            adv, perturb = net(img)
            
            # convert to numpy for YOLO
            def to_numpy(t):
                return (
                    t.squeeze(0)
                    .cpu()
                    .permute(1, 2, 0)
                    .mul(255)
                    .byte()
                    .numpy()
                )
            
            orig_np = to_numpy(img)
            adv_np = to_numpy(adv)
            
            # get YOLO predictions
            orig_results = yolo(orig_np, verbose=False)
            adv_results = yolo(adv_np, verbose=False)
            
            orig_conf = _max_person_confidence(orig_results[0])
            adv_conf = _max_person_confidence(adv_results[0])
            
            # convert to PIL and add prediction text
            orig_pil = _tensor_to_pil(img[0])
            adv_pil = _tensor_to_pil(adv[0])
            
            orig_pil = _draw_prediction_text(orig_pil, orig_conf, position="top")
            adv_pil = _draw_prediction_text(adv_pil, adv_conf, position="top")
            
            # convert back to tensors
            orig_tensor = TF.to_tensor(orig_pil)
            adv_tensor = TF.to_tensor(adv_pil)
            
            # normalize perturbation to [0,1] for visibility
            p = perturb[0]  # [3, 256, 256]
            
            # normalize each channel independently
            p_vis = torch.zeros_like(p)
            for c in range(p.shape[0]):
                p_c = p[c]
                p_min, p_max = p_c.min(), p_c.max()
                p_vis[c] = (p_c - p_min) / (p_max - p_min + 1e-8)
            
            grid_images.extend([orig_tensor, adv_tensor, p_vis])

    grid = vutils.make_grid(grid_images, nrow=3, padding=2, normalize=False)
    path = sample_dir / f"sample_s{step:06d}.png"
    vutils.save_image(grid, path)
    print(f"[trainer] Sample grid saved -> {path}")

    net.train()


# -- trainer --

class Trainer:
    """
    Encapsulates everything for a training run.
    """

    def __init__(
        self,
        config: Config = cfg,
        resume_from: str | Path | None = None,
    ):
        self.cfg = config
        self.device = torch.device(config.device)

        # models
        self.net = build_perturbation_net(config).to(self.device)
        self.yolo = _load_yolo_backbone(config.yolo_weights, self.device)
        self.yolo_wrapper = YOLO(config.yolo_weights)  # For inference & visualization

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.learning_rate)

        # loss
        self.criterion = AdversarialLoss(config)

        # state
        self.start_epoch = 0
        self.global_step = 0

        if resume_from is not None:
            self._load_checkpoint(Path(resume_from))

        # data (80% training)
        self.stream = build_stream(config, split_type="train")

        # cache a fixed set of images for reproducible sample grids (persons)
        print(f"[trainer] Caching {config.num_sample_images} sample images with person detections…")
        self.sample_images: list[torch.Tensor] = []
        
        attempts = 0
        max_attempts = config.num_sample_images * 20  # safety limit
        
        while len(self.sample_images) < config.num_sample_images and attempts < max_attempts:
            img = next(self.stream).to(self.device)
            
            # check if YOLO detects a person
            def to_numpy(t):
                return (
                    t.squeeze(0)
                    .permute(1, 2, 0)
                    .mul(255)
                    .byte()
                    .numpy()
                )
            
            img_np = to_numpy(img)
            results = self.yolo_wrapper(img_np, verbose=False)
            
            if _max_person_confidence(results[0]) > 0.0:
                self.sample_images.append(img)
                print(f"  [{len(self.sample_images)}/{config.num_sample_images}] Found image with person detection")
            
            attempts += 1
        
        if len(self.sample_images) < config.num_sample_images:
            print(f"[trainer] WARNING: Only found {len(self.sample_images)}/{config.num_sample_images} images with person detections after {attempts} attempts")

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt["step"]
        print(f"[trainer] Taken from {path} (epoch {self.start_epoch}, step {self.global_step})")

    def _train_step(self, orig_image: torch.Tensor) -> dict[str, float]:
        """
        Single optimization step.
        """
        orig_image = orig_image.to(self.device)

        self.optimizer.zero_grad()

        # 1: generate adversarial image
        adv_image, perturbation = self.net(orig_image)

        # 2: resize to YOLO input resolution (differentiable)
        adv_yolo = preprocess_for_yolo(adv_image, self.cfg.yolo_input_size)

        # 3: YOLO forward pass (frozen -> gradients flow)
        with torch.no_grad():
            pass

        # YOLO raw output: (1, 84, 8400)
        raw_output = self.yolo(adv_yolo)

        # index 0 is the raw prediction tensor
        if isinstance(raw_output, (tuple, list)):
            raw_tensor = raw_output[0]
        else:
            raw_tensor = raw_output

        # 4: compute loss
        total_loss, breakdown = self.criterion(
            raw_yolo_output=raw_tensor,
            adv_image=adv_image,
            orig_image=orig_image,
            perturbation=perturbation,
        )

        # 5: loss backpropagation
        total_loss.backward()
        self.optimizer.step()

        return breakdown

    def train(self) -> None:
        """Run the full training."""
        print(
            f"\n[trainer] Starting training | "
            f"epochs={self.cfg.epochs} | "
            f"steps/epoch={self.cfg.steps_per_epoch}\n"
        )

        self.net.train()

        for epoch in range(self.start_epoch, self.cfg.epochs):
            running = {"det": 0.0, "recon": 0.0, "tv": 0.0, "total": 0.0}
            count   = 0

            for local_step in range(self.cfg.steps_per_epoch):
                self.global_step += 1
                count            += 1

                image = next(self.stream).to(self.device)
                breakdown = self._train_step(image)

                for k in running:
                    running[k] += breakdown[k]

                # logging
                if self.global_step % self.cfg.log_every == 0:
                    avg = {k: running[k] / count for k in running}
                    lambda_info = ""
                    if self.cfg.use_adaptive_lambdas and 'lambda_recon' in avg:
                        lambda_info = f" [λ_r={avg['lambda_recon']:.1f} λ_t={avg['lambda_tv']:.3f}]"
                    print(
                        f"Epoch {epoch+1:02d}/{self.cfg.epochs} | "
                        f"Step {self.global_step:06d} | "
                        f"Loss {avg['total']:.4f} "
                        f"(det={avg['det']:.4f} "
                        f"recon={avg['recon']:.6f} "
                        f"tv={avg['tv']:.6f}){lambda_info}"
                    )
                    # reset running stats after each log
                    running = {k: 0.0 for k in running}
                    count   = 0

                # checkpoint
                if self.global_step % self.cfg.save_every == 0:
                    _save_checkpoint(
                        self.net,
                        self.optimizer,
                        epoch,
                        self.global_step,
                        self.cfg.checkpoint_dir,
                    )

                # sample grid
                if self.global_step % self.cfg.sample_every == 0:
                    _save_samples(
                        self.net,
                        self.sample_images,
                        self.yolo_wrapper,
                        self.global_step,
                        self.cfg.sample_dir,
                    )

        # final checkpoint at end of training
        _save_checkpoint(
            self.net,
            self.optimizer,
            self.cfg.epochs,
            self.global_step,
            self.cfg.checkpoint_dir,
        )
        print("\n[trainer] Training complete.")