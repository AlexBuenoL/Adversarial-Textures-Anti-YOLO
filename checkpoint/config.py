from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # paths
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    sample_dir: Path = Path("outputs/samples")

    # device
    device: str = "cpu"           
                                    
    # dataset
    hf_dataset_name: str = "bitmind/MS-COCO-unique-256_training_faces"
    hf_config_name: str = "base_transforms"
    hf_split: str = "train"
    hf_train_split_ratio: float = 0.8
    dataset_size: int = 7180     
    image_size: int = 256         

    # YOLO
    yolo_weights: str = "yolov8n.pt"
    yolo_input_size: int = 640     # YOLO expects 640x640
    target_class_id: int = 0       # COCO class 0 = person

    # top-k detection scores to suppress per image.
    topk_detections: int = 10

    # perturbation network (UNet)
    unet_base_channels: int = 8

    # adversarial constraint: maximum L-inf perturbation in [0, 1] pixel space.
    epsilon: float = 8 / 255.0

    # loss weights
    lambda_recon: float = 50.0     # reconstruction fidelity
    lambda_tv: float = 5           # total variation
    
    # adaptive lambda scheduling
    use_adaptive_lambdas: bool = True         # dynamically adjust lambdas based on detection loss
    det_loss_threshold: float = 0.05          # when det_loss < threshold, increase recon/tv
    lambda_recon_max: float = 250.0           # maximum reconstruction weight (when det_loss is very low)
    lambda_tv_max: float = 25.0               # maximum TV weight (when det_loss is very low)

    # training
    epochs: int = 1               
    steps_per_epoch: int | None = None  
    learning_rate: float = 1e-3
    log_every: int = 50           
    save_every: int = 250         
    sample_every: int = 250        
    num_sample_images: int = 4     

    # evaluation
    eval_steps: int | None = None 

    def __post_init__(self):
        # create output directories and calculate training/eval steps"""
        for d in (self.output_dir, self.checkpoint_dir, self.sample_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        
        # calculate steps_per_epoch to cover all training data
        if self.steps_per_epoch is None:
            train_size = int(self.dataset_size * self.hf_train_split_ratio)
            self.steps_per_epoch = train_size // self.epochs
        
        # calculate eval_steps to process all eval data
        if self.eval_steps is None:
            eval_partition_ratio = 1.0 - self.hf_train_split_ratio
            self.eval_steps = int(self.dataset_size * eval_partition_ratio)


cfg = Config()