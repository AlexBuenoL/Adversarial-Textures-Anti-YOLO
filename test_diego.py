import torch
import torch.nn.functional as F
from ultralytics import YOLO
import cv2
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_SIZE = 640
TARGET_CLASS = 0  # person class
EPOCHS = 300

EPSILON = 8 / 255.0   # max perturbation magnitude
LR = 0.01

LAMBDA_RECON = 50.0
LAMBDA_TV = 0.1


def total_variation(x):
    return (
        torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:])) +
        torch.mean(torch.abs(x[:, :-1, :] - x[:, 1:, :]))
    )


def optimize_adversarial_image(model, base_img_tensor):

    # Learnable perturbation
    delta = torch.zeros_like(base_img_tensor, requires_grad=True)

    optimizer = torch.optim.Adam([delta], lr=LR)

    print(f"Optimizing adversarial image on {DEVICE}...")

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        # Bound perturbation
        perturbation = EPSILON * torch.tanh(delta)

        # Create adversarial image
        adv_img = torch.clamp(base_img_tensor + perturbation, 0, 1)

        # Forward pass
        preds = model(adv_img)

        raw_tensor = preds[0]  # shape: (1, 84, 8400)

        # Person class confidence
        person_scores = raw_tensor[:, 4 + TARGET_CLASS, :]

        # Detection suppression loss
        topk_scores = torch.topk(person_scores.flatten(), k=20).values
        det_loss = topk_scores.mean()

        # Reconstruction loss
        recon_loss = F.mse_loss(adv_img, base_img_tensor)

        # Smooth perturbation
        tv_loss = total_variation(perturbation[0])

        # Final loss
        loss = (
            det_loss
            + LAMBDA_RECON * recon_loss
            + LAMBDA_TV * tv_loss
        )

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Det: {det_loss.item():.4f} | "
                f"Recon: {recon_loss.item():.6f} | "
                f"TV: {tv_loss.item():.6f}"
            )

    return adv_img, perturbation

def evaluate_image(yolo_model, image_path, title="Image"):

    results = yolo_model(image_path)

    result = results[0]

    print(f"\n--- {title} ---")

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("No detections.")
        return

    for i, box in enumerate(boxes):

        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        class_name = result.names[cls_id]

        print(
            f"Detection {i+1}: "
            f"{class_name} | "
            f"confidence = {conf:.4f}"
        )

if __name__ == '__main__':

    # Load YOLOv8
    print("Loading YOLOv8 model...")

    yolo = YOLO('yolov8n.pt')

    model = yolo.model.to(DEVICE)
    model.eval()

    # Freeze detector
    for param in model.parameters():
        param.requires_grad = False

    # Load image
    img_path = 'person.jpg'

    img_bgr = cv2.imread(img_path)

    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    base_img_tensor = (
        torch.from_numpy(img_rgb)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
        / 255.0
    )

    # Optimize
    adv_img, perturbation = optimize_adversarial_image(
        model,
        base_img_tensor
    )

    # Save adversarial image
    final_adv = (
        adv_img.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    final_adv = (final_adv * 255).astype(np.uint8)

    final_adv_bgr = cv2.cvtColor(final_adv, cv2.COLOR_RGB2BGR)

    cv2.imwrite('adversarial_image.jpg', final_adv_bgr)

    # Save perturbation visualization
    perturb_vis = (
        perturbation.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )

    # Normalize for visualization
    perturb_vis = perturb_vis - perturb_vis.min()
    perturb_vis = perturb_vis / perturb_vis.max()

    perturb_vis = (perturb_vis * 255).astype(np.uint8)

    cv2.imwrite('perturbation_visualization.png', perturb_vis)

    print("Saved adversarial_image.jpg")
    print("Saved perturbation_visualization.png")

    evaluate_image(yolo, img_path, title="Original Image")
    evaluate_image(yolo, 'adversarial_image.jpg', title="Adversarial Image")