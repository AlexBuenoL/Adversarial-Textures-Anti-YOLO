import torch
import torch.nn.functional as F
from ultralytics import YOLO
import cv2
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 640
PATCH_SIZE = 150  # Size of the patch
TARGET_CLASS = 0  # Label for person
EPOCHS = 100
PATCH_X, PATCH_Y = 250, 250  

def training(model, patch, optimizer, base_img_tensor):
    print(f"Starting patch optimization for {EPOCHS} epochs on {DEVICE}...")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Clamp the patch to valid pixel ranges [0, 1]
        with torch.no_grad():
            patch.clamp_(0, 1)
            
        # Apply patch to the image 
        adv_img = base_img_tensor.clone()
        adv_img[0, :, PATCH_Y:PATCH_Y+PATCH_SIZE, PATCH_X:PATCH_X+PATCH_SIZE] = patch
        
        # Forward pass
        preds = model(adv_img)
        raw_tensor = preds[0] 
        
        # Extract "Person" class scores (channel 4 is COCO class 0)
        person_scores = raw_tensor[:, 4:5, :] # Shape: (1, 1, 8400)
        
        # Calculate loss
        loss = person_scores.max()
        
        # Backpropagate the gradients back to the patch pixels
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Max person confidence logit: {loss.item():.4f}")
    
    return patch, adv_img

if __name__ == '__main__':
    # Load YOLOv8 Nano and extract PyTorch model
    print("Loading YOLOv8 model...")
    yolo = YOLO('yolov8n.pt')
    model = yolo.model.to(DEVICE)
    model.eval()  # Weights are not trained

    # Freeze all parameters 
    for param in model.parameters():
        param.requires_grad = False

    # Initialize patch with random noise
    # requires_grad=True as we want to train the patch weights
    patch = torch.rand((3, PATCH_SIZE, PATCH_SIZE), device=DEVICE, requires_grad=True)

    # Use Adam optimizer to update ONLY the patch pixels
    optimizer = torch.optim.Adam([patch], lr=0.05)

    # Load and preprocess image
    img_path = 'person.jpg' 
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensor [1, 3, 640, 640] normalized to [0, 1]
    base_img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

    # Training
    patch, adv_img = training(model, patch, optimizer, base_img_tensor)

    # Evaluation
    print("\nOptimization Complete. Exporting results...")

    # Detach and convert the optimized patch back to a Numpy image
    final_patch = patch.detach().cpu().permute(1, 2, 0).numpy()
    final_patch = (final_patch * 255).astype(np.uint8)

    # Convert the adversarial image back to BGR for OpenCV saving
    final_adv_img = adv_img.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    final_adv_img = (final_adv_img * 255).astype(np.uint8)
    final_adv_img = cv2.cvtColor(final_adv_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite('optimized_patch.png', cv2.cvtColor(final_patch, cv2.COLOR_RGB2BGR))
    cv2.imwrite('adversarial_person.jpg', final_adv_img)

    print("Saved 'optimized_patch.png' and 'adversarial_person.jpg'")