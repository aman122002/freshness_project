import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os
import sys

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Load the trained model ---
model_path = os.path.join(os.getcwd(), "..", "results", "model_v2.pth")
model = get_model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --- Load image from command line ---
if len(sys.argv) < 2:
    print("Usage: python predict.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# --- Predict ---
with torch.no_grad():
    output = model(image)
    prob = torch.sigmoid(output).item()
    label = "Fresh" if prob > 0.5 else "Rotten"

print(f"Prediction: {label}")
print(f"Confidence: {prob:.2f}")
