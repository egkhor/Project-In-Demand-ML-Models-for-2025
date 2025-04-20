import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess an example image (replace 'example_image.jpg' with your image)
image = Image.open("example_image.jpg").convert("RGB")
image_tensor = preprocess(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

# Load ImageNet class labels (simplified for example)
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

print(f"Predicted class: {labels[predicted_class]} (Confidence: {probabilities[predicted_class]:.2f})")

# Save the model
torch.save(model.state_dict(), "resnet50_classifier.pth")
print("Model saved as resnet50_classifier.pth")
