import torch
import torch.nn as nn
import torchvision.models as models

def get_model():
    # Load a pretrained ResNet18 model
    model = models.resnet18(pretrained=True)
    
    # Replace the last layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Output 1 value for fresh/not fresh
    
    return model
