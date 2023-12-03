import cv2
import numpy as np
import random
import torch
import torch.nn
from torchvision import models
from PIL import Image
from torchvision import transforms

rounds: int = int(input("Enter number of rounds: "))
# Base case
while rounds <= 0:
    print("Rounds must be greater than 0")
    rounds: int = int(input("Enter number of rounds: "))


def computer_move():
    moves = ["rock", "paper", "scissors"]
    return random.choice(moves)


def player_move_from_image(image, model_path, class_labels):
    model = models.resnet18(pretrained=False)
    num_classes = 4
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    image = Image.open(image).convert('RGB')

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    # Apply transformations to the input image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(input_batch)

    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

    class_labels = ['nothing', 'paper', 'rock', 'scissors']
    predicted_label = class_labels[predicted_class]

    return predicted_label


