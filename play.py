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


def determine_winner(move_computer, move_player):

    combinations = {
        "rock": "scissors",
        "paper": "rock",
        "scissors": "paper"
    }

    if move_player == combinations[move_computer]:
        return "computer"
    elif move_computer == combinations[move_computer]:
        return "player"
    else:
        return "tie"


def play():
    player_score = 0
    computer_score = 0
    round_number = 1

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot connect to video source, exiting...")
        exit()

    while True:
        ret, frame = cap.read()

        height, width, channels = frame.shape

        rect_width, rect_height = 300, 300
        rect_x, rect_y = int((width - rect_width) / 2) + 300, int((height - rect_height) / 2)

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
        text_x = rect_x + 20  # Shift the text to the right by adding 20 pixels
        cv2.putText(frame, "Player Move", (text_x-5, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)
        text = f"Round {round_number} of {rounds}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = 10
        text_y = text_size[1] + 20
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if round_number < rounds:
            text = f"Press 'c' to lock in your move"
        else:
            # TODO Make this text announce the actual winner
            text = f"DONE! Press 'ESC' to exit."

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Rock Paper Scissors", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break



play()
