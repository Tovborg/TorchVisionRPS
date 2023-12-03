import cv2
import numpy as np
import random
import torch
import torch.nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import time

rounds: int = int(input("Enter number of rounds: "))
# Base case
while rounds <= 0:
    print("Rounds must be greater than 0")
    rounds: int = int(input("Enter number of rounds: "))


def computer_move():
    moves = ["rock", "paper", "scissors"]
    return random.choice(moves)


def player_move_from_image(image, model_path):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image)
    model = models.resnet18(pretrained=False)
    num_classes = 4
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

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

        # Player move rectangle and text
        rect_x, rect_y = int((width - rect_width) / 2) + 300, int((height - rect_height) / 2)  # Adjusted rect_x

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

        text = "Player Move"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = rect_x + int((rect_width - text_size[0]) / 2)
        text_y = rect_y - 10  # Adjust vertical position above the rectangle

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Computer move rectangle and text
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
        rect_x, rect_y = int((width - rect_width) / 2) - 300, int((height - rect_height) / 2)  # Adjusted rect_x

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)

        text = "Computer Move"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = rect_x + int((rect_width - text_size[0]) / 2)
        text_y = rect_y - 10  # Adjust vertical position above the rectangle

        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Round text


        if round_number < rounds:
            text = f"Press 'c' to lock in your move"
        else:
            # TODO Make this text announce the actual winner
            text = f"DONE! Press 'ESC' to exit."

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)



        text = f"Round {round_number} of {rounds}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = 10
        text_y = text_size[1] + 20
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        play_key = cv2.waitKey(1)
        if play_key == ord('c'):
            # Player logic
            if round_number < rounds:
                round_number = round_number + 1 if round_number > 0 else round_number
                print(f"Starting round {round_number}")

                # Capture player move image
                rect_x, rect_y = int((width - rect_width) / 2) + 300, int((height - rect_height) / 2)  # Adjusted rect_x

                time.sleep(0.5)
                ret, frame = cap.read()
                roi = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
                player_move = player_move_from_image(roi, "model.pth")
                print(f"Player move: {player_move}")

            else:
                # Determine winner
                print("Done playing, determining winner...")
                if player_score > computer_score:
                    print("Player wins!")
                elif computer_score > player_score:
                    print("Computer wins!")
                else:
                    print("It's a tie!")

        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(1)
        if key == 27:
            print("Thanks for playing!")
            break



play()
