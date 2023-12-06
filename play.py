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


def generate_computer_move():
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

    if move_computer == move_player:
        return "tie"
    elif move_computer == "rock":
        if move_player == "paper":
            return "player"
        elif move_player == "scissors":
            return "computer"
    elif move_computer == "paper":
        if move_player == "rock":
            return "computer"
        elif move_player == "scissors":
            return "player"
    elif move_computer == "scissors":
        if move_player == "rock":
            return "player"
        elif move_player == "paper":
            return "computer"
    return "Error"

def determine_game_winner(player_score, computer_score):
    if player_score == computer_score:
        return "tie"
    winner = "computer" if computer_score > player_score else "player"
    return winner

def create_bold_text(text, font_scale, thickness, text_x, text_y, color, frame):

    rect_width, rect_height = 300, 300
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = text_x + int((rect_width - text_size[0]) / 2)
    text_y = text_y - 10  # Adjust vertical position above the rectangle
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)




def play():
    player_score = 0
    computer_score = 0
    round_number = 1

    player_move: str = ""
    move_computer: str = ""

    round_winner: str = ""
    game_finished: bool = False
    game_started: bool = False

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
        create_bold_text("Player Move", 0.7, 2, rect_x, rect_y, (255, 255, 255), frame)

        # Computer move rectangle and text
        rect_x, rect_y = int((width - rect_width) / 2) - 300, int((height - rect_height) / 2)  # Adjusted rect_x

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
        create_bold_text("Computer Move", 0.7, 2, rect_x, rect_y, (255, 255, 255), frame)
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
            if not game_started:
                game_started = True

            # Player logic
            if round_number < rounds:
                print("--------------------")
                rect_x, rect_y = int((width - rect_width) / 2) + 300, int((height - rect_height) / 2)  # Adjusted rect_x

                # Read player's move from rectangle
                ret, frame = cap.read()
                roi = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
                player_move = player_move_from_image(roi, "model.pth")

                if player_move == "nothing":
                    player_move = "rock"
                print(f"Player move: {player_move}")
                # Put image in player move rectangle

                round_number = round_number + 1 if round_number > 0 else round_number

                # Computer logic
                computer_move = generate_computer_move()
                print(f"Computer move: {computer_move}")
                print("\n")
                # Determine winner
                winner = determine_winner(computer_move, player_move)
                round_winner = winner
                if winner == "computer":
                    computer_score += 1
                elif winner == "player":
                    player_score += 1
                else:
                    pass
                print("Round winner: ", winner)


            else:
                # Determine winner
                game_finished = True

        # Put test text on screen
        if game_started:

            computer_move_image = cv2.imread(f"images/{computer_move.lower()}.jpeg")
            computer_move_image = cv2.resize(computer_move_image, (300, 300))
            rect_x, rect_y = int((width - rect_width) / 2) - 300, int((height - rect_height) / 2)
            frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width] = computer_move_image

            text = f"Player: {player_score} | Computer: {computer_score}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = int((width - text_size[0]) / 2)
            text_y = text_size[1] + 20
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if game_finished:
                winner = determine_game_winner(player_score, computer_score)
                text = f"Game winner: {winner}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = int((width - text_size[0]) / 2)
                text_y = text_size[1] + 50
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                text = f"Round winner: {round_winner}"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = int((width - text_size[0]) / 2)
                text_y = text_size[1] + 50
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)




        cv2.imshow("Rock Paper Scissors", frame)
        key = cv2.waitKey(1)
        if key == 27:
            print("Thanks for playing!")
            break



play()
