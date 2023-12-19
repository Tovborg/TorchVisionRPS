import cv2
import numpy as np
import random
import torch
import torch.nn
from torchvision import transforms
import time
import torch.nn as nn
import pretrainedmodels

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

rounds: int = int(input("Enter number of rounds: "))
# Base case
while rounds <= 0:
    print("Rounds must be greater than 0")
    rounds: int = int(input("Enter number of rounds: "))


def generate_computer_move() -> str:
    moves = ["rock", "paper", "scissors"]
    return random.choice(moves)


from PIL import Image
import torch
import pretrainedmodels
from torchvision import transforms

def player_move_from_image(image, model_path) -> str:
    # Convert numpy array to PIL Image
    image = Image.fromarray(image)

    # Load the pre-trained NASNetMobile model
    model_name = "nasnetamobile"
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)  # Don't load pretrained weights here
    num_ftrs = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(num_ftrs, 4)  # Adjust for 4 output classes
    model.load_state_dict(torch.load(model_path))  # Load model state_dict

    model.eval()  # Set the model to evaluation mode

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]),
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





def determine_winner(move_computer, move_player) -> str:

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
    elif move_player == "nothing":
        return "computer"
    return "Error"


def determine_game_winner(player_score, computer_score) -> str:
    if player_score == computer_score:
        return "tie"
    winner = "computer" if computer_score > player_score else "player"
    return winner


def create_bold_text(text, font_scale, thickness, text_x, text_y, color, frame) -> None:
    rect_width, rect_height = 300, 300
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = text_x + int((rect_width - text_size[0]) / 2)
    text_y = text_y - 10  # Adjust vertical position above the rectangle
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def play() -> None:
    player_score: int = 0
    computer_score: int = 0
    round_number: int = 1

    round_winner: str = ""
    game_finished: bool = False
    game_started: bool = False

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot connect to video source, exiting...")
        exit()

    while True:
        ret, frame = cap.read()

        height, width, _ = frame.shape

        rect_width, rect_height = 300, 300

        # Player move rectangle and text
        player_rect_x, player_rect_y = int((width - rect_width) / 2) + 300, int((height - rect_height) / 2)

        cv2.rectangle(frame, (player_rect_x, player_rect_y), (player_rect_x + rect_width, player_rect_y + rect_height), (0, 255, 0), 2)
        create_bold_text("Player Move", 0.7, 2, player_rect_x, player_rect_y, (255, 255, 255), frame)

        # Computer move rectangle and text
        computer_rect_x, computer_rect_y = int((width - rect_width) / 2) - 300, int((height - rect_height) / 2)

        cv2.rectangle(frame, (computer_rect_x, computer_rect_y), (computer_rect_x + rect_width, computer_rect_y + rect_height), (0, 255, 0), 2)
        create_bold_text("Computer Move", 0.7, 2, computer_rect_x, computer_rect_y, (255, 255, 255), frame)

        # Player instructions text
        if round_number <= rounds:
            instructions_text = f"Press 'c' to lock in your move"
        else:
            # TODO Make this text announce the actual winner
            instructions_text = f"DONE! Press 'ESC' to exit."

        instructions_text_size, _ = cv2.getTextSize(instructions_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        instructions_text_x = width - instructions_text_size[0] - 10
        instructions_text_y = instructions_text_size[1] + 10
        cv2.putText(frame, instructions_text, (instructions_text_x, instructions_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Round number text
        if round_number <= rounds:
            round_n_text = f"Round {round_number} of {rounds}"
        else:
            round_n_text = f"Game finished"

        round_n_text_size, _ = cv2.getTextSize(round_n_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        round_n_text_x = 10
        round_n_text_y = round_n_text_size[1] + 20
        cv2.putText(frame, round_n_text, (round_n_text_x, round_n_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Await player input
        play_key = cv2.waitKey(1)
        if play_key == ord('c'):
            # Start game
            if not game_started:
                game_started = True

            # Player logic
            if round_number <= rounds:
                print("--------------------")

                # Read player's move from rectangle
                ret, frame = cap.read()
                roi = frame[player_rect_y:player_rect_y + rect_height, player_rect_x:player_rect_x + rect_width]
                player_move = player_move_from_image(roi, "model.pth")
                # TODO needs fixing
                # if player_move == "nothing":
                #     player_move = "rock"
                print(f"Player move: {player_move}")
                # Put image in player move rectangle

                round_number = round_number + 1

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

        # Put test text on screen
        if round_number > rounds:
            game_finished = True

        if game_started:

            computer_move_image = cv2.imread(f"images/{computer_move.lower()}.jpeg")
            computer_move_image = cv2.resize(computer_move_image, (300, 300))

            frame[computer_rect_y:computer_rect_y + rect_height, computer_rect_x:computer_rect_x + rect_width] \
                = computer_move_image

            player_move_text = f"Predicted Player Move: {player_move}"
            text_size, _ = cv2.getTextSize(player_move_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = player_rect_x + int((rect_width - text_size[0]) / 2)
            text_y = player_rect_y + 325  # Adjust vertical position above the rectangle
            cv2.putText(frame, player_move_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            score_text = f"Player: {player_score} | Computer: {computer_score}"
            text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            score_text_x = int((width - text_size[0]) / 2)
            score_text_y = text_size[1] + 20
            cv2.putText(frame, score_text, (score_text_x, score_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

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
