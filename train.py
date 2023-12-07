import os.path
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import optim
from torchvision import models
import argparse

parser = argparse.ArgumentParser(description='Rock Paper Scissors argument parser')
parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to collect for each class')
parser.add_argument('--n_collections', type=int, default=1, help="How many times to collect data for each class, "
                                                                 "runs 'collect_data' n_collections times "
                                                                 "useful for collecting data for multiple locations")

args = parser.parse_args()

number_collections = 1


def collect_data(n_samples: int) -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot connect to video source, exiting...")
        exit()

    classes = ["rock", "paper", "scissors", "nothing"]

    current_class = 0
    image_counter = 0
    trigger = False

    while True:

        ret, frame = cap.read()

        height, width, channels = frame.shape

        rect_width, rect_height = 300, 300
        rect_x, rect_y = int((width - rect_width) / 2)+200, int((height - rect_height) / 2)

        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
        text_x = rect_x + 20  # Shift the text to the right by adding 20 pixels
        cv2.putText(frame, "Place your hand in the box", (text_x, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)

        if current_class < len(classes):
            text = f"Press 'c' to start collecting data for class {classes[current_class]}"
        else:
            text = f"DONE! Press 'ESC' to exit."
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = width - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if trigger is not True:
            key = cv2.waitKey(1)
            if key == ord('c'):
                trigger = True
                os.makedirs("data", exist_ok=True)

        if trigger is True:
            os.makedirs(f"data/{classes[current_class]}", exist_ok=True)

            ret, frame = cap.read()
            roi = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
            cv2.imwrite(f"data/{classes[current_class]}/{classes[current_class]}-{image_counter}-{number_collections}.png", roi)

            image_counter += 1
            cv2.putText(frame, f"Collected {image_counter} images", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), 2)

            if image_counter == n_samples:
                print(f"Collected {n_samples} samples for class {classes[current_class]}")
                image_counter = 0
                current_class += 1
                trigger = False

        cv2.imshow("Collecting data", frame)

        key = cv2.waitKey(1)
        # Exit when 'ESC' is pressed
        if key == 27:

            break


if args.n_collections > 1:
    collect_data(args.n_samples)
    for i in range(args.n_collections - 1):
        print(f"Collecting data for the {i+2} time")
        collect_data(args.n_samples)
        number_collections += 1
else:
    collect_data(args.n_samples)

# Path: train.py
print("Starting training...")
time.sleep(2)
print("Using Torch version:", torch.__version__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} for training")

data_dir = "data/"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Default values for ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.to(device)
model.train()

EPOCHS = 10
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10}")
            running_loss = 0.0

torch.save(model.state_dict(), "model.pth")

model = models.resnet18(pretrained=False)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')





