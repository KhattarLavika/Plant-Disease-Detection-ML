import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import multiprocessing

def train_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define data transformers
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    training_dataset = ImageFolder(root="C:\\Users\\HP\\Desktop\\PlantDisease\\PlantVillage", transform=transformation)
    train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)

    # Loading pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(training_dataset.classes))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model
    save_dir = 'C:\\Users\\HP\\Desktop\\PlantDisease'
    save_path = os.path.join(save_dir, 'model.pth')

    # Save the model's state dictionary to a .pth file
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()
