import os
import torch
import torchvision
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 data with data augmentation
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="data/cifar10", train=True, download=True, transform=train_transform
)
train_length = int(0.8 * len(train_dataset))
val_length = len(train_dataset) - train_length
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_length, val_length])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=2
)

test_dataset = torchvision.datasets.CIFAR10(
    root="data/cifar10", train=False, download=True, transform=test_transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2
)

# Define model, loss, optimizer
model = timm.create_model("resnet18", pretrained=False, num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

def train(epoch):
    model.train()
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Train Epoch {epoch + 1} [{i * len(inputs)}/{len(train_loader.dataset)}]: Loss: {total_loss / (i + 1):.4f}")
    return total_loss / len(train_loader)

def validate(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    print(f"Validation Epoch {epoch + 1}: Avg. Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    return val_accuracy, val_loss


def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / len(test_loader.dataset):.2f}%")

# Ensure the results directory exists
if not os.path.exists('./results'):
    os.makedirs('./results')

# Create a file for storing training logs
log_path = os.path.join("./results", "training_log.txt")
with open(log_path, "w") as log_file:
    log_file.write("Epoch, Train Loss, Validation Loss, Validation Accuracy\n")

best_val_acc = 0.0
save_path = "./results/best_model.pth"
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train(epoch)
    val_acc, val_loss = validate(epoch)
    scheduler.step()
    
    # Append logs to the file
    with open(log_path, "a") as log_file:
        log_file.write(f"{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}, {val_acc:.2f}\n")
    
    # Save model checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

# Load best model for testing
model.load_state_dict(torch.load(save_path))
test()
