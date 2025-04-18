import argparse
import deepspeed
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Model Parallelism with DeepSpeed")
    
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--deepspeed', action='store_true', help='Enable DeepSpeed')
    parser.add_argument('--deepspeed_config', type=str, required=True, help='Path to DeepSpeed config file')
    parser.add_argument('--local_rank')
    
    # Parse args
    return parser.parse_args()

# Define a simple VGG16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.vgg(x)
        return self.fc(x)

def train_model(model, train_loader, optimizer, criterion, device, model_engine=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.half()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(train_loader), 100 * correct / total

def main():
    # Parse arguments
    args = parse_args()
    print("Arguments:", vars(args))

    # Data transformation and loading
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Set device for model parallelism
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = VGG16().to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize DeepSpeed
    if args.deepspeed:
        # Make sure to pass the `deepspeed_config` argument explicitly here
        model_engine, optimizer, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model, optimizer=optimizer)

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        loss, accuracy = train_model(model_engine, train_loader, optimizer, criterion, device)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
