import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
import wandb



# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load dataset ---
dataset = load_dataset("timm/mini-imagenet", cache_dir="/home/mena.attia/mini-imagenet")


# --- Initialize wandb ---
wandb.init(
    project="vgg16-mini-imagenet",
    name="vgg16-run-3",
    config={
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-4,
        "model": "vgg16",
        "dataset": "mini-imagenet"
    }
)
config = wandb.config

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

def transform_example(example):
    example["image"] = transform(example["image"].convert("RGB"))
    return example

dataset = dataset.map(transform_example)
dataset["train"].set_format(type="torch", columns=["image", "label"])
dataset["validation"].set_format(type="torch", columns=["image", "label"])
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

# --- Model ---
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 100)  # 100 classes for mini-ImageNet
model = model.to(device)

# --- Loss and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)


# --- Training ---
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total

    # --- wandb log ---
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_accuracy": val_accuracy
    })

    print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }, f"checkpoint_epoch_{epoch+1}.pt")
    

print("Training complete.")

torch.save(model.state_dict(), "vgg16_mini_imagenet_final.pt")
print("Model saved to vgg16_mini_imagenet_final.pt")

wandb.finish()