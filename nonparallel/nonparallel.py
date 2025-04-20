import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from PIL import Image
import wandb
import logging
import time
import pynvml
import os
import torchvision

# --- Setup logging ---
def setup_logger(log_dir="logs", filename_prefix="trainbaseline"):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{filename_prefix}.log")

    # Basic config
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename , mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

logger = setup_logger(log_dir="logs", filename_prefix="vgg16_baseline")
logger.info("Logger initialized")

# --- Set device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Initialize NVML for GPU monitoring ---
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes single GPU

# --- Load dataset ---
# dataset = load_dataset("timm/mini-imagenet", cache_dir="/home/mena.attia/mini-imagenet")

# --- Initialize wandb ---
wandb.init(
    project="vgg16-mini-imagenet",
    name="vgg16-run-3",
    config={
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-4,
        "model": "vgg16",
        "dataset": "mini-imagenet",
        "data_path": "/home/mena.attia/Documents/ml710project/imagenet_mini/"
    }
)
config = wandb.config

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# def transform_example(example):
#     example["image"] = transform(example["image"].convert("RGB"))
#     return example

# dataset = dataset.map(transform_example)
# dataset["train"].set_format(type="torch", columns=["image", "label"])
# dataset["validation"].set_format(type="torch", columns=["image", "label"])
# train_dataset = dataset["train"]
# val_dataset = dataset["validation"]

# train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

trainset = torchvision.datasets.ImageFolder(os.path.join(config.data_path, 'train'), transform=transform)
train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, drop_last=True)

valset = torchvision.datasets.ImageFolder(os.path.join(config.data_path, 'validation'), transform=transform)
val_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, drop_last=True)

# --- Model ---
model = models.vgg16(weights='IMAGENET1K_V1')
model.classifier[6] = nn.Linear(4096, 100)  # 100 classes for mini-ImageNet
model = model.to(device)

# --- Loss and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# --- Training ---
start_time = time.time()
logger.info("Starting training...")

for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0
    epoch_start = time.time()

    for step, batch in enumerate(train_loader):
        inputs, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_samples += inputs.size(0)

        # GPU metrics per step
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # %

        logger.info(f"[Epoch {epoch+1} | Step {step+1}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Mem Alloc: {mem_alloc:.1f} MB | "
                    f"Mem Reserved: {mem_reserved:.1f} MB | "
                    f"GPU Util: {gpu_util}%")

        wandb.log({
            "step_loss": loss.item(),
            "gpu_mem_allocated_MB": mem_alloc,
            "gpu_mem_reserved_MB": mem_reserved,
            "gpu_utilization_percent": gpu_util
        })

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total

    # --- Epoch timing and throughput ---
    epoch_time = time.time() - epoch_start
    num_samples = len(train_loader.dataset)
    throughput = num_samples / epoch_time

    logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.2f}% | "
                f"Time Elapsed: {epoch_time:.2f} sec | "
                f"Throughput: {throughput:.2f} img/s | "
                f"Total Samples: {total_samples}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_accuracy": val_accuracy,
        "epoch_time_sec": epoch_time,
        "epoch_throughput_img_per_sec": throughput
    })

    # --- Save checkpoint ---
    # torch.save({
    #     'epoch': epoch + 1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'val_accuracy': val_accuracy
    # }, f"checkpoint_epoch_{epoch+1}.pt")

# --- Final summary ---
total_time = time.time() - start_time
logger.info(f"Training complete. Total time: {total_time:.2f} seconds.")
wandb.log({"total_training_time_sec": total_time})

torch.save(model.state_dict(), "vgg16_mini_imagenet_final.pt")
logger.info("Model saved to vgg16_mini_imagenet_final.pt")

wandb.finish()