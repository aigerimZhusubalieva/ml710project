import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import models, transforms
from datasets import load_dataset
import wandb
import logging
# from datetime import datetime
from torch.distributed import get_rank, is_initialized
import argparse
import pynvml
from torch.cuda.amp import GradScaler, autocast



def setup_logger(log_dir="logs", filename_prefix="trainddp"):
    os.makedirs(log_dir, exist_ok=True)
    rank = get_rank() if is_initialized() else 0
    log_filename = os.path.join(log_dir, f"{filename_prefix}_rank{rank}.log")

    # Basic config
    logging.basicConfig(
        level=logging.INFO,
        format=f"[RANK {rank}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename , mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return val_loss / total, correct / total

def train():
    logger = setup_logger(log_dir="logs", filename_prefix="vgg16_ddp")
    logger.info("Logger initialized")
    setup()
    scaler = GradScaler()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
      run = wandb.init(
        entity="ML701",   
            project="ML710_Project",
            name="ddp-baseline-run1",
            config={
                "learning_rate": 0.0001,
                "batch_size": args.batch_size,
                "epochs": 5,
                "architecture": "VGG16",
                "dataset": "ImageFolder"
            })

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])

    # def transform_example(example):
    #   example["image"] = transform(example["image"])
    #   return example

    # dataset = load_dataset("timm/mini-imagenet", cache_dir="/home/mena.attia/mini-imagenet")

    # dataset = dataset.map(transform_example)
    # dataset["train"].set_format(type="torch", columns=["image", "label"])
    # dataset["validation"].set_format(type="torch", columns=["image", "label"])
    # train_dataset = dataset["train"]
    # val_dataset = dataset["validation"]

    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, pin_memory=True)

    # Initialize pynvml for GPU monitoring
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load datasets
    trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    valset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'validation'), transform=transform)

    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, num_workers=4, pin_memory=True,
    persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, sampler=val_sampler , drop_last=True, num_workers=4, pin_memory=True,
    persistent_workers=True)
    

    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(4096, 100)  # 100 classes for mini-ImageNet
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        if rank == 0:
          print(f"[Epoch {epoch+1}] Starting training loop")
        model.train()
        train_sampler.set_epoch(epoch)

        start_time = time.time()
        total, correct, running_loss = 0, 0, 0.0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            # GPU Utilization Logging
            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu    # %

            if rank == 0:
                logger.info(f"[Epoch {epoch+1} | Step {step+1}] Loss: {loss.item():.4f} | "
                            f"Total samples {total} | "
                            f"Mem Alloc: {mem_allocated:.1f} MB | "
                            f"Mem Reserved: {mem_reserved:.1f} MB | "
                            f"GPU Util: {util}%")
                wandb.log({
                    "step_loss": loss.item(),
                    "gpu_mem_allocated_MB": mem_allocated,
                    "gpu_mem_reserved_MB": mem_reserved,
                    "gpu_utilization_percent": util
                })

        elapsed = time.time() - start_time
        train_loss = running_loss / total
        train_acc = correct / total
        throughput = total / elapsed
        stat_eff = throughput * train_acc

        if rank == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "throughput": throughput,
                "statistical_efficiency": stat_eff
            })
            logger.info(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Total samples {total} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                  f"Throughput: {throughput:.2f} img/s | Stat. Eff.: {stat_eff:.2f}")

    cleanup()

if __name__ == "__main__":
    train()