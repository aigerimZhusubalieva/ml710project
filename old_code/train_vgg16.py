import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import wandb
import pynvml
import time

def evaluate(model_engine, dataloader, criterion):
    model_engine.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
            if model_engine.fp16_enabled():
                images = images.half()
            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(dataloader.dataset)
    accuracy = correct / total
    model_engine.train()
    return val_loss, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Setup Weights & Biases
    if args.local_rank == 0:
        run = wandb.init(
            entity="ML701",   
            project="ML710_Project",
            name="ddp-baseline-run1",
            config={
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "architecture": "VGG16",
                "dataset": "ImageFolder"
            }
        )
    else:
        run = None

    # Initialize NVML for GPU monitoring
    pynvml.nvmlInit()
    device = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)

    print(f"Initializing rank {args.local_rank}")

    deepspeed.init_distributed()
    print("Starting training loop")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    # Use DistributedSampler
    train_sampler = DistributedSampler(trainset)
    # Create DataLoader with the sampler
    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)

    valset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'validation'), transform=transform)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

    if args.local_rank == 0:
        wandb.define_metric("epoch")
        wandb.define_metric("avg_train_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        wandb.define_metric("val_accuracy", step_metric="epoch")
        wandb.define_metric("throughput_samples_per_sec", step_metric="epoch")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_samples = 0

        start_time = time.time()

        for images, labels in trainloader:
            images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
            if model_engine.fp16_enabled():
                images = images.half()
            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            # GPU stats (can stay per-step)
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            # You can keep logging per-step loss if you want
            if args.local_rank == 0:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "gpu_memory_allocated_MB": memory_allocated,
                    "gpu_memory_reserved_MB": memory_reserved,
                    "gpu_utilization_percent": utilization
                })

        avg_train_loss = epoch_loss / num_samples
        elapsed_time = time.time() - start_time
        throughput = num_samples / elapsed_time

        val_loss, val_acc = evaluate(model_engine, valloader, criterion)
        print(f"[Epoch {epoch}] Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if args.local_rank == 0:
            wandb.log({
                "avg_train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "throughput_samples_per_sec": throughput,
                "epoch": epoch
            })

    if model_engine.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'vgg16_final.pth'))
    run.finish()
if __name__ == "__main__":
    import deepspeed
    main()