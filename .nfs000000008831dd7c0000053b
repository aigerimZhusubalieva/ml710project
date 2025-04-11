import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import wandb
import pynvml


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

    wandb.login(key="9070ac0892430e02d60f5f9927d269ca331fe9c2")
    
    # Setup Weights & Biases
    run = wandb.init(
        entity="ML701",   
        project="ML710_Project",  # <<<< change this to your project name
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "architecture": "VGG16",
            "dataset": "ImageFolder"
        }
    )

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

    model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Important: ensure different shuffling every epoch
        for images, labels in trainloader:
            images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
            if model_engine.fp16_enabled():
                images = images.half()
            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

            # 1. Get GPU memory
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # in MB

            # 2. Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # in %

            # 3. Log to wandb
            wandb.log({
                "train_loss": loss.item(),  # log loss too!
                "gpu_memory_allocated_MB": memory_allocated,
                "gpu_memory_reserved_MB": memory_reserved,
                "gpu_utilization_percent": utilization
            })

    if model_engine.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'vgg16_final.pth'))
    run.finish()
if __name__ == "__main__":
    import deepspeed
    main()
