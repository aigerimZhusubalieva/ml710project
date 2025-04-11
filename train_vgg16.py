import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os

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

    if model_engine.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'vgg16_final.pth'))

if __name__ == "__main__":
    import deepspeed
    main()
