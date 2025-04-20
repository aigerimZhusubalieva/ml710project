# Here we are doing Hybrid parallelism which is going to be a combination of model parallelism and data parallelism
# For the data parallelism we will divide the dataset into two different GPUs where each will be tackling half of the dataset 
# and for Model parallelism we will have each GPU train on a specific layer. (How will all this happen in two GPU??)
## For VGG we will have GPU 0 -> Conv layer and GPU 1 -> FC layer 
### with deepspeed I can use deepspeed ZERO layer 3 (Stage 3	Optimizer states + Gradients + Model parameters	✅✅✅ Saves maximum memory	Full model + data parallelism (Hybrid Parallelism))

import torch
from datetime import datetime
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import time
import wandb
import pynvml


def evaluate(model_engine, valloader):
    model_engine.eval()
    correct = 0
    total = 0
    val_loss_total = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
            if model_engine.fp16_enabled():
                images = images.half()
            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    avg_val_loss = val_loss_total / len(valloader)

    if model_engine.local_rank == 0:
        print(f"✅ Validation Accuracy: {accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}")
        wandb.log({
            "validation_accuracy": accuracy,
            "validation_loss": avg_val_loss
        })
    model_engine.train()


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
    run = wandb.init(
        entity="ML701",   
        project="Hybrid_parallelism_data_model",
        name=f"zero0-100_mini{datetime.now().strftime('%m%d-%H%M')}",
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

    # Load training dataset
    trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    train_sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)

    # # Load validation dataset
    val_path = os.path.join(args.data_path, 'validation')
    valset = torchvision.datasets.ImageFolder(val_path, transform=transform)
    val_sampler = DistributedSampler(valset, shuffle=False)
    valloader = DataLoader(valset, batch_size=args.batch_size, sampler=val_sampler)

    # Define model, loss, and optimizer
    model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Wrap in DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        running_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for images, labels in trainloader:
            images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
            if model_engine.fp16_enabled():
                images = images.half()

            outputs = model_engine(images)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            total_samples += images.size(0)

            # ✅ Count correct predictions for accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

            # ✅ GPU usage logging per mini-batch (as before)
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            wandb.log({
                "train_loss": loss.item(),
                "gpu_memory_allocated_MB": memory_allocated,
                "gpu_memory_reserved_MB": memory_reserved,
                "gpu_utilization_percent": utilization
            })

        # ✅ After epoch: compute stats
        epoch_time = time.time() - epoch_start
        throughput = total_samples / epoch_time
        train_accuracy = 100 * correct_predictions / total_samples

        if model_engine.local_rank == 0:
            print(f"📊 Epoch {epoch+1} | Loss: {running_loss:.4f} | Acc: {train_accuracy:.2f}% | Time: {epoch_time:.2f}s | Throughput: {throughput:.2f} samples/s")
            wandb.log({
                "epoch_train_loss": running_loss,
                "train/accuracy": train_accuracy,
                "train/throughput": throughput,
                "epoch_time_sec": epoch_time
            })

        # ✅ Run validation
        evaluate(model_engine, valloader)


    # Save model
    if model_engine.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'vgg16_100_Zero_3.pth'))

    run.finish()


if __name__ == "__main__":
    import deepspeed
    main()

# # Here we are doing Hybrid parallelism which is going to be a combination of model parallelism and data parallelism
# # For the data parallelism we will divide the dataset into two different GPUs where each will be tackling half of the dataset 
# # and for Model parallelism we will have each GPU train on a specific layer. (How will all this happen in two GPU??)
# ## For VGG we will have GPU 0 -> Conv layer and GPU 1 -> FC layer 
# ### with deepspeed I can use deepspeed ZERO layer 3 (Stage 3	Optimizer states + Gradients + Model parameters	✅✅✅ Saves maximum memory	Full model + data parallelism (Hybrid Parallelism))
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, DistributedSampler
# import argparse
# import os
# import wandb
# import pynvml


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data-path", type=str)
#     parser.add_argument("--output-dir", type=str)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--lr", type=float, default=0.01)
#     parser.add_argument("--local_rank", type=int, default=-1)
#     parser = deepspeed.add_config_arguments(parser)
#     args = parser.parse_args()

#     # Setup Weights & Biases
#     run = wandb.init(
#         entity="ML701",   
#         project="Hybrid_parallelism_data_model",  # <<<< change this to your project name
#         config={
#             "learning_rate": args.lr,
#             "batch_size": args.batch_size,
#             "epochs": args.epochs,
#             "architecture": "VGG16",
#             "dataset": "ImageFolder"
#         }
#     )

#     # Initialize NVML for GPU monitoring
#     pynvml.nvmlInit()
#     device = torch.cuda.current_device()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(device)

#     print(f"Initializing rank {args.local_rank}")

#     deepspeed.init_distributed()
#     print("Starting training loop")

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)

#     # Use DistributedSampler
#     train_sampler = DistributedSampler(trainset)

#     # Create DataLoader with the sampler
#     trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)

#     model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

#     model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

#     for epoch in range(args.epochs):
#         train_sampler.set_epoch(epoch)  # Important: ensure different shuffling every epoch
#         for images, labels in trainloader:
#             images, labels = images.to(model_engine.local_rank), labels.to(model_engine.local_rank)
#             if model_engine.fp16_enabled():
#                 images = images.half()
#             outputs = model_engine(images)
#             loss = criterion(outputs, labels)
#             model_engine.backward(loss)
#             model_engine.step()

#             # 1. Get GPU memory
#             memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
#             memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # in MB

#             # 2. Get GPU utilization
#             utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # in %

#             # 3. Log to wandb
#             wandb.log({
#                 "train_loss": loss.item(),  # log loss too!
#                 "gpu_memory_allocated_MB": memory_allocated,
#                 "gpu_memory_reserved_MB": memory_reserved,
#                 "gpu_utilization_percent": utilization
#             })

#     if model_engine.local_rank == 0:
#         torch.save(model.state_dict(), os.path.join(args.output_dir, 'vgg16_final.pth'))
#     run.finish()
# if __name__ == "__main__":
#     import deepspeed
#     main()