import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import wandb
import pynvml
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
import logging
from datetime import datetime
import time
from torch.distributed import get_rank, is_initialized
# from torch.utils.data import Subset
# import random

def setup_logger(log_dir="logs", filename_prefix="train"):
    os.makedirs(log_dir, exist_ok=True)
    rank = get_rank() if is_initialized() else 0

    log_filename = os.path.join(log_dir, f"{filename_prefix}_rank{rank}.log")

    logger_name = f"vgg16_pipeline_rank{rank}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent logs from duplicating in stdout

    # Clear old handlers
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    # File handler that overwrites on each run
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    formatter = logging.Formatter(f"[RANK {rank}] %(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# ------------------------------
# Args
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()

# ------------------------------
# Custom wrapper to split VGG16
# ------------------------------
def get_vgg16_pipeline_model(num_classes=100):
    # Define VGG16 as a pipeline
    vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    vgg16.classifier[6] = nn.Linear(4096, 100)  

    # Set all ReLU layers to inplace=False
    for module in vgg16.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    features = list(vgg16.features)
    avgpool = [vgg16.avgpool]
    classifier = list(vgg16.classifier)


    layers = [
        LayerSpec(nn.Sequential, *features[:16]),  
        LayerSpec(nn.Sequential, *features[16:], *avgpool, nn.Flatten(), *classifier)
    ]

    return PipelineModule(layers=layers,
                           loss_fn=nn.CrossEntropyLoss(),
                           num_stages=2,
                           partition_method="parameters",
                           activation_checkpoint_interval=0)

# ------------------------------
# Batch Iterator
# ------------------------------
class BatchIterator:
    def __init__(self, dataloader, device, fp16_enabled=False):
        self.dataloader = iter(dataloader)
        self.device = device
        self.fp16 = fp16_enabled

    def __iter__(self):
        return self

    def __next__(self):
        inputs, labels = next(self.dataloader)
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        if self.fp16:
            inputs = inputs.half()
        return inputs, labels

# ------------------------------
# Validation loop
# ------------------------------

# def evaluate(engine, val_loader, device):
#     engine.eval()
#     total_loss = 0.0
#     total_samples = 0

#     val_iter = BatchIterator(val_loader, device, fp16_enabled=engine.fp16_enabled())

#     for batch in val_iter:
#         repeated_batch = iter([batch] * engine.micro_batches)
#         loss = engine.eval_batch(repeated_batch)
#         total_loss += loss.item() * batch[1].size(0)
#         total_samples += batch[1].size(0)

#     avg_loss = total_loss / total_samples
#     engine.train()
#     return avg_loss

def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    logger = setup_logger(log_dir="logs", filename_prefix="vgg16_pipeline")
    logger.info("Logger initialized")

    # Setup wandb
    wandb.init(
        project="ML710_Project",
        name="pipeline-run",
        entity="ML701",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "architecture": "VGG16",
            "dataset": "ImageNet-Mini",
            "parallelism": "pipeline"
        }
    )

    # GPU monitoring
    pynvml.nvmlInit()
    device_id = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    # DeepSpeed distributed init
    deepspeed.init_distributed()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load datasets
    trainset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    valset = torchvision.datasets.ImageFolder(os.path.join(args.data_path, 'validation'), transform=transform)


    # # Set random seed for reproducibility
    # random.seed(42)

    # # Use 10% of the training set
    # subset_indices = random.sample(range(len(trainset)), int(0.05 * len(trainset)))
    # trainset = Subset(trainset, subset_indices)

    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, sampler=val_sampler , drop_last=True)

    model = get_vgg16_pipeline_model(num_classes=len(trainset.classes))

    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    loss_history = []
    window = 1
    total_time = 0
    global_samples = 0

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        start_time = time.time()
        logger.info(f"started training. time: {start_time} ")

        fp16_enabled = model_engine.fp16_enabled() if hasattr(model_engine, "fp16_enabled") else False
        train_iter = BatchIterator(trainloader, device=model_engine.device, fp16_enabled=fp16_enabled)

        total_samples = 0

        total_loss = 0.0
        step = 0

        # for batch in train_iter:
        while True:
            try:
                batch_group = [next(train_iter) for _ in range(2)]  # 2 micro-batches
                loss = model_engine.train_batch(iter(batch_group))
                total_loss += loss.item()
                step += 1
                total_samples += args.batch_size
                # Log GPU info + training loss
                mem_alloc = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                mem_res = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

                wandb.log({
                    "train_loss": loss.item(),
                    "gpu_memory_allocated_MB": mem_alloc,
                    "gpu_memory_reserved_MB": mem_res,
                    "gpu_utilization_percent": util
                })
            except StopIteration:
                # Safely drain pipeline and avoid timer errors
                model_engine.timers("train_batch").reset()
                model_engine.timers("batch_input").reset()
                break

        

        train_loss = total_loss / step
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")


        elapsed = time.time() - start_time  # End timing
        throughput = total_samples / elapsed  # images/sec

        total_time += elapsed
        global_samples += total_samples

        # val_loss= evaluate(model_engine, valloader, device=model_engine.device)
        loss_history.append(train_loss)

        if len(loss_history) > window:
            delta_loss = abs(loss_history[-1] - loss_history[-1 - window])
            stat_eff = delta_loss / (window * step)
        else:
            stat_eff = 0

        # if model_engine.global_rank == 0:
        wandb.log({
        # "val_loss": val_loss,
        # "val_accuracy": val_acc,
        "throughput": throughput,
        "statistical_efficiency": stat_eff,
        "epoch": epoch
    }),
        logger.info(
            f"Rank {model_engine.global_rank} "
            f"[Epoch {epoch+1}] "
            f"Throughput: {throughput:.2f} img/s | "
            # f"Validation loss: {val_loss:.4f} | "
            f"train loss: {train_loss} "
            f"steps: {step} "
            f"total samples in epoch: {total_samples} "
            f"time elapsed: {elapsed} "
            f"overall time elapsed: {total_time} "
            f"overall samples: {global_samples} "
            f"Statistical Efficiency: {stat_eff:.2f}"
        )

        print(f"[Epoch {epoch+1}] completed")

    # if model_engine.global_rank == 0:
    #     model_engine.save_checkpoint(args.output_dir, tag="vgg16_pipeline")
        # os.makedirs(args.output_dir, exist_ok=True)
        # torch.save(model_engine.state_dict(), os.path.join(args.output_dir, "vgg16_pipeline.pth"))

    success = model_engine.save_checkpoint(args.output_dir, tag="vgg16_pipeline")
    if model_engine.global_rank == 0:
        print(f"[Checkpoint] Saved: {success}")

    wandb.finish()


if __name__ == "__main__":
    main()