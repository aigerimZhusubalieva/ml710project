import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import wandb
import pynvml
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

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
def get_vgg16_pipeline_model(num_classes=1000):
    # Define VGG16 as a pipeline
    vgg16 = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    features = list(vgg16.features)
    avgpool = [vgg16.avgpool]
    classifier = list(vgg16.classifier)


    layers = [
    LayerSpec(nn.Sequential, *features[:20]),                    # First 20 layers
    LayerSpec(nn.Sequential, *features[20:], *avgpool, *classifier),  # Remaining + FC
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
def evaluate(engine, val_loader, device):
    engine.eval()
    val_iter = BatchIterator(val_loader, device, fp16_enabled=engine.fp16_enabled())
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for batch in val_iter:
            output = engine.eval_batch(batch)
            logits, labels = output[0], output[1]
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy



def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)

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

    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler)
    valloader = DataLoader(valset, batch_size=args.batch_size, sampler=val_sampler)

    model = get_vgg16_pipeline_model(num_classes=len(trainset.classes))

    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )

    for epoch in range(args.epochs):
        model_engine.train()

        fp16_enabled = model_engine.fp16_enabled() if hasattr(model_engine, "fp16_enabled") else False
        train_iter = BatchIterator(trainloader, device=model_engine.device, fp16_enabled=fp16_enabled)

        for _ in range(len(trainloader)):
            model_engine.train_batch(train_iter)

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

        val_loss, val_acc = evaluate(model_engine, valloader, device=model_engine.device)
        if model_engine.global_rank == 0:
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            })

        print(f"[Epoch {epoch+1}] completed")

    if model_engine.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model_engine.state_dict(), os.path.join(args.output_dir, "vgg16_pipeline.pth"))

    wandb.finish()


if __name__ == "__main__":
    main()