# train_utils.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from utils import *
from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, rank):
    model.train()
    for images, labels in dataloader:
        images = images.to(rank)
        labels = labels.to(rank)

        


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, criterion, rank):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(rank)
            labels = labels.to(rank)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if rank == 0:
        print(f"Accuracy: {100 * correct / total:.2f}%")


def save_split(split_name):
    split = dataset[split_name]
    image_id = 0
    for example in tqdm(split, desc=f"Saving {split_name}"):
        label = example["label"]
        label_name = dataset["train"].features["label"].int2str(label)
        image = example["image"]
        image: Image.Image  # PIL image

        # Build directory path
        split_dir = os.path.join(output_root, split_name, label_name)
        os.makedirs(split_dir, exist_ok=True)

        # Create a unique filename
        image_id+=1
        filename = f"{image_id}.jpg"
        filepath = os.path.join(split_dir, filename)

        # Save image
        image.save(filepath)
