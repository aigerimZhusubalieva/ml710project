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