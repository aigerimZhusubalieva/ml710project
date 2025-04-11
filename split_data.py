import os
import shutil
import os
from PIL import Image
import os
from tqdm import tqdm
from datasets import load_dataset


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

output_root = "imagenet_mini"
dataset = load_dataset("timm/mini-imagenet")
save_split("train")
save_split("validation")

src_root = "imagenet_mini"
dst_root = "imagenet_2class"
classes_to_keep = ["n01532829", "n01558993"]

for split in ["validation"]:
    os.makedirs(os.path.join(dst_root, split), exist_ok=True)
    for cls in classes_to_keep:
        src = os.path.join(src_root, split, cls)
        dst = os.path.join(dst_root, split, cls)
        shutil.copytree(src, dst)
