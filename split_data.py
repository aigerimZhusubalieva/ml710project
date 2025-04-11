import os
import shutil

src_root = "imagenet_mini"
dst_root = "imagenet_2class"
classes_to_keep = ["n01532829", "n01558993"]

for split in ["validation"]:
    os.makedirs(os.path.join(dst_root, split), exist_ok=True)
    for cls in classes_to_keep:
        src = os.path.join(src_root, split, cls)
        dst = os.path.join(dst_root, split, cls)
        shutil.copytree(src, dst)
