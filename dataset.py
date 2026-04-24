import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OSDataset(Dataset):
    def __init__(self, json_path, size=256):
        self.size = size

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file {json_path} does not exist.")

        self.root_dir = os.path.dirname(json_path)
        filename = os.path.basename(json_path)

        if "train" in filename:
            self.image_folder = os.path.join(self.root_dir, "train")
        elif "val" in filename:
            self.image_folder = os.path.join(self.root_dir, "val")
        elif "test" in filename:
            self.image_folder = os.path.join(self.root_dir, "test")
        else:
            self.image_folder = self.root_dir

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} items from {json_path} (Folder: {self.image_folder})")

        self.transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        opt_name = item["image_name"]
        prompt = item["prompt"]

        sar_name = opt_name.replace("opt", "sar")

        opt_path = os.path.join(self.image_folder, opt_name)
        sar_path = os.path.join(self.image_folder, sar_name)

        if not os.path.exists(opt_path):
            raise FileNotFoundError(f"Source image missing: {opt_path}")
        if not os.path.exists(sar_path):
            raise FileNotFoundError(f"Target image missing: {sar_path}")

        control_image = Image.open(opt_path).convert("L").convert("RGB")
        target_image = Image.open(sar_path).convert("L").convert("RGB")

        source_tensor = self.transforms(control_image)
        target_tensor = self.transforms(target_image)

        return {
            "source": source_tensor,
            "target": target_tensor,
            "prompt": prompt,
            "base_name": opt_name,
            "sar_name": sar_name
        }