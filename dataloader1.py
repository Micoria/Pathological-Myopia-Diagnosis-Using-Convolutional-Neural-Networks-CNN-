import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        print(f"✅ Loaded {len(self.imgs_info)} images from {txt_path}")
        if len(self.imgs_info) == 0:
            raise ValueError(f"❌ {txt_path} is empty! Check split_dataset.py.")

        self.train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = [line.strip().split(' ') for line in f.readlines()]
        print(f"🔍 First 5 parsed lines from {txt_path}: {imgs_info[:5]}")
        return imgs_info

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ File not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = self.train_tf(img) if self.train_flag else self.val_tf(img)
        return img, int(label)

    def __len__(self):
        return len(self.imgs_info)