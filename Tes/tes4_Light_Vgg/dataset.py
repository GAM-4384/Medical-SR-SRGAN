# dataset.py
import os
from PIL import Image
import torch
from torch import device
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import warnings
from functools import partial


class XRayDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None, cache_size=100, augment=False):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.augment = augment
        self.cache_size = cache_size

        # 获取文件列表
        self.image_files = [f for f in os.listdir(lr_dir)
                            if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 初始化缓存
        self.cache = {}

        # 设置基础变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),  # 明确指定数据类型
            transforms.Normalize(mean=[0.0], std=[1.0])
        ]) if transform is None else transform

        # 预加载部分数据
        self._preload_data()

    def _preload_data(self):
        """预加载部分数据到内存"""
        print(f"预加载 {self.cache_size} 张图像...")
        for idx in tqdm(range(min(self.cache_size, len(self.image_files)))):
            try:
                img_name = self.image_files[idx]
                lr_path = os.path.join(self.lr_dir, img_name)
                hr_path = os.path.join(self.hr_dir, img_name)

                lr_img = self._load_and_transform_image(lr_path, is_lr=True)
                hr_img = self._load_and_transform_image(hr_path, is_lr=False)

                if lr_img is not None and hr_img is not None:
                    self.cache[idx] = (lr_img, hr_img)
            except Exception as e:
                warnings.warn(f"预加载图像 {idx} 失败: {str(e)}")
                continue

        print(f"成功预加载 {len(self.cache)} 张图像")

    def _load_and_transform_image(self, img_path, is_lr=True):
        try:
            with Image.open(img_path) as img:
                img = img.convert('L')
                # 确保输入图像质量
                size = (256, 256) if is_lr else (1024, 1024)
                img = img.resize(size, Image.BICUBIC)

                # 转换为张量并归一化到[0,1]
                img = torch.FloatTensor(np.array(img)).unsqueeze(0) / 255.0

                return img
        except Exception as e:
            print(f"加载图像失败 {img_path}: {str(e)}")
            return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取数据集中的一项"""
        try:
            # 尝试从缓存获取
            if idx in self.cache:
                lr_img, hr_img = self.cache[idx]
            else:
                # 从磁盘加载
                img_name = self.image_files[idx]
                lr_path = os.path.join(self.lr_dir, img_name)
                hr_path = os.path.join(self.hr_dir, img_name)

                lr_img = self._load_and_transform_image(lr_path, is_lr=True)
                hr_img = self._load_and_transform_image(hr_path, is_lr=False)

                if lr_img is None or hr_img is None:
                    return torch.zeros(1, 256, 256), torch.zeros(1, 1024, 1024)

            # 数据增强
            if self.augment:
                if torch.rand(1) > 0.5:
                    lr_img = torch.flip(lr_img, [1])
                    hr_img = torch.flip(hr_img, [1])
                if torch.rand(1) > 0.5:
                    lr_img = torch.flip(lr_img, [2])
                    hr_img = torch.flip(hr_img, [2])

            return lr_img, hr_img

        except Exception as e:
            warnings.warn(f"获取图像 {idx} 失败: {str(e)}")
            return torch.zeros(1, 256, 256), torch.zeros(1, 1024, 1024)

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        torch.cuda.empty_cache()


class DataPrefetcher:
    """数据预取器，用于提高数据加载效率"""

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_lr = None
        self.next_hr = None
        self._preload()

    def _preload(self):
        try:
            self.next_lr, self.next_hr = next(self.loader)
        except StopIteration:
            self.next_lr = None
            self.next_hr = None
            return

        with torch.cuda.stream(self.stream):
            self.next_lr = self.next_lr.to(self.device, non_blocking=True)
            self.next_hr = self.next_hr.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        lr = self.next_lr
        hr = self.next_hr
        self._preload()
        return lr, hr