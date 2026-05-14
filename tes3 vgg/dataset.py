import os
import random
import mmap
import io
from datetime import time
from queue import Queue
from threading import Thread
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor
import warnings
from functools import partial


class ImageAugment:
    """图像增强类"""

    def __init__(self, config):
        self.enabled = config['augmentation']['enabled']
        if not self.enabled:
            return

        self.flip = config['augmentation']['flip']
        self.rotate = config['augmentation']['rotate']
        self.brightness = config['augmentation']['brightness']
        self.contrast = config['augmentation']['contrast']

    def __call__(self, lr_img, hr_img):
        if not self.enabled:
            return lr_img, hr_img

        # 保持LR和HR图像一致的转换
        if self.flip and random.random() > 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)

        if self.rotate and random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)

        if self.brightness and random.random() > 0.5:
            brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            lr_img = TF.adjust_brightness(lr_img, brightness_factor)
            hr_img = TF.adjust_brightness(hr_img, brightness_factor)

        if self.contrast and random.random() > 0.5:
            contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            lr_img = TF.adjust_contrast(lr_img, contrast_factor)
            hr_img = TF.adjust_contrast(hr_img, contrast_factor)

        return lr_img, hr_img


class XRayDataset(Dataset):
    """优化后的X射线图像数据集"""

    def __init__(self, lr_dir, hr_dir, config, is_training=True):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.config = config
        self.is_training = is_training
        self.img_size = config['data']['img_size']

        # 获取图像文件列表
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not self.image_files:
            raise RuntimeError(f"未在{lr_dir}中找到图像文件")

        # 配置数据增强
        self.augment = ImageAugment(config) if is_training else None

        # 设置转换pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

        # 配置缓存
        self.cache_size = config['data']['cache_size']
        self.cache = {}

        if self.cache_size > 0:
            self._preload_data()

    def _load_image(self, img_path):
        """优化的图像加载函数"""
        try:
            img = Image.open(img_path).convert('L')
            return img
        except Exception as e:
            warnings.warn(f"加载图像失败 {img_path}: {str(e)}")
            return None

    def _process_image_pair(self, lr_img, hr_img):
        """处理图像对"""
        # 数据增强
        if self.is_training and self.augment is not None:
            lr_img, hr_img = self.augment(lr_img, hr_img)

        # 转换为tensor
        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)

        return lr_tensor, hr_tensor

    def _preload_data(self):
        """并行预加载数据到缓存"""
        print(f"预加载{min(self.cache_size, len(self.image_files))}张图像到缓存...")

        def load_and_process(img_file):
            lr_path = os.path.join(self.lr_dir, img_file)
            hr_path = os.path.join(self.hr_dir, img_file)

            lr_img = self._load_image(lr_path)
            hr_img = self._load_image(hr_path)

            if lr_img is None or hr_img is None:
                return None

            return self._process_image_pair(lr_img, hr_img)

        # 使用线程池并行加载
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for idx, tensor_pair in enumerate(executor.map(load_and_process,
                                                       self.image_files[:self.cache_size])):
                if tensor_pair is not None:
                    self.cache[idx] = tensor_pair

        print(f"成功缓存{len(self.cache)}张图像对")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]

        # 常规加载流程
        img_name = self.image_files[idx]
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)

        lr_img = self._load_image(lr_path)
        hr_img = self._load_image(hr_path)

        if lr_img is None or hr_img is None:
            # 返回零张量作为替代
            return torch.zeros(1, self.img_size, self.img_size), \
                   torch.zeros(1, self.img_size * 4, self.img_size * 4)

        # 处理图像对
        lr_tensor, hr_tensor = self._process_image_pair(lr_img, hr_img)

        # 添加到缓存
        if len(self.cache) < self.cache_size:
            self.cache[idx] = (lr_tensor, hr_tensor)

        return lr_tensor, hr_tensor

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_dataloaders(config):
    """创建优化后的数据加载器"""
    # 验证路径
    for path in [config['paths']['train_lr_dir'], config['paths']['train_hr_dir'],
                 config['paths']['val_lr_dir'], config['paths']['val_hr_dir']]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")

    # 创建训练集
    train_dataset = XRayDataset(
        config['paths']['train_lr_dir'],
        config['paths']['train_hr_dir'],
        config,
        is_training=True
    )

    # 创建验证集
    val_dataset = XRayDataset(
        config['paths']['val_lr_dir'],
        config['paths']['val_hr_dir'],
        config,
        is_training=False
    )

    # 修改数据加载器配置以避免多进程问题
    loader_kwargs = {
        'batch_size': config['train']['batch_size'],
        'pin_memory': True,
        'num_workers': 0,  # 设置为0以避免多进程问题
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader