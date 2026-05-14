import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import warnings
from functools import partial
import torch.nn.functional as F


class XRayDataset(Dataset):
    """增强版X射线图像数据集加载器"""

    def __init__(self, lr_dir, hr_dir, config, is_training=True):
        """初始化数据集

        Args:
            lr_dir (str): 低分辨率图像目录
            hr_dir (str): 高分辨率图像目录
            config (dict): 配置字典
            is_training (bool): 是否为训练模式
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.config = config
        self.is_training = is_training

        # 获取图像文件列表
        self.image_files = self._get_image_files()

        # 设置数据增强和转换
        self.transform = self._setup_transforms()

        # 初始化缓存
        self.cache_size = min(config['data']['cache_size'], len(self.image_files))
        self.cache = {}

        # 预加载数据
        if self.cache_size > 0:
            self._preload_data()

    def _get_image_files(self):
        """获取有效的图像文件列表"""
        try:
            # 获取所有PNG文件
            lr_files = set(f for f in os.listdir(self.lr_dir) if f.endswith('.png'))
            hr_files = set(f for f in os.listdir(self.hr_dir) if f.endswith('.png'))

            # 只保留同时存在于两个目录的文件
            valid_files = sorted(list(lr_files.intersection(hr_files)))

            if not valid_files:
                raise ValueError("No valid image pairs found")

            print(f"Found {len(valid_files)} valid image pairs")
            return valid_files

        except Exception as e:
            raise RuntimeError(f"Error loading image files: {str(e)}")

    def _setup_transforms(self):
        """设置图像转换和增强"""
        transform_list = [
            transforms.ToTensor(),
        ]

        if self.is_training and self.config['data']['augmentation']['enabled']:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=self.config['data']['augmentation']['flip_probability']),
                transforms.RandomRotation(10) if self.config['data']['augmentation'][
                                                     'rotate_probability'] > random.random() else None,
                transforms.ColorJitter(
                    brightness=self.config['data']['augmentation']['brightness_range'],
                    contrast=self.config['data']['augmentation']['contrast_range']
                )
            ])

        transform_list = [t for t in transform_list if t is not None]
        return transforms.Compose(transform_list)

    def _load_and_process_image(self, img_path):
        """加载并处理单个图像"""
        try:
            with Image.open(img_path) as img:
                img = img.convert('L')  # 转换为灰度图
                if self.transform:
                    img = self.transform(img)
                return img.float()
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {str(e)}")
            return None

    def _preload_data(self):
        """并行预加载数据"""
        print(f"Preloading {self.cache_size} images...")

        def load_pair(idx):
            img_name = self.image_files[idx]
            lr_path = os.path.join(self.lr_dir, img_name)
            hr_path = os.path.join(self.hr_dir, img_name)

            lr_img = self._load_and_process_image(lr_path)
            hr_img = self._load_and_process_image(hr_path)

            if lr_img is not None and hr_img is not None:
                return idx, (lr_img, hr_img)
            return None

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                for result in executor.map(load_pair, range(self.cache_size)):
                    if result is not None:
                        idx, pair = result
                        self.cache[idx] = pair

            print(f"Successfully preloaded {len(self.cache)} images")

        except Exception as e:
            warnings.warn(f"Error during preloading: {str(e)}")
            self.cache.clear()

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取数据对"""
        try:
            # 检查缓存
            if idx in self.cache:
                return self.cache[idx]

            # 加载图像
            img_name = self.image_files[idx]
            lr_path = os.path.join(self.lr_dir, img_name)
            hr_path = os.path.join(self.hr_dir, img_name)

            lr_img = self._load_and_process_image(lr_path)
            hr_img = self._load_and_process_image(hr_path)

            if lr_img is None or hr_img is None:
                # 返回零张量作为替代
                return torch.zeros(1, 256, 256), torch.zeros(1, 1024, 1024)

            # 保存到缓存
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (lr_img, hr_img)

            return lr_img, hr_img

        except Exception as e:
            print(f"Error loading image pair {idx}: {str(e)}")
            return torch.zeros(1, 256, 256), torch.zeros(1, 1024, 1024)

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_data_loaders(config):
    """创建数据加载器

    Args:
        config (dict): 配置字典

    Returns:
        tuple: (训练数据加载器, 验证数据加载器)
    """
    try:
        # 验证数据目录
        required_dirs = [
            config['paths']['train_lr_dir'],
            config['paths']['train_hr_dir'],
            config['paths']['val_lr_dir'],
            config['paths']['val_hr_dir']
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        # 创建训练集
        train_dataset = XRayDataset(
            lr_dir=config['paths']['train_lr_dir'],
            hr_dir=config['paths']['train_hr_dir'],
            config=config,
            is_training=True
        )

        # 创建验证集
        val_dataset = XRayDataset(
            lr_dir=config['paths']['val_lr_dir'],
            hr_dir=config['paths']['val_hr_dir'],
            config=config,
            is_training=False
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=config['data']['shuffle'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=config['data']['prefetch_factor'],
            persistent_workers=config['data']['persistent_workers'],
            drop_last=config['data']['drop_last']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['validation']['batch_size'],
            shuffle=False,
            num_workers=max(1, config['data']['num_workers'] // 2),
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=config['data']['prefetch_factor'],
            persistent_workers=config['data']['persistent_workers']
        )

        print(f"Created data loaders - Training: {len(train_dataset)} images, "
              f"Validation: {len(val_dataset)} images")

        return train_loader, val_loader

    except Exception as e:
        raise RuntimeError(f"Error creating data loaders: {str(e)}")


class DataPrefetcher:
    """数据预取器，用于加速数据加载"""

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_lr = None
        self.next_hr = None
        self._preload()

    def _preload(self):
        """预加载下一批数据"""
        try:
            self.next_lr, self.next_hr = next(self.loader)
        except StopIteration:
            self.next_lr = None
            self.next_hr = None
            return

        with torch.cuda.stream(self.stream):
            self.next_lr = self.next_lr.to(self.device, non_blocking=True)
            self.next_hr = self.next_hr.to(self.device, non_blocking=True)

            # 转换为 channels_last 内存格式以提高性能
            self.next_lr = self.next_lr.to(memory_format=torch.channels_last)
            self.next_hr = self.next_hr.to(memory_format=torch.channels_last)

    def next(self):
        """获取下一批数据"""
        torch.cuda.current_stream().wait_stream(self.stream)
        lr = self.next_lr
        hr = self.next_hr
        self._preload()
        return lr, hr


def get_dataloader_stats(loader):
    """获取数据加载器的统计信息"""
    total_samples = len(loader.dataset)
    batch_size = loader.batch_size
    num_batches = len(loader)
    num_workers = loader.num_workers

    stats = {
        'total_samples': total_samples,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'num_workers': num_workers,
        'pin_memory': loader.pin_memory,
        'shuffle': loader.shuffle,
        'drop_last': loader.drop_last
    }

    return stats