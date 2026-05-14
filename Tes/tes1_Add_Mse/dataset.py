import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
import threading
from functools import partial


class XRayDataset(Dataset):
    """优化后的X射线图像数据集加载器"""

    def __init__(self, lr_dir, hr_dir, transform=None, cache_size=100):
        """
        初始化数据集
        Args:
            lr_dir (str): 低分辨率图像目录
            hr_dir (str): 高分辨率图像目录
            transform (callable, optional): 自定义的转换函数
            cache_size (int): 缓存大小，默认100
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png')]
        self.cache_size = min(cache_size, len(self.image_files))  # 限制缓存大小
        self.cache = {}

        # 设置转换函数
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float32)
            ])

        # 预加载数据
        self._preload_data()

    def _load_image(self, img_path):
        """加载单个图像"""
        try:
            with Image.open(img_path) as img:
                img = img.convert('L')
                if self.transform:
                    img = self.transform(img)
                return img
        except Exception as e:
            warnings.warn(f"加载图像 {img_path} 时出错: {str(e)}")
            return None

    def _load_image_pair(self, idx):
        """加载单对图像"""
        try:
            img_name = self.image_files[idx]
            lr_path = os.path.join(self.lr_dir, img_name)
            hr_path = os.path.join(self.hr_dir, img_name)

            lr_img = self._load_image(lr_path)
            hr_img = self._load_image(hr_path)

            if lr_img is None or hr_img is None:
                return idx, None

            return idx, (lr_img, hr_img)

        except Exception as e:
            warnings.warn(f"加载图像对 {idx} 时出错: {str(e)}")
            return idx, None

    def _preload_data(self):
        """优化的并行预加载逻辑"""
        preload_num = min(self.cache_size, len(self.image_files))
        print(f"预加载 {preload_num} 张图像...")

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for idx in range(preload_num):
                    futures.append(executor.submit(self._load_image_pair, idx))

                for future in tqdm(futures, desc="预加载图像中"):
                    result = future.result()
                    if result is not None:
                        idx, image_pair = result
                        if image_pair is not None:
                            self.cache[idx] = image_pair

            print(f"成功预加载 {len(self.cache)} 张图像")

        except Exception as e:
            print(f"预加载过程中出错: {str(e)}")
            self.cache.clear()
            torch.cuda.empty_cache()

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取单个数据对"""
        try:
            # 首先检查缓存
            if idx in self.cache:
                return self.cache[idx]

            # 如果不在缓存中，加载图像
            result = self._load_image_pair(idx)
            if result is None or result[1] is None:
                # 返回一个全零张量作为替代
                return torch.zeros(1, 256, 256), torch.zeros(1, 1024, 1024)

            _, image_pair = result

            # 仅在缓存未满时添加到缓存
            if len(self.cache) < self.cache_size:
                self.cache[idx] = image_pair

            return image_pair

        except Exception as e:
            print(f"获取图像 {idx} 时出错: {str(e)}")
            return torch.zeros(1, 256, 256), torch.zeros(1, 1024, 1024)

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        torch.cuda.empty_cache()


def create_data_loaders(config):
    try:
        # 检查数据目录是否存在
        for dir_path in [
            config['paths']['train_lr_dir'],
            config['paths']['train_hr_dir'],
            config['paths']['val_lr_dir'],
            config['paths']['val_hr_dir']
        ]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"数据目录不存在: {dir_path}")

        print(f"发现 {len(os.listdir(config['paths']['train_lr_dir']))} 张训练图像")
        print(f"发现 {len(os.listdir(config['paths']['val_lr_dir']))} 张验证图像")

        # 优化的数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

        # 创建训练数据集
        train_dataset = XRayDataset(
            lr_dir=config['paths']['train_lr_dir'],
            hr_dir=config['paths']['train_hr_dir'],
            transform=transform,
            cache_size=config['data']['cache_size']
        )

        # 创建验证数据集
        val_dataset = XRayDataset(
            lr_dir=config['paths']['val_lr_dir'],
            hr_dir=config['paths']['val_hr_dir'],
            transform=transform,
            cache_size=config['data']['cache_size'] // 2
        )

        # 优化后的数据加载器配置
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=config['data']['prefetch_factor'],
            persistent_workers=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['validation']['batch_size'],  # 使用验证集的batch size
            shuffle=False,
            num_workers=max(1, config['data']['num_workers'] // 2),
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=config['data']['prefetch_factor'],
            persistent_workers=True
        )

        return train_loader, val_loader

    except Exception as e:
        print(f"创建数据加载器时出错: {str(e)}")
        raise