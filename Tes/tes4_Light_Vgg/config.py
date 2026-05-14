
import os
import torch


def get_training_config():
    """获取平衡版本的训练配置"""
    config = {
        # 模型参数 - 轻量级配置
        'model': {
            'num_channels': 32,     # 增加到32通道
            'num_blocks': 6,        # 增加到6个块
            'debug_mode': False
        },

        # 训练参数 - 平衡配置
        'train': {
            'batch_size': 4,
            'num_epochs': 100,
            'learning_rate': 2e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'warmup_epochs': 5,
            'grad_clip_norm': 1.0,
            'amp_enabled': True,  # 暂时禁用AMP
            'compile_enabled': False,
            'gradient_accumulation_steps': 4  # 简化训练流程
        },

        'optimizer': {
            'type': 'AdamW',
            'betas': (0.9, 0.999),  # 调整beta2
            'eps': 1e-8,
            'amsgrad': True,
            'weight_decay': 1e-4
        },

        'scheduler': {
            'type': 'OneCycleLR',
            'pct_start': 0.1,
            'div_factor': 25.0,
            'final_div_factor': 1e4
        },

        # 数据加载配置 - 优化内存使用
        'data': {
            'drop_last': True,
            'shuffle': True,
            'img_size': 256,
            'batch_size': 4,
            'num_workers': 2,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'cache_size': 200,
        },


        # 损失函数权重
        'loss_weights': {
            'l1': 1.0,
            'ssim': 0.1
        },

        # 日志设置
        'logging': {
            'print_freq': 10,
            'save_freq': 5,
            'val_freq': 1,  # 频繁验证以监控训练
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_dir': 'runs'
        },

        # CUDA配置 - 保守设置
        'cuda': {
            'benchmark': True,
            'deterministic': False,
            'allow_tf32': True,
            'max_split_size_mb': 512  # 限制CUDA内存分配
        },

        # 路径设置
        'paths': {
            'train_lr_dir': r'E:\My struggle\PY SRRESNET\processed\train\lr',
            'train_hr_dir': r'E:\My struggle\PY SRRESNET\processed\train\hr',
            'val_lr_dir': r'E:\My struggle\PY SRRESNET\processed\val\lr',
            'val_hr_dir': r'E:\My struggle\PY SRRESNET\processed\val\hr',
            'test_lr_dir': r'E:\My struggle\PY SRRESNET\processed\test\lr',
            'test_hr_dir': r'E:\My struggle\PY SRRESNET\processed\test\hr'
        },
        # 恢复训练设置
        'resume': {
            'enabled': False,
            'checkpoint_path': None

        },

        # 数据增强设置 - 保持基本增强
        'augmentation': {
            'enabled': True,
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation': False,  # 禁用复杂的增强
        },

        # 验证设置
        'validation': {
            'batch_size': 4,  # 减小验证batch size
            'max_images': 100,  # 限制验证图像数量
            'metrics': ['psnr', 'ssim']
        }
    }

    return config


def setup_device():
    """设置训练设备和优化"""
    if torch.cuda.is_available():
        # 基本CUDA优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # 使用 TF32（如果可用）
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 清理 GPU 缓存
        torch.cuda.empty_cache()

        # 设置 GPU 内存分配策略
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制GPU内存使用

        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        # CPU优化
        torch.set_num_threads(4)  # 限制CPU线程数

    return device


def create_dirs(config):
    """创建必要的目录"""
    dirs = [
        config['logging']['checkpoint_dir'],
        config['logging']['log_dir'],
        config['logging']['tensorboard_dir']
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def get_optimizer(model_parameters, config):
    """创建优化器"""
    return torch.optim.AdamW(
        model_parameters,
        lr=config['train']['learning_rate'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps'],
        weight_decay=config['train']['weight_decay'],
        amsgrad=config['optimizer']['amsgrad']
    )


def get_scheduler(optimizer, config, steps_per_epoch):
    """创建学习率调度器"""
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['train']['learning_rate'],
        epochs=config['train']['num_epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=config['scheduler']['pct_start'],
        div_factor=config['scheduler']['div_factor'],
        final_div_factor=config['scheduler']['final_div_factor'],
        last_epoch=-1  # 显式设置初始状态
    )

    # 确保初始学习率正确设置
    optimizer.zero_grad()
    optimizer.step()

    return scheduler