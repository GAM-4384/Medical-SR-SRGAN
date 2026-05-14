import os
import torch


def get_training_config():
    """获取优化后的训练配置"""
    config = {
        # 模型参数优化 - 降低模型复杂度
        'model': {
            'num_channels': 16,  # 从32减到16降低复杂度
            'num_blocks': 3,     # 从4减到3降低复杂度
            'debug_mode': False,
            'initialize_weights': True
        },

        # 训练参数优化
        'train': {
            'batch_size': 16,    # 设置合适的batch size
            'num_epochs': 50,
            'learning_rate': 5e-4,  # 略微提高学习率加快收敛
            'min_lr': 1e-6,
            'weight_decay': 1e-4,
            'warmup_epochs': 2,
            'grad_accum_steps': 2,  # 增加梯度累积减少显存占用
            'grad_clip_norm': 0.5,
            'amp_enabled': True,   # 启用混合精度训练
            'compile_enabled': False
        },

        'loss': {
            'type': 'CombinedLoss',
            'weights': {
                'l1': 1.0,
                'ssim': 0.1,
                'mse': 0.1
            },
            'epsilon': 1e-3
        },

        # 数据加载优化
        'data': {
            'num_workers': 4,     # 根据CPU核心数设置
            'prefetch_factor': 4, # 增加预取因子提高效率
            'pin_memory': True,
            'persistent_workers': True,
            'cache_size': 50,    # 减小缓存降低内存占用
            'drop_last': True,
            'shuffle': True,
            'img_size': 256
        },

        # 优化器设置保持不变
        'optimizer': {
            'type': 'AdamW',
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': True
        },

        # 学习率调度器设置
        'scheduler': {
            'type': 'OneCycleLR',
            'pct_start': 0.3,
            'div_factor': 25.0,
            'final_div_factor': 1e4
        },

        # 损失函数权重保持不变
        'loss_weights': {
            'l1': 1.0,
            'ssim': 0.1,
            'mse': 0.1
        },

        # 日志设置优化 - 减少验证频率
        'logging': {
            'print_freq': 20,    # 增加打印间隔
            'save_freq': 5,
            'val_freq': 5,       # 减少验证频率
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'tensorboard_dir': 'runs',
            'enable_profiling': False
        },

        # CUDA优化设置
        'cuda': {
            'benchmark': True,
            'deterministic': False,
            'allow_tf32': True,
            'max_split_size_mb': 16  # 减小显存分配块大小
        },

        # 路径设置保持不变
        'paths': {
            'train_lr_dir': r'E:\My struggle\PY SRRESNET\processed\train\lr',
            'train_hr_dir': r'E:\My struggle\PY SRRESNET\processed\train\hr',
            'val_lr_dir': r'E:\My struggle\PY SRRESNET\processed\val\lr',
            'val_hr_dir': r'E:\My struggle\PY SRRESNET\processed\val\hr',
            'test_lr_dir': r'E:\My struggle\PY SRRESNET\processed\test\lr',
            'test_hr_dir': r'E:\My struggle\PY SRRESNET\processed\test\hr'
        },

        # 恢复训练设置保持不变
        'resume': {
            'enabled': False,
            'checkpoint_path': r'E:\My struggle\PY SRRESNET\checkpoints\interrupted.pth'
        },

        # 数据增强设置保持不变
        'augmentation': {
            'enabled': True,
            'flip': True,
            'rotate': True,
            'brightness': 0.1,
            'contrast': 0.1,
        },

        # 验证设置优化
        'validation': {
            'batch_size': 16,    # 增大验证batch size
            'metrics': ['psnr', 'ssim'],
            'save_images': True,
            'max_save_images': 10
        }
    }

    return config


def setup_training_device():
    """优化训练环境设置"""
    if torch.cuda.is_available():
        # 优化CUDA设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 清理GPU缓存
        torch.cuda.empty_cache()

        # 优化GPU内存分配
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            'max_split_size_mb=16,'  # 减小分配块大小
            'garbage_collection_threshold=0.8,'
            'roundup_power2_divisions=16'
        )

        # 优化CPU线程数
        torch.set_num_threads(2)  # 减少CPU线程数

        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

# 其他函数保持不变
def create_experiment_dirs(config):
    """创建实验所需的目录"""
    dirs = [
        config['logging']['checkpoint_dir'],
        config['logging']['log_dir'],
        config['logging']['tensorboard_dir']
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    if config['validation']['save_images']:
        os.makedirs(os.path.join(config['logging']['log_dir'], 'val_images'), exist_ok=True)


def get_optimizer(model_parameters, config):
    """创建优化器"""
    if config['optimizer']['type'] == 'AdamW':
        return torch.optim.AdamW(
            model_parameters,
            lr=config['train']['learning_rate'],
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
            weight_decay=config['train']['weight_decay'],
            amsgrad=config['optimizer']['amsgrad']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']['type']}")


def get_scheduler(optimizer, config, steps_per_epoch):
    """创建学习率调度器"""
    if config['scheduler']['type'] == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['train']['learning_rate'],
            epochs=config['train']['num_epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=config['scheduler']['pct_start'],
            div_factor=config['scheduler']['div_factor'],
            final_div_factor=config['scheduler']['final_div_factor']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {config['scheduler']['type']}")