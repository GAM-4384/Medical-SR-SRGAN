import os
import torch


def get_training_config():
    """获取训练配置"""
    config = {
        # 模型配置
        'model': {
            'num_channels': 16,  # 与预训练模型匹配
            'num_blocks': 8,  # 残差块数量
            'kernel_size': 7,  # 卷积核大小
            'debug_mode': False  # 调试模式
        },

        # 训练参数
        'train': {
            'batch_size': 8,  # 批量大小
            'num_epochs': 100,  # 训练轮数
            'learning_rate': 1e-4,  # 初始学习率
            'min_lr': 1e-6,  # 最小学习率
            'warmup_epochs': 3,  # 预热轮数
            'weight_decay': 1e-4,  # 权重衰减
            'grad_clip_norm': 0.5,  # 梯度裁剪阈值
            'amp_enabled': True,  # 是否启用混合精度训练
            'compile_enabled': True,  # 是否启用torch.compile
        },

        # 数据加载配置
        'data': {
            'num_workers': 4,  # 数据加载线程数
            'prefetch_factor': 2,  # 预取因子
            'pin_memory': True,  # 是否启用内存锁页
            'persistent_workers': True,  # 持久化worker
            'cache_size': 100,  # 数据缓存大小
            'drop_last': True,  # 是否丢弃不完整的batch
            'shuffle': True,  # 是否打乱数据
            'train_val_split': 0.9,  # 训练集验证集分割比例
            'augmentation': {  # 数据增强设置
                'enabled': True,
                'flip_probability': 0.5,
                'rotate_probability': 0.3,
                'brightness_range': 0.1,
                'contrast_range': 0.1
            }
        },

        # 优化器配置
        'optimizer': {
            'type': 'AdamW',  # 优化器类型
            'betas': (0.9, 0.999),  # Adam动量参数
            'eps': 1e-8,  # 数值稳定性参数
            'amsgrad': True  # 是否使用AMSGrad变体
        },

        # 学习率调度器
        'scheduler': {
            'type': 'OneCycleLR',  # 调度器类型
            'pct_start': 0.3,  # 预热阶段比例
            'div_factor': 25.0,  # 初始学习率除数
            'final_div_factor': 1e4,  # 最终学习率除数
            'three_phase': True,  # 是否使用三阶段调度
            'cycle_momentum': True  # 是否调度动量
        },

        # 损失函数配置
        'loss': {
            'ssim_weight': 0.7,  # SSIM损失权重
            'l1_weight': 0.2,  # L1损失权重
            'edge_weight': 0.1,  # 边缘损失权重
            'charbonnier_eps': 1e-3  # Charbonnier损失参数
        },

        # 验证配置
        'validation': {
            'batch_size': 16,  # 验证批量大小
            'metrics': ['psnr', 'ssim', 'edge_similarity'],  # 验证指标
            'eval_frequency': 1,  # 每多少epoch验证一次
            'save_images': True,  # 是否保存验证图像
            'max_save_images': 10,  # 最大保存图像数
            'early_stopping': {  # 早停设置
                'enabled': True,
                'patience': 10,
                'min_delta': 1e-4
            }
        },

        # 监控配置
        'monitoring': {
            'tensorboard': {  # Tensorboard设置
                'enabled': True,
                'log_freq': 100,  # 记录频率
                'flush_secs': 30
            },
            'checkpointing': {  # 模型保存设置
                'save_freq': 5,  # 保存频率
                'max_to_keep': 3,  # 保留检查点数量
                'save_best_only': True  # 是否只保存最佳模型
            },
            'profiling': {  # 性能分析设置
                'enabled': False,
                'wait': 100,
                'warmup': 100,
                'active': 100,
                'repeat': 3
            }
        },

        # 恢复训练设置
        'resume': {
            'enabled': True,  # 是否恢复训练
            'checkpoint_path': 'checkpoints/latest.pth',  # 检查点路径
            'strict_loading': False  # 是否严格加载权重
        },

        # 路径配置
        'paths': {
            'train_lr_dir': r'E:\My struggle\PY SRRESNET\processed\train\lr',
            'train_hr_dir': r'E:\My struggle\PY SRRESNET\processed\train\hr',
            'val_lr_dir': r'E:\My struggle\PY SRRESNET\processed\val\lr',
            'val_hr_dir': r'E:\My struggle\PY SRRESNET\processed\val\hr',
            'test_lr_dir': r'E:\My struggle\PY SRRESNET\processed\test\lr',
            'test_hr_dir': r'E:\My struggle\PY SRRESNET\processed\test\hr',
            'checkpoint_dir': r'E:\My struggle\PY SRRESNET\checkpoints',
            'log_dir': r'E:\My struggle\PY SRRESNET\logs',
            'output_dir': r'E:\My struggle\PY SRRESNET\outputs'
        },

        # CUDA设置
        'cuda': {
            'benchmark': True,  # cuDNN基准测试
            'deterministic': False,  # 确定性计算
            'allow_tf32': True,  # 是否允许TF32
            'max_split_size_mb': 512  # 最大分配块大小
        }
    }

    return config


def setup_training_device():
    """设置训练设备并优化CUDA配置"""
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # 获取主配置
    config = get_training_config()

    try:
        # 设置CUDA设备
        device = torch.device('cuda')

        # 优化CUDA设置
        if config['cuda']['benchmark']:
            torch.backends.cudnn.benchmark = True

        if config['cuda']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = config['cuda']['allow_tf32']
            torch.backends.cudnn.allow_tf32 = config['cuda']['allow_tf32']

        # 设置GPU内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            f'max_split_size_mb={config["cuda"]["max_split_size_mb"]},'
            'garbage_collection_threshold=0.8,'
            'roundup_power2_divisions=8'
        )

        # 清理GPU缓存
        torch.cuda.empty_cache()

        # 设置CPU线程数
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(4)

        return device

    except Exception as e:
        print(f"Error setting up CUDA device: {str(e)}")
        print("Falling back to CPU")
        return torch.device('cpu')


def create_experiment_dirs(config):
    """创建实验所需的目录结构"""
    dirs = [
        config['paths']['checkpoint_dir'],
        config['paths']['log_dir'],
        config['paths']['output_dir']
    ]

    if config['validation']['save_images']:
        dirs.append(os.path.join(config['paths']['output_dir'], 'validation_images'))

    if config['monitoring']['tensorboard']['enabled']:
        dirs.append(os.path.join(config['paths']['log_dir'], 'tensorboard'))

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def get_optimizer(model_parameters, config):
    """创建优化器"""
    optimizer_config = config['optimizer']

    if optimizer_config['type'].lower() == 'adamw':
        return torch.optim.AdamW(
            model_parameters,
            lr=config['train']['learning_rate'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=config['train']['weight_decay'],
            amsgrad=optimizer_config['amsgrad']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")


def get_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler_config = config['scheduler']

    if scheduler_config['type'] == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['train']['learning_rate'],
            epochs=config['train']['num_epochs'],
            steps_per_epoch=1,  # 将在训练时更新
            pct_start=scheduler_config['pct_start'],
            div_factor=scheduler_config['div_factor'],
            final_div_factor=scheduler_config['final_div_factor'],
            three_phase=scheduler_config['three_phase'],
            cycle_momentum=scheduler_config['cycle_momentum']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")


def update_config(base_config, update_dict):
    """更新配置"""

    def _update_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_recursive(d[k], v)
            else:
                d[k] = v
        return d

    return _update_recursive(base_config.copy(), update_dict)


def validate_config(config):
    """验证配置的合法性"""
    try:
        # 验证必要参数
        assert 'model' in config, "Missing model configuration"
        assert 'train' in config, "Missing training configuration"
        assert 'paths' in config, "Missing paths configuration"

        # 验证数值参数的范围
        assert config['train']['batch_size'] > 0, "Batch size must be positive"
        assert config['train']['num_epochs'] > 0, "Number of epochs must be positive"
        assert config['train']['learning_rate'] > 0, "Learning rate must be positive"

        # 验证路径存在性
        for key, path in config['paths'].items():
            if 'dir' in key and key not in ['checkpoint_dir', 'log_dir', 'output_dir']:
                assert os.path.exists(path), f"Data path does not exist: {path}"

        return True
    except AssertionError as e:
        print(f"Configuration validation failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error in configuration validation: {str(e)}")
        return False


if __name__ == '__main__':
    # 测试配置
    config = get_training_config()
    if validate_config(config):
        print("Configuration validation passed")
        device = setup_training_device()
        print(f"Using device: {device}")
        create_experiment_dirs(config)