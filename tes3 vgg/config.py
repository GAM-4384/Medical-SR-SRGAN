import os
import torch

def get_training_config():
    """获取训练配置"""
    config = {
        'paths': {
            'train_lr_dir': 'processed/train/lr',
            'train_hr_dir': 'processed/train/hr',
            'val_lr_dir': 'processed/val/lr',
            'val_hr_dir': 'processed/val/hr',
            'test_lr_dir': 'processed/test/lr',
            'test_hr_dir': 'processed/test/hr'
        },
        'data': {
            # 图像和数据处理配置
            'img_size': 16,  # 进一步减小图像大小
            'scale_factor': 4,
            'lr_size': 4,
            'channels': 1,
            'normalize': True,
            'mean': [0.5],
            'std': [0.5],
            'cache_size': 20,  # 进一步减小缓存大小

            # 数据加载配置
            'num_workers': 0,
            'pin_memory': False,  # 禁用锁页内存
            'prefetch_factor': 2,
            'persistent_workers': False
        },
        'augmentation': {  # 添加数据增强配置
            'enabled': False,  # 暂时禁用数据增强以节省内存
            'flip': True,
            'rotate': True,
            'brightness': 0.1,
            'contrast': 0.1
        },
        'train': {
            'batch_size': 1,  # 保持最小批次
            'accumulation_steps': 32,  # 进一步增加梯度累积步数
            'num_epochs': 100,
            'amp_enabled': True,
            'grad_clip_norm': 0.5,
            'compile_enabled': False
        },
        'model': {
            'num_channels': 4,   # 进一步减小通道数
            'num_blocks': 2,     # 保持最小块数
            'debug_mode': False
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-5,  # 降低初始学习率
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        },
        'scheduler': {
            'type': 'cosine_plateau',
            'min_lr': 1e-6,
            'warmup_epochs': 2,
        },
        'loss_weights': {
            'lambda_char': 1.0,
            'lambda_ssim': 0.5,
            'lambda_perceptual': 0.0  # 禁用感知损失以节省内存
        },
        'validation': {
            'val_freq': 1,
            'save_images': False,  # 暂时禁用图像保存
            'max_save_images': 2
        },
        'logging': {
            'log_dir': 'runs',
            'tensorboard_dir': 'runs/tensorboard',
            'checkpoint_dir': 'runs/checkpoints',
            'save_freq': 10,
            'print_freq': 100
        },
        'resume': {
            'enabled': False,
            'checkpoint_path': None
        },
        'performance': {
            'cudnn_benchmark': True,
            'cudnn_deterministic': False,
            'gpu_mem_fraction': 0.5,  # 进一步降低GPU内存使用限制
            'enable_tf32': True,
            'enable_cudnn_auto_tuner': True,
            'gradient_checkpointing': True,
            'empty_cache_freq': 1,    # 每个批次都清理缓存
        }
    }

    # 创建必要的目录
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)

    return config

def get_optimizer(parameters, config):
    """获取优化器"""
    try:
        optimizer_type = config['optimizer']['type'].lower()
        if optimizer_type == 'adam':
            return torch.optim.Adam(
                parameters,
                lr=config['optimizer']['lr'],
                betas=config['optimizer'].get('betas', (0.9, 0.999)),
                eps=config['optimizer'].get('eps', 1e-8),
                weight_decay=config['optimizer'].get('weight_decay', 0)
            )
        elif optimizer_type == 'adamw':
            return torch.optim.AdamW(
                parameters,
                lr=config['optimizer']['lr'],
                betas=config['optimizer'].get('betas', (0.9, 0.999)),
                eps=config['optimizer'].get('eps', 1e-8),
                weight_decay=config['optimizer'].get('weight_decay', 0.01)
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                parameters,
                lr=config['optimizer']['lr'],
                momentum=config['optimizer'].get('momentum', 0.9),
                weight_decay=config['optimizer'].get('weight_decay', 0),
                nesterov=config['optimizer'].get('nesterov', False)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    except Exception as e:
        raise Exception(f"创建优化器失败: {str(e)}")


def get_scheduler(optimizer, config, steps_per_epoch):
    """获取学习率调度器"""
    try:
        scheduler_type = config['scheduler']['type'].lower()
        total_steps = steps_per_epoch * config['train']['num_epochs']
        warmup_steps = steps_per_epoch * config['scheduler']['warmup_epochs']

        if scheduler_type == 'cosine_plateau':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config['scheduler']['min_lr']
            )

        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30 * steps_per_epoch,
                gamma=0.1
            )

        elif scheduler_type == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config['optimizer']['lr'],
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )

        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    except Exception as e:
        raise Exception(f"创建调度器时出错: {str(e)}")


def setup_training_device():
    """设置训练设备和环境"""
    if torch.cuda.is_available():
        # 基本CUDA设置
        torch.cuda.empty_cache()

        # 选择第一个可用的GPU
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

        # 读取配置
        config = get_training_config()
        perf_config = config['performance']

        # 设置GPU内存使用限制（如果可用）
        try:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(
                    perf_config['gpu_mem_fraction']
                )
        except Exception as e:
            print(f"警告: 无法设置GPU内存限制: {str(e)}")

        # 优化CUDA设置
        torch.backends.cudnn.benchmark = perf_config['cudnn_benchmark']
        torch.backends.cudnn.deterministic = perf_config['cudnn_deterministic']

        # 设置TF32（如果可用）
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = perf_config['enable_tf32']
            torch.backends.cudnn.allow_tf32 = perf_config['enable_tf32']

        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

        # 打印配置
        print("\n性能优化配置:")
        print(f"cuDNN基准测试: {perf_config['cudnn_benchmark']}")
        print(f"TF32启用状态: {perf_config.get('enable_tf32', False)}")
        print(f"GPU内存使用比例: {perf_config['gpu_mem_fraction'] * 100}%")

    else:
        device = torch.device('cpu')
        print("警告: 未检测到GPU，将使用CPU训练")

    return device


def create_experiment_dirs(config):
    """创建实验所需的目录"""
    try:
        # 创建主要目录
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
        os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)

        # 创建验证图像目录
        if config['validation']['save_images']:
            os.makedirs(os.path.join(config['logging']['log_dir'], 'val_images'),
                        exist_ok=True)

    except Exception as e:
        raise Exception(f"创建目录失败: {str(e)}")


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss,
                    best_epoch, config, filename):
    """保存检查点"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'config': config
        }

        save_path = os.path.join(config['logging']['checkpoint_dir'], filename)
        torch.save(checkpoint, save_path)
        print(f"保存检查点到 {save_path}")

    except Exception as e:
        raise Exception(f"保存检查点失败: {str(e)}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    """加载检查点"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"成功加载检查点: {checkpoint_path}")
        return (checkpoint['epoch'], checkpoint['best_val_loss'],
                checkpoint['best_epoch'], checkpoint['config'])

    except Exception as e:
        raise Exception(f"加载检查点失败: {str(e)}")