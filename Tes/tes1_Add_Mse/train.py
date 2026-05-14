import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import logging
from tqdm import tqdm
import GPUtil
from torch.utils.tensorboard import SummaryWriter

# 从config文件导入所需函数
from Tes.tes1_Add_Mse.config import (
    setup_training_device,
    create_experiment_dirs,
    get_optimizer,
    get_scheduler
)

# 从model文件导入模型和损失函数
from Tes.tes1_Add_Mse.model import XRaySR, L1CharbonnierLoss, SSIM, CombinedLoss

# 从dataset文件导入数据加载器
from Tes.tes1_Add_Mse.dataset import create_data_loaders


class Trainer:
    def __init__(self, config):
        """初始化训练器"""
        self.logger = self._setup_logger()
        self.logger.info("Initializing Trainer...")

        self.config = self._optimize_config(config)
        self.device = setup_training_device()

        # 创建tensorboard writer
        self.writer = SummaryWriter(log_dir=config['logging']['log_dir'])

        # 创建目录
        create_experiment_dirs(config)

        # 初始化数据加载器和模型
        self._initialize_training_components()

        self.ssim_module = SSIM().to(self.device)

        # 添加训练步数计数器
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.training_start_time = None

        self.logger.info("Trainer initialized successfully")

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('TrainingLogger')
        logger.setLevel(logging.INFO)

        # 创建logs目录
        os.makedirs('../../logs', exist_ok=True)

        # 文件处理器
        file_handler = logging.FileHandler('../../logs/training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter('%(message)s')
        )

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _optimize_config(self, config):
        """优化配置参数"""
        # 减小batch size以加快训练
        config['train']['batch_size'] = 8
        # 减少预加载的图像数量
        config['data']['cache_size'] = 100
        # 增加打印频率以提高透明度
        config['logging']['print_freq'] = 10
        # 减少验证频率
        config['logging']['val_freq'] = 2
        return config

    def _initialize_training_components(self):
        """初始化训练组件"""
        try:
            # 创建数据加载器
            self.train_loader, self.val_loader = create_data_loaders(self.config)
            self.logger.info(f"Created data loaders - Train: {len(self.train_loader.dataset)} images, "
                             f"Val: {len(self.val_loader.dataset)} images")

            # 创建模型
            self._create_model()

            # 创建优化器和调度器
            self._create_optimizer_and_scheduler()

            # 初始化训练状态
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            self.training_start_time = None

            # 混合精度训练
            self.scaler = GradScaler()

        except Exception as e:
            self.logger.error(f"Error in initialization: {str(e)}")
            raise

    def _create_model(self):
        """创建和优化模型"""
        try:
            self.model = XRaySR(
                num_channels=self.config['model']['num_channels'],
                num_blocks=self.config['model']['num_blocks'],
                debug_mode=self.config['model']['debug_mode']
            ).to(self.device)

            # 初始化权重
            if self.config['model']['initialize_weights']:
                self.model.initialize_weights()

            # 创建损失函数
            self.criterion = CombinedLoss(
                l1_weight=self.config['loss']['weights']['l1'],
                ssim_weight=self.config['loss']['weights']['ssim'],
                mse_weight=self.config['loss']['weights']['mse'],
                epsilon=self.config['loss']['epsilon']
            ).to(self.device)

            # GPU并行化
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

            self.model.train()
            self.logger.info("Model created successfully")

            # 记录模型配置
            self.logger.info(f"Model config - Channels: {self.config['model']['num_channels']}, "
                             f"Blocks: {self.config['model']['num_blocks']}")
            self.logger.info(f"Loss weights - L1: {self.config['loss']['weights']['l1']}, "
                             f"SSIM: {self.config['loss']['weights']['ssim']}, "
                             f"MSE: {self.config['loss']['weights']['mse']}")

        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise

    def _create_optimizer_and_scheduler(self):
        """创建优化器和学习率调度器"""
        try:
            # 创建优化器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                betas=self.config['optimizer']['betas'],
                eps=self.config['optimizer']['eps'],
                weight_decay=self.config['train']['weight_decay'],
                amsgrad=self.config['optimizer']['amsgrad']
            )

            # 创建学习率调度器
            total_steps = len(self.train_loader) * self.config['train']['num_epochs']
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['train']['learning_rate'],
                total_steps=total_steps,
                pct_start=self.config['scheduler']['pct_start'],
                div_factor=self.config['scheduler']['div_factor'],
                final_div_factor=self.config['scheduler']['final_div_factor']
            )

            self.logger.info("Optimizer and scheduler created successfully")

        except Exception as e:
            self.logger.error(f"Error creating optimizer: {str(e)}")
            raise

    def train_step(self, lr_imgs, hr_imgs, batch_idx):
        """单步训练"""
        try:
            # 确保输入张量维度正确
            if len(lr_imgs.shape) != 4 or len(hr_imgs.shape) != 4:
                lr_imgs = lr_imgs.unsqueeze(1) if len(lr_imgs.shape) == 3 else lr_imgs
                hr_imgs = hr_imgs.unsqueeze(1) if len(hr_imgs.shape) == 3 else hr_imgs

            # 移动数据到设备并设置格式
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)
            lr_imgs = lr_imgs.to(memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(memory_format=torch.channels_last)

            # 记录张量形状用于调试
            self.logger.debug(f"LR shape: {lr_imgs.shape}, HR shape: {hr_imgs.shape}")

            # 使用混合精度训练
            with autocast(enabled=self.config['train']['amp_enabled']):
                sr_imgs = self.model(lr_imgs)
                losses = self.criterion(sr_imgs, hr_imgs)

                if not isinstance(losses, dict):
                    raise ValueError("Criterion should return a dictionary of losses")

            # 反向传播和优化
            self.scaler.scale(losses['total']).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['train']['grad_clip_norm']
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # 计算PSNR (使用MSE损失来计算)
            with torch.no_grad():
                mse = losses['mse'].clamp(min=1e-10)  # 防止除零错误
                psnr = 10 * torch.log10(1.0 / mse)
                ssim = 1 - losses['ssim']

            # 返回之前确保所有值都是标量
            return (
                losses['total'].item(),
                psnr.item(),
                ssim.item()
            )

        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}\n"
                              f"LR tensor shape: {lr_imgs.shape if lr_imgs is not None else 'None'}\n"
                              f"HR tensor shape: {hr_imgs.shape if hr_imgs is not None else 'None'}")
            raise

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = {'loss': 0, 'psnr': 0, 'ssim': 0}
        batch_times = []
        data_loading_times = []
        processed_batches = 0  # 添加计数器

        try:
            pbar = tqdm(enumerate(self.train_loader),
                        total=len(self.train_loader),
                        desc=f'Epoch [{epoch}/{self.config["train"]["num_epochs"]}]',
                        dynamic_ncols=True)

            batch_start = time.time()
            for batch_idx, (lr_imgs, hr_imgs) in pbar:
                try:
                    # 记录数据加载时间
                    data_loading_time = time.time() - batch_start
                    data_loading_times.append(data_loading_time)

                    # 训练步骤
                    step_start = time.time()
                    loss, psnr, ssim = self.train_step(lr_imgs, hr_imgs, batch_idx)
                    processed_batches += 1

                    # 更新指标
                    epoch_metrics['loss'] += loss
                    epoch_metrics['psnr'] += psnr
                    epoch_metrics['ssim'] += ssim

                    # 记录batch时间
                    batch_time = time.time() - step_start
                    batch_times.append(batch_time)

                    # 更新进度条
                    if batch_idx % self.config['logging']['print_freq'] == 0:
                        avg_batch_time = sum(batch_times[-50:]) / max(len(batch_times[-50:]), 1)
                        avg_data_time = sum(data_loading_times[-50:]) / max(len(data_loading_times[-50:]), 1)

                        pbar.set_postfix({
                            'loss': f'{loss:.4f}',
                            'PSNR': f'{psnr:.2f}',
                            'SSIM': f'{ssim:.4f}',
                            'batch_t': f'{avg_batch_time:.3f}s',
                            'data_t': f'{avg_data_time:.3f}s',
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                        })

                    # 更新学习率
                    self.scheduler.step()

                    # 准备下一个batch的计时
                    batch_start = time.time()

                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue  # 跳过出错的batch，继续处理下一个

        except Exception as e:
            self.logger.error(f"Error in epoch {epoch}: {str(e)}")
            raise

        finally:
            # 安全计算epoch平均指标
            if processed_batches > 0:
                for key in epoch_metrics:
                    epoch_metrics[key] /= processed_batches

                # 输出epoch统计信息
                avg_batch_time = sum(batch_times) / max(len(batch_times), 1)
                self.logger.info(
                    f"Epoch {epoch} - "
                    f"Loss: {epoch_metrics['loss']:.4f}, "
                    f"PSNR: {epoch_metrics['psnr']:.2f}, "
                    f"SSIM: {epoch_metrics['ssim']:.4f}, "
                    f"Avg batch time: {avg_batch_time:.3f}s"
                )
            else:
                self.logger.warning(f"No batches were successfully processed in epoch {epoch}")

        return epoch_metrics

    def validate(self, epoch):
        """验证模型性能的函数

        Args:
            epoch (int): 当前训练的轮次

        Returns:
            dict: 包含验证指标的字典(loss, psnr, ssim)
        """
        # 将模型设置为评估模式
        self.model.eval()

        # 初始化验证指标字典
        val_metrics = {'loss': 0, 'psnr': 0, 'ssim': 0}
        num_batches = len(self.val_loader)

        # 创建进度条
        pbar = tqdm(self.val_loader,
                    desc=f'Validating Epoch {epoch}',
                    dynamic_ncols=True)

        try:
            with torch.no_grad():  # 在验证时不需要计算梯度
                for lr_imgs, hr_imgs in pbar:
                    # 将图像数据移动到指定设备(GPU/CPU)
                    lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                    hr_imgs = hr_imgs.to(self.device, non_blocking=True)

                    # 使用混合精度训练以提高效率
                    with autocast(enabled=self.config['train']['amp_enabled']):
                        # 通过模型获得超分辨率图像
                        sr_imgs = self.model(lr_imgs)
                        # 计算损失值（现在返回的是一个字典）
                        losses = self.criterion(sr_imgs, hr_imgs)

                    # 累加各项指标
                    # 注意：这里使用losses['total']而不是直接使用loss
                    val_metrics['loss'] += losses['total'].item()

                    # 计算PSNR (Peak Signal-to-Noise Ratio)
                    mse = losses['mse']  # 直接使用字典中的MSE值
                    psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
                    val_metrics['psnr'] += psnr.item()

                    # 计算SSIM (Structural Similarity Index)
                    val_metrics['ssim'] += (1 - losses['ssim']).item()

                    # 更新进度条显示的信息
                    pbar.set_postfix({
                        'Loss': f"{val_metrics['loss'] / num_batches:.4f}",
                        'PSNR': f"{val_metrics['psnr'] / num_batches:.2f}dB",
                        'SSIM': f"{val_metrics['ssim'] / num_batches:.4f}"
                    })

        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise

        # 计算所有批次的平均值
        for key in val_metrics:
            val_metrics[key] /= num_batches

        # 将验证结果记录到tensorboard
        self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
        self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)

        # 输出验证结果到日志
        self.logger.info(
            f"Validation - "
            f"Loss: {val_metrics['loss']:.4f}, "
            f"PSNR: {val_metrics['psnr']:.2f}dB, "
            f"SSIM: {val_metrics['ssim']:.4f}"
        )

        return val_metrics

    def save_checkpoint(self, epoch, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }

        save_path = os.path.join(self.config['logging']['checkpoint_dir'], filename)
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")


    def resume_from_checkpoint(self, checkpoint_path):
        """从检查点恢复训练"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch = checkpoint['best_epoch']

            self.logger.info(f"Resumed training from epoch {self.current_epoch}")

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def train(self):
        """完整训练流程"""
        self.logger.info("Starting training...")
        self.training_start_time = time.time()

        # 如果需要从检查点恢复
        if self.config['resume']['enabled'] and self.config['resume']['checkpoint_path']:
            self.resume_from_checkpoint(self.config['resume']['checkpoint_path'])

        try:
            for epoch in range(self.current_epoch + 1, self.config['train']['num_epochs'] + 1):
                epoch_start_time = time.time()

                # 训练epoch
                train_metrics = self.train_epoch(epoch)

                # 定期验证
                if epoch % self.config['logging']['val_freq'] == 0:
                    val_metrics = self.validate(epoch)

                    # 保存最佳模型
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.best_epoch = epoch
                        self.save_checkpoint(epoch, 'best_model.pth')

                # 记录epoch时间
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

                # 保存定期检查点
                if epoch % self.config['logging']['save_freq'] == 0:
                    self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save_checkpoint(epoch, 'interrupted.pth')

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

        finally:
            total_time = time.time() - self.training_start_time
            self.logger.info(
                f"Training completed in {total_time / 3600:.2f} hours\n"
                f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}"
            )
            self.writer.close()

def main():
    try:
        # 加载配置
        from config import get_training_config, setup_training_device
        config = get_training_config()

        # 创建训练器并开始训练
        trainer = Trainer(config)
        trainer.train()

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()
