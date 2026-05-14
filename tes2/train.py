import os
import random
import time
import warnings
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import logging
from tqdm import tqdm
import psutil
import GPUtil
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

from tes.tes2.model import CombinedLoss, create_model, load_model
from tes.tes2.dataset import create_data_loaders, DataPrefetcher
from tes.tes2.config import get_training_config, setup_training_device, create_experiment_dirs

def setup_environment():
    """设置训练环境"""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置警告过滤
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 设置线程数
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(4)

class Trainer:
    def __init__(self, config=None):
        """初始化训练器"""
        self.config = config or get_training_config()
        self.logger = self._setup_logger()
        self.device = setup_training_device()

        # 创建必要的目录
        create_experiment_dirs(self.config)
        self.writer = SummaryWriter(log_dir=self.config['paths']['log_dir'])

        # 初始化早停相关参数
        self.patience = self.config['validation']['early_stopping']['patience']
        self.min_delta = self.config['validation']['early_stopping']['min_delta']
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

        # 初始化组件和训练状态
        self._initialize_components()
        self._initialize_training_state()

        self.logger.info("Trainer initialized successfully")
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('TrainingLogger')
        logger.setLevel(logging.INFO)

        # 清除已存在的处理器
        logger.handlers.clear()

        # 创建日志目录
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        # 文件处理器
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def _initialize_components(self):
        """初始化训练组件"""
        try:
            # 创建模型
            self.model = create_model(self.config).to(self.device)

            # 创建损失函数
            self.criterion = CombinedLoss(
                ssim_weight=self.config['loss']['ssim_weight'],
                l1_weight=self.config['loss']['l1_weight'],
                edge_weight=self.config['loss']['edge_weight']
            ).to(self.device)

            # 创建优化器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                betas=self.config['optimizer']['betas'],
                eps=self.config['optimizer']['eps'],
                weight_decay=self.config['train']['weight_decay'],
                amsgrad=self.config['optimizer']['amsgrad']
            )

            # 创建数据加载器
            self.train_loader, self.val_loader = create_data_loaders(self.config)

            # 创建学习率调度器
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['train']['learning_rate'],
                epochs=self.config['train']['num_epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=self.config['scheduler']['pct_start'],
                div_factor=self.config['scheduler']['div_factor'],
                final_div_factor=self.config['scheduler']['final_div_factor']
            )

            # 创建混合精度训练的scaler
            self.scaler = GradScaler(enabled=self.config['train']['amp_enabled'])

            self.logger.info("Successfully initialized all components")

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

    def _initialize_training_state(self):
        """初始化训练状态"""
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = -float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.training_start_time = None

    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = {
            'loss': 0,
            'psnr': 0,
            'ssim': 0
        }

        # 创建数据预取器
        prefetcher = DataPrefetcher(self.train_loader, self.device)
        batch_idx = 0
        lr_imgs, hr_imgs = prefetcher.next()

        pbar = tqdm(total=len(self.train_loader), desc=f'Epoch [{epoch}/{self.config["train"]["num_epochs"]}]')

        while lr_imgs is not None:
            try:
                batch_start = time.time()

                # 确保输入数据是float32类型
                lr_imgs = lr_imgs.float()
                hr_imgs = hr_imgs.float()

                # 前向传播
                with autocast(enabled=self.config['train']['amp_enabled']):
                    sr_imgs = self.model(lr_imgs)
                    loss = self.criterion(sr_imgs, hr_imgs)

                # 反向传播
                self.optimizer.zero_grad(set_to_none=True)

                if self.config['train']['amp_enabled']:
                    # 使用GradScaler进行反向传播和优化
                    self.scaler.scale(loss).backward()

                    # Unscale 优化器
                    self.scaler.unscale_(self.optimizer)

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['train']['grad_clip_norm']
                    )

                    # 更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 普通的反向传播
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['train']['grad_clip_norm']
                    )
                    # 更新参数
                    self.optimizer.step()

                # 调度器步进
                self.scheduler.step()

                # 计算指标
                with torch.no_grad():
                    mse = F.mse_loss(sr_imgs.float(), hr_imgs.float())
                    psnr = -10 * torch.log10(mse + 1e-8)
                    ssim = 1 - self.criterion.ssim_module(sr_imgs.float(), hr_imgs.float())

                # 更新平均指标
                epoch_metrics['loss'] += loss.item()
                epoch_metrics['psnr'] += psnr.item()
                epoch_metrics['ssim'] += ssim.item()

                # 更新进度条
                batch_time = time.time() - batch_start
                lr = self.optimizer.param_groups[0]['lr']

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'PSNR': f"{psnr.item():.2f}",
                    'SSIM': f"{ssim.item():.4f}",
                    'lr': f"{lr:.2e}",
                    'time': f"{batch_time:.3f}s"
                })
                pbar.update(1)

                # 记录到tensorboard
                step = epoch * len(self.train_loader) + batch_idx
                self._log_training_step(step, loss.item(), psnr.item(), ssim.item(), lr, batch_time)

                # 获取下一批数据
                lr_imgs, hr_imgs = prefetcher.next()
                batch_idx += 1

            except Exception as e:
                self.logger.error(f"Error in training batch: {str(e)}")
                continue

        pbar.close()

        # 计算epoch平均指标
        for key in epoch_metrics:
            epoch_metrics[key] /= len(self.train_loader)

        return epoch_metrics

    def _validate(self, epoch):
        """验证模型"""
        self.model.eval()
        val_metrics = {
            'loss': 0,
            'psnr': 0,
            'ssim': 0
        }

        pbar = tqdm(self.val_loader, desc='Validating')

        with torch.no_grad():
            for lr_imgs, hr_imgs in pbar:
                try:
                    # 移动数据到设备并确保类型正确
                    lr_imgs = lr_imgs.to(self.device).float()
                    hr_imgs = hr_imgs.to(self.device).float()

                    # 前向传播
                    with autocast(enabled=self.config['train']['amp_enabled']):
                        sr_imgs = self.model(lr_imgs)
                        loss = self.criterion(sr_imgs, hr_imgs)

                    # 计算指标
                    mse = F.mse_loss(sr_imgs.float(), hr_imgs)
                    psnr = -10 * torch.log10(mse + 1e-8)
                    ssim = 1 - self.criterion.ssim_module(sr_imgs.float(), hr_imgs)

                    # 更新指标
                    val_metrics['loss'] += loss.item()
                    val_metrics['psnr'] += psnr.item()
                    val_metrics['ssim'] += ssim.item()

                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'PSNR': f"{psnr.item():.2f}",
                        'SSIM': f"{ssim.item():.4f}"
                    })

                except Exception as e:
                    self.logger.error(f"Error in validation batch: {str(e)}")
                    continue

        pbar.close()

        # 计算平均指标
        for key in val_metrics:
            val_metrics[key] /= len(self.val_loader)

        # 记录到tensorboard
        self._log_validation_epoch(epoch, val_metrics)

        return val_metrics

    def _save_checkpoint(self, epoch, val_metrics, is_best=False):
        """保存检查点"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'val_metrics': val_metrics,
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch
            }

            # 保存最新检查点
            latest_path = Path(self.config['paths']['checkpoint_dir']) / 'latest.pth'
            torch.save(checkpoint, latest_path)

            # 如果是最佳模型，额外保存一份
            if is_best:
                best_path = Path(self.config['paths']['checkpoint_dir']) / 'best_model.pth'
                torch.save(checkpoint, best_path)
                self.logger.info(f"Saved best model checkpoint to {best_path}")

            # 定期保存
            if epoch % self.config['monitoring']['checkpointing']['save_freq'] == 0:
                epoch_path = Path(self.config['paths']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth'
                torch.save(checkpoint, epoch_path)

            self.logger.info(f"Saved checkpoint for epoch {epoch}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")

    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        try:
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型权重
            load_model(self.model, checkpoint_path, self.device)

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # 恢复训练状态
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch = checkpoint['best_epoch']

            self.logger.info(f"Successfully resumed from epoch {self.current_epoch}")

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def _log_training_step(self, step, loss, psnr, ssim, lr, batch_time):
        """记录训练步骤的指标"""
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.add_scalar('train/psnr', psnr, step)
        self.writer.add_scalar('train/ssim', ssim, step)
        self.writer.add_scalar('train/learning_rate', lr, step)
        self.writer.add_scalar('train/batch_time', batch_time, step)

        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            self.writer.add_scalar('system/gpu_util', gpu.load * 100, step)
            self.writer.add_scalar('system/gpu_memory', gpu.memoryUsed, step)

        self.writer.add_scalar('system/cpu_util', psutil.cpu_percent(), step)
        self.writer.add_scalar('system/memory_util', psutil.virtual_memory().percent, step)

    def _log_validation_epoch(self, epoch, metrics):
        """记录验证epoch的指标"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)



    def train(self):
        """完整的训练流程"""
        self.logger.info("Starting training...")
        self.training_start_time = time.time()

        try:
            # 尝试恢复训练
            if self.config['resume']['enabled']:
                checkpoint_path = self.config['resume']['checkpoint_path']
                if os.path.exists(checkpoint_path):
                    try:
                        self._load_checkpoint(checkpoint_path)
                        self.logger.info(f"Resumed training from epoch {self.current_epoch}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load checkpoint: {str(e)}")
                        self.logger.info("Starting training from scratch")
                        try:
                            os.remove(checkpoint_path)
                            self.logger.info(f"Removed corrupted checkpoint file: {checkpoint_path}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove corrupted checkpoint: {str(e)}")
                else:
                    self.logger.info("No checkpoint found, starting from scratch")

            # 使用torch.compile优化模型(如果启用)
            if self.config['train']['compile_enabled'] and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                self.logger.info("Model compilation enabled")

            # 主训练循环
            while self.current_epoch < self.config['train']['num_epochs']:
                epoch_start_time = time.time()
                self.current_epoch += 1

                # 训练一个epoch
                self.logger.info(f"\nEpoch {self.current_epoch}/{self.config['train']['num_epochs']}")
                train_metrics = self._train_epoch(self.current_epoch)

                # 定期验证
                if self.current_epoch % self.config['validation']['eval_frequency'] == 0:
                    val_metrics = self._validate(self.current_epoch)

                    # 检查是否需要早停
                    should_stop, is_best = self._check_early_stopping(val_metrics['loss'])

                    # 保存检查点
                    self._save_checkpoint(
                        self.current_epoch,
                        val_metrics,
                        is_best=is_best
                    )

                    # 记录本轮训练信息
                    epoch_time = time.time() - epoch_start_time
                    self.logger.info(
                        f"Epoch {self.current_epoch} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train PSNR: {train_metrics['psnr']:.2f}, "
                        f"Train SSIM: {train_metrics['ssim']:.4f}, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val PSNR: {val_metrics['psnr']:.2f}, "
                        f"Val SSIM: {val_metrics['ssim']:.4f}, "
                        f"Time: {epoch_time:.2f}s"
                    )

                    if should_stop:
                        self.logger.info(
                            f"Early stopping triggered after {self.epochs_without_improvement} "
                            f"epochs without improvement"
                        )
                        break

                # 清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 训练完成
            total_time = time.time() - self.training_start_time
            self.logger.info(
                f"\nTraining completed in {total_time / 3600:.2f}h. "
                f"Best validation loss: {self.best_val_loss:.4f} "
                f"at epoch {self.best_epoch}"
            )

            # 加载最佳模型
            best_model_path = os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth')
            if os.path.exists(best_model_path):
                try:
                    self._load_checkpoint(best_model_path)
                    self.logger.info("Loaded best model for final evaluation")
                except Exception as e:
                    self.logger.error(f"Failed to load best model: {str(e)}")

            # 关闭TensorBoard写入器
            self.writer.close()

            return {
                'best_epoch': self.best_epoch,
                'best_val_loss': self.best_val_loss,
                'total_time': total_time
            }

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.logger.info("Saving checkpoint...")
            try:
                self._save_checkpoint(
                    self.current_epoch,
                    {'loss': float('inf')},
                    is_best=False
                )
                self.logger.info("Checkpoint saved successfully")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _check_early_stopping(self, current_loss):
        """
        检查是否应该触发早停机制

        Args:
            current_loss (float): 当前的验证损失值

        Returns:
            tuple: (should_stop, is_best)
                should_stop: 布尔值，指示是否应该停止训练
                is_best: 布尔值，指示当前模型是否是最佳模型
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            return False, True

        if current_loss < self.best_loss - self.min_delta:
            # 如果当前损失值比最佳损失值改善了超过 min_delta
            self.best_loss = current_loss
            self.counter = 0
            return False, True

        # 如果没有足够的改善
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return True, False

        return False, False

if __name__ == '__main__':
    # 设置环境
    setup_environment()

    try:
        # 获取配置
        config = get_training_config()

        # 创建训练器实例
        trainer = Trainer(config)

        # 开始训练
        training_results = trainer.train()

        print("\nTraining completed successfully!")
        print(f"Best epoch: {training_results['best_epoch']}")
        print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        print(f"Total training time: {training_results['total_time'] / 3600:.2f} hours")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()