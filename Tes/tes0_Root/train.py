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
from Tes.tes0_Root.config import (
    setup_training_device,
    create_experiment_dirs,
    get_optimizer,
    get_scheduler
)

# 从model文件导入模型和损失函数
from Tes.tes0_Root.model import XRaySR, L1CharbonnierLoss, SSIM

# 从dataset文件导入数据加载器
from Tes.tes0_Root.dataset import create_data_loaders


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

    def train_step(self, lr_imgs, hr_imgs):
        """单步训练"""
        try:
            # 移动数据到设备并设置格式
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)
            lr_imgs = lr_imgs.to(memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(memory_format=torch.channels_last)

            # 使用混合精度训练
            with autocast(enabled=self.config['train']['amp_enabled']):
                sr_imgs = self.model(lr_imgs)
                loss = self.criterion(sr_imgs, hr_imgs)

            # 反向传播和优化
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['train']['grad_clip_norm']
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # 计算指标
            with torch.no_grad():
                mse = nn.MSELoss()(sr_imgs.float(), hr_imgs.float())
                psnr = 10 * torch.log10(1 / mse)
                ssim = self.ssim_module(sr_imgs, hr_imgs)

            return loss.item(), psnr.item(), ssim.item()

        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            raise

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_metrics = {'loss': 0, 'psnr': 0, 'ssim': 0}
        batch_times = []
        data_loading_times = []

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch [{epoch}/{self.config["train"]["num_epochs"]}]',
                    dynamic_ncols=True)

        try:
            batch_start = time.time()
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
                # 记录数据加载时间
                data_loading_time = time.time() - batch_start
                data_loading_times.append(data_loading_time)

                # 训练步骤
                step_start = time.time()
                loss, psnr, ssim = self.train_step(lr_imgs, hr_imgs)

                # 更新指标
                epoch_metrics['loss'] += loss
                epoch_metrics['psnr'] += psnr
                epoch_metrics['ssim'] += ssim

                # 记录batch时间
                batch_time = time.time() - step_start
                batch_times.append(batch_time)

                # 更新进度条
                if batch_idx % self.config['logging']['print_freq'] == 0:
                    avg_batch_time = sum(batch_times[-50:]) / len(batch_times[-50:])
                    avg_data_time = sum(data_loading_times[-50:]) / len(data_loading_times[-50:])

                    # 获取GPU信息
                    if torch.cuda.is_available():
                        gpu = GPUtil.getGPUs()[0]
                        gpu_info = f'GPU: {gpu.memoryUsed}/{gpu.memoryTotal}MB {gpu.temperature}°C'
                    else:
                        gpu_info = 'CPU only'

                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'PSNR': f'{psnr:.2f}',
                        'SSIM': f'{ssim:.4f}',
                        'batch_t': f'{avg_batch_time:.3f}s',
                        'data_t': f'{avg_data_time:.3f}s',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'GPU': gpu_info
                    })

                    # 记录到tensorboard
                    step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('train/loss', loss, step)
                    self.writer.add_scalar('train/psnr', psnr, step)
                    self.writer.add_scalar('train/ssim', ssim, step)
                    self.writer.add_scalar('train/learning_rate',
                                           self.optimizer.param_groups[0]['lr'], step)

                # 更新学习率
                self.scheduler.step()

                # 准备下一个batch的计时
                batch_start = time.time()

                # 定期清理GPU缓存
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"Error in epoch {epoch}: {str(e)}")
            raise

        finally:
            # 计算epoch平均指标
            for key in epoch_metrics:
                epoch_metrics[key] /= len(self.train_loader)

            # 输出epoch统计信息
            self.logger.info(
                f"Epoch {epoch} - "
                f"Loss: {epoch_metrics['loss']:.4f}, "
                f"PSNR: {epoch_metrics['psnr']:.2f}, "
                f"SSIM: {epoch_metrics['ssim']:.4f}, "
                f"Avg batch time: {sum(batch_times) / len(batch_times):.3f}s"
            )

        return epoch_metrics

    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        val_metrics = {'loss': 0, 'psnr': 0, 'ssim': 0}
        num_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader,
                    desc=f'Validating Epoch {epoch}',
                    dynamic_ncols=True)

        try:
            with torch.no_grad():
                for lr_imgs, hr_imgs in pbar:
                    lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                    hr_imgs = hr_imgs.to(self.device, non_blocking=True)

                    with autocast(enabled=self.config['train']['amp_enabled']):
                        sr_imgs = self.model(lr_imgs)
                        loss = self.criterion(sr_imgs, hr_imgs)

                    mse = nn.MSELoss()(sr_imgs, hr_imgs)
                    psnr = 10 * torch.log10(1 / mse)
                    ssim = self.ssim_module(sr_imgs, hr_imgs)

                    val_metrics['loss'] += loss.item()
                    val_metrics['psnr'] += psnr.item()
                    val_metrics['ssim'] += ssim.item()

                    pbar.set_postfix({
                        'Loss': f"{val_metrics['loss'] / num_batches:.4f}",
                        'PSNR': f"{val_metrics['psnr'] / num_batches:.2f}dB",
                        'SSIM': f"{val_metrics['ssim'] / num_batches:.4f}"
                    })

        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise

        finally:
            # 计算平均指标
            for key in val_metrics:
                val_metrics[key] /= num_batches

            # 记录到tensorboard
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
            self.writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)

            # 输出验证结果
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


    def _create_model(self):
        """创建和优化模型"""
        try:
            self.model = XRaySR(
                num_channels=self.config['model']['num_channels'],
                num_blocks=self.config['model']['num_blocks'],
                debug_mode=self.config['model']['debug_mode']
            ).to(self.device)

            self.model = self.model.to(memory_format=torch.channels_last)
            if torch.cuda.device_count() > 1:
                self.model = nn.parallel.DistributedDataParallel(self.model)  # 使用DDP替代DataParallel

            # 启用torch.compile
            if self.config['train']['compile_enabled'] and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)

            # GPU并行化
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)

            self.model.initialize_weights()

            # 创建损失函数
            self.criterion = L1CharbonnierLoss().to(self.device)
            self.ssim_module = SSIM().to(self.device)

            self.logger.info("Model created successfully")

            torch.backends.cudnn.benchmark = True


        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise

    def _create_optimizer_and_scheduler(self):
        """创建优化器和学习率调度器"""
        try:
            # 创建优化器
            self.optimizer = get_optimizer(self.model.parameters(), self.config)

            # 创建学习率调度器
            steps_per_epoch = len(self.train_loader)
            self.scheduler = get_scheduler(self.optimizer, self.config, steps_per_epoch)

            self.logger.info("Optimizer and scheduler created successfully")

        except Exception as e:
            self.logger.error(f"Error creating optimizer: {str(e)}")
            raise

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
    """主函数"""
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