# 导入自定义模块

import os
import time
import logging
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import GPUtil
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# 导入自定义模块
from Tes.tes3_Refer_to_Vgg.config import (
    get_training_config,
    setup_training_device,
    create_experiment_dirs,
    get_optimizer,
    get_scheduler,
    save_checkpoint
)
from Tes.tes3_Refer_to_Vgg.model import (
    OptimizedXRaySR,
    OptimizedCombinedLoss
)
from Tes.tes3_Refer_to_Vgg.dataset import create_dataloaders
class OptimizedTrainer:
    def __init__(self):
        """初始化优化后的训练器"""
        self._setup_logger()
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.info("初始化训练器...")
        self._setup_environment()
        self.config = get_training_config()
        self._setup_memory_saving_config()
        self.device = setup_training_device()
        self._initialize_training()

    def _setup_environment(self):
        """设置训练环境"""
        if torch.cuda.is_available():
            # 清理并限制GPU内存
            torch.cuda.empty_cache()

            # 设置基本的CUDA内存分配器配置
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

            # 优化CUDA设置
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True

    def _setup_memory_saving_config(self):
        """设置内存节省配置"""
        # 设置梯度累积步数
        self.gradient_accumulation_steps = self.config['train'].get('accumulation_steps', 4)

        # 计算有效批次大小
        self.config['train']['effective_batch_size'] = (
                self.config['train']['batch_size'] * self.gradient_accumulation_steps
        )

        # 配置梯度检查点
        self.use_gradient_checkpointing = self.config['performance'].get('gradient_checkpointing', False)

        # 设置CUDA图形
        self.cuda_graphs_enabled = (
                torch.cuda.is_available() and
                self.config['train'].get('use_cuda_graphs', False)
        )

        # 配置混合精度训练
        self.amp_enabled = self.config['train'].get('amp_enabled', True)

        self.logger.info(f"内存优化配置:")
        self.logger.info(f"- 梯度累积步数: {self.gradient_accumulation_steps}")
        self.logger.info(f"- 有效批次大小: {self.config['train']['effective_batch_size']}")
        self.logger.info(f"- 梯度检查点: {self.use_gradient_checkpointing}")
        self.logger.info(f"- CUDA图形: {self.cuda_graphs_enabled}")
        self.logger.info(f"- 混合精度训练: {self.amp_enabled}")

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('TrainingLogger')
        logger.setLevel(logging.INFO)

        # 创建logs目录
        os.makedirs('../../logs', exist_ok=True)

        # 文件处理器
        fh = logging.FileHandler('../../logs/training.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))

        # 清除现有的处理器
        logger.handlers.clear()

        # 添加新的处理器
        logger.addHandler(fh)
        logger.addHandler(ch)

    def _initialize_training(self):
        """初始化训练组件"""
        try:
            # 创建必要的目录
            create_experiment_dirs(self.config)

            # 初始化tensorboard
            self.writer = SummaryWriter(self.config['logging']['tensorboard_dir'])

            # 创建数据加载器
            self.train_loader, self.val_loader = create_dataloaders(self.config)
            self.logger.info(f"数据加载器创建成功 - 训练集: {len(self.train_loader.dataset)}张图像, "
                             f"验证集: {len(self.val_loader.dataset)}张图像")

            # 设置CUDA内存分配器
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(
                    self.config['performance']['gpu_mem_fraction']
                )

            # 创建和配置模型
            self.model = OptimizedXRaySR(
                num_channels=self.config['model']['num_channels'],
                num_blocks=self.config['model']['num_blocks'],
                debug_mode=self.config['model']['debug_mode']
            ).to(self.device, memory_format=torch.channels_last)

            # 启用梯度检查点（如果配置）
            if self.use_gradient_checkpointing:
                self.model.enable_gradient_checkpointing()
                self.logger.info("已启用梯度检查点")

            # 初始化模型权重
            self.model.initialize_weights()

            # 创建损失函数
            self.criterion = OptimizedCombinedLoss(
                lambda_char=self.config['loss_weights']['lambda_char'],
                lambda_ssim=self.config['loss_weights']['lambda_ssim'],
                lambda_perceptual=self.config['loss_weights']['lambda_perceptual']
            ).to(self.device)

            # 创建优化器并使用torch.cuda.amp
            self.optimizer = get_optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.config
            )

            # 创建学习率调度器
            steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
            self.scheduler = get_scheduler(self.optimizer, self.config, steps_per_epoch)

            # 创建梯度缩放器
            self.scaler = GradScaler(enabled=self.amp_enabled)

            # 训练状态初始化
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            self.training_start_time = time.time()

            self.logger.info("训练初始化完成")

        except Exception as e:
            self.logger.error(f"初始化失败: {str(e)}")
            raise

    def _initialize_cuda_graphs(self):
        """初始化CUDA图"""
        try:
            if not torch.cuda.is_available():
                return

            self.logger.info("初始化CUDA图...")

            # 创建静态输入
            self.static_input = torch.randn(
                self.config['train']['batch_size'],
                1,
                self.config['data']['img_size'],
                self.config['data']['img_size'],
                device=self.device
            )
            self.static_target = torch.randn(
                self.config['train']['batch_size'],
                1,
                self.config['data']['img_size'] * 4,
                self.config['data']['img_size'] * 4,
                device=self.device
            )

            # 预热运行
            self.model.eval()
            with torch.cuda.amp.autocast(enabled=self.config['train']['amp_enabled']):
                for _ in range(3):  # 进行多次预热
                    _ = self.model(self.static_input)

            # 捕获CUDA图
            self.g = torch.cuda.CUDAGraph()

            # 设置为训练模式
            self.model.train()

            # 开始记录图
            with torch.cuda.graph(self.g):
                with torch.cuda.amp.autocast(enabled=self.config['train']['amp_enabled']):
                    self.static_output = self.model(self.static_input)
                    self.static_loss = self.criterion(self.static_output, self.static_target)
                    self.static_scaled_loss = self.scaler.scale(self.static_loss)

            self.logger.info("CUDA图初始化完成")

        except Exception as e:
            self.logger.warning(f"CUDA图初始化失败: {str(e)}")
            self.cuda_graphs_enabled = False

    def _save_validation_images(self, lr_imgs, sr_imgs, hr_imgs, epoch):
        """保存验证图像"""
        n = min(lr_imgs.size(0), self.config['validation']['max_save_images'])

        # 创建保存目录
        save_dir = os.path.join(self.config['logging']['log_dir'], 'val_images', f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)

        for i in range(n):
            # 拼接图像：LR - SR - HR
            comparison = torch.cat([
                torch.nn.functional.interpolate(lr_imgs[i:i + 1], scale_factor=4, mode='nearest'),
                sr_imgs[i:i + 1],
                hr_imgs[i:i + 1]
            ], dim=-1)

            save_path = os.path.join(save_dir, f'compare_{i}.png')
            save_image(comparison, save_path, normalize=True)

    def _update_cuda_graph_inputs(self, lr_imgs, hr_imgs):
        """更新CUDA图的输入"""
        self.static_input.copy_(lr_imgs)
        self.static_target.copy_(hr_imgs)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'total': 0, 'char': 0, 'ssim': 0, 'perceptual': 0}
        processed_batches = 0
        failed_batches = 0

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch [{self.current_epoch}/{self.config["train"]["num_epochs"]}]',
                    dynamic_ncols=True)

        try:
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
                try:
                    loss, loss_components = self.train_step(lr_imgs, hr_imgs, batch_idx)

                    # 处理训练步骤失败的情况
                    if loss is None:
                        failed_batches += 1
                        if failed_batches > 10:  # 如果连续失败次数过多
                            self.logger.warning("连续失败次数过多，跳过当前epoch")
                            return None
                        continue

                    # 更新损失记录
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        epoch_losses['total'] += loss
                        if loss_components:
                            epoch_losses['char'] += loss_components['charbonnier']
                            epoch_losses['ssim'] += loss_components['ssim']
                            epoch_losses['perceptual'] += loss_components.get('perceptual', 0)
                        processed_batches += 1

                    # 更新进度条
                    if batch_idx % self.config['logging']['print_freq'] == 0:
                        gpu = GPUtil.getGPUs()[0]
                        current_lr = self.optimizer.param_groups[0]['lr']

                        pbar.set_postfix({
                            'loss': f'{loss:.4f}' if loss else 'N/A',
                            'lr': f'{current_lr:.2e}',
                            'GPU': f'{gpu.memoryUsed:.0f}MB',
                            'Failed': failed_batches
                        })

                except Exception as e:
                    self.logger.error(f"批次 {batch_idx} 处理失败: {str(e)}")
                    continue

                # 定期保存检查点
                if processed_batches > 0 and processed_batches % 1000 == 0:
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.scaler, self.current_epoch, self.best_val_loss,
                        self.best_epoch, self.config,
                        f'checkpoint_epoch_{self.current_epoch}_batch_{batch_idx}.pth'
                    )

        except Exception as e:
            self.logger.error(f"Epoch {self.current_epoch} 训练失败: {str(e)}")
            raise

        finally:
            # 计算平均损失
            if processed_batches > 0:
                for key in epoch_losses:
                    epoch_losses[key] /= processed_batches

            # 清理内存
            torch.cuda.empty_cache()

        return epoch_losses

    def train_step(self, lr_imgs, hr_imgs, batch_idx):
        """单步训练"""
        try:
            # 清理不需要的缓存
            if batch_idx % self.config['performance']['empty_cache_freq'] == 0:
                torch.cuda.empty_cache()

            # 移动数据到设备
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            # 常规前向传播
            with autocast(enabled=self.amp_enabled):
                # 分离计算图以节省内存
                lr_imgs = lr_imgs.detach()
                hr_imgs = hr_imgs.detach()

                sr_imgs = self.model(lr_imgs)
                loss, loss_components = self.criterion(sr_imgs, hr_imgs)
                loss = loss / self.gradient_accumulation_steps

            # 反向传播
            self.scaler.scale(loss).backward()

            # 在累积步骤结束时更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['train']['grad_clip_norm']
                )

                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # 调度器步进
                self.scheduler.step()

            # 主动释放内存
            del sr_imgs
            torch.cuda.empty_cache()

            return loss.item() * self.gradient_accumulation_steps, loss_components

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                self.logger.warning(f"内存不足(batch_idx={batch_idx})")
                return None, None
            raise e

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_losses = {'total': 0, 'char': 0, 'ssim': 0, 'perceptual': 0}
        num_batches = len(self.val_loader)

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar = tqdm(self.val_loader,
                    desc=f'Validating Epoch {self.current_epoch}',
                    dynamic_ncols=True)

        try:
            with torch.no_grad():
                for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
                    # 定期清理缓存
                    if batch_idx % self.config['performance']['empty_cache_freq'] == 0:
                        torch.cuda.empty_cache()

                    try:
                        # 移动数据到设备
                        lr_imgs = lr_imgs.to(self.device, non_blocking=True)
                        hr_imgs = hr_imgs.to(self.device, non_blocking=True)

                        # 使用混合精度
                        with autocast(enabled=self.config['train']['amp_enabled']):
                            sr_imgs = self.model(lr_imgs)
                            loss, loss_components = self.criterion(sr_imgs, hr_imgs)

                        # 更新损失
                        val_losses['total'] += loss.item()
                        val_losses['char'] += loss_components['charbonnier']
                        val_losses['ssim'] += loss_components['ssim']
                        val_losses['perceptual'] += loss_components['perceptual']

                        # 保存验证图像样本
                        if batch_idx == 0 and self.config['validation']['save_images']:
                            self._save_validation_images(lr_imgs, sr_imgs, hr_imgs, self.current_epoch)

                        # 更新进度条
                        current_loss = {k: v / (batch_idx + 1) for k, v in val_losses.items()}
                        pbar.set_postfix({
                            'Loss': f"{current_loss['total']:.4f}",
                            'PSNR': f"{10 * torch.log10(1 / current_loss['total']):.2f}dB",
                            'SSIM': f"{1 - current_loss['ssim']:.4f}"
                        })

                        # 释放内存
                        del sr_imgs
                        torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            self.logger.warning(f"验证时内存不足(batch_idx={batch_idx})")
                            continue
                        raise e

        except Exception as e:
            self.logger.error(f"验证失败: {str(e)}")
            raise

        finally:
            # 计算平均损失
            for key in val_losses:
                val_losses[key] /= num_batches

            # 记录到tensorboard
            self.writer.add_scalar('val/total_loss', val_losses['total'], self.current_epoch)
            self.writer.add_scalar('val/char_loss', val_losses['char'], self.current_epoch)
            self.writer.add_scalar('val/ssim_loss', val_losses['ssim'], self.current_epoch)
            self.writer.add_scalar('val/perceptual_loss', val_losses['perceptual'], self.current_epoch)
            self.writer.add_scalar('val/psnr', 10 * torch.log10(1 / val_losses['total']), self.current_epoch)

            # 输出验证结果
            self.logger.info(
                f"验证结束 - "
                f"Loss: {val_losses['total']:.4f}, "
                f"PSNR: {10 * torch.log10(1 / val_losses['total']):.2f}dB, "
                f"SSIM: {1 - val_losses['ssim']:.4f}"
            )

            # 清理内存
            torch.cuda.empty_cache()

        return val_losses

    def train(self):
        """完整训练流程"""
        self.logger.info("开始训练...")
        self.training_start_time = time.time()

        try:
            for epoch in range(self.current_epoch + 1, self.config['train']['num_epochs'] + 1):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # 训练epoch
                train_losses = self.train_epoch()

                # 如果训练失败，尝试降低学习率并继续
                if train_losses is None:
                    self.logger.warning(f"Epoch {epoch} 训练失败，尝试调整学习率")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    continue

                # 记录训练损失
                self.writer.add_scalar('train/total_loss', train_losses['total'], epoch)
                self.writer.add_scalar('train/char_loss', train_losses['char'], epoch)
                self.writer.add_scalar('train/ssim_loss', train_losses['ssim'], epoch)
                if train_losses.get('perceptual', 0) > 0:
                    self.writer.add_scalar('train/perceptual_loss', train_losses['perceptual'], epoch)

                # 定期验证
                if epoch % self.config['validation']['val_freq'] == 0:
                    val_losses = self.validate()

                    # 保存最佳模型
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.best_epoch = epoch
                        save_checkpoint(
                            self.model, self.optimizer, self.scheduler,
                            self.scaler, epoch, self.best_val_loss,
                            self.best_epoch, self.config, 'best_model.pth'
                        )

                    # 记录验证结果
                    self.logger.info(
                        f"验证结果 - Loss: {val_losses['total']:.4f}, "
                        f"PSNR: {10 * torch.log10(1 / val_losses['total']):.2f}dB"
                    )

                # 记录epoch时间和学习率
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {epoch} 完成 - "
                    f"耗时: {epoch_time:.2f}s, "
                    f"学习率: {current_lr:.2e}, "
                    f"损失: {train_losses['total']:.4f}"
                )

                # 保存定期检查点
                if epoch % self.config['logging']['save_freq'] == 0:
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.scaler, epoch, self.best_val_loss,
                        self.best_epoch, self.config,
                        f'checkpoint_epoch_{epoch}.pth'
                    )

                # 如果连续多个epoch没有改善，降低学习率
                if epoch - self.best_epoch >= 5:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    self.logger.info(f"降低学习率至 {param_group['lr']:.2e}")

        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.scaler, epoch, self.best_val_loss,
                self.best_epoch, self.config, 'interrupted.pth'
            )

        except Exception as e:
            self.logger.error(f"训练错误: {str(e)}")
            raise

        finally:
            total_time = time.time() - self.training_start_time
            self.logger.info(
                f"训练完成, 总耗时: {total_time / 3600:.2f}小时\n"
                f"最佳验证损失: {self.best_val_loss:.4f} (Epoch {self.best_epoch})"
            )

            # 保存最终模型
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.scaler, epoch, self.best_val_loss,
                self.best_epoch, self.config, 'final_model.pth'
            )

            self.writer.close()
            torch.cuda.empty_cache()

def main():
    """主函数"""
    try:
        trainer = OptimizedTrainer()
        trainer.train()

    except Exception as e:
        logging.error(f"训练失败: {str(e)}")
        raise


if __name__ == '__main__':
    main()