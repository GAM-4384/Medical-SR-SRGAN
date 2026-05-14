import os
import logging
from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from Tes.tes4_Light_Vgg.model import BalancedXRaySR, BalancedLoss, ModelEMA, MetricsCalculator
from Tes.tes4_Light_Vgg.dataset import XRayDataset
from torch.utils.data import DataLoader
from Tes.tes4_Light_Vgg.config import get_training_config, setup_device, create_dirs


class Trainer:
    """训练器类，用于处理模型训练、验证和测试"""

    def __init__(self, config):
        """
        初始化训练器

        Args:
            config: 配置字典，包含训练所需的所有参数
        """
        try:
            print("初始化训练器...")
            self.config = config

            # 设置设备
            self.device = setup_device()
            print(f"使用设备: {self.device}")

            # 创建必要的目录
            create_dirs(config)

            # 设置基本参数
            self.current_epoch = 0
            self.step_count = 0

            # 初始化数据加载器
            self._init_data_loader()

            # 创建模型
            self.model = BalancedXRaySR(
                num_channels=config['model']['num_channels'],
                num_blocks=config['model']['num_blocks']
            ).to(self.device)

            if config['train']['compile_enabled'] and hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)

            # 初始化损失函数
            self.criterion = BalancedLoss(
                lambda_l1=config['loss_weights']['l1'],
                lambda_ssim=config['loss_weights']['ssim']
            ).to(self.device)

            # 初始化优化器
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,  # 增大初始学习率
                betas=(0.9, 0.99),
                weight_decay=1e-4
            )

            # 修改学习率调度器
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=1e-3,
                epochs=config['train']['num_epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=25,
                final_div_factor=1000
            )

            # 初始化梯度缩放器（用于混合精度训练）
            self.scaler = GradScaler(enabled=config['train']['amp_enabled'])

            # 初始化EMA（如果启用）
            if config.get('ema', {}).get('enabled', True):
                self.ema = ModelEMA(
                    self.model,
                    decay=config.get('ema', {}).get('decay', 0.9999),
                    device=self.device
                )
            else:
                self.ema = None

            # 初始化指标计算器
            self.metrics_calculator = MetricsCalculator()

            # 加载检查点（如果需要）
            if config['resume']['enabled']:
                self._load_checkpoint(config['resume']['checkpoint_path'])

            # 打印模型参数量
            total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"模型参数量: {total_params:.2f}M")

            print("训练器初始化完成")

        except Exception as e:
            print(f"训练器初始化失败: {str(e)}")
            raise e

    def _init_data_loader(self):
        """初始化数据加载器"""
        try:
            paths_config = self.config['paths']
            data_config = self.config['data']

            print("预加载训练数据...")
            train_dataset = XRayDataset(
                hr_dir=paths_config['train_hr_dir'],
                lr_dir=paths_config['train_lr_dir'],
                cache_size=data_config['cache_size']
            )

            print("预加载验证数据...")
            val_dataset = XRayDataset(
                hr_dir=paths_config['val_hr_dir'],
                lr_dir=paths_config['val_lr_dir'],
                cache_size=data_config['cache_size']
            )

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['train']['batch_size'],
                shuffle=data_config['shuffle'],
                num_workers=data_config['num_workers'],
                pin_memory=data_config['pin_memory'],
                drop_last=data_config['drop_last'],
                persistent_workers=data_config['persistent_workers'],
                prefetch_factor=data_config['prefetch_factor']
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['validation']['batch_size'],
                shuffle=False,
                num_workers=data_config['num_workers'],
                pin_memory=data_config['pin_memory']
            )

            print(f"成功加载数据 - 训练集: {len(train_dataset)}张, 验证集: {len(val_dataset)}张")

        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise e

    def _load_checkpoint(self, checkpoint_path):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        try:
            print(f"加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 加载其他状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.step_count = checkpoint.get('step_count', 0)

            # 加载EMA状态（如果存在）
            if self.ema and 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])

            print(f"成功加载检查点，当前epoch: {self.current_epoch}")

        except Exception as e:
            print(f"加载检查点失败: {str(e)}")
            raise e

    def save_checkpoint(self, is_best=False):
        """
        保存检查点

        Args:
            is_best: 是否为最佳模型
        """
        try:
            # 准备检查点数据
            checkpoint = {
                'epoch': self.current_epoch,
                'step_count': self.step_count,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'ema_state_dict': self.ema.state_dict() if self.ema else None
            }

            # 获取检查点目录
            checkpoint_dir = self.config['logging']['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存最新检查点
            latest_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{self.current_epoch + 1}.pth"
            )
            torch.save(checkpoint, latest_path)
            print(f"保存检查点: {latest_path}")

            # 如果是最佳模型，额外保存一份
            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                print(f"保存最佳模型: {best_path}")

        except Exception as e:
            print(f"保存检查点失败: {str(e)}")
            raise e

    def train_step(self, lr_imgs, hr_imgs):
        try:
            # 确保输入数据在GPU上且类型正确
            lr_imgs = lr_imgs.to(self.device, dtype=torch.float32, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, dtype=torch.float32, non_blocking=True)

            # 清除优化器的梯度
            self.optimizer.zero_grad(set_to_none=True)

            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                sr_imgs = self.model(lr_imgs)
                loss = self.criterion(sr_imgs, hr_imgs)

            # 使用梯度累积
            loss = loss / self.config['train']['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()

            # 在累积足够的梯度后更新模型
            if (self.step_count + 1) % self.config['train']['gradient_accumulation_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['train']['grad_clip_norm']
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()

            self.step_count += 1

            # 计算评估指标
            with torch.no_grad():
                psnr, ssim = self.metrics_calculator.calculate_metrics(sr_imgs.detach(), hr_imgs)

            return loss.item(), psnr, ssim  # 返回三个值

        except Exception as e:
            print(f"训练步骤出错: {str(e)}")
            raise e

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        val_steps = 0
        max_images = self.config['validation']['max_images']

        try:
            print("\n开始验证...")
            for i, (lr_imgs, hr_imgs) in enumerate(self.val_loader):
                if max_images and i * self.config['validation']['batch_size'] >= max_images:
                    break

                # 确保数据类型和设备正确
                lr_imgs = lr_imgs.to(self.device, dtype=torch.float32)
                hr_imgs = hr_imgs.to(self.device, dtype=torch.float32)

                with autocast(enabled=self.config['train']['amp_enabled']):
                    if self.ema is not None:
                        sr_imgs = self.ema(lr_imgs)
                    else:
                        sr_imgs = self.model(lr_imgs)
                    loss = self.criterion(sr_imgs, hr_imgs)

                psnr, ssim = self.metrics_calculator.calculate_metrics(sr_imgs, hr_imgs)

                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                val_steps += 1

            # 计算平均值
            avg_loss = total_loss / val_steps
            avg_psnr = total_psnr / val_steps
            avg_ssim = total_ssim / val_steps

            print(f"验证结果 - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
            return avg_loss, avg_psnr, avg_ssim

        except Exception as e:
            print(f"验证过程出错: {str(e)}")
            raise e
        finally:
            self.model.train()

    def train(self):
        try:
            print("开始训练...")
            torch.backends.cudnn.benchmark = True

            for epoch in range(self.current_epoch, self.config['train']['num_epochs']):
                self.model.train()
                epoch_loss = 0.0
                epoch_psnr = 0.0
                epoch_ssim = 0.0

                # 使用tqdm创建进度条
                pbar = tqdm(self.train_loader,
                            desc=f'Epoch {epoch + 1}/{self.config["train"]["num_epochs"]}')

                for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
                    # 训练一个批次
                    loss, psnr, ssim = self.train_step(lr_imgs, hr_imgs)  # 正确解包三个返回值

                    # 累积指标
                    epoch_loss += loss
                    epoch_psnr += psnr
                    epoch_ssim += ssim

                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'psnr': f'{psnr:.2f}',
                        'ssim': f'{ssim:.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })

                    # 定期验证
                    if (batch_idx + 1) % self.config['logging']['val_freq'] == 0:
                        self.validate()
                        self.model.train()

                # 计算epoch平均指标
                avg_loss = epoch_loss / len(self.train_loader)
                avg_psnr = epoch_psnr / len(self.train_loader)
                avg_ssim = epoch_ssim / len(self.train_loader)

                print(f"\nEpoch {epoch + 1} 完成:")
                print(f"平均损失: {avg_loss:.4f}")
                print(f"平均PSNR: {avg_psnr:.2f}")
                print(f"平均SSIM: {avg_ssim:.4f}")

                # 保存检查点
                if (epoch + 1) % self.config['logging']['save_freq'] == 0:
                    self.save_checkpoint()

        except Exception as e:
            print(f"训练出错: {str(e)}")
            raise e


def main():
    """主函数"""
    try:
        # 配置logging
        logging.basicConfig(level=logging.INFO)

        # 获取配置
        config = get_training_config()

        # 创建训练器实例
        trainer = Trainer(config)

        # 开始训练
        trainer.train()

    except Exception as e:
        logging.error(f"训练失败: {str(e)}")
        raise e

if __name__ == '__main__':
    main()