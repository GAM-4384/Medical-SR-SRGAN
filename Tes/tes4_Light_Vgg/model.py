import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.cuda.amp import autocast
import copy
import torch
import math


class LightweightResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 使用分组卷积和通道重排来加速计算
        self.conv_block = nn.Sequential(
            # 使用1x1卷积减少通道数，实现特征降维
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.PReLU(),
            # 使用分组卷积降低计算复杂度
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1,
                      groups=channels // 2, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.PReLU(),
            # 使用1x1卷积恢复通道数，实现特征升维
            nn.Conv2d(channels // 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 简化的通道注意力机制，帮助模型关注重要特征
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 可学习的缩放因子，用于调节残差连接的影响
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 主路径处理
        out = self.conv_block(x)
        # 应用通道注意力
        attention = self.channel_attention(out)
        out = out * attention
        # 残差连接
        return x + self.scale * out


class BalancedXRaySR(nn.Module):
    def __init__(self, num_channels=32, num_blocks=6):
        super().__init__()

        # 特征提取
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # 主干网络
        trunk = []
        for _ in range(num_blocks):
            trunk.append(LightweightResidualBlock(num_channels))
        self.trunk = nn.Sequential(*trunk)

        # 特征融合
        self.fusion = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # 将上采样分成两个阶段，每次2倍
        self.upsampler1 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.upsampler2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # 最终重建
        self.final = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """改进的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化，适合ReLU类激活函数
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @autocast()
    def forward(self, x):
        # 保持在显存中的中间结果最少
        feat1 = self.first_conv(x)
        feat2 = self.trunk(feat1)
        feat2 = self.fusion(feat2)
        feat2 = feat2 + feat1

        # 分两步进行上采样，减少峰值显存使用
        out = self.upsampler1(feat2)
        torch.cuda.empty_cache()  # 清理第一次上采样的中间变量

        out = self.upsampler2(out)
        torch.cuda.empty_cache()  # 清理第二次上采样的中间变量

        out = self.final(out)
        return out


class BalancedLoss(nn.Module):
    def __init__(self, lambda_l1=0.5, lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.ssim = SSIM()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # 确保值域在[0,1]范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 计算L1损失
        l1_loss = self.l1_loss(pred, target)

        # 计算SSIM损失
        ssim_value = self.ssim(pred, target)
        ssim_loss = 1 - ssim_value

        # 总损失，调整权重
        total_loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss

        return total_loss



class SSIM(nn.Module):
    """结构相似性(SSIM)计算模块"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        # 确保window张量与输入类型匹配
        if channel == self.channel and self.window.dtype == img1.dtype and self.window.device == img1.device:
            window = self.window
        else:
            window = self._create_window(self.window_size).to(device=img1.device, dtype=img1.dtype)
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                              for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # 确保所有输入张量的类型一致
        img1 = img1.type_as(window)
        img2 = img2.type_as(window)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class MetricsCalculator:
    """图像质量评估指标计算器"""

    def __init__(self, data_range=1.0):
        self.data_range = torch.tensor(data_range)
        self.ssim_module = SSIM()

    def calculate_psnr(self, sr_imgs, hr_imgs):
        """
        计算峰值信噪比(PSNR)
        """
        # 确保数据类型和设备一致
        sr_imgs = sr_imgs.float()
        hr_imgs = hr_imgs.float()
        self.data_range = self.data_range.to(sr_imgs.device, sr_imgs.dtype)

        # 计算MSE
        mse = F.mse_loss(sr_imgs, hr_imgs, reduction='none').mean(dim=(1, 2, 3))
        mse = torch.clamp(mse, min=1e-10)

        # 计算PSNR
        psnr = 20 * torch.log10(self.data_range) - 10 * torch.log10(mse)

        return psnr.mean()

    def calculate_metrics(self, sr_imgs, hr_imgs):
        """计算所有评估指标"""
        with torch.no_grad():
            # 确保混合精度兼容性
            sr_imgs = sr_imgs.float()  # 转换为float32
            hr_imgs = hr_imgs.float()

            psnr = self.calculate_psnr(sr_imgs, hr_imgs)
            ssim = self.ssim_module(sr_imgs, hr_imgs)

            return psnr.item(), ssim.item()


class ModelEMA:
    """
    Model Exponential Moving Average
    在训练过程中对模型权重进行平滑，通常可以提高模型性能和稳定性
    Args:
        model: 需要进行EMA的模型
        decay: EMA的衰减率 (默认: 0.9999)
        updates: EMA更新次数 (默认: 0)
        device: 运行设备 (默认: None，即使用model的设备)
    """

    def __init__(self, model, decay=0.9999, updates=0, device=None):
        # 创建EMA
        self.ema = copy.deepcopy(model)
        self.ema.eval()  # EMA模型设置为评估模式

        # 保存参数
        self.updates = updates  # 记录更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 衰减率函数
        self.device = device if device else next(model.parameters()).device

        # 将EMA模型移动到指定设备
        self.ema.to(self.device)

        # 关闭EMA模型的梯度计算
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """
        更新EMA模型的权重

        Args:
            model: 当前训练的模型
        """
        self.updates += 1
        d = self.decay(self.updates)

        # 更新EMA模型的参数
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                if model_p.dtype.is_floating_point:  # 只更新浮点类型的参数
                    ema_p.data.mul_(d).add_(model_p.data, alpha=1 - d)

            # 更新BN层的running_mean和running_var
            for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
                if model_b.dtype.is_floating_point:
                    ema_b.data.mul_(d).add_(model_b.data, alpha=1 - d)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        更新EMA模型的属性

        Args:
            model: 源模型
            include: 需要包含的属性列表
            exclude: 需要排除的属性列表
        """
        copy_attr(self.ema, model, include, exclude)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        """
        使EMA模型可以像普通模型一样被调用
        """
        return self.ema(*args, **kwargs)

    def state_dict(self):
        """
        返回EMA模型的状态字典
        """
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        """
        加载状态字典到EMA模型

        Args:
            state_dict: 要加载的状态字典
        """
        self.ema.load_state_dict(state_dict)


def copy_attr(a, b, include=(), exclude=()):
    """
    复制模型属性

    Args:
        a: 目标对象
        b: 源对象
        include: 需要包含的属性列表
        exclude: 需要排除的属性列表
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)