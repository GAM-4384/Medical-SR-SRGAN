import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.cuda.amp import autocast

class ResidualBlock(nn.Module):
    """优化后的残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class XRaySR(nn.Module):
    """优化后的X射线图像超分辨率网络"""
    def __init__(self, num_channels=32, num_blocks=4, debug_mode=False):
        super(XRaySR, self).__init__()
        self.debug_mode = debug_mode
        self.first_forward = True

        # 初始特征提取
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=7, padding=3, bias=False),
            nn.PReLU()
        )

        # 特征提取主干
        trunk = []
        for _ in range(num_blocks):
            trunk.append(ResidualBlock(num_channels))
        self.trunk = nn.Sequential(*trunk)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )

        # 上采样模块 (4x)
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # 最终重建
        self.final = nn.Conv2d(num_channels, 1, kernel_size=7, padding=3)

        # 图像增强模块
        self.enhance = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    @autocast()
    def forward(self, x):
        # 特征提取
        feat1 = self.first_conv(x)

        # 主干处理
        feat2 = self.trunk(feat1)

        # 特征融合
        feat3 = self.fusion(feat2)
        feat3 = feat3 + feat1  # 全局残差连接

        # 上采样和重建
        sr = self.upsampler(feat3)
        sr = self.final(sr)

        # 图像增强
        enhance_mask = self.enhance(sr)
        out = sr * enhance_mask

        return out

    def initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class L1CharbonnierLoss(nn.Module):
    """改进的 Charbonnier 损失函数，返回所有必要的损失指标"""
    def __init__(self, epsilon=1e-3):
        super(L1CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
        self.ssim_module = SSIM()  # 添加 SSIM 模块

    def forward(self, pred, target):
        # 确保输入是浮点类型
        pred = pred.float()
        target = target.float()

        # 计算 L1 Charbonnier 损失
        diff = pred - target
        l1_loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon).mean()

        # 计算 MSE 损失
        mse_loss = F.mse_loss(pred, target)

        # 计算 SSIM 损失
        ssim_value = self.ssim_module(pred, target)
        ssim_loss = 1 - ssim_value

        # 计算总损失（可以调整权重）
        total_loss = l1_loss + 0.1 * ssim_loss + 0.1 * mse_loss

        # 返回损失字典
        return {
            'total': total_loss,
            'l1': l1_loss,
            'ssim': ssim_loss,
            'mse': mse_loss
        }

class SSIM(nn.Module):
    """结构相似性计算模块"""
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        window = torch.Tensor([exp(-(x - self.window_size // 2) ** 2 / float(2 * 1.5 ** 2))
                               for x in range(self.window_size)])
        window = window.unsqueeze(1) * window.unsqueeze(0)
        window = window / window.sum()
        self.register_buffer('window', window.unsqueeze(0).unsqueeze(0).float())

    @autocast()
    def forward(self, img1, img2):
        return self._ssim(img1, img2)

    def _ssim(self, img1, img2):
        img1 = img1.float()
        img2 = img2.float()
        self.window = self.window.to(img1.dtype)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class CombinedLoss(nn.Module):
    """组合损失函数，结合L1 Charbonnier、SSIM和MSE损失"""

    def __init__(self, l1_weight=1.0, ssim_weight=0.1, mse_weight=0.1, epsilon=1e-3):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.epsilon = epsilon

        # 初始化SSIM模块
        self.ssim_module = SSIM()

    def forward(self, pred, target):
        # 确保输入是浮点类型
        pred = pred.float()
        target = target.float()

        # 计算L1 Charbonnier损失
        diff = pred - target
        l1_loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon).mean()

        # 计算SSIM损失
        ssim_value = self.ssim_module(pred, target)
        ssim_loss = 1 - ssim_value

        # 计算MSE损失，保持为张量形式
        mse_loss = F.mse_loss(pred, target)

        # 组合损失
        total_loss = (self.l1_weight * l1_loss +
                      self.ssim_weight * ssim_loss +
                      self.mse_weight * mse_loss)

        return {
            'total': total_loss,
            'l1': l1_loss,  # 注意：不再使用.item()
            'ssim': ssim_loss,  # 不再使用.item()
            'mse': mse_loss  # 不再使用.item()
        }
