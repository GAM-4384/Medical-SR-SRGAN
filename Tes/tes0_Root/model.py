import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.cuda.amp import autocast


class ResidualBlock(nn.Module):
    """残差块实现
    结构:
        - 两个3x3卷积层
        - 每个卷积后跟BatchNorm
        - 中间使用PReLU激活函数
        - 残差连接将输入直接加到输出上
    """

    def __init__(self, channels):

        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            # 第一个卷积层：保持空间尺寸不变(padding=1)
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),  # 参数化ReLU激活函数
            # 第二个卷积层
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):

        return x + self.conv_block(x)  # 残差连接


class XRaySR(nn.Module):
    """X射线图像超分辨率网络
    网络结构特点：
    - 使用大卷积核(7x7)进行初始和最终特征提取
    - 采用PReLU作为激活函数
    - 包含全局残差学习
    - 使用PixelShuffle进行上采样
    - 附加图像增强模块
    """

    def __init__(self, num_channels=32, num_blocks=4, debug_mode=False):

        super(XRaySR, self).__init__()
        self.debug_mode = debug_mode
        self.first_forward = True

        # 初始特征提取：1通道 -> num_channels通道
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=7, padding=3, bias=False),  # 大卷积核
            nn.PReLU()
        )

        # 特征提取主干：堆叠多个残差块
        trunk = []
        for _ in range(num_blocks):
            trunk.append(ResidualBlock(num_channels))
        self.trunk = nn.Sequential(*trunk)

        # 特征融合：处理残差块输出
        self.fusion = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )

        # 上采样模块：将分辨率提升4倍
        self.upsampler = nn.Sequential(
            # 第一次2倍上采样
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),  # 像素重排，将通道数转换为空间分辨率
            nn.PReLU(),
            # 第二次2倍上采样
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # 最终重建：将特征图转换回单通道
        self.final = nn.Conv2d(num_channels, 1, kernel_size=7, padding=3)

        # 图像增强模块：自适应增强重建结果
        self.enhance = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # 输出0-1之间的增强权重
        )

    @autocast()
    def forward(self, x):
        """前向传播
        Args:
            x (Tensor): 输入低分辨率图像 [B, 1, H, W]
        Returns:
            Tensor: 超分辨率重建结果 [B, 1, 4H, 4W]
        """
        # 1. 特征提取
        feat1 = self.first_conv(x)

        # 2. 主干网络处理
        feat2 = self.trunk(feat1)

        # 3. 特征融合
        feat3 = self.fusion(feat2)
        feat3 = feat3 + feat1  # 全局残差连接

        # 4. 上采样和重建
        sr = self.upsampler(feat3)  # 4倍上采样
        sr = self.final(sr)  # 最终重建

        # 5. 自适应增强
        enhance_mask = self.enhance(sr)  # 生成增强权重图
        out = sr * enhance_mask  # 应用增强

        return out

    def initialize_weights(self):
        """初始化网络权重

        采用以下策略：
        - 卷积层：（kaiming_normal_）
        - BatchNorm层：权重设为1，偏置设为0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class L1CharbonnierLoss(nn.Module):
    """Charbonnier损失函数

    L1损失的一个变体，在原点附近更平滑，对异常值更鲁棒。
    损失函数形式：sqrt(x^2 + ε^2)
    """

    def __init__(self, epsilon=1e-3):
        """
        Args:
            epsilon (float): 平滑参数，默认1e-3
        """
        super(L1CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """计算损失值

        Args:
            pred (Tensor): 预测值
            target (Tensor): 目标值

        Returns:
            Tensor: 损失值标量
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()


class SSIM(nn.Module):
    """结构相似性(SSIM)计算模块

    SSIM是一种衡量图像质量的度量，比单纯的均方误差或PSNR更符合人眼感知。
    计算公式：SSIM = (2μxμy + C1)(2σxy + C2) / (μx² + μy² + C1)(σx² + σy² + C2)
    """

    def __init__(self, window_size=11):
        """
        Args:
            window_size (int): 滑动窗口大小，默认11
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        # 创建高斯窗口
        window = torch.Tensor([exp(-(x - self.window_size // 2) ** 2 / float(2 * 1.5 ** 2))
                               for x in range(self.window_size)])
        window = window.unsqueeze(1) * window.unsqueeze(0)  # 2D高斯核
        window = window / window.sum()  # 归一化
        self.register_buffer('window', window.unsqueeze(0).unsqueeze(0).float())

    @autocast()
    def forward(self, img1, img2):
        """计算SSIM

        Args:
            img1 (Tensor): 第一张图像
            img2 (Tensor): 第二张图像

        Returns:
            Tensor: SSIM值，范围[-1,1]，1表示完全相同
        """
        return self._ssim(img1, img2)

    def _ssim(self, img1, img2):
        """SSIM的具体计算过程

        Args:
            img1 (Tensor): 第一张图像
            img2 (Tensor): 第二张图像

        Returns:
            Tensor: SSIM图，取平均得到最终SSIM值
        """
        img1 = img1.float()
        img2 = img2.float()
        self.window = self.window.to(img1.dtype)
        # 确保高斯窗口(window)的数据类型与输入图像(img1)的数据类型保持一致
        # 计算均值
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2)

        # 计算方差和协方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2) - mu1_mu2

        # SSIM常数项
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()