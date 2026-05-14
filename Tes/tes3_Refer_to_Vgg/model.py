import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.cuda.amp import autocast
import torchvision.models as models


class EfficientAttention(nn.Module):
    """高效注意力模块"""

    def __init__(self, channels):
        super(EfficientAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 生成Q、K、V
        q = self.query(x).view(batch_size, -1, height * width)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        # 计算注意力
        attn = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return x + self.gamma * out


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        padding = kernel_size // 2

        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        # 逐点卷积
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class OptimizedResidualBlock(nn.Module):
    """优化的残差块"""

    def __init__(self, channels):
        super(OptimizedResidualBlock, self).__init__()

        # 主要路径
        self.conv1 = DepthwiseSeparableConv(channels, channels, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=3)

        # SE注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.conv2(out)

        # SE注意力
        se_weight = self.se(out)
        out = out * se_weight

        # 残差连接
        out = out + identity
        return out


class OptimizedUpscaleBlock(nn.Module):
    """优化的上采样块"""

    def __init__(self, in_channels, scale_factor=2):
        super(OptimizedUpscaleBlock, self).__init__()

        self.conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                in_channels * (scale_factor ** 2),
                kernel_size=3
            ),
            nn.PixelShuffle(scale_factor)
        )

        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class OptimizedXRaySR(nn.Module):
    """优化后的X射线图像超分辨率网络"""

    def __init__(self, num_channels=32, num_blocks=4, debug_mode=False):
        super(OptimizedXRaySR, self).__init__()
        self.debug_mode = debug_mode
        self.use_checkpoint = False

        # 初始特征提取
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )

        # 特征提取主干
        trunk = []
        for _ in range(num_blocks):
            trunk.append(OptimizedResidualBlock(num_channels))
        self.trunk = nn.Sequential(*trunk)

        # 最后的卷积层
        self.final_conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # 上采样模块 (4x)
        self.upsampler = nn.Sequential(
            OptimizedUpscaleBlock(num_channels),  # 2x
            OptimizedUpscaleBlock(num_channels)  # 4x
        )

        # 最终重建
        self.reconstruction = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

    def _forward_trunk(self, x):
        """使用检查点的主干前向传播"""
        if self.use_checkpoint and self.training:
            for block in self.trunk:
                x = torch.utils.checkpoint.checkpoint(block, x)
        else:
            x = self.trunk(x)
        return x

    def forward(self, x):
        # 特征提取
        feat1 = self.first_conv(x)

        # 主干处理
        feat2 = self._forward_trunk(feat1)
        feat2 = self.final_conv(feat2)
        feat2 = feat2 + feat1  # 全局残差连接

        # 上采样和重建
        out = self.upsampler(feat2)
        out = self.reconstruction(out)

        return out

    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self.use_checkpoint = True

    def initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.normal_(m.weight, mean=0.1, std=0.02)
class OptimizedCombinedLoss(nn.Module):
    """优化的组合损失函数"""

    def __init__(self, lambda_char=1.0, lambda_ssim=1.0, lambda_perceptual=0.1):
        super(OptimizedCombinedLoss, self).__init__()
        self.charbonnier = L1CharbonnierLoss()
        self.ssim = SSIM()

        # 优化VGG感知损失
        try:
            vgg = models.vgg19(pretrained=True).features[:36].eval()
            # 冻结VGG参数
            for param in vgg.parameters():
                param.requires_grad = False

            # 优化：只使用关键层
            self.vgg_layers = nn.ModuleList([
                vgg[:4],  # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:18]  # relu3_4
            ])
            self.use_perceptual = True
        except Exception as e:
            print(f"VGG感知损失初始化失败: {str(e)}, 将不使用感知损失")
            self.use_perceptual = False

        self.lambda_char = lambda_char
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual if self.use_perceptual else 0.0

        # 注册归一化参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """归一化函数"""
        x = torch.clamp(x, 0, 1)
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def perceptual_loss(self, x, target):
        """优化的感知损失计算"""
        if not self.use_perceptual:
            return torch.tensor(0.0).to(x.device)

        x = self.normalize(x)
        target = self.normalize(target)

        loss = 0
        weights = [1.0 / 4, 1.0 / 2, 1.0]  # 不同层的权重

        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            with torch.no_grad():
                target = layer(target)
            loss += weights[i] * F.l1_loss(x, target)

        return loss

    def forward(self, pred, target):
        # 计算各个损失
        loss_char = self.charbonnier(pred, target)
        loss_ssim = 1 - self.ssim(pred, target)
        loss_perceptual = self.perceptual_loss(pred, target) if self.use_perceptual else 0.0

        # 合并损失
        total_loss = (self.lambda_char * loss_char +
                      self.lambda_ssim * loss_ssim +
                      self.lambda_perceptual * loss_perceptual)

        return total_loss, {
            'charbonnier': loss_char.item(),
            'ssim': loss_ssim.item(),
            'perceptual': loss_perceptual.item() if self.use_perceptual else 0.0
        }

class L1CharbonnierLoss(nn.Module):
    """Charbonnier损失函数"""

    def __init__(self, epsilon=1e-3):
        super(L1CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()

class SSIM(nn.Module):
    """优化的结构相似性计算模块"""

    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1

        # 创建高斯窗口
        window = torch.Tensor([exp(-(x - self.window_size // 2) ** 2 / float(2 * 1.5 ** 2))
                               for x in range(self.window_size)])
        window = window.unsqueeze(1) * window.unsqueeze(0)
        window = window / window.sum()
        self.register_buffer('window', window.unsqueeze(0).unsqueeze(0))

    @torch.jit.script_method
    def forward(self, img1, img2):
        """优化的SSIM计算"""
        # 确保输入为float类型
        img1 = img1.float()
        img2 = img2.float()

        # 计算均值
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # SSIM常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()