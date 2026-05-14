import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.cuda.amp import autocast


class EnhancedResidualBlock(nn.Module):
    """增强型残差块，添加注意力机制"""

    def __init__(self, channels):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 4, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // 4, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Channel attention
        attn = self.avg_pool(out)
        attn = self.fc1(attn)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)

        out = out * attn + residual
        return out


class EnhancedXRaySR(nn.Module):
    """改进后的X射线超分辨率网络"""

    def __init__(self, num_channels=16, num_blocks=8):  # 修改默认通道数为16
        super(EnhancedXRaySR, self).__init__()

        # 初始特征提取
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=7, padding=3),  # 修改卷积核大小为7x7
            nn.PReLU()
        )

        # 残差块
        trunk = []
        for _ in range(num_blocks):
            trunk.append(EnhancedResidualBlock(num_channels))
        self.trunk = nn.Sequential(*trunk)

        # 特征融合
        self.fusion = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(num_channels)

        # 上采样模块
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # 最终重建
        self.final = nn.Conv2d(num_channels, 1, kernel_size=7, padding=3)  # 修改卷积核大小为7x7

        self.initialize_weights()

    def forward(self, x):
        """前向传播"""
        # 输入检查和归一化
        if x.max() > 1:
            x = x / 255.0

        feat1 = self.first_conv(x)
        trunk_feat = self.trunk(feat1)

        # 特征融合
        feat2 = self.fusion(trunk_feat)
        feat2 = self.fusion_bn(feat2)
        feat2 = feat2 + feat1  # 全局残差连接

        up_feat = self.upsampler(feat2)
        out = self.final(up_feat)

        # 确保输出在合理范围内
        out = torch.clamp(out, 0, 1)
        return out

    def initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class L1CharbonnierLoss(nn.Module):
    """Charbonnier损失函数（L1的改进版本）"""

    def __init__(self, epsilon=1e-3):
        super(L1CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """计算Charbonnier损失

        Args:
            pred: 预测图像
            target: 目标图像

        Returns:
            损失值
        """
        # 确保输入在正确的范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 计算差值
        diff = pred - target
        # 使用Charbonnier公式
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return loss.mean()

class SSIM(nn.Module):
    """改进的结构相似性计算模块"""

    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.register_buffer('window', self._create_window(window_size))
        # 预计算常数
        self.register_buffer('C1', torch.tensor(0.01 ** 2))
        self.register_buffer('C2', torch.tensor(0.03 ** 2))

    def forward(self, img1, img2):
        """计算SSIM值"""
        # 确保输入在0-1范围内
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        # 确保窗口在正确的设备上
        window = self.window.to(img1.device, dtype=img1.dtype)

        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=1)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=1) - mu1_mu2

        # SSIM计算
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = numerator / (denominator + 1e-6)  # 添加eps防止除零

        # 确保输出在0-1范围内
        ssim_value = ssim_map.mean()
        return torch.clamp(ssim_value, 0, 1)

    def _create_window(self, window_size):
        """创建高斯窗口"""
        _1D_window = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2))
                                   for x in range(window_size)])
        _1D_window = _1D_window / _1D_window.sum()
        _2D_window = _1D_window.unsqueeze(1) * _1D_window.unsqueeze(0)
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        return window


class CombinedLoss(nn.Module):
    """组合损失函数，专注于医学图像的结构保持"""

    def __init__(self, ssim_weight=0.7, l1_weight=0.2, edge_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.ssim_module = SSIM()
        self.l1_loss = L1CharbonnierLoss()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight

        # 注册Sobel算子缓冲区
        self.register_buffer('sobel_x', torch.FloatTensor([[1, 0, -1],
                                                           [2, 0, -2],
                                                           [1, 0, -1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', torch.FloatTensor([[1, 2, 1],
                                                           [0, 0, 0],
                                                           [-1, -2, -1]]).unsqueeze(0).unsqueeze(0))

    def edge_loss(self, pred, target):
        """计算边缘损失"""
        # 确保输入在0-1范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # 计算水平和垂直边缘
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

        # 计算边缘损失
        return F.l1_loss(pred_edge, target_edge)

    def forward(self, pred, target):
        """计算总损失"""
        # 确保输入在0-1范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        # SSIM损失 (1-SSIM因为需要最小化)
        ssim_loss = 1 - self.ssim_module(pred, target)

        # L1 Charbonnier损失
        l1_loss = self.l1_loss(pred, target)

        # 边缘损失
        edge_loss = self.edge_loss(pred, target)

        # 组合损失
        total_loss = (self.ssim_weight * ssim_loss +
                      self.l1_weight * l1_loss +
                      self.edge_weight * edge_loss)

        return total_loss


def create_model(config):
    """根据配置创建模型"""
    model = EnhancedXRaySR(
        num_channels=config['model']['num_channels'],
        num_blocks=config['model']['num_blocks']
    )
    return model


def load_model(model, checkpoint_path, device):
    """加载预训练模型"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 获取当前模型的状态字典
        model_state = model.state_dict()
        # 加载的预训练权重
        pretrained_state = checkpoint['model_state_dict']

        # 智能匹配权重
        matched_state_dict = {}
        for name, param in model_state.items():
            if name in pretrained_state:
                if param.shape == pretrained_state[name].shape:
                    matched_state_dict[name] = pretrained_state[name]
                else:
                    print(f"Shape mismatch for {name}: current {param.shape} vs loaded {pretrained_state[name].shape}")
                    matched_state_dict[name] = param
            else:
                print(f"Missing weight: {name}")
                matched_state_dict[name] = param

        # 加载匹配的权重
        model.load_state_dict(matched_state_dict, strict=False)
        print("Successfully loaded pretrained weights")

        return True
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return False

# 这一版本的效果并不是很好，过度专注ssim指标 psrn指标则是过低
