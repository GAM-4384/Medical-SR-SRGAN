import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Tes.tes0_Root.model import XRaySR, SSIM
from Tes.tes0_Root.config import get_training_config, setup_training_device

class ModelTester:
    def __init__(self, checkpoint_path, test_lr_dir, test_hr_dir, save_dir='test_results',
                 sample_size=200, random_seed=42):
        self.checkpoint_path = checkpoint_path
        self.test_lr_dir = test_lr_dir
        self.test_hr_dir = test_hr_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.setup_logging()
        self.config = get_training_config()
        self.device = setup_training_device()
        self.setup_model()
        self.setup_metrics()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
        self.use_amp = False
        self.test_files = self.get_test_files()
        self.logger.info(f"Selected {len(self.test_files)} images for testing")
    def setup_logging(self):
        """配置日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.save_dir / f'test_results_{timestamp}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_model(self):
        """设置模型并加载检查点，处理通道数不匹配和缺失参数的情况"""
        try:
            self.model = XRaySR(
                num_channels=self.config['model']['num_channels'],
                num_blocks=self.config['model']['num_blocks']
            ).to(self.device)

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    epoch = checkpoint.get('epoch', 'unknown')
                else:
                    state_dict = checkpoint
                    epoch = 'unknown'
            else:
                state_dict = checkpoint
                epoch = 'unknown'

            # 获取当前模型的状态字典
            model_state_dict = self.model.state_dict()

            # 创建新的状态字典
            new_state_dict = {}

            # 遍历现有模型的参数
            for k, v in model_state_dict.items():
                if k in state_dict:#检查参数是否存在于加载的检查点中。如果存在，获取旧参数值
                    old_param = state_dict[k]
                    if old_param.shape == v.shape:#如果新旧参数形状完全相同，直接使用旧参数
                        new_state_dict[k] = old_param
                    else:
                        # 处理通道数不匹配的情况
                        if len(v.shape) == 4:  # 卷积层权重 输出通道数，输入通道数，卷积核高度，卷积核宽度
                            out_channels, in_channels, kh, kw = v.shape
                            #识别参数类型（确认是卷积层参数）
                            #提取新旧模型的具体维度信息
                            #为后续的维度调整做准备，方便比较和修改
                            old_out, old_in, old_kh, old_kw = old_param.shape

                            # 调整输出通道数 比较新模型和旧模型的输出通道数是否不同
                            if out_channels != old_out:
                                # 如果新的通道数更少，截取前面的通道
                                if out_channels < old_out:
                                    old_param = old_param[:out_channels]
                                # 如果新的通道数更多，重复已有通道
                                else:
                                    repeat_times = out_channels // old_out  # 整除得到完整重复次数
                                    remainder = out_channels % old_out      # 取余得到需要额外补充的通道数
                                    old_param = torch.cat([
                                        old_param.repeat(repeat_times, 1, 1, 1), #                                                                                                                                                                                                        重复已有通道
                                        old_param[:remainder]               # 补充剩余通道
                                    ])

                            # 调整输入通道数
                            if in_channels != old_in:
                                old_param = old_param.transpose(0, 1)
                                if in_channels < old_in:
                                    old_param = old_param[:in_channels]
                                else:
                                    repeat_times = in_channels // old_in
                                    remainder = in_channels % old_in
                                    old_param = torch.cat([
                                        old_param.repeat(repeat_times, 1, 1, 1),
                                        old_param[:remainder]
                                    ])
                                old_param = old_param.transpose(0, 1)

                            # 调整kernel size
                            if kh != old_kh or kw != old_kw:
                                old_param = F.interpolate(
                                    old_param.view(-1, 1, old_kh, old_kw),
                                    size=(kh, kw),
                                    mode='bilinear',
                                    align_corners=False
                                ).view(out_channels, in_channels, kh, kw)

                            new_state_dict[k] = old_param

                        elif len(v.shape) == 1:  # 批归一化层参数或偏置项
                            if v.shape[0] != old_param.shape[0]:
                                if v.shape[0] < old_param.shape[0]:
                                    new_state_dict[k] = old_param[:v.shape[0]]
                                else:
                                    repeat_times = v.shape[0] // old_param.shape[0]
                                    remainder = v.shape[0] % old_param.shape[0]
                                    new_state_dict[k] = torch.cat([
                                        old_param.repeat(repeat_times),
                                        old_param[:remainder]
                                    ])
                        else:
                            self.logger.warning(f"Parameter {k} shape mismatch: {old_param.shape} vs {v.shape}")
                            new_state_dict[k] = v
                else:
                    self.logger.info(f"Initializing new parameter {k}")
                    new_state_dict[k] = v

            # 加载调整后的参数
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model = self.model.float()
            self.model.eval()

            self.logger.info(f"Successfully loaded and adapted checkpoint (epoch: {epoch})")
            self.logger.info("Parameter adaptation summary:")
            self.logger.info(f"Total parameters in model: {len(model_state_dict)}")
            self.logger.info(f"Parameters in checkpoint: {len(state_dict)}")
            self.logger.info(f"Parameters adapted: {len(new_state_dict)}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def setup_metrics(self):
        """这段代码是设置模型评估指标的函数，设置了三个常用的图像质量评估指标："""
        self.ssim_module = SSIM().to(self.device)
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()


    def calculate_nrmse(self, sr_img, hr_img):
        """计算归一化均方根误差"""
        mse = F.mse_loss(sr_img, hr_img) # 计算两个图像之间的像素差的平方的平均值
        rmse = torch.sqrt(mse)
        range_max = torch.max(hr_img) # 计算数值范围：
        range_min = torch.min(hr_img)
        range_val = range_max - range_min
        if range_val == 0: # 如果数值范围为0（图像完全相同），返回0
            return torch.tensor(0.0).to(self.device)
        return rmse / range_val


    def calculate_perceptual_loss(self, sr_img, hr_img):
        """计算感知损失"""
        def sobel_edges(img):
            sobel_x = torch.tensor([[-1, 0, 1],#水平方向的Sobel算子，用于检测垂直边缘
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=torch.float32).to(self.device)
            sobel_y = torch.tensor([[-1, -2, -1], # 垂直方向的Sobel算子，用于检测水平边缘
                                  [0, 0, 0],
                                  [1, 2, 1]], dtype=torch.float32).to(self.device)
            sobel_x = sobel_x.view(1, 1, 3, 3) # 将算子转换为卷积层可用的形状：[输出通道，输入通道，核高度，核宽度]
            sobel_y = sobel_y.view(1, 1, 3, 3)
            edges_x = F.conv2d(img, sobel_x, padding=1)# 使用卷积操作分别计算x和y方向的梯度
            edges_y = F.conv2d(img, sobel_y, padding=1)# padding=1 保持输出尺寸与输入相同
            return torch.sqrt(edges_x.pow(2) + edges_y.pow(2))# 计算边缘强度
        # 计算感知损失
        sr_edges = sobel_edges(sr_img)
        hr_edges = sobel_edges(hr_img)
        return F.mse_loss(sr_edges, hr_edges)

    def calculate_uiqi(self, sr_img, hr_img, window_size=8):
        """计算通用图像质量指数 (UIQI)"""
        try:
            # 确保输入是4D张量 [批次B, 通道C, 高度H, 宽度W]
            #基本设置和输入预处理
            if sr_img.dim() != 4:
                sr_img = sr_img.unsqueeze(0)
            if hr_img.dim() != 4:
                hr_img = hr_img.unsqueeze(0)
            # 确保是float32类型 确保计算精度
            sr_img = sr_img.float()
            hr_img = hr_img.float()
            # 将图像展平为二维数组 [批次, 像素]
            N = window_size ** 2
            sr = sr_img.view(sr_img.size(0), -1)
            hr = hr_img.view(hr_img.size(0), -1)
            # 计算均值
            sr_mean = sr.mean(dim=1, keepdim=True)
            hr_mean = hr.mean(dim=1, keepdim=True)
            # 计算方差和协方差
            sr_var = ((sr - sr_mean) ** 2).mean(dim=1, keepdim=True)
            hr_var = ((hr - hr_mean) ** 2).mean(dim=1, keepdim=True)
            sr_hr_cov = ((sr - sr_mean) * (hr - hr_mean)).mean(dim=1, keepdim=True)
            # 这些统计量用于评估图像的结构相似性

            # 计算UIQI
            numerator = 4 * sr_hr_cov * sr_mean * hr_mean
            denominator = (sr_var + hr_var) * (sr_mean ** 2 + hr_mean ** 2)
            # 避免除零
            denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-8, denominator)
            quality = numerator / denominator
            return quality.mean()
        except Exception as e:
            self.logger.error(f"Error calculating UIQI: {str(e)}")
            return torch.tensor(0.0).to(self.device)

    def calculate_metrics(self, sr_img, hr_img):
        """计算多个评估指标"""
        try:
            metrics = {}
            # 确保数据在同一设备上且为float32类型
            sr_img = sr_img.float().to(self.device)
            hr_img = hr_img.float().to(self.device)
            # 计算各种指标
            mse = self.mse_criterion(sr_img, hr_img)
            metrics['psnr'] = 10 * torch.log10(1 / (mse + 1e-8))
            metrics['ssim'] = self.ssim_module(sr_img, hr_img)
            metrics['l1'] = self.l1_criterion(sr_img, hr_img)
            metrics['nrmse'] = self.calculate_nrmse(sr_img, hr_img)
            metrics['perceptual_loss'] = self.calculate_perceptual_loss(sr_img, hr_img)
            metrics['uiqi'] = self.calculate_uiqi(sr_img, hr_img)
            return {k: v.item() for k, v in metrics.items()} # 将所有张量值转换为Python标量
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return None
    # 在ModelTester类中修改以下两个方法

    def plot_metrics_summary(self, results, save_path=None):
        """生成评估指标的可视化图表"""
        try:
            # 从结果中提取各项指标的数据 创建一个字典来存储六种不同的评估指标数据
            metrics_data = {
                'PSNR(峰值信噪比)': [],
                'SSIM(结构相似度)': [],
                'L1(平均绝对误差)': [],
                'NRMSE(归一化均方根误差)': [],
                'Perceptual Loss(感知损失)': [],
                'UIQI(通用图像质量指数)': []
            }

            # 收集所有图片的指标数据
            for img_results in results['individual_results'].values():
                metrics_data['PSNR(峰值信噪比)'].append(img_results['psnr'])
                metrics_data['SSIM(结构相似度)'].append(img_results['ssim'])
                metrics_data['L1(平均绝对误差)'].append(img_results['l1'])
                metrics_data['NRMSE(归一化均方根误差)'].append(img_results['nrmse'])
                metrics_data['Perceptual Loss(感知损失)'].append(img_results['perceptual_loss'])
                metrics_data['UIQI(通用图像质量指数)'].append(img_results['uiqi'])

            #设置图表样式
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

            # 创建一个3x2的子图布局
            fig, axs = plt.subplots(3, 2, figsize=(15, 20))
            fig.suptitle('评估指标分布总览', size=16, y=0.95)

            # 为每个指标创建组合图 小提琴图绘制
            for (metric_name, metric_values), ax in zip(metrics_data.items(), axs.flat):
                parts = ax.violinplot(metric_values, points=100, widths=0.7,
                                      showmeans=True, showextrema=True, showmedians=True)

                # 设置颜色
                for pc in parts['bodies']:
                    pc.set_facecolor('#3498db')
                    pc.set_alpha(0.6)
                parts['cmeans'].set_color('#e74c3c')
                parts['cmedians'].set_color('#2ecc71')

                # 添加box plot 箱线图叠加：
                bp = ax.boxplot(metric_values, positions=[1], widths=0.15,# 箱线图的位置，设为1使其与小提琴图重叠
                                patch_artist=True, showfliers=True) #显示离群点
                bp['boxes'][0].set_facecolor('#f39c12') # 橙色填充
                bp['boxes'][0].set_alpha(0.5)

                # 添加统计信息
                mean_val = np.mean(metric_values)
                median_val = np.median(metric_values)
                std_val = np.std(metric_values)

                stats_text = f'平均值: {mean_val:.4f}\n中位数: {median_val:.4f}\n标准差: {std_val:.4f}'
                ax.text(1.4, np.median(metric_values), stats_text,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

                # 设置标题和样式
                ax.set_title(f'{metric_name}分布', pad=20)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xticks([])

            plt.tight_layout()

            # 保存图表
            if save_path is None:
                save_path = self.save_dir / 'metrics_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"指标分布图已保存至 {save_path}")

        except Exception as e:
            self.logger.error(f"生成指标分布图时出错: {str(e)}")

    def save_comparison_plot(self, lr_img, sr_img, hr_img, save_path):
        """保存对比图"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

            plt.figure(figsize=(15, 5))

            images = [] #将每个输入图像转换为 NumPy 数组
            for img in [lr_img, sr_img, hr_img]:
                img_np = img.float().squeeze().cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                images.append(img_np)

            titles = ['低分辨率', '超分辨率重建', '高分辨率原图']

            for i, (img, title) in enumerate(zip(images, titles)):
                plt.subplot(1, 3, i + 1)
                plt.imshow(img, cmap='gray')
                plt.title(title)
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            self.logger.error(f"保存对比图时出错: {str(e)}")


    def get_test_files(self):
        """获取要测试的图片文件列表"""
        try:
            all_files = [f for f in os.listdir(self.test_lr_dir)
                        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if not all_files:
                raise ValueError(f"No valid image files found in {self.test_lr_dir}")
            if len(all_files) > self.sample_size:
                #如果文件总数超过预设的样本大小(self.sample_size)  使用random.sample随机选择指定数量的文件
                selected_files = random.sample(all_files, self.sample_size)
                self.logger.info(f"Randomly sampled {self.sample_size} images from {len(all_files)} total images")
            else:
                #如果文件数量不足样本大小 使用所有可用文件
                selected_files = all_files
                self.logger.info(f"Using all {len(selected_files)} available images")
            return selected_files
        except Exception as e:
            self.logger.error(f"Error while getting test files: {str(e)}")
            return []

    def test_single_image(self, img_name):
        """测试单张图像"""
        try:
            lr_path = os.path.join(self.test_lr_dir, img_name)
            hr_path = os.path.join(self.test_hr_dir, img_name)
            if not os.path.exists(lr_path) or not os.path.exists(hr_path):
                self.logger.error(f"Image file not found: {img_name}")
                return None
            # 加载和转换图像，确保数据类型为float32
            lr_img = self.transform(Image.open(lr_path).convert('L')).unsqueeze(0)
            hr_img = self.transform(Image.open(hr_path).convert('L')).unsqueeze(0)
            # 明确转换为float32类型并移动到GPU
            lr_img = lr_img.float().to(self.device)
            hr_img = hr_img.float().to(self.device)
            # 确保模型也是float32类型
            self.model = self.model.float()
            with torch.no_grad():
                sr_img = self.model(lr_img)

            metrics = self.calculate_metrics(sr_img, hr_img)
            if metrics is not None:
                plot_path = self.save_dir / f'comparison_{os.path.splitext(img_name)[0]}.png'
                self.save_comparison_plot(lr_img, sr_img, hr_img, plot_path)
            return metrics

        except Exception as e:
            self.logger.error(f"Error processing image {img_name}: {str(e)}")
            return None

    def test_model(self):
        """测试模型性能"""
        if not self.test_files:
            self.logger.error("No test files available")
            return None
        results = {
            'individual_results': {},
            'average_metrics': {},
            'test_config': {
                'checkpoint_path': str(self.checkpoint_path),
                'test_lr_dir': str(self.test_lr_dir),
                'test_hr_dir': str(self.test_hr_dir),
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(self.test_files)
            }
        }
        successful_tests = 0
        metrics_sum = {}
        for img_name in tqdm(self.test_files, desc="Testing images"):
            metrics = self.test_single_image(img_name)
            if metrics is not None:
                results['individual_results'][img_name] = metrics
                successful_tests += 1
                for metric_name, value in metrics.items():
                    metrics_sum[metric_name] = metrics_sum.get(metric_name, 0) + value
        if successful_tests > 0:
            results['average_metrics'] = {
                metric_name: value / successful_tests

                for metric_name, value in metrics_sum.items()
            }
            self.logger.info(f"\nSuccessfully processed {successful_tests}/{len(self.test_files)} images")
            self.logger.info("\nTest Results:")
            for metric_name, value in results['average_metrics'].items():
                self.logger.info(f"Average {metric_name.upper()}: {value:.4f}")
            try:
                # 保存JSON结果
                with open(self.save_dir / 'test_results.json', 'w') as f:
                    json.dump(results, f, indent=4)
                # 生成并保存指标分布图
                self.plot_metrics_summary(results)
            except Exception as e:
                self.logger.error(f"Error saving results: {str(e)}")
        return results

def main():
    torch.manual_seed(42)#设置随机种子
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    checkpoint_path = r"E:\PY SRRESNET\checkpoints\tes4\best_model.pth"
    test_lr_dir = r"E:\PY SRRESNET\processed\val\lr"
    test_hr_dir = r"E:\PY SRRESNET\processed\val\hr"
    try:
        torch.set_num_threads(1)
        tester = ModelTester(
            checkpoint_path=checkpoint_path,
            test_lr_dir=test_lr_dir,
            test_hr_dir=test_hr_dir,
            sample_size=200
        )
        results = tester.test_model()
        if results is None:
            logging.error("Testing failed to produce results")
    except Exception as e:
        logging.error(f"Testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()