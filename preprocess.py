import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import shutil
import matplotlib.pyplot as plt


def plot_dataset_statistics(train_count, test_count, val_count, save_dir):
    """绘制数据集数量统计柱状图并保存"""
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    categories = ['训练集', '测试集', '验证集']
    counts = [train_count, test_count, val_count]

    bars = plt.bar(categories, counts, color=['skyblue', 'lightgreen', 'salmon'])

    plt.title('数据集分布统计')
    plt.xlabel('数据集类型')
    plt.ylabel('图片数量')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    try:
        save_path = os.path.join(save_dir, 'dataset_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"柱状图已成功保存到: {save_path}")
    except Exception as e:
        print(f"保存图表时发生错误: {e}")
    finally:
        plt.close()  # 关闭图表，释放内存
class ImageDegrader:
    """图像降质类，模拟低剂量X射线成像的特征"""

    def __init__(self,
                 noise_level=(10, 25),
                 blur_range=(0.5, 1.5),
                 contrast_range=(0.7, 0.9),
                 scale_factor=4):
        self.noise_level = noise_level
        self.blur_range = blur_range
        self.contrast_range = contrast_range
        self.scale_factor = scale_factor

    def add_noise(self, img):
        noise_sigma = random.uniform(*self.noise_level)
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def reduce_contrast(self, img):
        contrast_factor = random.uniform(*self.contrast_range)
        mean_value = np.mean(img)
        reduced_contrast = (img - mean_value) * contrast_factor + mean_value
        return np.clip(reduced_contrast, 0, 255).astype(np.uint8)

    def apply_blur(self, img):
        kernel_size = random.uniform(*self.blur_range)
        if kernel_size < 1:
            return img
        kernel_size = int(kernel_size * 2) * 2 + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def reduce_resolution(self, img):
        h, w = img.shape
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor
        return cv2.resize(img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

    def degrade_image(self, hr_img):
        img = np.array(hr_img)
        img = self.reduce_contrast(img)
        img = self.add_noise(img)
        img = self.apply_blur(img)
        img = self.reduce_resolution(img)
        return Image.fromarray(img)


def create_dirs(base_dir):
    """创建必要的目录结构"""
    dirs = [
        os.path.join(base_dir, 'train/hr'),
        os.path.join(base_dir, 'train/lr'),
        os.path.join(base_dir, 'val/hr'),
        os.path.join(base_dir, 'val/lr'),
        os.path.join(base_dir, 'test/hr'),
        os.path.join(base_dir, 'test/lr'),
        os.path.join(base_dir, 'statistics')  # 添加统计数据目录
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def process_image(img_path, save_hr_path, save_lr_path, degrader):
    """处理单张图像，生成HR-LR对"""
    try:
        img = Image.open(img_path).convert('L')
        w, h = img.size
        w = w - (w % degrader.scale_factor)
        h = h - (h % degrader.scale_factor)
        if w == 0 or h == 0:
            return False

        img = img.resize((w, h))
        img_array = np.array(img)
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        hr_img = Image.fromarray(img_array)
        lr_img = degrader.degrade_image(hr_img)

        hr_img.save(save_hr_path)
        lr_img.save(save_lr_path)
        return True

    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        return False


def collect_all_images(train_folders, raw_dir):
    """收集所有训练图像路径"""
    all_images = []
    for folder in train_folders:
        images_dir = os.path.join(raw_dir, 'train', folder, 'images')
        if os.path.exists(images_dir):
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            all_images.extend([(folder, f) for f in image_files])
    return all_images


def main(test_mode=False, train_ratio=0.8):
    """主函数：处理和组织图像数据集

    Args:
        test_mode (bool): 是否运行测试模式（仅处理少量图像）
        train_ratio (float): 训练集占比（0-1之间）
    """
    # 基础路径设置
    raw_dir = 'raw'
    processed_dir = 'processed'

    # 创建图像降质器
    degrader = ImageDegrader(
        noise_level=(10, 25),
        blur_range=(0.5, 1.5),
        contrast_range=(0.7, 0.9),
        scale_factor=4
    )

    # 创建必要的目录
    dirs = create_dirs(processed_dir)
    statistics_dir = os.path.join(processed_dir, 'statistics')
    os.makedirs(statistics_dir, exist_ok=True)

    # 训练文件夹列表
    train_folders = ['images_001', 'images_004', 'images_007']

    # 收集所有训练图像
    print("\n收集图像文件...")
    all_images = collect_all_images(train_folders, raw_dir)

    if test_mode:
        all_images = all_images[:20]
        print("测试模式：只处理20张图像")

    # 处理图像并进行训练集/测试集划分
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # 初始化计数器
    total_processed = 0
    train_count = 0
    test_count = 0
    failed_images = []

    # 处理训练集图像
    print("\n处理训练集图像...")
    for folder, img_name in tqdm(train_images, desc="处理训练集"):
        img_path = os.path.join(raw_dir, 'train', folder, 'images', img_name)
        save_hr_path = os.path.join(processed_dir, 'train/hr', img_name)
        save_lr_path = os.path.join(processed_dir, 'train/lr', img_name)

        try:
            if process_image(img_path, save_hr_path, save_lr_path, degrader):
                total_processed += 1
                train_count += 1
            else:
                failed_images.append(img_path)
        except Exception as e:
            print(f"\n处理图像失败 {img_path}: {str(e)}")
            failed_images.append(img_path)

    # 处理测试集图像
    print("\n处理测试集图像...")
    for folder, img_name in tqdm(test_images, desc="处理测试集"):
        img_path = os.path.join(raw_dir, 'train', folder, 'images', img_name)
        save_hr_path = os.path.join(processed_dir, 'test/hr', img_name)
        save_lr_path = os.path.join(processed_dir, 'test/lr', img_name)

        try:
            if process_image(img_path, save_hr_path, save_lr_path, degrader):
                total_processed += 1
                test_count += 1
            else:
                failed_images.append(img_path)
        except Exception as e:
            print(f"\n处理图像失败 {img_path}: {str(e)}")
            failed_images.append(img_path)

    # 处理验证集
    val_count = 0
    val_folder = os.path.join(raw_dir, 'val/images_012')

    if os.path.exists(val_folder):
        print("\n处理验证集...")
        val_images_dir = os.path.join(val_folder, 'images')

        if os.path.exists(val_images_dir):
            for img_name in tqdm(os.listdir(val_images_dir), desc="处理验证集"):
                if img_name.endswith('.png'):
                    img_path = os.path.join(val_images_dir, img_name)
                    save_hr_path = os.path.join(processed_dir, 'val/hr', img_name)
                    save_lr_path = os.path.join(processed_dir, 'val/lr', img_name)

                    try:
                        if process_image(img_path, save_hr_path, save_lr_path, degrader):
                            val_count += 1
                        else:
                            failed_images.append(img_path)
                    except Exception as e:
                        print(f"\n处理图像失败 {img_path}: {str(e)}")
                        failed_images.append(img_path)
        else:
            print(f"警告: 验证集图像目录不存在: {val_images_dir}")
    else:
        print(f"警告: 验证集文件夹不存在: {val_folder}")

    # 生成并保存数据集统计图
    try:
        print("\n生成数据集统计图...")
        plot_dataset_statistics(train_count, test_count, val_count, statistics_dir)
        print("统计图生成完成")
    except Exception as e:
        print(f"生成统计图时出错: {str(e)}")

    # 打印处理结果
    print("\n处理完成:")
    print(f"- 训练集: {train_count} 张图像")
    print(f"- 测试集: {test_count} 张图像")
    print(f"- 验证集: {val_count} 张图像")
    print(f"总计成功处理: {total_processed + val_count} 张图像")

    if failed_images:
        print(f"\n处理失败的图像数量: {len(failed_images)}")
        print("失败的图像列表已保存到 failed_images.txt")
        with open(os.path.join(statistics_dir, 'failed_images.txt'), 'w') as f:
            f.write('\n'.join(failed_images))

    # 返回处理结果
    return {
        'train_count': train_count,
        'test_count': test_count,
        'val_count': val_count,
        'total_processed': total_processed + val_count,
        'failed_count': len(failed_images)
    }

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)

    # 运行主程序
    main(test_mode=False, train_ratio=0.8)