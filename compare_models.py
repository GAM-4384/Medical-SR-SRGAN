import argparse
from tkinter import filedialog, Tk
import tkinter as tk
from pathlib import Path
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
import seaborn as sns
from model_test import ModelTester


class ModelSelector:
    def __init__(self):
        self.selected_models = []
        # 添加日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def gui_select_models(self):
        """通过图形界面选择模型"""
        root = Tk()
        root.withdraw()  # 隐藏主窗口
        default_checkpoint_dir = Path.cwd() / 'checkpoints'
        while True:
            model_path = filedialog.askopenfilename(
                title='选择模型检查点文件',
                filetypes=[('PyTorch模型', '*.pth'), ('所有文件', '*.*')],
                initialdir=str(default_checkpoint_dir)
            )

            if not model_path:  # 用户取消选择
                break

            model_name = Path(model_path).stem
            self.selected_models.append({
                'path': model_path,
                'name': model_name
            })

            self.logger.info(f"已选择模型: {model_name}")

            # 创建新窗口询问是否继续
            root_confirm = Tk()
            root_confirm.title("确认")
            root_confirm.geometry("300x100")
            root_confirm.lift()  # 将窗口提到前面
            root_confirm.focus_force()  # 强制获取焦点

            continue_selection = tk.BooleanVar(value=False)

            def on_yes():
                continue_selection.set(True) # 设置继续选择标志为True
                root_confirm.quit()          # 退出主循环
                root_confirm.destroy()       # 销毁确认窗口

            def on_no():
                continue_selection.set(False)
                root_confirm.quit()
                root_confirm.destroy()

            # 创建提示标签，显示当前选择的模型名称并询问是否继续
            label = tk.Label(root_confirm, text=f"已选择模型: {model_name}\n是否继续添加模型？")
            label.pack(pady=10) # 设置垂直间距为10像素

            # 创建"是"和"否"按钮
            yes_btn = tk.Button(root_confirm, text="是", command=on_yes)
            no_btn = tk.Button(root_confirm, text="否", command=on_no)
            # 将按钮放置在窗口左右两侧，水平间距为50像素
            yes_btn.pack(side=tk.LEFT, padx=50)
            no_btn.pack(side=tk.RIGHT, padx=50)

            root_confirm.mainloop()

            if not continue_selection.get():
                self.logger.info("结束选择模型")
                break

        root.destroy()

        # 确保至少选择了一个模型
        if not self.selected_models:
            self.logger.warning("未选择任何模型！")
            return []

        self.logger.info(f"共选择了 {len(self.selected_models)} 个模型:")
        for model in self.selected_models:
            self.logger.info(f"- {model['name']}: {model['path']}")

        return self.selected_models

class ModelComparator:
    def __init__(self, model_configs, test_lr_dir, test_hr_dir, save_dir='comparison_results',
                 sample_size=50):
        """
        初始化模型比较器
        Args:
            model_configs: 要比较的模型配置列表，每个配置包含模型路径和名称
                         格式：[{'path': 'path/to/checkpoint', 'name': 'model_name'}, ...]
            test_lr_dir: 测试用低分辨率图像目录
            test_hr_dir: 测试用高分辨率图像目录
            save_dir: 结果保存目录
            sample_size: 用于测试的图片数量
        """
        self.model_configs = model_configs
        self.test_lr_dir = test_lr_dir
        self.test_hr_dir = test_hr_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

        self.setup_logging()

    def setup_logging(self):
        """配置日志系统"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.save_dir / f'comparison_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_comparison(self):
        """运行模型比较"""
        results = {}
        failed_models = []

        # 测试每个模型
        for config in self.model_configs:
            self.logger.info(f"测试模型: {config['name']}")

            try:
                # 创建模型测试器
                tester = ModelTester(
                    checkpoint_path=config['path'],
                    test_lr_dir=self.test_lr_dir,
                    test_hr_dir=self.test_hr_dir,
                    sample_size=self.sample_size,
                    save_dir=self.save_dir / config['name']
                )

                # 运行测试并保存结果
                model_results = tester.test_model()
                if model_results is not None:
                    results[config['name']] = model_results
                else:
                    failed_models.append(config['name'])

            except Exception as e:
                self.logger.error(f"测试模型 {config['name']} 时出错: {str(e)}")
                failed_models.append(config['name'])
                continue

        if failed_models:
            self.logger.warning(f"以下模型测试失败: {', '.join(failed_models)}")

        if results:
            # 生成比较报告
            self.logger.info("开始生成比较报告...")
            self.generate_comparison_report(results)
            self.logger.info("比较报告生成完成")
        else:
            self.logger.error("所有模型测试都失败了，无法生成比较报告")

        return results

    def generate_comparison_report(self, results):
        """生成模型比较报告"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self._plot_metrics_comparison(results)
        self._plot_metrics_boxplot(results)
        self._save_comparison_table(results)
        self._generate_statistical_tests(results)

    def _plot_metrics_comparison(self, results):
        """绘制指标对比图"""
        metrics = ['psnr', 'ssim', 'l1', 'nrmse', 'perceptual_loss', 'uiqi']
        metric_names = {
            'psnr': 'PSNR(峰值信噪比)',
            'ssim': 'SSIM(结构相似度)',
            'l1': 'L1(平均绝对误差)',
            'nrmse': 'NRMSE(归一化均方根误差)',
            'perceptual_loss': 'Perceptual Loss(感知损失)',
            'uiqi': 'UIQI(通用图像质量指数)'
        }

        fig, axs = plt.subplots(3, 2, figsize=(15, 20))
        fig.suptitle('模型性能对比', size=16)

        for (metric, ax) in zip(metrics, axs.flat):
            metric_values = []
            model_names = []

            for model_name, model_results in results.items():
                values = [r[metric] for r in model_results['individual_results'].values()]
                metric_values.extend(values)
                model_names.extend([model_name] * len(values))

            sns.violinplot(x=model_names, y=metric_values, ax=ax)
            ax.set_title(f'{metric_names[metric]}对比')
            ax.set_xlabel('模型')
            ax.set_ylabel('指标值')
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metrics_boxplot(self, results):
        """绘制箱线图对比"""
        metrics = ['psnr', 'ssim', 'uiqi']  # 选择主要指标
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        metric_names = {
            'psnr': 'PSNR(峰值信噪比)',
            'ssim': 'SSIM(结构相似度)',
            'uiqi': 'UIQI(通用图像质量指数)'
        }

        import matplotlib
        mpl_version = matplotlib.__version__
        is_new_mpl = tuple(map(int, mpl_version.split('.'))) >= (3, 9)

        for idx, metric in enumerate(metrics):
            data = []
            labels = []

            for model_name, model_results in results.items():
                values = [r[metric] for r in model_results['individual_results'].values()]
                data.append(values)
                labels.append(model_name)

            # 根据 Matplotlib 版本使用相应的参数名
            if is_new_mpl:
                axs[idx].boxplot(data, tick_labels=labels)
            else:
                axs[idx].boxplot(data, labels=labels)

            axs[idx].set_title(f'{metric_names[metric]}')
            plt.setp(axs[idx].get_xticklabels(), rotation=45)

            # 添加网格线以提高可读性
            axs[idx].grid(True, linestyle='--', alpha=0.7)

            # 优化Y轴标签
            if metric == 'psnr':
                axs[idx].set_ylabel('分贝(dB)')
            elif metric == 'ssim' or metric == 'uiqi':
                axs[idx].set_ylabel('指标值')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
    def _save_comparison_table(self, results):
        """保存比较结果表格"""
        comparison_table = {
            'model_comparison': {
                model_name: results[model_name]['average_metrics']
                for model_name in results.keys()
            }
        }

        with open(self.save_dir / 'comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_table, f, indent=4, ensure_ascii=False)

    def _generate_statistical_tests(self, results):
        """生成统计测试结果"""
        from scipy import stats

        metrics = ['psnr', 'ssim', 'uiqi'] # 要比较的三个评估指标
        model_names = list(results.keys()) # 获取所有模型名称
        stat_results = {}

        for metric in metrics:
            stat_results[metric] = {}
            # 两两比较所有模型
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]

                    # 提取两个模型在特定指标上的所有测试结果
                    values1 = [r[metric] for r in results[model1]['individual_results'].values()]
                    values2 = [r[metric] for r in results[model2]['individual_results'].values()]

                    # 执行t检验
                    t_stat, p_value = stats.ttest_ind(values1, values2)

                    stat_results[metric][f'{model1}_vs_{model2}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value)
                    }
        # 将统计检验结果保存为JSON文件
        with open(self.save_dir / 'statistical_tests.json', 'w', encoding='utf-8') as f:
            json.dump(stat_results, f, indent=4, ensure_ascii=False)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("启动模型比较工具...")

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='模型比较工具')
    parser.add_argument('--mode', choices=['gui', 'cli'], default='gui',
                        help='选择模式：gui(图形界面) 或 cli(命令行)')
    parser.add_argument('--models', nargs='*',
                        help='模型路径列表 (仅CLI模式使用)')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='用于测试的图片数量')
    parser.add_argument('--test-lr-dir',
                        default=r"E:\PY SRRESNET\processed\val\lr",
                        help='测试用低分辨率图像目录')
    parser.add_argument('--test-hr-dir',
                        default=r"E:\PY SRRESNET\processed\val\hr",
                        help='测试用高分辨率图像目录')

    args = parser.parse_args()

    try:
        # 根据模式选择模型
        if args.mode == 'gui':
            logger.info("使用图形界面模式选择模型...")
            selector = ModelSelector()
            model_configs = selector.gui_select_models()
        else:  # CLI模式
            logger.info("使用命令行模式选择模型...")
            if not args.models:
                parser.error("CLI模式需要指定模型路径")
            model_configs = [
                {'path': path, 'name': Path(path).stem}
                for path in args.models
            ]

        if not model_configs:
            logger.error("未选择任何模型，程序退出")
            return

        logger.info("\n选择的模型:")
        for config in model_configs:
            logger.info(f"- {config['name']}: {config['path']}")

        # 创建比较器并运行比较
        logger.info("开始创建模型比较器...")
        comparator = ModelComparator(
            model_configs=model_configs,
            test_lr_dir=args.test_lr_dir,
            test_hr_dir=args.test_hr_dir,
            sample_size=args.sample_size
        )

        # 运行比较
        logger.info("开始运行模型比较...")
        results = comparator.run_comparison()

        logger.info("模型比较完成！")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()