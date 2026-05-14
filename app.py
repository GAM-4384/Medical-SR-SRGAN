import sys
import os
import torch
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QFrame, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import Qt
from torchvision import transforms
from Tes.tes0_Root.model import XRaySR

# 配置文件路径
MODEL_PATH = r"E:\My struggle\PY SRRESNET\checkpoints\checkpoint_epoch_10.pth"
DEFAULT_IMAGE_DIR = r"E:\My struggle\PY SRRESNET\processed\val\lr"


class ImageDisplay(QLabel):
    """图像显示区域"""

    def __init__(self, title, hint="", parent=None):
        """初始化函数分析
        参数:
            title: 显示区域的标题
            hint: 提示文本，默认为空字符串
            parent: 父窗口组件，默认为None
        """
        # 调用父类(QLabel)的初始化方法
        super().__init__(parent)

        # 保存标题和提示文本到实例变量
        self.title = title  # 保存标题
        self.hint = hint  # 保存提示文本

        # 设置固定大小为500x500像素
        self.setFixedSize(500, 500)

        # 设置内容居中对齐
        self.setAlignment(Qt.AlignCenter)

        # 启用拖放功能
        self.setAcceptDrops(True)

        # 定义默认样式 - 定义了普通状态下的外观
        self.default_style = """
            QLabel {
                background-color: white;            # 背景色为白色
                border: 2px dashed #cccccc;        # 2像素虚线边框，颜色为淡灰色
                border-radius: 10px;               # 10像素圆角
            }
        """

        # 定义鼠标悬停样式 - 定义了拖动文件悬停时的外观
        self.hover_style = """
            QLabel {
                background-color: #f0f7ff;         # 背景色为浅蓝色
                border: 2px dashed #2d8cf0;        # 2像素虚线边框，颜色为蓝色
                border-radius: 10px;               # 10像素圆角
            }
        """

        # 应用默认样式
        self.setStyleSheet(self.default_style)

        # 显示初始提示信息
        self.showHint()

    def showHint(self):
        """显示提示信息方法分析
        功能：设置控件显示的HTML格式文本，包含标题、提示和操作说明
        """
        # 使用HTML格式设置文本内容，包含三个部分：
        self.setText(f"<div style='text-align:center;'>"  # 创建居中对齐的div容器
                     f"<p style='font-size:16px;font-weight:bold;'>{self.title}</p>"  # 显示标题，16px大小，加粗
                     f"<p style='color:#666666;'>{self.hint}</p>"  # 显示提示文本，灰色(#666666)
                     f"<p style='color:#888888;font-size:12px;'>支持拖拽图片到此处</p></div>")  # 显示操作提示，浅灰色，12px大小

    def dragEnterEvent(self, e: QDragEnterEvent):
        """拖动进入事件处理方法分析
        参数:
            e: QDragEnterEvent类型的事件对象
        功能：处理当用户开始拖动文件到控件上时的事件
        """
        # 检查拖入的数据是否包含URL（文件路径）
        if e.mimeData().hasUrls():
            # 接受拖放操作
            e.accept()
            # 切换到悬停样式，提供视觉反馈
            self.setStyleSheet(self.hover_style)
        else:
            # 如果不是文件，则忽略该事件
            e.ignore()

    def dragLeaveEvent(self, e):
        """拖动离开事件处理方法分析
        参数:
            e: 拖动离开事件对象
        功能：处理当拖动的文件离开控件区域时的事件
        """
        # 恢复到默认样式
        self.setStyleSheet(self.default_style)

    def dragMoveEvent(self, e):
        """拖动移动事件处理方法分析
        参数:
            e: 拖动移动事件对象
        功能：处理拖动过程中的事件
        """
        # 如果正在拖动的数据包含URL，则接受该事件
        if e.mimeData().hasUrls():
            e.accept()
        else:
            # 否则忽略该事件
            e.ignore()

    def dropEvent(self, e: QDropEvent):
        """拖放完成事件处理方法分析
        参数:
            e: QDropEvent类型的事件对象
        功能：处理文件放下时的事件
        """
        # 恢复默认样式
        self.setStyleSheet(self.default_style)

        # 获取拖放的文件URL列表
        urls = e.mimeData().urls()

        if urls:  # 如果存在URL
            # 获取第一个文件的本地路径
            path = urls[0].toLocalFile()

            # 检查文件扩展名是否为支持的图片格式
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                # 调用主窗口的加载图片方法
                # parent().parent().parent() 获取主窗口引用
                self.parent().parent().parent().loadImage(path)
            else:
                # 显示错误提示框
                QMessageBox.warning(self, "错误", "不支持的文件格式")

    def mousePressEvent(self, event):
        """鼠标按下事件处理方法分析
        参数:
            event: 鼠标事件对象
        功能：处理鼠标点击事件
        """
        # 检查是否为左键点击
        if event.button() == Qt.LeftButton:
            # 调用主窗口的选择图片方法
            self.parent().parent().parent().selectImage()
class XRayEnhancer(QMainWindow):
    def __init__(self):
        """XRayEnhancer类的初始化方法分析
        功能：初始化主窗口，设置UI和加载模型
        """
        # 调用父类QMainWindow的初始化方法
        super().__init__()

        # 初始化用户界面
        self.initUI()

        # 加载深度学习模型
        self.loadModel()

    def loadModel(self):
        """模型加载方法分析
        功能：加载并初始化深度学习模型
        """
        try:
            # 检测是否可用GPU，否则使用CPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 创建模型实例并移动到指定设备
            self.model = XRaySR(num_channels=16, num_blocks=3).to(self.device)

            # 加载模型权重
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 设置模型为评估模式
            self.model.eval()

            # 定义图像预处理转换
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # 将图像转换为张量
                transforms.ConvertImageDtype(torch.float32)  # 转换为浮点数类型
            ])

            print(f"模型加载成功, 使用设备: {self.device}")

        except Exception as e:
            # 如果加载失败，打印错误信息
            print(f"模型加载失败: {str(e)}")
            self.model = None
            # 显示错误提示框
            QMessageBox.critical(self, "错误", "模型加载失败，请检查模型文件路径")

    def initUI(self):
        """初始化用户界面方法的详细分析
        功能：设置并初始化应用程序的图形用户界面
        """

        # 1. 设置主窗口的基本属性
        self.setWindowTitle('X光图像超分辨率处理')  # 设置窗口标题
        self.setStyleSheet("background-color: #f5f5f5;")  # 设置窗口背景色为浅灰色
        self.resize(1200, 600)  # 设置窗口初始大小为1200x600像素

        # 2. 创建并设置中央部件
        central_widget = QWidget()  # 创建中央部件
        self.setCentralWidget(central_widget)  # 设置为主窗口的中央部件
        layout = QVBoxLayout(central_widget)  # 创建垂直布局管理器
        layout.setSpacing(20)  # 设置布局中部件之间的间距为20像素

        # 3. 创建标题标签
        title = QLabel("X光图像超分辨率处理")  # 创建标题标签
        title.setFont(QFont('Microsoft YaHei', 20, QFont.Bold))  # 设置标题字体为微软雅黑，20号，粗体
        title.setAlignment(Qt.AlignCenter)  # 设置标题居中对齐
        layout.addWidget(title)  # 将标题添加到布局中

        # 4. 创建图像处理区域的水平布局
        image_area = QHBoxLayout()  # 创建水平布局用于放置图像处理相关控件

        # 5. 创建输入图像区域
        input_container = QFrame()  # 创建输入区域容器
        # 设置输入区域样式：白色背景，圆角边框
        input_container.setStyleSheet("background-color: white; border-radius: 10px;")
        input_layout = QVBoxLayout(input_container)  # 创建输入区域的垂直布局
        # 创建输入图像显示控件
        self.input_display = ImageDisplay("输入图像", "拖拽图片到这里或点击选择文件")
        # 创建选择图片按钮
        input_btn = QPushButton("选择图片")
        # 设置按钮样式：蓝色背景，白色文字，圆角等
        input_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d8cf0;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2b85e4;
            }
        """)
        input_btn.clicked.connect(self.selectImage)  # 连接按钮点击事件到selectImage方法
        input_layout.addWidget(self.input_display)  # 添加显示控件到布局
        input_layout.addWidget(input_btn)  # 添加按钮到布局

        # 6. 创建输出图像区域
        output_container = QFrame()  # 创建输出区域容器
        # 设置输出区域样式：白色背景，圆角边框
        output_container.setStyleSheet("background-color: white; border-radius: 10px;")
        output_layout = QVBoxLayout(output_container)  # 创建输出区域的垂直布局
        # 创建输出图像显示控件
        self.output_display = ImageDisplay("输出图像", "处理后的图像将显示在这里")
        # 创建保存结果按钮
        save_btn = QPushButton("保存结果")
        # 设置按钮样式：绿色背景，白色文字，圆角等
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #19be6b;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #18b566;
            }
        """)
        save_btn.clicked.connect(self.saveResult)  # 连接按钮点击事件到saveResult方法
        output_layout.addWidget(self.output_display)  # 添加显示控件到布局
        output_layout.addWidget(save_btn)  # 添加按钮到布局

        # 7. 创建处理按钮区域
        process_container = QFrame()  # 创建处理按钮容器
        process_layout = QVBoxLayout(process_container)  # 创建垂直布局
        process_btn = QPushButton("处理")  # 创建处理按钮
        process_btn.setFixedSize(100, 40)  # 设置按钮固定大小
        # 设置处理按钮样式：深蓝色背景，白色文字，圆角等
        process_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e70d7;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1962c0;
            }
        """)
        process_btn.clicked.connect(self.processImage)  # 连接按钮点击事件到processImage方法
        process_layout.addWidget(process_btn)  # 添加按钮到布局

        # 8. 组装所有区域到主布局
        image_area.addWidget(input_container)  # 添加输入区域
        image_area.addWidget(process_container)  # 添加处理按钮区域
        image_area.addWidget(output_container)  # 添加输出区域
        layout.addLayout(image_area)  # 将整个图像处理区域添加到主布局

        # 9. 初始化图像变量
        self.input_image = None  # 存储输入图像
        self.output_image = None  # 存储输出图像

    def selectImage(self):
        """选择图片方法分析
        功能：打开文件选择对话框并加载选中的图片
        """
        # 打开文件选择对话框，返回选中的文件路径和使用的过滤器
        file_name, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口
            "选择图片",  # 对话框标题
            DEFAULT_IMAGE_DIR,  # 默认打开目录
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"  # 文件类型过滤器
        )

        # 如果选择了文件（用户没有取消）
        if file_name:
            # 调用加载图片方法
            self.loadImage(file_name)

    def loadImage(self, path):
        """加载图片方法分析
        参数:
            path: 图片文件路径
        功能：加载并显示图片
        """
        try:
            # 打开图片并转换为灰度图像
            self.input_image = Image.open(path).convert('L')
            # 在输入显示区域显示图片
            self.displayImage(self.input_image, self.input_display)
        except Exception as e:
            # 显示错误提示框
            QMessageBox.warning(self, "错误", f"图片加载失败: {str(e)}")

    def processImage(self):
        """图片处理方法分析
        功能：使用深度学习模型处理图片
        """
        # 检查是否已加载输入图片
        if self.input_image is None:
            QMessageBox.warning(self, "提示", "请先选择输入图片")
            return

        # 检查模型是否正确加载
        if self.model is None:
            QMessageBox.warning(self, "错误", "模型未正确加载")
            return

        try:
            # 预处理：将图片转换为张量，并添加batch维度
            input_tensor = self.transform(self.input_image).unsqueeze(0)
            # 将张量移动到指定设备（GPU/CPU）
            input_tensor = input_tensor.to(self.device)

            # 使用模型进行推理
            with torch.no_grad():  # 不计算梯度
                output_tensor = self.model(input_tensor)

            # 后处理：将输出张量转换回图像
            # 1. 移除批次维度并转移到CPU
            output_array = output_tensor.squeeze().cpu().numpy()
            # 2. 将值缩放到0-255范围内并转换为8位无符号整数
            output_array = (output_array * 255).clip(0, 255).astype(np.uint8)
            # 3. 创建PIL图像对象
            self.output_image = Image.fromarray(output_array)
            # 4. 显示处理后的图像
            self.displayImage(self.output_image, self.output_display)

        except Exception as e:
            # 显示错误提示框
            QMessageBox.critical(self, "错误", f"图片处理失败: {str(e)}")

    def saveResult(self):
        """保存结果方法分析
        功能：将处理后的图片保存到文件
        """
        # 检查是否有处理后的图片
        if self.output_image is None:
            QMessageBox.warning(self, "提示", "没有可保存的结果")
            return

        # 打开保存文件对话框
        file_name, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口
            "保存图片",  # 对话框标题
            "",  # 默认保存路径
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"  # 文件类型过滤器
        )

        # 如果选择了保存路径（用户没有取消）
        if file_name:
            try:
                # 保存图片
                self.output_image.save(file_name)
                # 显示成功提示框
                QMessageBox.information(self, "成功", "图片保存成功")
            except Exception as e:
                # 显示错误提示框
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def displayImage(self, img, display):
        """显示图片方法分析
        参数:
            img: PIL.Image对象，要显示的图片
            display: ImageDisplay对象，显示区域
        功能：将图片显示到指定的显示区域
        """
        try:
            # 将PIL图像转换为numpy数组
            img_array = np.array(img)
            # 获取图像尺寸
            height, width = img_array.shape
            # 计算每行字节数（对于灰度图像，等于宽度）
            bytes_per_line = width

            # 创建QImage对象
            # Format_Grayscale8表
            image = QImage(img_array.data,  # 图像数据
                           width,  # 图像宽度
                           height,  # 图像高度
                           bytes_per_line,  # 每行字节数
                           QImage.Format_Grayscale8)  # 图像格式

            # 将QImage转换为QPixmap
            pixmap = QPixmap.fromImage(image)

            # 缩放图像以适应显示区域，保持纵横比
            scaled_pixmap = pixmap.scaled(480, 480,  # 目标大小
                                          Qt.KeepAspectRatio,  # 保持纵横比
                                          Qt.SmoothTransformation)  # 使用平滑缩放

            # 在显示区域显示图像
            display.setPixmap(scaled_pixmap)

        except Exception as e:
            # 显示错误提示框
            QMessageBox.critical(self, "错误", f"图片显示失败: {str(e)}")


def main():
    """主函数分析
       功能：程序的入口点，创建并运行应用程序
       """
    # 创建QApplication实例
    # sys.argv传入命令行参数
    app = QApplication(sys.argv)

    # 创建主窗口实例
    window = XRayEnhancer()

    # 显示主窗口
    window.show()

    # 运行应用程序，进入事件循环
    # sys.exit确保程序正确退出
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()