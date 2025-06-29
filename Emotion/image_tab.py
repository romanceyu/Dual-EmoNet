# image_tab.py
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QScrollArea, QSplitter,
                             QGridLayout, QGroupBox, QFrame, QProgressBar,
                             QComboBox, QMessageBox, QToolBar)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QBuffer, QIODevice, QByteArray, QSize
import datetime


class ImageProcessThread(QThread):
    """处理图像的线程类，避免UI冻结"""
    finished = pyqtSignal(dict)  # 发送处理结果的信号
    progress = pyqtSignal(int)  # 发送进度的信号
    status = pyqtSignal(str)  # 发送处理状态的信号

    def __init__(self, emotion_recognizer, image):
        super().__init__()
        self.emotion_recognizer = emotion_recognizer
        self.image = image

    def run(self):
        """线程执行的主要方法"""
        try:
            # 转换为OpenCV格式
            self.status.emit("准备处理...")
            self.progress.emit(10)

            height, width, channel = self.image.shape
            self.status.emit("检测人脸...")
            self.progress.emit(20)

            # 详细反馈处理过程
            self.status.emit("对齐人脸...")
            self.progress.emit(40)

            self.status.emit("提取特征...")
            self.progress.emit(60)

            self.status.emit("分析情绪...")
            self.progress.emit(80)

            # 调用模型进行预测
            result = self.emotion_recognizer.predict(self.image)

            self.status.emit("生成结果...")
            self.progress.emit(95)

            # 发送结果信号
            self.finished.emit(result)
            self.status.emit("处理完成！")
            self.progress.emit(100)

        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            self.status.emit(f"错误: {str(e)}")
            self.finished.emit({"error": str(e)})


class ImageTab(QWidget):
    """图像分析标签页"""

    def __init__(self, emotion_recognizer):
        super().__init__()
        self.emotion_recognizer = emotion_recognizer
        self.current_image = None
        self.original_pixmap = None
        self.processed_pixmap = None
        self.results = None
        self.zoom_factor = 1.0
        self.current_mode = "fit"  # 默认适应窗口
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)  # 减少边距
        main_layout.setSpacing(5)  # 减少组件间距
        self.setLayout(main_layout)

        # 顶部控制区域 - 使用网格布局让按钮和控件更紧凑
        top_layout = QGridLayout()
        top_layout.setVerticalSpacing(5)
        top_layout.setHorizontalSpacing(10)

        # 第一行: 操作按钮
        self.load_btn = QPushButton("加载图像...")
        self.load_btn.clicked.connect(self.loadImage)
        self.load_btn.setFixedHeight(40)
        top_layout.addWidget(self.load_btn, 0, 0)

        self.analyze_btn = QPushButton("分析表情")
        self.analyze_btn.clicked.connect(self.analyzeImage)
        self.analyze_btn.setEnabled(False)  # 初始时禁用
        self.analyze_btn.setFixedHeight(40)
        top_layout.addWidget(self.analyze_btn, 0, 1)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.saveResults)
        self.save_btn.setEnabled(False)  # 初始时禁用
        self.save_btn.setFixedHeight(40)
        top_layout.addWidget(self.save_btn, 0, 2)

        # 第二行: 缩放控制和状态标签放在一行
        zoom_label = QLabel("缩放:")
        zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_layout.addWidget(zoom_label, 1, 0)

        self.zoom_combo = QComboBox()
        self.zoom_combo.addItem("适应窗口", "fit")
        self.zoom_combo.addItem("原始大小", "original")
        self.zoom_combo.addItem("放大", "zoom_in")
        self.zoom_combo.addItem("缩小", "zoom_out")
        self.zoom_combo.setCurrentIndex(0)
        self.zoom_combo.currentIndexChanged.connect(self.changeZoomMode)
        self.zoom_combo.setEnabled(False)
        top_layout.addWidget(self.zoom_combo, 1, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        top_layout.addWidget(self.status_label, 1, 2)
        # 添加清除按钮
        self.clear_btn = QPushButton("清除图像")
        self.clear_btn.clicked.connect(self.clearImage)
        self.clear_btn.setEnabled(False)  # 初始时禁用
        top_layout.addWidget(self.clear_btn, 1, 3)

        main_layout.addLayout(top_layout)

        # 进度条 - 减小高度使其更紧凑
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 创建分割器，实现左右分栏布局
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)  # 使用拉伸因子1使其占据大部分空间

        # 左侧图像显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)  # 消除内边距

        # 图像滚动区域
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignCenter)

        # 图像容器和标签
        self.image_container = QWidget()
        self.image_container_layout = QVBoxLayout(self.image_container)
        self.image_container_layout.setAlignment(Qt.AlignCenter)
        self.image_container_layout.setContentsMargins(0, 0, 0, 0)  # 消除内边距

        self.image_label = QLabel("请加载图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.image_container_layout.addWidget(self.image_label)

        self.image_scroll.setWidget(self.image_container)
        left_layout.addWidget(self.image_scroll)
        splitter.addWidget(left_widget)

        # 右侧分析结果区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 5, 0)  # 减少内边距

        # 结果组
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(5, 5, 5, 5)  # 减少内边距

        # 结果标签
        self.results_label = QLabel("加载图像并点击分析表情按钮以获取结果...")
        self.results_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_label.setWordWrap(True)

        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setWidget(self.results_label)

        results_layout.addWidget(results_scroll)
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        splitter.addWidget(right_widget)

        # 设置初始分割比例 (左侧区域占更多空间)
        splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])

    def clearImage(self):
        """清除当前加载的图像和分析结果"""
        # 重置图像相关变量
        self.current_image = None
        self.original_pixmap = None
        self.processed_pixmap = None
        self.results = None

        # 重置UI状态
        self.image_label.setText("请加载图像...")
        self.image_label.setPixmap(QPixmap())  # 清除图像
        self.results_label.setText("加载图像并点击分析表情按钮以获取结果...")

        # 禁用相关按钮
        self.analyze_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.zoom_combo.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # 重置缩放
        self.zoom_factor = 1.0
        self.zoom_combo.setCurrentIndex(0)  # 重置为"适应窗口"

        # 更新状态
        self.status_label.setText("图像已清除")

    def loadImage(self):
        """加载图像按钮的点击处理程序"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*)",
            options=options)

        if file_path:
            try:
                # 使用OpenCV加载图像
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise Exception("无法读取图像文件")

                # 转换为Qt格式并存储
                self.original_pixmap = self.convertCvToPixmap(self.current_image)

                # 重置缩放设置
                self.zoom_factor = 1.0
                self.zoom_combo.setCurrentIndex(0)  # 默认适应窗口
                self.zoom_combo.setEnabled(True)

                # 显示图像
                self.displayPixmap(self.original_pixmap)

                # 更新状态
                self.analyze_btn.setEnabled(True)
                self.clear_btn.setEnabled(True)  # 启用清除按钮
                self.save_btn.setEnabled(False)
                self.results_label.setText(f"已加载图像：{os.path.basename(file_path)}\n点击分析表情按钮以进行分析。")
                self.results = None
                self.processed_pixmap = None
                self.status_label.setText(f"已加载: {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像时出错：{str(e)}")

    def convertCvToPixmap(self, cv_img):
        """将OpenCV格式的图像转换为QPixmap"""
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        # OpenCV使用BGR顺序，需要转换为RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # 转换为QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # 转换为QPixmap
        return QPixmap.fromImage(qt_image)

    def displayPixmap(self, pixmap):
        """根据当前模式显示QPixmap"""
        if pixmap is None:
            return

        if self.current_mode == "fit":
            # 适应窗口显示
            scaled_pixmap = self.scalePixmapToFit(pixmap)
            self.image_label.setPixmap(scaled_pixmap)
        elif self.current_mode == "original":
            # 原始大小
            self.image_label.setPixmap(pixmap)
        else:
            # 应用缩放因子
            scaled_width = int(pixmap.width() * self.zoom_factor)
            scaled_height = int(pixmap.height() * self.zoom_factor)
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

        # 设置标签大小
        self.image_label.adjustSize()

    def scalePixmapToFit(self, pixmap):
        """缩放图像以适应视图大小"""
        scroll_size = self.image_scroll.size()
        # 计算合适的缩放比例
        scaled = pixmap.scaled(
            scroll_size.width() - 20,  # 留一些边距
            scroll_size.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        return scaled

    def changeZoomMode(self, index):
        """处理缩放模式变化"""
        mode = self.zoom_combo.itemData(index)
        self.current_mode = mode

        # 根据模式调整缩放因子
        if mode == "zoom_in":
            self.zoom_factor *= 1.2
        elif mode == "zoom_out":
            self.zoom_factor *= 0.8
        elif mode == "original":
            self.zoom_factor = 1.0
        elif mode == "fit":
            # 适应窗口时不使用缩放因子
            pass

        # 显示当前活跃的图片（原始或处理后）
        if self.processed_pixmap and self.results:
            self.displayPixmap(self.processed_pixmap)
        elif self.original_pixmap:
            self.displayPixmap(self.original_pixmap)

    def resizeEvent(self, event):
        """窗口大小变化时重新调整图像"""
        super().resizeEvent(event)
        # 如果是适应窗口模式，则在窗口大小改变时重新调整图像
        if self.current_mode == "fit":
            if self.processed_pixmap and self.results:
                self.displayPixmap(self.processed_pixmap)
            elif self.original_pixmap:
                self.displayPixmap(self.original_pixmap)

    def analyzeImage(self):
        """分析图像按钮的点击处理程序"""
        if self.current_image is None:
            return

        # 禁用按钮，显示进度条
        self.analyze_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.status_label.setText("准备分析...")

        # 创建并启动处理线程
        self.process_thread = ImageProcessThread(
            self.emotion_recognizer, self.current_image.copy())
        self.process_thread.finished.connect(self.handleResults)
        self.process_thread.progress.connect(self.updateProgress)
        self.process_thread.status.connect(self.updateStatus)
        self.process_thread.start()

    def updateProgress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def updateStatus(self, message):
        """更新状态消息"""
        self.status_label.setText(message)

    def handleResults(self, results):
        """处理分析结果"""
        # 重新启用按钮
        self.analyze_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 检查是否有错误
        if "error" in results:
            QMessageBox.critical(self, "处理错误", f"分析图像时出错：{results['error']}")
            self.status_label.setText("处理失败")
            return

        # 存储结果
        self.results = results

        # 存储和显示标注后的图像
        self.processed_pixmap = self.convertCvToPixmap(results["annotated_image"])
        self.displayPixmap(self.processed_pixmap)

        # 启用保存按钮
        self.save_btn.setEnabled(True)

        # 格式化结果显示
        faces = results["detected_faces"]
        if not faces:
            result_text = "未检测到人脸！"
        else:
            result_text = f"检测到 {len(faces)} 个人脸:\n\n"
            for i, face in enumerate(faces):
                emotion = face["emotion"]
                confidence = face["confidence"]

                result_text += f"人脸 #{i + 1}:\n"
                result_text += f"- 主要情绪: {emotion}\n"
                result_text += f"- 检测置信度: {confidence:.2f}\n"
                result_text += "- 情绪概率分布:\n"

                # 添加每种情绪的概率
                for emotion_name, score in face["emotion_scores"].items():
                    result_text += f"  {emotion_name}: {score:.2f}\n"

                result_text += "\n"

        self.results_label.setText(result_text)
        self.status_label.setText("分析完成")

    def saveResults(self):
        """保存结果按钮的点击处理程序"""
        if not self.results:
            return

        # 弹出保存文件对话框
        options = QFileDialog.Options()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"emotion_result_{timestamp}"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", default_name,
            "图像文件 (*.png *.jpg);;文本报告 (*.txt);;所有文件 (*)",
            options=options)

        if not file_path:
            return

        try:
            # 根据文件扩展名决定保存方式
            ext = os.path.splitext(file_path)[1].lower()

            if ext in ['.png', '.jpg', '.jpeg']:
                # 保存图像
                cv2.imwrite(file_path, self.results["annotated_image"])

                # 同时保存文本报告
                txt_path = os.path.splitext(file_path)[0] + ".txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_label.text())

            elif ext in ['.txt']:
                # 保存文本报告
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_label.text())

                # 询问是否同时保存图像
                reply = QMessageBox.question(self, '保存图像',
                                             '是否也保存标注后的图像？',
                                             QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)

                if reply == QMessageBox.Yes:
                    img_path = os.path.splitext(file_path)[0] + ".png"
                    cv2.imwrite(img_path, self.results["annotated_image"])

            # 使用安全的方法显示状态消息
            try:
                # 尝试获取MainWindow的状态栏
                main_window = self.window()
                if hasattr(main_window, 'statusBar') and callable(main_window.statusBar):
                    main_window.statusBar().showMessage(f"结果已保存到：{file_path}", 3000)
                else:
                    # 如果没有状态栏，使用自己的状态标签
                    self.status_label.setText(f"结果已保存到：{os.path.basename(file_path)}")
            except Exception:
                # 如果出错，至少显示在自己的状态标签中
                self.status_label.setText(f"结果已保存到：{os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存结果时出错：{str(e)}")
            self.status_label.setText("保存失败")
