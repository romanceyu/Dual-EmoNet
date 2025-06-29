import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QScrollArea, QSplitter,
                             QGridLayout, QGroupBox, QFrame, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QBuffer, QIODevice, QByteArray
import datetime


class ImageProcessThread(QThread):
    """处理图像的线程类，避免UI冻结"""
    finished = pyqtSignal(dict)  # 发送处理结果的信号
    progress = pyqtSignal(int)  # 发送进度的信号

    def __init__(self, emotion_recognizer, image):
        super().__init__()
        self.emotion_recognizer = emotion_recognizer
        self.image = image

    def run(self):
        """线程执行的主要方法"""
        try:
            # 模拟处理进度
            self.progress.emit(10)

            # 转换为OpenCV格式
            height, width, channel = self.image.shape
            self.progress.emit(30)

            # 调用模型进行预测
            result = self.emotion_recognizer.predict(self.image)
            self.progress.emit(90)

            # 发送结果信号
            self.finished.emit(result)
            self.progress.emit(100)

        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            self.finished.emit({"error": str(e)})


class ImageTab(QWidget):
    """图像分析标签页"""

    def __init__(self, emotion_recognizer):
        super().__init__()
        self.emotion_recognizer = emotion_recognizer
        self.current_image = None
        self.results = None
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 创建顶部按钮区域
        buttons_layout = QHBoxLayout()

        # "加载图像"按钮
        self.load_btn = QPushButton("加载图像...")
        self.load_btn.clicked.connect(self.loadImage)
        self.load_btn.setFixedHeight(40)
        buttons_layout.addWidget(self.load_btn)

        # "分析"按钮
        self.analyze_btn = QPushButton("分析表情")
        self.analyze_btn.clicked.connect(self.analyzeImage)
        self.analyze_btn.setEnabled(False)  # 初始时禁用
        self.analyze_btn.setFixedHeight(40)
        buttons_layout.addWidget(self.analyze_btn)

        # "保存结果"按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.saveResults)
        self.save_btn.setEnabled(False)  # 初始时禁用
        self.save_btn.setFixedHeight(40)
        buttons_layout.addWidget(self.save_btn)

        main_layout.addLayout(buttons_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 创建分割器，实现左右分栏布局
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧图像显示区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # 图像标签
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QLabel("请加载图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.image_scroll.setWidget(self.image_label)

        left_layout.addWidget(self.image_scroll)
        splitter.addWidget(left_widget)

        # 右侧分析结果区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 结果组
        results_group = QGroupBox("分析结果")
        results_layout = QVBoxLayout()

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

        # 设置初始分割比例
        splitter.setSizes([int(self.width() * 0.7), int(self.width() * 0.3)])

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

                # 显示图像
                self.displayImage(self.current_image)

                # 更新状态
                self.analyze_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.results_label.setText(f"已加载图像：{os.path.basename(file_path)}\n点击分析表情按钮以进行分析。")
                self.results = None

            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "错误", f"加载图像时出错：{str(e)}")

    def displayImage(self, cv_img):
        """在UI中显示OpenCV格式的图像"""
        # 转换为Qt格式
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        # OpenCV使用BGR顺序，需要转换为RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # 转换为QImage
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 调整大小以适应标签但保持纵横比
        pixmap = QPixmap.fromImage(convert_to_qt_format)

        # 显示图像
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def analyzeImage(self):
        """分析图像按钮的点击处理程序"""
        if self.current_image is None:
            return

        # 禁用按钮，显示进度条
        self.analyze_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # 创建并启动处理线程
        self.process_thread = ImageProcessThread(
            self.emotion_recognizer, self.current_image.copy())
        self.process_thread.finished.connect(self.handleResults)
        self.process_thread.progress.connect(self.updateProgress)
        self.process_thread.start()

    def updateProgress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def handleResults(self, results):
        """处理分析结果"""
        # 重新启用按钮
        self.analyze_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        # 检查是否有错误
        if "error" in results:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "处理错误", f"分析图像时出错：{results['error']}")
            return

        # 存储结果
        self.results = results

        # 显示标注后的图像
        self.displayImage(results["annotated_image"])

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
                from PyQt5.QtWidgets import QMessageBox
                reply = QMessageBox.question(self, '保存图像',
                                             '是否也保存标注后的图像？',
                                             QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)

                if reply == QMessageBox.Yes:
                    img_path = os.path.splitext(file_path)[0] + ".png"
                    cv2.imwrite(img_path, self.results["annotated_image"])

            # 显示成功消息
            self.parent().statusBar().showMessage(f"结果已保存到：{file_path}", 3000)

        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "保存错误", f"保存结果时出错：{str(e)}")