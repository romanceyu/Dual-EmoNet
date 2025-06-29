import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QComboBox, QGroupBox,
                             QGridLayout, QCheckBox, QSpinBox, QSplitter,
                             QScrollArea, QMessageBox, QFrame, QApplication)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
import sys
import os

class VideoProcessThread(QThread):
    """处理视频的线程类，避免UI冻结"""
    frame_ready = pyqtSignal(dict)  # 发送处理后的帧
    error = pyqtSignal(str)  # 发送错误信息
    finished = pyqtSignal()  # 发送完成信号

    def __init__(self, emotion_recognizer, source_type='file', source_path=None):
        super().__init__()
        self.emotion_recognizer = emotion_recognizer
        self.source_type = source_type  # 'file' 或 'camera'
        self.source_path = source_path  # 视频文件路径或摄像头索引
        self.running = False
        self.mutex = QMutex()  # 线程同步锁
        self.skip_frames = 1  # 跳过的帧数，用于降低CPU使用率
        self.show_probabilities = True  # 是否显示概率
        self.frame_count = 0  # 帧计数

    def run(self):
        """线程执行的主要方法"""
        try:
            # 设置视频源
            if self.source_type == 'file' and self.source_path:
                cap = cv2.VideoCapture(self.source_path)
            elif self.source_type == 'camera':
                cap = cv2.VideoCapture(0)  # 使用默认摄像头
            else:
                self.error.emit("无效的视频源")
                return

            # 检查视频是否成功打开
            if not cap.isOpened():
                self.error.emit("无法打开视频源")
                return

            # 设置标志为运行中
            self.mutex.lock()
            self.running = True
            self.mutex.unlock()

            # 处理视频帧
            while self.isRunning():
                # 读取帧
                ret, frame = cap.read()

                # 如果读取失败或视频结束
                if not ret:
                    # 如果是文件视频结束，发送完成信号
                    if self.source_type == 'file':
                        self.finished.emit()
                    break

                # 帧计数递增
                self.frame_count += 1

                # 根据跳帧设置决定是否处理当前帧
                if (self.frame_count % (self.skip_frames + 1)) != 0:
                    continue

                # 调用情绪识别器处理帧
                result = self.emotion_recognizer.predict(frame)

                # 获取处理后的帧并发送
                self.frame_ready.emit(result)

            # 释放视频捕获资源
            cap.release()

        except Exception as e:
            self.error.emit(f"处理视频时出错: {str(e)}")
        finally:
            # 发送完成信号
            self.finished.emit()

    def isRunning(self):
        """检查线程是否应该继续运行"""
        self.mutex.lock()
        running = self.running
        self.mutex.unlock()
        return running

    def stop(self):
        """停止线程"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()

    def setSkipFrames(self, value):
        """设置跳过的帧数"""
        self.skip_frames = value

    def setShowProbabilities(self, value):
        """设置是否显示概率"""
        self.show_probabilities = value


class VideoTab(QWidget):
    """视频分析标签页"""

    def __init__(self, emotion_recognizer, status_bar=None):
        super().__init__()
        self.emotion_recognizer = emotion_recognizer
        self.status_bar = status_bar  # 状态栏引用
        self.video_thread = None
        self.current_frame = None
        self.results = []  # 存储分析结果历史
        self.recording = False
        self.output_video = None
        self.recording_path = None  # 存储录制视频的路径
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 创建顶部按钮区域
        top_layout = QGridLayout()  # 使用网格布局更加灵活

        # 视频源选择区域 - 第一行
        self.source_combo = QComboBox()
        self.source_combo.addItem("摄像头")
        self.source_combo.addItem("视频文件")
        self.source_combo.currentIndexChanged.connect(self.sourceChanged)
        top_layout.addWidget(QLabel("视频源:"), 0, 0)
        top_layout.addWidget(self.source_combo, 0, 1)

        # "打开视频"按钮
        self.open_btn = QPushButton("打开视频...")
        self.open_btn.clicked.connect(self.openVideo)
        self.open_btn.setVisible(False)  # 初始设为摄像头模式
        self.open_btn.setFixedHeight(40)
        top_layout.addWidget(self.open_btn, 0, 2)

        # 添加一个空白填充
        top_layout.setColumnStretch(3, 1)

        # "开始/停止"按钮 - 第一行后面
        self.start_stop_btn = QPushButton("开始分析")
        self.start_stop_btn.clicked.connect(self.toggleVideoAnalysis)
        self.start_stop_btn.setFixedHeight(40)
        top_layout.addWidget(self.start_stop_btn, 0, 4)

        # 第二行 - 录制相关控制
        # "录制准备"按钮 (新增)
        self.prepare_record_btn = QPushButton("准备录制...")
        self.prepare_record_btn.clicked.connect(self.prepareRecording)
        self.prepare_record_btn.setEnabled(False)  # 初始禁用
        self.prepare_record_btn.setFixedHeight(40)
        top_layout.addWidget(self.prepare_record_btn, 1, 0, 1, 2)  # 跨两列

        # "开始/停止录制"按钮
        self.record_btn = QPushButton("开始录制")
        self.record_btn.clicked.connect(self.toggleRecording)
        self.record_btn.setEnabled(False)  # 初始禁用
        self.record_btn.setFixedHeight(40)
        top_layout.addWidget(self.record_btn, 1, 2)

        # "导出结果"按钮
        self.export_btn = QPushButton("导出分析报告")
        self.export_btn.clicked.connect(self.exportResults)
        self.export_btn.setEnabled(False)  # 初始禁用
        self.export_btn.setFixedHeight(40)
        top_layout.addWidget(self.export_btn, 1, 4)

        # 添加到主布局
        main_layout.addLayout(top_layout)

        # 视频格式选择（新增）
        self.format_layout = QHBoxLayout()
        self.format_layout.addWidget(QLabel("录制格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("MP4 (.mp4)", "mp4")
        self.format_combo.addItem("AVI (.avi)", "avi")
        self.format_layout.addWidget(self.format_combo)
        self.format_layout.addStretch()

        # 状态标签
        self.status_label = QLabel("就绪")
        self.format_layout.addWidget(self.status_label)

        main_layout.addLayout(self.format_layout)

        # 创建分割器，实现上下分栏布局
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # 上部视频显示区域
        top_widget = QWidget()
        top_widget_layout = QVBoxLayout(top_widget)

        # 视频标签
        self.video_label = QLabel("点击开始分析按钮以启动视频分析...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc;")
        self.video_label.setMinimumHeight(480)

        top_widget_layout.addWidget(self.video_label)
        splitter.addWidget(top_widget)

        # 下部控制和结果区域
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)

        # 左侧控制面板
        control_group = QGroupBox("分析控制")
        control_layout = QGridLayout()

        # 跳帧控制
        control_layout.addWidget(QLabel("处理帧速率:"), 0, 0)
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(0, 10)
        self.skip_frames_spin.setValue(1)
        self.skip_frames_spin.setToolTip("值越高，处理越快但帧率越低")
        self.skip_frames_spin.valueChanged.connect(self.updateSkipFrames)
        control_layout.addWidget(self.skip_frames_spin, 0, 1)

        # 显示控制
        self.show_prob_check = QCheckBox("显示情绪概率")
        self.show_prob_check.setChecked(True)
        self.show_prob_check.stateChanged.connect(self.updateShowProbabilities)
        control_layout.addWidget(self.show_prob_check, 1, 0, 1, 2)

        # 其他控制项可以在此添加...

        control_group.setLayout(control_layout)
        bottom_layout.addWidget(control_group)

        # 右侧结果面板
        results_group = QGroupBox("实时分析结果")
        results_layout = QVBoxLayout()

        # 结果标签
        self.results_label = QLabel("尚未开始分析...")
        self.results_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_label.setWordWrap(True)

        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setWidget(self.results_label)

        results_layout.addWidget(results_scroll)
        results_group.setLayout(results_layout)

        # 右侧结果面板占据更多空间
        bottom_layout.addWidget(results_group, 2)

        splitter.addWidget(bottom_widget)

        # 设置初始分割比例
        splitter.setSizes([int(self.height() * 0.7), int(self.height() * 0.3)])

    def sourceChanged(self, index):
        """视频源改变时的处理函数"""
        # 索引0是摄像头，索引1是视频文件
        if index == 0:  # 摄像头
            self.open_btn.setVisible(False)
        else:  # 视频文件
            self.open_btn.setVisible(True)

        # 如果正在分析，停止它
        self.stopVideo()
        self.start_stop_btn.setText("开始分析")
        self.prepare_record_btn.setEnabled(False)
        self.record_btn.setEnabled(False)

        # 重置录制状态
        if self.recording:
            self.stopRecording()

    def openVideo(self):
        """打开视频文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.wmv);;所有文件 (*)",
            options=options)

        if file_path:
            self.video_path = file_path
            self.video_label.setText(f"已选择视频: {os.path.basename(file_path)}\n点击开始分析按钮以开始分析")
            self.status_label.setText(f"已加载: {os.path.basename(file_path)}")

    def toggleVideoAnalysis(self):
        """开始/停止视频分析"""
        if self.video_thread and self.video_thread.isRunning():
            self.stopVideo()
            self.start_stop_btn.setText("开始分析")
            self.prepare_record_btn.setEnabled(False)
            self.record_btn.setEnabled(False)

            # 如果正在录制，也停止录制
            if self.recording:
                self.stopRecording()
        else:
            self.startVideo()
            self.start_stop_btn.setText("停止分析")
            self.prepare_record_btn.setEnabled(True)

    def startVideo(self):
        """启动视频分析"""
        # 清空结果历史
        self.results = []
        self.export_btn.setEnabled(False)

        # 根据选择创建不同的视频源
        source_type = 'camera' if self.source_combo.currentIndex() == 0 else 'file'
        source_path = None if source_type == 'camera' else self.video_path if hasattr(self,
                                                                                      'video_path') else None

        # 如果选择了文件但没有路径
        if source_type == 'file' and not source_path:
            QMessageBox.warning(self, "提示", "请先选择视频文件")
            return

        # 创建并启动视频处理线程
        self.video_thread = VideoProcessThread(
            self.emotion_recognizer, source_type, source_path)
        self.video_thread.frame_ready.connect(self.updateFrame)
        self.video_thread.error.connect(self.handleError)
        self.video_thread.finished.connect(self.handleVideoFinished)

        # 设置跳帧和显示选项
        self.video_thread.setSkipFrames(self.skip_frames_spin.value())
        self.video_thread.setShowProbabilities(self.show_prob_check.isChecked())

        # 启动线程
        self.video_thread.start()
        self.status_label.setText("分析中...")

    def stopVideo(self):
        """停止视频分析"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()  # 等待线程结束
            self.status_label.setText("分析已停止")

    def updateFrame(self, result):
        """更新视频帧和结果"""
        # 保存当前帧和结果
        self.current_frame = result["annotated_image"].copy()
        self.results.append({
            "timestamp": datetime.datetime.now(),
            "faces": result["detected_faces"]
        })

        # 如果结果列表太长，只保留最近的100个
        if len(self.results) > 100:
            self.results = self.results[-100:]

        # 如果正在录制，写入帧
        if self.recording and self.output_video:
            self.output_video.write(self.current_frame)

        # 显示帧
        self.displayFrame(self.current_frame)

        # 更新结果显示
        self.updateResults(result)

        # 启用导出按钮
        if not self.export_btn.isEnabled() and self.results:
            self.export_btn.setEnabled(True)

    def displayFrame(self, cv_img):
        """在UI中显示视频帧"""
        # 转换为Qt格式
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        # OpenCV使用BGR顺序，需要转换为RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # 转换为QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 调整大小以适应标签但保持纵横比
        pixmap = QPixmap.fromImage(qt_image)

        # 获取标签大小
        label_size = self.video_label.size()

        # 缩放图像以适应标签
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 显示图像
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

    def updateResults(self, result):
        """更新结果显示"""
        faces = result["detected_faces"]

        # 格式化结果显示
        if not faces:
            result_text = "当前帧未检测到人脸"
        else:
            result_text = f"当前帧检测到 {len(faces)} 个人脸:\n\n"

            # 统计情绪分布
            emotion_counts = {}
            for face in faces:
                emotion = face["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # 显示总体统计
            result_text += "情绪分布:\n"
            for emotion, count in emotion_counts.items():
                result_text += f"- {emotion}: {count}人\n"

            result_text += "\n详细信息:\n"
            for i, face in enumerate(faces):
                emotion = face["emotion"]
                confidence = face["confidence"]

                result_text += f"人脸 #{i + 1}: {emotion} (置信度: {confidence:.2f})\n"

        self.results_label.setText(result_text)

    def handleError(self, error_msg):
        """处理视频处理错误"""
        QMessageBox.critical(self, "视频处理错误", error_msg)
        self.stopVideo()
        self.start_stop_btn.setText("开始分析")
        self.status_label.setText(f"错误: {error_msg}")

    def handleVideoFinished(self):
        """处理视频处理完成"""
        # 如果是自然结束而不是手动停止
        if self.video_thread and not self.video_thread.isRunning():
            self.start_stop_btn.setText("开始分析")
            self.prepare_record_btn.setEnabled(False)
            self.record_btn.setEnabled(False)

            # 如果是文件视频结束，显示统计信息
            if self.source_combo.currentIndex() == 1:
                # 计算总体统计信息
                total_faces = sum(len(result["faces"]) for result in self.results)
                emotion_counts = {}

                for result in self.results:
                    for face in result["faces"]:
                        emotion = face["emotion"]
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                # 显示统计信息
                stats_text = "视频分析完成\n\n"
                stats_text += f"总处理帧数: {len(self.results)}\n"
                stats_text += f"检测到的人脸总数: {total_faces}\n\n"
                stats_text += "情绪统计:\n"

                for emotion, count in emotion_counts.items():
                    percentage = (count / total_faces) * 100 if total_faces > 0 else 0
                    stats_text += f"- {emotion}: {count} ({percentage:.1f}%)\n"

                self.results_label.setText(stats_text)
                self.status_label.setText("分析完成")

                # 如果正在录制，停止录制
                if self.recording:
                    self.stopRecording()

    def prepareRecording(self):
        """准备录制 - 设置保存路径但不立即开始"""
        try:
            # 获取保存路径
            options = QFileDialog.Options()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 根据选择的格式设置默认文件后缀
            format_ext = self.format_combo.currentData()
            default_name = f"emotion_video_{timestamp}.{format_ext}"

            file_path, _ = QFileDialog.getSaveFileName(
                self, "设置录制文件", default_name,
                f"{format_ext.upper()} 视频 (*.{format_ext});;所有文件 (*)", options=options)

            if not file_path:
                return

            # 保存路径，但不立即创建写入器
            self.recording_path = file_path
            self.record_btn.setEnabled(True)
            self.status_label.setText(f"准备录制到: {os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.critical(self, "准备录制错误", f"设置录制路径时出错：{str(e)}")
            self.status_label.setText("录制准备失败")

    def toggleRecording(self):
        """开始/停止录制"""
        if not self.recording:
            # 检查是否有设置录制路径
            if not hasattr(self, 'recording_path') or not self.recording_path:
                self.prepareRecording()
                # 如果用户取消了保存对话框
                if not hasattr(self, 'recording_path') or not self.recording_path:
                    return

            # 开始录制
            try:
                if self.current_frame is not None:
                    # 获取当前帧的尺寸
                    height, width, _ = self.current_frame.shape

                    # 根据文件扩展名选择编码器
                    filename, ext = os.path.splitext(self.recording_path)
                    if ext.lower() == '.mp4':
                        # 使用MP4编码器
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
                    else:
                        # 默认使用AVI编码器
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')

                    # 创建视频写入器
                    self.output_video = cv2.VideoWriter(self.recording_path, fourcc, 20.0, (width, height))

                    if not self.output_video.isOpened():
                        raise Exception("无法创建视频写入器")

                    self.recording = True
                    self.record_btn.setText("停止录制")
                    self.status_label.setText(f"录制中: {os.path.basename(self.recording_path)}")
                    self.prepare_record_btn.setEnabled(False)  # 录制时禁用准备按钮
                else:
                    QMessageBox.warning(self, "警告", "没有可录制的视频帧！")
            except Exception as e:
                QMessageBox.critical(self, "录制错误", f"开始录制时出错：{str(e)}")
                self.status_label.setText("录制失败")
        else:
            # 停止录制
            self.stopRecording()

    def stopRecording(self):
        """停止录制"""
        if self.output_video:
            self.output_video.release()
            self.output_video = None

        self.recording = False
        self.record_btn.setText("开始录制")
        self.prepare_record_btn.setEnabled(True)  # 停止录制后重新启用准备按钮

        if hasattr(self, 'recording_path') and self.recording_path:
            self.status_label.setText(f"录制已保存: {os.path.basename(self.recording_path)}")

            # 显示成功消息
            QMessageBox.information(self, "录制完成", f"视频已保存到：{self.recording_path}")

            # 清除录制路径，必须重新准备
            self.recording_path = None
            self.record_btn.setEnabled(False)

    def exportResults(self):
        """导出分析结果"""
        if not self.results:
            return

        try:
            # 打开保存对话框
            options = QFileDialog.Options()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出结果", f"emotion_analysis_{timestamp}.txt",
                "文本文件 (*.txt);;CSV文件 (*.csv);;所有文件 (*)", options=options)

            if not file_path:
                return

            # 根据文件类型决定导出格式
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.csv':
                # 导出为CSV格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 写入标题行
                    f.write("时间戳,人脸ID,情绪,置信度\n")

                    # 写入每个结果
                    for result in self.results:
                        timestamp = result["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        faces = result["faces"]

                        for i, face in enumerate(faces):
                            emotion = face["emotion"]
                            confidence = face["confidence"]
                            f.write(f"{timestamp},{i + 1},{emotion},{confidence:.2f}\n")
            else:
                # 导出为文本格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("面部表情分析结果\n")
                    f.write("=" * 50 + "\n\n")

                    # 写入总体统计
                    f.write(f"分析总帧数: {len(self.results)}\n")

                    # 计算情绪分布
                    all_emotions = []
                    for result in self.results:
                        for face in result["faces"]:
                            all_emotions.append(face["emotion"])

                    if all_emotions:
                        emotion_counts = {}
                        for emotion in all_emotions:
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                        f.write("\n情绪分布统计:\n")
                        for emotion, count in emotion_counts.items():
                            percentage = (count / len(all_emotions)) * 100
                            f.write(f"- {emotion}: {count} ({percentage:.1f}%)\n")

                    # 写入详细结果
                    f.write("\n\n详细帧分析:\n" + "=" * 50 + "\n")

                    for i, result in enumerate(self.results):
                        timestamp = result["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        faces = result["faces"]

                        f.write(f"\n帧 #{i + 1} - {timestamp}\n")
                        if not faces:
                            f.write("  未检测到人脸\n")
                        else:
                            f.write(f"  检测到 {len(faces)} 个人脸:\n")
                            for j, face in enumerate(faces):
                                emotion = face["emotion"]
                                confidence = face["confidence"]
                                f.write(f"  人脸 #{j + 1}: {emotion} (置信度: {confidence:.2f})\n")

            # 显示成功消息在状态标签中
            self.status_label.setText(f"结果已导出: {os.path.basename(file_path)}")
            QMessageBox.information(self, "导出成功", f"结果已导出到：{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出结果时出错：{str(e)}")
            self.status_label.setText("导出失败")

    def updateSkipFrames(self, value):
        """更新跳帧设置"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.setSkipFrames(value)

    def updateShowProbabilities(self, state):
        """更新是否显示概率设置"""
        show = (state == Qt.Checked)
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.setShowProbabilities(show)

    def resizeEvent(self, event):
        """窗口大小改变时更新视频显示"""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.displayFrame(self.current_frame)