
# MainWindow代码
import os
import sys
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QApplication, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QAction,
                             QMenuBar, QMenu, QFileDialog, QStatusBar)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QSettings
# sys.path.append(os.path.split(sys.path[0])[0])
# 注意：根据需要调整导入路径
from image_tab import ImageTab
from video_tab import VideoTab
from emotion_recognizer import EmotionRecognizer


class MainWindow(QMainWindow):
    """
    面部表情识别系统的主窗口类
    """

    def __init__(self):
        super().__init__()

        # 初始化设置
        self.settings = QSettings("EmotionDetection", "FacialEmotionRecognizer")

        # 尝试初始化情绪识别器
        try:
            self.emotion_recognizer = EmotionRecognizer()
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            QMessageBox.warning(self, "模型加载失败",
                                f"无法加载情绪识别模型：{str(e)}\n请检查模型路径和依赖项！")

        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        # 设置窗口基本属性
        self.setWindowTitle("面部表情识别系统")
        self.setMinimumSize(1200, 900)

        # 恢复窗口大小和位置
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # 创建菜单栏
        self.createMenuBar()

        # 创建主要布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页控件
        self.tab_widget = QTabWidget()

        # 创建两个标签页
        self.image_tab = ImageTab(self.emotion_recognizer)
        self.video_tab = VideoTab(self.emotion_recognizer)

        # 添加标签页到标签页控件
        self.tab_widget.addTab(self.image_tab, "图像分析")
        self.tab_widget.addTab(self.video_tab, "视频分析")

        # 添加标签页控件到主布局
        main_layout.addWidget(self.tab_widget)

        # 确保状态栏被创建
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

        # 设置样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QComboBox {
                padding: 5px 10px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QScrollArea {
                border: 1px solid #cccccc;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 2px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)

        # 如果模型未加载，禁用分析功能
        if not self.model_loaded:
            self.image_tab.setEnabled(False)
            self.video_tab.setEnabled(False)
            self.statusBar().showMessage("警告：模型未加载，功能受限")

    def createMenuBar(self):
        """创建应用的菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 加载模型动作
        load_model_action = QAction('加载模型...', self)
        load_model_action.triggered.connect(self.loadModel)
        file_menu.addAction(load_model_action)

        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 帮助菜单
        help_menu = menubar.addMenu('帮助')

        # 关于动作
        about_action = QAction('关于', self)
        about_action.triggered.connect(self.showAboutDialog)
        help_menu.addAction(about_action)

    def loadModel(self):
        """加载模型对话框"""
        # 打开文件对话框选择模型目录
        model_dir = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if model_dir:
            # 如果选择了目录，显示选择模型文件对话框
            model_file, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", model_dir, "模型文件 (*.pth)")

            if model_file:
                try:
                    # 尝试重新加载模型
                    self.emotion_recognizer = EmotionRecognizer(
                        model_dir=os.path.dirname(model_file),
                        model_file=os.path.basename(model_file)
                    )
                    self.model_loaded = True

                    # 更新标签页中的模型引用
                    self.image_tab.emotion_recognizer = self.emotion_recognizer
                    self.video_tab.emotion_recognizer = self.emotion_recognizer

                    # 启用标签页
                    self.image_tab.setEnabled(True)
                    self.video_tab.setEnabled(True)

                    self.statusBar().showMessage(f"成功加载模型: {model_file}")
                    QMessageBox.information(self, "模型加载", "模型加载成功！")

                except Exception as e:
                    QMessageBox.critical(self, "模型加载失败", f"无法加载模型：{str(e)}")

    def showAboutDialog(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于面部表情识别系统",
                          "面部表情识别系统 v1.0\n\n"
                          "基于PyQt5和深度学习的实时面部表情识别应用程序。\n"
                          "使用Dual_EmoNet模型进行情绪分类。\n\n"
                          "支持图像和视频分析，提供详细的情绪概率分布。")

    def closeEvent(self, event):
        """应用关闭时保存设置"""
        # 关闭视频处理线程(如果有)
        if hasattr(self.video_tab, 'stopVideo'):
            self.video_tab.stopVideo()

        # 保存窗口大小和位置
        self.settings.setValue("geometry", self.saveGeometry())

        # 接受关闭事件
        event.accept()