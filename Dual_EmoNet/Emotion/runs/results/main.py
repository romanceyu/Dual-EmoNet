#!/usr/bin/env python3
"""
面部表情识别系统 - 主程序
这是一个基于PyQt5和深度学习的面部表情识别桌面应用程序。

作者: [您的名字]
日期: 2025-04-08
"""
import sys
import os
import logging
from PyQt5.QtWidgets import QApplication, QMessageBox

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入应用程序组件
from Emotion.main_window import MainWindow


def setup_logging():
    """设置日志记录"""
    # 确保logs目录存在
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )


def handle_exception(exc_type, exc_value, exc_traceback):
    """处理未捕获的异常"""
    if issubclass(exc_type, KeyboardInterrupt):
        # 如果是键盘中断，使用默认处理器
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 记录未捕获的异常
    logging.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))

    # 显示错误消息框
    error_msg = f"程序遇到了一个错误:\n{exc_type.__name__}: {exc_value}"
    QMessageBox.critical(None, "程序错误", error_msg)


def main():
    """应用程序入口点"""
    # 设置异常处理器
    sys.excepthook = handle_exception

    # 设置日志记录
    setup_logging()

    # 启动应用程序
    logging.info("正在启动面部表情识别系统...")
    app = QApplication(sys.argv)
    app.setApplicationName("面部表情识别系统")

    # 显示启动画面(可选)
    # 您可以添加一个自定义的启动屏幕图像到assets/splash.png
    """
    splash_pix = QPixmap("assets/splash.png")
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    """

    # 设置应用程序图标(如果有)
    """
    app_icon = QIcon("assets/icons/app_icon.png")
    app.setWindowIcon(app_icon)
    """

    # 启动主窗口
    logging.info("正在初始化主窗口...")
    window = MainWindow()
    window.show()

    # 关闭启动画面(如果使用了)
    # QTimer.singleShot(1000, splash.close)

    # 运行应用程序
    logging.info("面部表情识别系统已启动")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())