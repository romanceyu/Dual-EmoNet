import os
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QBuffer, QIODevice, QByteArray
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import datetime
import json


def cv_to_qt_image(cv_img):
    """
    将OpenCV图像(numpy数组)转换为QImage

    参数:
        cv_img: OpenCV格式的图像(numpy数组)

    返回:
        QImage对象
    """
    # 检查图像是否有效
    if cv_img is None or cv_img.size == 0:
        return None

    # 获取图像尺寸
    height, width, channels = cv_img.shape

    # OpenCV使用BGR顺序，需要转换为RGB
    if channels == 3:
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = channels * width
        return QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    elif channels == 4:
        # 如果是RGBA格式
        bytes_per_line = channels * width
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    else:
        # 单通道图像
        bytes_per_line = width
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)


def cv_to_qt_pixmap(cv_img, max_width=None, max_height=None):
    """
    将OpenCV图像转换为QPixmap，并可选择调整大小

    参数:
        cv_img: OpenCV格式的图像
        max_width: 最大宽度(可选)
        max_height: 最大高度(可选)

    返回:
        QPixmap对象
    """
    # 转换为QImage
    qt_img = cv_to_qt_image(cv_img)
    if qt_img is None:
        return QPixmap()

    # 转换为QPixmap
    pixmap = QPixmap.fromImage(qt_img)

    # 如果指定了最大尺寸，进行调整
    if max_width or max_height:
        if max_width and max_height:
            pixmap = pixmap.scaled(max_width, max_height,
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
        elif max_width:
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
        elif max_height:
            if pixmap.height() > max_height:
                pixmap = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)

    return pixmap


def save_results(parent_widget, results, image_data, prefix="emotion_result"):
    """
    保存分析结果(图像和数据)

    参数:
        parent_widget: 父窗口部件
        results: 包含分析结果的字典
        image_data: 标注后的图像数据
        prefix: 文件名前缀

    返回:
        bool: 保存成功返回True，否则返回False
    """
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{prefix}_{timestamp}"

    # 弹出保存文件对话框
    options = QFileDialog.Options()
    file_path, selected_filter = QFileDialog.getSaveFileName(
        parent_widget, "保存结果", default_name,
        "所有支持的格式 (*.png *.jpg *.json *.zip);;图像文件 (*.png *.jpg);;JSON数据 (*.json);;完整数据包 (*.zip)",
        options=options)

    if not file_path:
        return False

    try:
        # 根据所选类型处理
        base_path = os.path.splitext(file_path)[0]

        if "图像文件" in selected_filter:
            # 只保存图像
            cv2.imwrite(file_path, image_data)
            parent_widget.statusBar().showMessage(f"图像已保存到：{file_path}", 3000)

        elif "JSON数据" in selected_filter:
            # 只保存JSON数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            parent_widget.statusBar().showMessage(f"数据已保存到：{file_path}", 3000)

        elif "完整数据包" in selected_filter:
            # 保存为ZIP压缩包(包含图像和JSON)
            import zipfile

            # 确保路径有.zip扩展名
            if not file_path.lower().endswith('.zip'):
                file_path += '.zip'

            # 创建临时文件
            img_path = f"{base_path}_image.png"
            json_path = f"{base_path}_data.json"

            # 保存图像和JSON
            cv2.imwrite(img_path, image_data)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 创建ZIP文件
            with zipfile.ZipFile(file_path, 'w') as zipf:
                zipf.write(img_path, os.path.basename(img_path))
                zipf.write(json_path, os.path.basename(json_path))

            # 删除临时文件
            os.remove(img_path)
            os.remove(json_path)

            parent_widget.statusBar().showMessage(f"完整数据包已保存到：{file_path}", 3000)

        else:
            # 默认保存图像
            cv2.imwrite(file_path, image_data)

            # 同时保存JSON
            json_path = f"{base_path}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            parent_widget.statusBar().showMessage(f"结果已保存到：{file_path} 和 {json_path}", 3000)

        return True

    except Exception as e:
        QMessageBox.critical(parent_widget, "保存错误", f"保存结果时出错：{str(e)}")
        return False


def format_emotion_results(results):
    """
    格式化情绪分析结果为易读的文本

    参数:
        results: 情绪分析结果字典

    返回:
        str: 格式化的文本
    """
    faces = results.get("detected_faces", [])

    if not faces:
        return "未检测到人脸！"

    text = f"检测到 {len(faces)} 个人脸:\n\n"
    for i, face in enumerate(faces):
        emotion = face["emotion"]
        confidence = face["confidence"]

        text += f"人脸 #{i + 1}:\n"
        text += f"- 主要情绪: {emotion}\n"
        text += f"- 检测置信度: {confidence:.2f}\n"
        text += "- 情绪概率分布:\n"

        # 添加每种情绪的概率
        for emotion_name, score in face["emotion_scores"].items():
            # 添加星号标记当前情绪
            marker = " *" if emotion_name == emotion else ""
            text += f"  {emotion_name}: {score:.2f}{marker}\n"

        text += "\n"

    return text