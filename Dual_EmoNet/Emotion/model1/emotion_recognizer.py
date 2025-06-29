import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from retinaface import RetinaFace
from approach.DA_EmoNet2 import DA_EmoNet  # 确保此模块在Python路径中

class EmotionRecognizer:
    """封装情绪识别模型的类"""

    def __init__(self, model_dir="/root/autodl-tmp/ResEmoteNet/runs/20250312-202814/models",
                 model_file="best_test_model.pth"):
        """
        初始化情绪识别器

        参数:
            model_dir: 模型文件目录
            model_file: 模型文件名
        """
        self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载模型
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在，请检查路径！")

        self.model = DA_EmoNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 定义图像预处理转换
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"EmotionRecognizer已初始化，使用{self.device}设备")

    def predict(self, image):
        """
        预测输入图像的情绪

        参数:
            image: numpy数组，BGR格式的图像(OpenCV默认格式)

        返回:
            预测结果字典，包含:
            - detected_faces: 检测到的人脸列表，每个包含位置和情绪
            - annotated_image: 标注了结果的图像
        """
        # 复制输入图像以避免修改原始图像
        annotated_image = image.copy()

        # 转换为RGB图像用于RetinaFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用RetinaFace检测人脸
        faces = RetinaFace.detect_faces(rgb_image)

        # 如果未检测到人脸，返回空结果
        if not isinstance(faces, dict):
            return {"detected_faces": [], "annotated_image": annotated_image}

        detected_faces = []

        # 处理每个检测到的人脸
        for face_idx in faces:
            face_data = faces[face_idx]

            # 获取边界框坐标
            x1, y1, x2, y2 = [int(coord) for coord in face_data["facial_area"]]
            w, h = x2 - x1, y2 - y1

            # 获取置信度得分
            score = face_data["score"]

            # 只处理高置信度的检测结果
            if score > 0.8:
                # 裁剪人脸区域
                face_crop = image[y1:y2, x1:x2]

                # 预测情绪
                emotion_scores = self._detect_emotion(face_crop)
                max_index = np.argmax(emotion_scores)
                max_emotion = self.emotions[max_index]

                # 将此人脸添加到结果列表
                face_result = {
                    "position": (x1, y1, x2, y2),
                    "confidence": score,
                    "emotion": max_emotion,
                    "emotion_scores": dict(zip(self.emotions, emotion_scores))
                }
                detected_faces.append(face_result)

                # 在图像上标注结果
                self._annotate_face(annotated_image, face_result)

        return {
            "detected_faces": detected_faces,
            "annotated_image": annotated_image
        }

    def _detect_emotion(self, face_image):
        """
        检测裁剪后的人脸图像中的情绪

        参数:
            face_image: 人脸区域的BGR格式图像

        返回:
            各情绪的概率值列表
        """
        # 将BGR转换为RGB
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

        # 转换图像为模型输入格式并进行推理
        img_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)

        scores = probabilities.cpu().numpy().flatten()
        rounded_scores = [float(round(score, 2)) for score in scores]
        return rounded_scores

    def _annotate_face(self, image, face_result):
        """
        在图像上标注人脸和情绪结果

        参数:
            image: 要标注的图像
            face_result: 人脸分析结果
        """
        x1, y1, x2, y2 = face_result["position"]
        emotion = face_result["emotion"]
        confidence = face_result["confidence"]

        # 设置文本显示参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)  # 绿色
        thickness = 2
        line_type = cv2.LINE_AA

        # 绘制人脸边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示情绪标签
        cv2.putText(image, emotion, (x1, y1 - 10), font, font_scale,
                    font_color, thickness, line_type)

        # 显示置信度
        cv2.putText(image, f"Conf: {confidence:.2f}", (x1, y2 + 20), font,
                    font_scale, (255, 0, 0), thickness, line_type)

        # 显示各情绪的概率值
        emotion_scores = face_result["emotion_scores"]
        y_offset = y1
        for emotion, score in emotion_scores.items():
            y_offset += 20
            if y_offset > y2:  # 如果文本超出人脸框，放到右侧
                y_offset = y1
                x_offset = x2 + 10
            else:
                x_offset = x1

            emotion_text = f"{emotion}: {score:.2f}"
            cv2.putText(image, emotion_text, (x_offset, y_offset), font,
                        0.5, (255, 255, 0), 1, line_type)