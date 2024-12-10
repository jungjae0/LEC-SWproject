import cv2
import numpy as np
import timm

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random

# ----------- Condition Classification With ViT Model
class VITModel(nn.Module):
    def __init__(self, num_classes):
        super(VITModel, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, image):
        return self.model(image)


def with_vit(image, seg_masks):
    condition_classification_model = VITModel(num_classes=3)
    condition_classification_model_path = r"weights/condition_vit_model.pth"
    condition_classification_model.load_state_dict(torch.load(condition_classification_model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    condition_classification_model.to(device)
    condition_classification_model.eval()

    preprocess = A.Compose([
        A.Resize(224, 224),
        ToTensorV2()])

    image_np = np.array(image)
    condition_dict = {'0': ["Fresh", (0, 255, 0)], '1': ["Spoiled", (255, 0, 0)], "2": ['Too Much Water', (0, 0, 255)]}

    for idx, mask in enumerate(seg_masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            masked_image = image_np[y:y + h, x:x + w] * mask[y:y + h, x:x + w, None]  # RGB 채널 유지

            input_tensor = preprocess(image=masked_image)['image'].unsqueeze(0).type(torch.float32).to(device)

            with torch.no_grad():
                output = condition_classification_model(input_tensor)

            probabilities = F.softmax(output, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            result = str(predicted.item())

            threshold = 0.5

            if max_prob.item() < threshold:
                continue
            else:

                label = condition_dict[result][0]
                color = condition_dict[result][1]

                cv2.drawContours(image_np, [contour], -1, color, 2)

                text = f"{label}"

                cv2.putText(image_np, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_np


# ----------- Condition Classification With Color
def check_color_dominance_from_mask(image, mask):
    # 이미지 HSV 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    # 초록색 계열 범위 정의
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])

    # 갈색~노란색 계열 범위 정의
    brown_yellow_lower = np.array([10, 100, 20])
    brown_yellow_upper = np.array([40, 255, 255])

    # 초록색과 갈색~노란색 마스크 생성
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    brown_yellow_mask = cv2.inRange(hsv_image, brown_yellow_lower, brown_yellow_upper)

    # 마스크 영역에 해당하는 픽셀 수 계산
    green_pixels = cv2.countNonZero(green_mask[mask > 0])
    brown_yellow_pixels = cv2.countNonZero(brown_yellow_mask[mask > 0])

    # 결과 반환
    if green_pixels > brown_yellow_pixels:
        return "fresh"
    elif brown_yellow_pixels > green_pixels:
        return "too much water"
    else:
        return None


def with_color(image, seg_masks):
    annotated_image = image.copy()

    for mask in seg_masks:
        label = check_color_dominance_from_mask(image, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            color = (0, 255, 0) if label == "fresh" else (0, 140, 255) if label == "too much water" else (255, 255, 255)
            cv2.drawContours(annotated_image, [contour], -1, color, 2)
            text_position = tuple(contour[0][0])
            cv2.putText(annotated_image, label if label else "Unknown", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1)

    return annotated_image



# ----------- Random Labeling
def random_classification():
    """
    랜덤으로 "fresh" 또는 "too much water"를 반환
    """
    return random.choice(["fresh", "too much water"])


def with_random_color(image, seg_masks):
    """
    HSV를 기반으로 하지 않고 랜덤으로 라벨을 지정하여 마스크를 처리.
    """
    annotated_image = image.copy()

    for mask in seg_masks:
        # 랜덤 분류
        label = random_classification()

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 라벨에 따라 색상 결정
            color = (0, 255, 0) if label == "fresh" else (0, 140, 255)
            cv2.drawContours(annotated_image, [contour], -1, color, 2)

            # 텍스트 위치 지정 및 라벨 추가
            text_position = tuple(contour[0][0])
            cv2.putText(annotated_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated_image