import os

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import plant_segmentation, condition_classification, leaf_segmentation

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def run(img_path, plant_segmentation_method, conditon_classification_method):
    img = cv2.imread(img_path)
    seg_masks = []

    if plant_segmentation_method == "CLIPSeg":
        img = Image.open(img_path)
        plant_object = plant_segmentation.with_clip(img)
        seg_masks = leaf_segmentation.with_sam(plant_object)
        print("Completed leaf segmentation with CLIPSeg")

    elif plant_segmentation_method == "OnlySAM":
        img = cv2.imread(img_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_masks = leaf_segmentation.with_sam(image)
        print("Completed leaf segmentation with OnlySAM")

    elif plant_segmentation_method == "YOLOSeg":
        img = cv2.imread(img_path)
        plant_object = plant_segmentation.with_yolo(img)
        seg_masks = leaf_segmentation.with_sam(plant_object)

        print("Completed leaf segmentation with YOLOSeg")

    elif plant_segmentation_method == "GroundedSAM":
        img = Image.open(img_path).convert("RGB")
        plant_object = plant_segmentation.with_groundedsam(img)
        seg_masks = leaf_segmentation.with_sam(plant_object)
        print("Completed leaf segmentation with GroundedSAM")

    if conditon_classification_method == "ViT":
        # --- classification with ViT
        classification_img = condition_classification.with_vit(img, seg_masks)
        print("Completed condition classification with ViT")
        return classification_img

    elif conditon_classification_method == "Color":
        classification_img = condition_classification.with_color(img, seg_masks)
        print("Completed condition classification with Color")

        return classification_img

def main():
    img_path = "./samples/03.jpg"
    # plant_segmentation_method = "YOLOSeg" # CLIPSeg, OnlySAM, YOLOSeg, GroundedSAM
    # conditon_classification_method = "ViT" # ViT, Color

    img_path = input(rf"식물 이미지 경로 입력(ex. C:\Users\001.png): ")
    # plant_segmentation_method = input("식물 Detection & Segmentation에 사용할 모델 입력 [CLIPSeg, OnlySAM, YOLOSeg, GroundedSAM]: ")
    # conditon_classification_method = input("식물 상태 판단에 사용할 방법 입력[Color, ViT]: ")
    while True:
        plant_segmentation_method = input(
            "식물 Detection & Segmentation에 사용할 모델 입력 [CLIPSeg, OnlySAM, YOLOSeg, GroundedSAM]: ")
        if plant_segmentation_method in ["CLIPSeg", "OnlySAM", "YOLOSeg", "GroundedSAM"]:
            break
        else:
            print("잘못된 입력입니다. 올바른 모델을 선택해주세요 [CLIPSeg, OnlySAM, YOLOSeg, GroundedSAM].")

    # 식물 상태 판단에 사용할 방법 입력
    while True:
        conditon_classification_method = input("식물 상태 판단에 사용할 방법 입력 [Color, ViT]: ")
        if conditon_classification_method in ["Color", "ViT"]:
            break
        else:
            print("잘못된 입력입니다. 올바른 방법을 선택해주세요 [Color, ViT].")

    result_file_name = input("결과 이미지 저장 파일명 입력(ex. result1): ")

    classification_img = run(img_path, plant_segmentation_method, conditon_classification_method)

    result_save_dir = "./results"
    os.makedirs(result_save_dir, exist_ok=True)
    result_save_path = os.path.join(result_save_dir, f"{result_file_name}.png")
    cv2.imwrite(result_save_path, classification_img)

    print(f"Completed save result: {result_save_path}")

if __name__ == '__main__':
    main()