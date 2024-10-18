import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
import os

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

sam_checkpoint = "../sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

import math
def convert_segmentation_to_image(segmentation_array):
    return np.array(segmentation_array, dtype=np.uint8) * 255

def apply_mask_to_image(image, mask):
    mask_3d = np.expand_dims(mask, axis=-1)

    masked_image = image * mask_3d

    return masked_image

def plot_images_grid(images, images_per_row=4, size=(8, 8)):
    # 총 이미지 수와 행의 개수를 계산
    total_images = len(images)
    rows = math.ceil(total_images / images_per_row)

    # 플롯 생성
    fig, axes = plt.subplots(rows, images_per_row, figsize=size)
    axes = axes.flatten()  # 축을 1D 배열로 변환

    # 이미지를 축에 맞춰 그리기
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')  # 축 제거

    # 빈 칸을 숨기기
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



def main():
    image = "./img_2.png"
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.95,
        # stability_score_thresh=0.72,
        # crop_n_layers=1,
        # crop_n_points_downscale_factor=2,
        min_mask_region_area=900
    )
    masks = mask_generator.generate(image)

    image_np = np.array(image)
    seg_imgs = [apply_mask_to_image(image_np, mask['segmentation']) for mask in masks]

    plot_images_grid(seg_imgs, images_per_row=4, size=(100, 100))

    seg_imgs_dir = './classification_imgs/'

    for idx, img in enumerate(seg_imgs):
        sg_path = os.path.join(seg_imgs_dir, f'{idx + 1}.png')
        img = Image.fromarray(img.astype(np.uint8))
        img.save(sg_path)


if __name__ == '__main__':
    main()