import os
import cv2
from PIL import Image
import numpy as np
import gradio as gr

import plant_segmentation, condition_classification, leaf_segmentation

def run(img, plant_segmentation_method, condition_classification_method):
    # Convert Gradio image (numpy array) to OpenCV format
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    seg_masks = []

    if plant_segmentation_method == "CLIPSeg":
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plant_object = plant_segmentation.with_clip(img_pil)
        seg_masks = leaf_segmentation.with_sam(plant_object)
        print("Completed leaf segmentation with CLIPSeg")

    elif plant_segmentation_method == "OnlySAM":
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_masks = leaf_segmentation.with_sam(image_rgb)
        print("Completed leaf segmentation with OnlySAM")

    elif plant_segmentation_method == "YOLOSeg":
        plant_object = plant_segmentation.with_yolo(img)
        seg_masks = leaf_segmentation.with_sam(plant_object)
        print("Completed leaf segmentation with YOLOSeg")

    elif plant_segmentation_method == "GroundedSAM":
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plant_object = plant_segmentation.with_groundedsam(img_pil)
        seg_masks = leaf_segmentation.with_sam(plant_object)
        print("Completed leaf segmentation with GroundedSAM")

    if condition_classification_method == "ViT":
        classification_img = condition_classification.with_vit(img, seg_masks)
        # print("Completed condition classification with ViT")
        # classification_img = condition_classification.with_random_color(img, seg_masks)
    elif condition_classification_method == "Color":
        classification_img = condition_classification.with_color(img, seg_masks)
        # print("Completed condition classification with Color")
        # classification_img = condition_classification.with_random_color(img, seg_masks)


    return cv2.cvtColor(classification_img, cv2.COLOR_BGR2RGB)

# Gradio Interface Setup
def gradio_interface(image, segmentation_method, classification_method):
    result = run(image, segmentation_method, classification_method)
    return result


# 예제 이미지 경로 설정
ex_images = [
    [os.path.join(os.path.dirname(__file__), "images/sample1.png")],
    [os.path.join(os.path.dirname(__file__), "images/sample2.jpeg")]
]

with gr.Blocks() as demo:
    gr.Markdown("# Plant Segmentation and Condition Classification")

    # 인터페이스 컴포넌트 정의
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Plant Image")
        result_image = gr.Image(label="Segmented and Classified Image")
    segmentation_method = gr.Radio(
        ["CLIPSeg", "OnlySAM", "YOLOSeg", "GroundedSAM"],
        label="Select Plant Segmentation Method"
    )
    classification_method = gr.Radio(
        ["Color", "ViT"],
        label="Select Condition Classification Method"
    )
    process_button = gr.Button("Process Image")

    process_button.click(
        gradio_interface,
        inputs=[image_input, segmentation_method, classification_method],
        outputs=result_image
    )


    gr.Examples(
        examples=ex_images,
        inputs=[image_input, segmentation_method, classification_method]
    )

demo.launch()

